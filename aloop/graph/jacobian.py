# -*- coding: utf-8 -*-
"""v2.3.3 P2-10: Batched linear Jacobian / residual evaluator (structure-aware fast path).

Background: build_multibreak_residual's J(x) uses numerical forward differences, and for each breakpoint k it redoes
one DAG propagation (O(d) node evaluations). Cascade large-scale models (2000/5000blk) have breakpoints reaching
998/2498, and J(0) measured at 2000blk takes 345s -- this is the absolute end-to-end bottleneck.

Method: for each node (including breakpoints), probe whether its output is *linear* in all predecessor inputs:
    f(preds) = sum_j w[j]*pred[j] + b
The probe is completed with a small number of random inputs (superposition verification). If the whole graph is linear, vectorize the propagation as
a single numpy operation on a (d+1, n) matrix, obtaining all perturbation results at once -> J is read off directly.

Effect (measured target): 2000blk J(0) 345s -> ~0.2s, r(0) 0.35s -> ~0.02s.
Nonlinear models (DeadZone/Relay/LUT/Quantizer) fail the probe -> fall back to the original numerical difference path.
"""
import numpy as np

from ..loopeval import _isfin


def probe_linearity(n, adj, node_fn, topo, breaks, atol=1e-6):
    """Probe whether each node's output is linear in its predecessor inputs.

    Args:
        n: number of nodes
        adj: (n,n) adjacency matrix
        node_fn: block evaluation function node_fn(u, {pred:val}) -> float
        topo: non-breakpoint topological order
        breaks: list of breakpoints

    Returns:
        (weights, biases, all_linear, eval_nodes)
          weights[u] = {pred_idx: coef}, biases[u] = constant term
          all_linear: whether the whole graph is linear (if False, the result is unusable)
          eval_nodes: set of nodes that need evaluation (topo + breaks)
    """
    bk = set(breaks)
    all_nodes = list(topo) + [v for v in breaks]
    weights = {}
    biases = {}
    # v2.3.8: prebuild predecessor adjacency list, replacing O(n^2) preds lookup.
    # 5000blk probe_linearity reduced from ~3.6s to ~0.4s.
    _pred_adj = {u: [] for u in range(n)}
    _r, _c = np.nonzero(adj)
    for _j, _u in zip(_r.tolist(), _c.tolist()):
        _pred_adj[_u].append(_j)

    def safe_eval(u, vals):
        try:
            v = float(node_fn(u, vals))
            return v if _isfin(v) else 0.0
        except Exception:
            raise ValueError("nonlinear-or-error")

    for u in all_nodes:
        preds = _pred_adj[u]
        if not preds:
            try:
                biases[u] = safe_eval(u, {})
            except Exception:
                return {}, {}, False, []
            weights[u] = {}
            continue
        # f(0)
        try:
            b0 = safe_eval(u, {j: 0.0 for j in preds})
        except Exception:
            return {}, {}, False, []
        ws = {}
        ok = True
        for j in preds:
            vals = {p: 0.0 for p in preds}
            vals[j] = 1.0
            try:
                fj = safe_eval(u, vals)
            except Exception:
                ok = False
                break
            ws[j] = fj - b0
        if not ok:
            return {}, {}, False, []
        # Superposition verification (all-1 input)
        try:
            f1 = safe_eval(u, {j: 1.0 for j in preds})
        except Exception:
            return {}, {}, False, []
        est = b0 + sum(ws.values())
        if abs(f1 - est) > atol * max(1.0, abs(f1)):
            return {}, {}, False, []
        # ---- v2.3.3 fix: homogeneity verification (2x input).
        # For single-predecessor nodes (preds has only 1 entry), superposition verification degenerates into an identity:
        #   any function f can be written as f(1)=w*1+b, so the "all-1 superposition" cannot expose nonlinearity.
        # Example: f(x)=sin(x) with a single predecessor is judged linear (sin is approximately linear at small test amplitudes),
        # causing fast-J to produce an incorrect Jacobian for nonlinearity at zero crossings.
        # Add a homogeneity test: f(2*e_j)==2*f(e_j)-b0 and f(all-2)==b0+2*sum(w).
        # Linear functions hold at any amplitude; nonlinear ones (sin/DeadZone/LUT/Quantizer) usually
        # expose differences at amplitude 2 (sin(2)=0.909 vs 2*sin(1)=1.683).
        try:
            for j in preds:
                vals2 = {p: 0.0 for p in preds}
                vals2[j] = 2.0
                f2j = safe_eval(u, vals2)
                if abs(f2j - (2.0 * ws[j] + b0)) > atol * max(1.0, abs(f2j)):
                    return {}, {}, False, []
            f2all = safe_eval(u, {j: 2.0 for j in preds})
            est2 = b0 + 2.0 * sum(ws.values())
            if abs(f2all - est2) > atol * max(1.0, abs(f2all)):
                return {}, {}, False, []
        except Exception:
            return {}, {}, False, []
        # ---- v2.3.3 hardening: random superposition verification (deterministic RNG).
        # Fixed test points (all-0/all-1/all-2) may happen to land on special inputs (e.g. symmetry/zero crossings).
        # Use 2 sets of random vectors u,v to verify additivity f(u+v)==f(u)+f(v)-f(0).
        # Linear functions satisfy this for any input (floating-point error ~1e-16 << atol); nonlinear functions almost
        # certainly expose it at random amplitudes (probability ~1), further eliminating fast-J misjudgments.
        try:
            _rng = np.random.RandomState(0)
            for _ in range(2):
                uvec = {j: float(_rng.uniform(-2.0, 2.0)) for j in preds}
                vvec = {j: float(_rng.uniform(-2.0, 2.0)) for j in preds}
                fu = safe_eval(u, uvec)
                fv = safe_eval(u, vvec)
                fuv = safe_eval(u, {j: uvec[j] + vvec[j] for j in preds})
                if abs(fuv - (fu + fv - b0)) > atol * max(1.0, abs(fuv)):
                    return {}, {}, False, []
        except Exception:
            return {}, {}, False, []
        weights[u] = ws
        biases[u] = b0
    return weights, biases, True, all_nodes


def build_fast_rJ(n, adj, node_fn, topo, breaks, weights, biases,
                  eps=1e-6):
    """Build batched evaluation of r(x) and J(x) (pure numpy, single propagation).

    Returns:
        (fast_r, fast_J) has the same semantics as the original build_multibreak_residual.
    """
    d = len(breaks)
    bk = {v: k for k, v in enumerate(breaks)}
    # Nodes that need propagation: non-breakpoint topological order
    prop = topo
    # The breakpoint residual also needs phi (the breakpoint's output); the breakpoint is known in the matrix (=x), but the residual
    # r_k = x_k - phi_k; phi_k = sum_j w[v][j]*full[:,j] + b[v]
    eval_breaks = list(breaks)

    def fast_r(x):
        xa = np.asarray(x, dtype=float).ravel()
        if xa.shape[0] != d:
            raise ValueError("dim mismatch")
        full = np.zeros((d + 1, n))
        full[0, breaks] = xa
        # Perturbation rows: row k+1 perturbs only breakpoint k by eps, other breakpoints remain at xa
        for k in range(d):
            full[k + 1, breaks] = xa
            full[k + 1, breaks[k]] = xa[k] + eps
        for u in prop:
            if u in bk:
                continue
            acc = np.full(d + 1, biases[u])
            for j, w in weights[u].items():
                acc += w * full[:, j]
            full[:, u] = acc
        # Residuals (for all rows)
        rmat = np.zeros((d + 1, d))
        for k, v in enumerate(eval_breaks):
            acc = np.full(d + 1, biases[v])
            for j, w in weights[v].items():
                acc += w * full[:, j]
            rmat[:, k] = full[:, v] - acc
        return rmat[0, :], rmat  # r(x), perturbation residual matrix (d+1, d)

    def fast_J(x):
        r0, rmat = fast_r(x)
        # rmat[i+1, j] = residual j under perturbed breakpoint i = dr_j/dx_i; standard convention
        # J[i][j] = dr_i/dx_j -> transpose
        return ((rmat[1:, :] - r0) / eps).T

    def fast_r_only(x):
        r0, _ = fast_r(x)
        return r0

    return fast_r_only, fast_J
