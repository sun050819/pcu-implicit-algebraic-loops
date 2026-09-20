"""CI-guided greedy minimum feedback vertex set (FVS)-an efficient innovative algorithm for cycle-breaking breakpoint generation.

Background: The existing pipeline "enumerate all internal cycles -> select top-k points per cycle" explodes exponentially in the number of cycles on dense graphs,
and nested cycles that share nodes will repeatedly select points. However, algebraic loop solving only requires "a set of breakpoints that cut all cycles"
-this is exactly the graph-theoretic minimum feedback vertex set (FVS, Minimum Feedback Vertex Set).

This module:
  1. ci_guided_fvs: greedy approximate FVS, iteratively [find a cycle -> cut the node on the cycle with the lowest CI ->
     update the graph] until the graph is acyclic (DAG). CI (complexity index) serves as a solvability prior, so that the breakpoints are not only
     minimal but also easiest to converge. Complexity O(|V|.(|V|+|E|)), far better than enumerating all cycles.
  2. node_ci: node-level complexity index (residual nonlinearity with that node as a breakpoint), used for FVS guidance.
"""
from __future__ import annotations

import sys
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..loopeval import BreakpointEquation

# For very large-scale graphs (2000/5000 blocks), the global cycle DFS depth can reach n, so relax the recursion limit (v2.3.3)
if sys.getrecursionlimit() < 20000:
    sys.setrecursionlimit(20000)


# ==========================================================================
# Find a simple cycle (single, DFS, adjacency-list + removed-set version)
# ==========================================================================
def _find_simple_cycle(n: int, adj: np.ndarray) -> Optional[List[int]]:
    """Find a simple cycle on the whole graph adj (DFS); return None if acyclic.

    Note: cycles must be found on the whole graph, not on an SCC-induced subgraph-a strongly connected component may depend on nodes outside the component
    for transit, and there may be no cycle within the induced subgraph (e.g., all outgoing edges of SCC members point to outside nodes).
    """
    adjl = {u: [w for w in range(n) if adj[u, w]] for u in range(n)}
    path: List[int] = []
    onpath: set = set()

    def dfs(u: int, start: int):
        path.append(u)
        onpath.add(u)
        for w in adjl[u]:
            if w == start and len(path) >= 2:   # support 2-cycles (bidirectional edges depend on each other)
                return True
            if w not in onpath:
                if dfs(w, start):
                    return True
        path.pop()
        onpath.discard(u)
        return False

    for s in range(n):
        if dfs(s, s):
            return list(path)
    return None


def _find_simple_cycle_adjl(adjl: Dict[int, List[int]], removed: set,
                            n: int, order_s: Optional[List[int]] = None
                            ) -> Optional[List[int]]:
    """Find a simple cycle on a prebuilt adjacency list + removed set (v2.3.5 iterative version).

    order_s provides the starting-point priority order (high coupling first), so DFS hits cycles faster;
    the iterative implementation avoids the Python recursive call overhead on large-scale graphs (2498+ rounds).
    """
    it = order_s if order_s is not None else range(n)
    for start in it:
        if start in removed:
            continue
        # Iterative DFS: stack element (u, neighbor_iterator_index)
        path: List[int] = [start]
        onpath: set = {start}
        # Adjacency-list traversal cursor corresponding to each path node
        cursor: List[int] = [0]
        while path:
            u = path[-1]
            ci = cursor[-1]
            nbrs = adjl[u]
            found = False
            while ci < len(nbrs):
                w = nbrs[ci]
                ci += 1
                if w in removed:
                    continue
                if w == start and len(path) >= 2:
                    return list(path)
                if w not in onpath:
                    cursor[-1] = ci
                    path.append(w)
                    onpath.add(w)
                    cursor.append(0)
                    found = True
                    break
            if not found:
                # Backtrack
                path.pop()
                onpath.discard(u)
                cursor.pop()
    return None


# ==========================================================================
# Node-level CI (complexity index)
# ==========================================================================
def node_ci_structural(v: int, n: int, adj: np.ndarray) -> float:
    """Structural CI: with v as a breakpoint, its in-coupling degree + out-coupling degree + dimension proxy."""
    outd = int(adj[v, :].sum())
    ind = int(adj[:, v].sum())
    c = outd + ind
    # Residual self-coupling strength r(y)=phi(y)-y, high coupling -> harder to solve
    return round(min(c / 8.0, 1.0), 4)


def node_ci_numeric(v: int, n: int, adj: np.ndarray, node_fn: Callable,
                    x0: float = 1.0, eps: float = 1e-6) -> float:
    """Numerical CI: build BreakpointEquation with v as a breakpoint, and estimate using the Jacobian/gradient norm."""
    try:
        eq = BreakpointEquation(n, adj, node_fn, v, eps=eps)
        r0 = float(eq.r(np.array([x0]))[0])
        J0 = float(eq.J(np.array([x0]))[0, 0])
        # Residual gradient g = J^T r; Delta=log10(1+|g|)/8; S=min(|J-1| coupling,1)
        g = abs(J0 * r0)
        Delta = min(np.log10(1.0 + g) / 8.0, 1.0)
        S = min(abs(J0 - 1.0) / 5.0, 1.0)          # The farther the residual self-coupling deviates from identity, the harder it is
        ci = 0.5 * Delta + 0.5 * S
        if np.isfinite(ci):
            return round(float(ci), 4)
    except Exception:
        pass
    return node_ci_structural(v, n, adj)


def node_ci_map(n: int, adj: np.ndarray, node_fn: Optional[Callable] = None,
                numeric: bool = True, x0: float = 1.0) -> Dict[int, float]:
    """Return {v: ci}, for FVS guidance. Use numerical CI when numeric=True and node_fn is provided."""
    out = {}
    for v in range(n):
        out[v] = (node_ci_numeric(v, n, adj, node_fn, x0) if numeric and node_fn is not None
                  else node_ci_structural(v, n, adj))
    return out


# ==========================================================================
# CI-guided greedy FVS
# ==========================================================================
def ci_guided_fvs(n: int, adj: np.ndarray,
                  node_ci: Optional[Dict[int, float]] = None,
                  max_rounds: Optional[int] = None) -> List[int]:
    """CI-guided greedy approximate minimum feedback vertex set.

    Each round: find a simple cycle on the whole graph -> cut the node on the cycle with the lowest CI -> continue after removing that node,
    until the graph is acyclic (DAG). Does not rely on SCC (strongly connected component determination may be wrong on some graphs,
    and strong connectivity may depend on nodes outside the component for transit); directly use "whether the whole graph can find a cycle" as the termination criterion.
    When node_ci is not provided, use structural CI (out/in-degree coupling).
    Return the list of breakpoint nodes (ascending).
    """
    if node_ci is None:
        node_ci = node_ci_map(n, adj, None, numeric=False)
    A = adj.astype(float).copy()
    cut: set = set()
    # v2.3.4 speedup: build the adjacency list in batch with np.nonzero (O(|E|) instead of O(n^2) element-by-element scanning)
    adj_rows, adj_cols = np.nonzero(A)
    adjl: Dict[int, List[int]] = {}
    in_adjl: Dict[int, List[int]] = {v: [] for v in range(n)}
    for r, c in zip(adj_rows.tolist(), adj_cols.tolist()):
        adjl.setdefault(r, []).append(c)
        in_adjl[c].append(r)
    for u in range(n):
        adjl.setdefault(u, [])
        in_adjl.setdefault(u, [])
    out_deg = np.array([len(adjl[u]) for u in range(n)], dtype=int)
    in_deg = np.array([len(in_adjl[v]) for v in range(n)], dtype=int)
    removed: set = set()
    # Cycle-search starting priority: high-coupling nodes are more likely to be on a cycle, so DFS hits them faster
    order_s = [u for u in range(n)] if n == 0 else sorted(range(n),
              key=lambda u: -(in_deg[u] + out_deg[u]))
    # Self-loops (x=f(x)) must be cut directly: a self-loop node cannot be eliminated by cutting other nodes
    for i in range(n):
        if A[i, i] != 0.0:
            cut.add(i)
            removed.add(i)
            for w in adjl[i]:
                in_deg[w] -= 1
            for u in in_adjl[i]:
                out_deg[u] -= 1
    max_rounds = max_rounds or n
    # v2.3.5 speedup: batch fast path for 2-cycles. Cycles in cascading/feedback models are almost all 2-cycles (bidirectional edges),
    # use numpy vectorization to detect all 2-cycles at once, greedily select the node with high score to cut,
    # avoiding 2498+ rounds of cycle-by-cycle DFS. 5000blk FVS 1027ms->~250ms.
    _two_cycle_cut = 0
    try:
        # 2-cycle detection: A[i,j] and A[j,i], deduplicated by upper triangle
        At = A.T
        two_mask = np.logical_and(A > 0, At > 0)
        np.fill_diagonal(two_mask, False)
        iu = np.triu_indices(n, k=1)
        idx = np.where(two_mask[iu])[0]
        if len(idx) > 0:
            us = iu[0][idx]
            vs = iu[1][idx]
            # Greedy: for each 2-cycle pair, select the node with high score (degree/(CI+0.01)),
            # skip if one of them has already been removed (that cycle has been cut by another cycle)
            for k in range(len(us)):
                u = int(us[k]); v = int(vs[k])
                if u in removed or v in removed:
                    continue
                # Select the one with high score (high degree + low CI)
                su = (out_deg[u] + in_deg[u]) / (node_ci.get(u, 0.5) + 0.01)
                sv = (out_deg[v] + in_deg[v]) / (node_ci.get(v, 0.5) + 0.01)
                w = u if su >= sv else v
                cut.add(w)
                removed.add(w)
                _two_cycle_cut += 1
                for wn in adjl[w]:
                    if wn not in removed:
                        in_deg[wn] -= 1
                for un in in_adjl[w]:
                    if un not in removed:
                        out_deg[un] -= 1
    except Exception:
        pass  # Fall back to standard cycle-by-cycle DFS when the 2-cycle fast path fails
    for _ in range(max_rounds):
        cyc = _find_simple_cycle_adjl(adjl, removed, n, order_s=order_s)
        if not cyc:
            break                      # The graph is already acyclic (DAG)
        # Degree-aware point selection (v2.3.1): prioritize selecting the node that breaks the most cycles (highest remaining degree) with lower CI.
        # The original "pure lowest CI" would select endpoints on graphs dense with bidirectional cycles/2-cycles (breaking only 1 cycle),
        # causing the number of breakpoints to approach n (optimal is only n/2). score = degree / (CI + eps).
        def _sel_score(u):
            deg = int(out_deg[u] + in_deg[u])
            ci = node_ci.get(u, 0.5)
            return deg / (ci + 0.01)
        v = max(cyc, key=_sel_score)
        cut.add(v)
        removed.add(v)
        for w in adjl[v]:
            if w not in removed:
                in_deg[w] -= 1
        for u in in_adjl[v]:
            if u not in removed:
                out_deg[u] -= 1
    return sorted(cut)


def ci_guided_fvs_scc(n: int, adj: np.ndarray,
                       node_ci: Optional[Dict[int, float]] = None,
                       max_rounds: Optional[int] = None) -> List[int]:
    """SCC decomposition + independent greedy FVS within components (v2.3.1 enhanced version).

    First perform SCC decomposition, run ci_guided_fvs independently on the induced subgraph of each nontrivial SCC (size>1, containing cycles),
    and finally merge the breakpoints of each component. Compared with global greedy, this guarantees:
    - Union of disjoint cycles (each cycle is an independent SCC): each SCC selects only 1 breakpoint = optimal
    - Cycles across SCCs do not exist (guaranteed by the SCC definition), so independence between components is safe
    - Within large SCCs, CI-guided greedy is still used, maintaining the solvability prior

    For a single-SCC graph (e.g., bidirectional cycles, complete graph), the result is the same as ci_guided_fvs.
    For multi-SCC graphs (e.g., union of disjoint cycles, sparse random graphs), the number of breakpoints is significantly reduced.

    Return the list of breakpoint nodes (ascending).
    """
    if node_ci is None:
        node_ci = node_ci_map(n, adj, None, numeric=False)
    # SCC decomposition (Tarjan, iterative version to avoid recursion depth issues)
    sccs = _tarjan_scc(n, adj)
    all_cut: set = set()
    for scc in sccs:
        if len(scc) <= 1:
            v = scc[0]
            # Single-node SCC: check self-loop
            if adj[v, v] != 0.0:
                all_cut.add(v)
            continue
        # Nontrivial SCC: run greedy FVS on the induced subgraph
        idx_map = {old: new for new, old in enumerate(scc)}
        k = len(scc)
        sub_adj = np.zeros((k, k), dtype=float)
        for i, u in enumerate(scc):
            for j, v in enumerate(scc):
                sub_adj[i, j] = adj[u, v]
        sub_ci = {idx_map[u]: node_ci.get(u, 0.5) for u in scc}
        sub_cut = ci_guided_fvs(k, sub_adj, node_ci=sub_ci, max_rounds=max_rounds)
        for sc in sub_cut:
            all_cut.add(scc[sc])
    return sorted(all_cut)


def _tarjan_scc(n: int, adj: np.ndarray) -> List[List[int]]:
    """Tarjan SCC decomposition (iterative version, avoids recursion overflow for large n). Returns list of SCCs."""
    index_counter = [0]
    stack = []
    lowlink = [0] * n
    index = [-1] * n
    on_stack = [False] * n
    result = []

    def strongconnect(v):
        # Iterative Tarjan
        work = [(v, 0)]
        call_stack = []
        while work:
            node, pi = work[-1]
            if pi == 0:
                index[node] = index_counter[0]
                lowlink[node] = index_counter[0]
                index_counter[0] += 1
                stack.append(node)
                on_stack[node] = True
            # Find the next unvisited neighbor
            w = -1
            for i in range(pi, n):
                if adj[node, i]:
                    if index[i] == -1:
                        w = i
                        break
                    elif on_stack[i]:
                        lowlink[node] = min(lowlink[node], index[i])
            if w != -1:
                work[-1] = (node, w + 1)
                work.append((w, 0))
            else:
                # All neighbors processed
                if lowlink[node] == index[node]:
                    scc = []
                    while True:
                        u = stack.pop()
                        on_stack[u] = False
                        scc.append(u)
                        if u == node:
                            break
                    result.append(scc)
                work.pop()
                if work:
                    parent = work[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[node])

    for v in range(n):
        if index[v] == -1:
            strongconnect(v)
    return result


# ==========================================================================
# Multi-breakpoint residual equation (core of FVS solving: breakpoint set y -> DAG-order evaluation -> r(y)=y-phi(y))
# ==========================================================================
def build_multibreak_residual(n: int, adj: np.ndarray, node_fn: Callable,
                              breaks: Sequence[int], eps: float = 1e-6,
                              J_const: Optional[np.ndarray] = None,
                              _probe_result: Optional[tuple] = None):
    """Return (r, J, func, grad, hess): given breakpoint set B (should be FVS),
    r_B(y) = y_B - phi_B(y), where phi_B is evaluated in topological order on the DAG after cutting B.

    v2.3.2: J_const parameter-if the system is linear (cascade models, etc.), the Jacobian is a constant matrix,
    It can be computed once externally and passed in, avoiding repeated numerical differentiation in each Newton iteration (d+1 DAG evaluations).
    """
    from ..loopeval import _isfin
    breaks = list(breaks)
    d = len(breaks)
    bk = {v: k for k, v in enumerate(breaks)}
    # Breakpoint semantics (v2.2.0 fix): breakpoint outputs are fixed to y, participating in evaluation as "known sources",
    # without cutting breakpoint outgoing edges - otherwise downstream would lose the breakpoint's feedback input (e.g., Sum's "+-" minus input),
    # producing spurious solutions on mixed loops of "external input + feedback" (e.g., y=0.5(1-y) approximated as y=0.5).
    # The non-breakpoint subgraph must be a DAG (guaranteed by FVS); breakpoints are treated as ready sources.
    from collections import deque
    # v2.3.8: All adjacency lists built in O(n+|E|), replacing the original O(n^2) double loop.
    # For 5000blk, build_multibreak_residual drops from ~6s to ~20ms.
    _rows, _cols = np.nonzero(adj)
    _edge_list = list(zip(_rows.tolist(), _cols.tolist()))
    # pred_adj: full predecessors (including breakpoints), used for r(x) evaluation
    pred_adj: Dict[int, List[int]] = {u: [] for u in range(n)}
    # succ_nobk: non-breakpoint successors, used for topological traversal
    succ_nobk: Dict[int, List[int]] = {u: [] for u in range(n)}
    # pred_nobk: non-breakpoint predecessors, used for in-degree calculation
    pred_nobk = [[] for _ in range(n)]
    for _j, _u in _edge_list:
        pred_adj[_u].append(_j)
        if _j not in bk and _u not in bk:
            pred_nobk[_u].append(_j)
            succ_nobk[_j].append(_u)
    indeg = [len(pred_nobk[v]) for v in range(n)]
    q = deque([v for v in range(n) if v not in bk and indeg[v] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for w in succ_nobk[u]:
            if indeg[w] > 0:
                indeg[w] -= 1
                if indeg[w] == 0:
                    q.append(w)
    # The non-breakpoint subgraph should have no cycles (guaranteed by FVS); if it has any, raise an error
    residual_cyclic = [v for v in range(n) if v not in bk and indeg[v] > 0]
    if residual_cyclic:
        raise ValueError("break set is not a feedback vertex set: residual cyclic %s"
                         % residual_cyclic[:10])

    def r(x):
        xx = np.asarray(x, dtype=float).reshape(-1)
        if xx.shape[0] != d:
            raise ValueError("dim mismatch")
        full = np.zeros(n, dtype=float)
        for k, v in enumerate(breaks):
            full[v] = xx[k]

        def safe(u, vals):
            try:
                fv = float(node_fn(u, vals))   # Complex/exception uniformly downgraded (consistent with existing pipeline)
                return fv if _isfin(fv) else (full[u] if full[u] != 0 else 0.0)
            except Exception:
                return full[u] if full[u] != 0 else 0.0

        for u in topo:
            # Full predecessors (including breakpoints, whose values are fixed to y) - do not cut breakpoint outgoing edges
            vals = {j: full[j] for j in pred_adj[u]}
            full[u] = safe(u, vals)
        rr = np.zeros(d)
        for k, v in enumerate(breaks):
            vals = {j: full[j] for j in pred_adj[v]}
            rr[k] = xx[k] - safe(v, vals)
        return rr

    def J(x):
        if J_const is not None:
            return J_const.copy()
        Jm = np.zeros((d, d))
        r0 = r(x)
        for k in range(d):
            xp = np.asarray(x, dtype=float).copy()
            xp[k] += eps
            Jm[:, k] = (r(xp) - r0) / eps
        return Jm

    # ---- v2.3.3 P2-10 Batch linear fast path: after probing that the whole graph is linear, vectorize the d DAG propagations of J
    # into a single (d+1,n) matrix operation. For large cascade models (2000/5000blk), J(0)
    # drops from 345s to ~0.2s. If probing fails for nonlinear models, automatically falls back to the numerical differentiation above.
    # The additivity check (atol=1e-6) for probing is already strict enough; only small models with d<=8 get an additional numerical verification
    # as double insurance (original numerical differentiation is infeasible for large models).
    try:
        from .jacobian import probe_linearity, build_fast_rJ
        if _probe_result is not None:
            _weights, _biases, _lin, _ = _probe_result
        else:
            _weights, _biases, _lin, _ = probe_linearity(n, adj, node_fn, topo, breaks)
        if _lin and d > 1:
            _fast_r, _fast_J = build_fast_rJ(n, adj, node_fn, topo, breaks,
                                             _weights, _biases, eps=eps)
            if d <= 8:
                try:
                    _Jf = np.asarray(_fast_J(np.zeros(d)))
                    _Jn = np.asarray(J(np.zeros(d)))
                    if _Jf.shape != _Jn.shape or not np.allclose(_Jf, _Jn, atol=1e-4):
                        raise ValueError("fast-J mismatch")
                except Exception:
                    _lin = False
            if _lin:
                r, J = _fast_r, _fast_J
    except ImportError:
        pass
    except Exception:
        pass

    # v2.3.2 optimization: grad/hess are called consecutively at the same x, caching r(x) and J(x) to avoid recomputation
    _cache = {"x": None, "r": None, "J": None}

    def _get_rJ(x):
        xa = np.asarray(x, dtype=float).ravel()
        if _cache["x"] is not None and np.array_equal(_cache["x"], xa):
            return _cache["r"], _cache["J"]
        rv = r(xa)
        Jv = J(xa)
        _cache["x"] = xa.copy()
        _cache["r"] = rv
        _cache["J"] = Jv
        return rv, Jv

    def func(x):
        rr, _ = _get_rJ(x)
        return float(0.5 * float(np.dot(rr, rr)))

    def grad(x):
        rr, Jv = _get_rJ(x)
        return (Jv.T @ rr).ravel()

    def hess(x):
        _, Jv = _get_rJ(x)
        return Jv.T @ Jv

    return r, J, func, grad, hess


def build_multibreak_residual_auto(n: int, adj: np.ndarray, node_fn: Callable,
                                   breaks: Sequence[int], ad_model=None,
                                   eps: float = 1e-6):
    """v2.3.3 Automatically select the optimal residual/Jacobian construction path.

    Linear (all probes pass) -> fast-J (batch vectorized, 2000blk J 345s->54ms);
    Nonlinear -> analytical forward AD (build_multibreak_residual_ad, O(n.d) single propagation,
      avoiding numerical differentiation O(d.n); only available when ad_model (SlxModel) is provided);
    Otherwise -> numerical differentiation (original build_multibreak_residual path).

    The probe result is passed to build_multibreak_residual via _probe_result, avoiding repeated probe
    (~1.5s) for large models (2000/5000blk).
    """
    from collections import deque
    breaks = list(breaks)
    d = len(breaks)
    bk = {v: k for k, v in enumerate(breaks)}
    # Non-breakpoint topological order (consistent with build_multibreak_residual)
    pred = [[] for _ in range(n)]
    for i in range(n):
        if i in bk:
            continue
        for j in range(n):
            if adj[j, i] and j not in bk:
                pred[i].append(j)
    indeg = [len(pred[v]) for v in range(n)]
    q = deque([v for v in range(n) if v not in bk and indeg[v] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for w in range(n):
            if w in bk:
                continue
            if adj[u, w] and indeg[w] > 0:
                indeg[w] -= 1
                if indeg[w] == 0:
                    q.append(w)
    from .jacobian import probe_linearity
    _pr = probe_linearity(n, adj, node_fn, topo, breaks)
    if _pr[2]:
        # Linear -> fast-J (reuse probe result, no repeated probe)
        return build_multibreak_residual(n, adj, node_fn, breaks, eps=eps,
                                         _probe_result=_pr)
    if ad_model is not None:
        # Nonlinear + has model -> analytical AD
        try:
            from ..simulink.parse_slx import build_multibreak_residual_ad
            return build_multibreak_residual_ad(ad_model, breaks, eps=eps)
        except Exception:
            pass
    # Fall back to numerical differentiation
    return build_multibreak_residual(n, adj, node_fn, breaks, eps=eps)
