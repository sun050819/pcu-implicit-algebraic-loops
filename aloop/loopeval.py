"""Foundations of loop evaluation and breakpoint solving (ported from the breakpoint-selection project).

Given a directed signal flow graph and per-node output functions
    x_v = node_fn(v, {j: x_j for j in pred(v)}),
At breakpoint v, freeze the output of v as the iteration variable y, and induce the feedforward mapping phi_v(y) from the "cut graph"
(Gauss-Seidel sweep), then iterate y <- phi_v(y) (Aitken acceleration; Newton fallback on divergence).
For each candidate breakpoint, record the true iteration count/final residual/runtime - this protocol simultaneously serves as the
label generator (for GAT training) and the "breakpointization" entry point of the solving layer.

This module also provides:
    - build_breakpoint_equation : breakpoint residual r(y)=phi(y)-y and its Jacobian (for
      general solvers such as three-stage, dim=1).
    - build_system_equation    : full-loop direct system r(x) and its Jacobian (for real implicit
      algebraic loops direct solving, dim=d).
"""
from __future__ import annotations

import contextlib
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------- Standard block types (aligned with the paper) ----------------
T_GAIN, T_SUM, T_MATH, T_PRODUCT = 0, 1, 2, 3


@contextlib.contextmanager
def _quiet():
    """Silent context: overflow from divergent breakpoints during evaluation/iteration is expected; suppress warnings."""
    with np.errstate(all="ignore"):
        with contextlib.suppress(ImportError):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield
                return
        yield


def make_block_node_fn(bt: int, params: List[float]):
    """Node output functions for the four standard block types. params=[k,b,c], all nonlinearities bounded."""
    k, b, c = float(params[0]), float(params[1]), float(params[2])

    def fn(v, pred_vals):
        s = sum(pred_vals.values())
        if bt == T_GAIN:
            return k * s
        if bt == T_SUM:
            return s + b
        if bt == T_MATH:
            return np.tanh(c * s)
        if bt == T_PRODUCT:
            return s + c * s * np.tanh(s)
        raise ValueError("unknown block type")

    return fn


def _isfin(v) -> bool:
    try:
        return bool(np.all(np.isfinite(v)))
    except Exception:
        return False


# ==========================================================================
# Cut-graph feedforward mapping evaluation phi_v(y)
# ==========================================================================
def _eval_loop_impl(n: int, adj: np.ndarray, node_fn, break_v: int, y: float,
                    tol_in: float = 1e-10, max_in: int = 200):
    """Return the feedforward mapping phi_v(y) induced after cutting break_v.

    If the cut graph is a DAG (single loop / fully broken), evaluate in one topological order; if residual cycles remain (nested/overlapping
    loops not fully broken), perform an internal Gauss-Seidel sweep. Return (new output of break_v, inner iterations, residual).
    """
    from collections import deque
    A = adj.copy()
    A[break_v, :] = 0                       # Cut the outgoing edges of break_v
    dt_ = np.complex128 if np.iscomplexobj(y) else np.float64
    # Neutral initialization: all nodes take the breakpoint value, avoiding degenerate false convergence in the first round
    x = np.full(n, y, dtype=dt_)
    x[break_v] = y

    def safe_eval(u, vals):
        v = node_fn(u, vals)
        return v if _isfin(v) else (x[u] if x[u] != 0 else y)

    pred = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if A[j, i]:
                pred[i].append(j)
    # Kahn topological order (cut graph)
    indeg = [len(pred[v]) for v in range(n)]
    indeg[break_v] = 0
    q = deque([v for v in range(n) if v != break_v and indeg[v] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for w in range(n):
            if A[u, w] and indeg[w] > 0:
                indeg[w] -= 1
                if indeg[w] == 0:
                    q.append(w)
    cyclic = [v for v in range(n) if v != break_v and indeg[v] > 0]
    if not cyclic:
        for u in topo:
            if pred[u]:
                x[u] = safe_eval(u, {j: x[j] for j in pred[u]})
        y_new = node_fn(break_v, {j: x[j] for j in pred[break_v]})
        return y_new, 1, abs(y_new - y)
    # Residual cycles -> Gauss-Seidel sweep over the cyclic part + DAG part
    iters = 0
    for _ in range(max_in):
        iters += 1
        newx = x.copy()
        for u in topo:
            if pred[u] and u not in cyclic:
                newx[u] = safe_eval(u, {j: newx[j] for j in pred[u]})
        for u in cyclic:
            newx[u] = safe_eval(u, {j: newx[j] for j in pred[u]})
        x = newx
        y_new = node_fn(break_v, {j: x[j] for j in pred[break_v]})
        res = abs(y_new - y)
        if res < tol_in:
            return y_new, iters, res
        y = y_new
    return y, iters, abs(y - y_new)


def eval_loop(n: int, adj: np.ndarray, node_fn, break_v: int, y: float,
              tol_in: float = 1e-10, max_in: int = 200):
    """Public entry: calls the implementation after silencing divergence noise."""
    with _quiet():
        return _eval_loop_impl(n, adj, node_fn, break_v, y, tol_in=tol_in,
                               max_in=max_in)


# ==========================================================================
# True single-breakpoint solving
# ==========================================================================
def _solve_breakpoint_impl(n: int, adj: np.ndarray, node_fn, break_v: int,
                           x0: float = 1.0, tol: float = 1e-6, max_iter: int = 800,
                           use_atiken: bool = True, use_newton: bool = True) -> Dict:
    """Truly solve at breakpoint v. iter = outer fixed-point iterations + inner residual-cycle sweep."""
    t0 = time.time()
    y = x0
    y_prev2 = None
    y_prev1 = None
    total_inner = 0
    n_unbounded = 0
    for k in range(max_iter):
        y_new, in_iters, _ = eval_loop(n, adj, node_fn, break_v, y)
        total_inner += in_iters
        if not _isfin(y_new):
            break
        res = abs(y_new - y)
        if res < tol:
            dt = (time.time() - t0) * 1000.0
            return {"iter": k + 1 + total_inner, "residual": res,
                    "time_ms": dt, "converged": True}
        # Early divergence detection: unbounded value range -> mark divergence immediately, skip expensive Newton fallback
        if abs(y_new) > 1e6 or abs(y) > 1e6:
            n_unbounded += 1
            if n_unbounded >= 3:
                return {"iter": k + 1 + total_inner, "residual": float("nan"),
                        "time_ms": (time.time() - t0) * 1000.0, "converged": False}
        else:
            n_unbounded = 0
        if use_atiken and y_prev2 is not None and y_prev1 is not None \
                and _isfin(y_new) and _isfin(y_prev1) and _isfin(y_prev2):
            denom = y_new - 2.0 * y_prev1 + y_prev2
            if abs(denom) > 1e-12:
                y_acc = y_prev2 - (y_prev1 - y_prev2) ** 2 / denom
                if np.isfinite(y_acc):
                    y = y_acc
                    y_prev2 = None
                    y_prev1 = None
                    continue
        y_prev2, y_prev1 = y_prev1, y_new
        y = y_new
    # Divergence -> Newton fallback (real values only, and final state bounded)
    from scipy.optimize import root
    if np.iscomplexobj(y) or not use_newton or not _isfin(y) or abs(y) > 1e4:
        return {"iter": max_iter, "residual": float("nan"),
                "time_ms": (time.time() - t0) * 1000.0, "converged": False}

    def phi(yy):
        yy0 = float(np.asarray(yy).reshape(-1)[0])
        if not np.isfinite(yy0):
            return float("inf")
        out = eval_loop(n, adj, node_fn, break_v, yy0, tol_in=1e-10)[0]
        return out if np.isfinite(out) else float("inf")

    y0 = float(np.asarray(y).reshape(-1)[0]) if np.isfinite(y) else 0.5
    try:
        sol = root(lambda yy: phi(yy) - yy, np.array([y0]),
                   method="hybr", options={"maxfev": 80})
        if sol.success and np.isfinite(float(sol.x[0])):
            yy = float(sol.x[0])
            r = abs(phi(yy) - yy)
            return {"iter": max_iter + int(sol.nfev) + total_inner, "residual": r,
                    "time_ms": (time.time() - t0) * 1000.0, "converged": True}
    except Exception:
        pass
    r = abs(phi(y) - y) if _isfin(y) else float("nan")
    return {"iter": max_iter, "residual": r,
            "time_ms": (time.time() - t0) * 1000.0, "converged": False}


def solve_breakpoint(n: int, adj: np.ndarray, node_fn, break_v: int,
                     x0: float = 1.0, tol: float = 1e-6, max_iter: int = 800,
                     use_atiken: bool = True, use_newton: bool = True) -> Dict:
    """Public entry: calls the implementation after silencing divergence noise."""
    with _quiet():
        return _solve_breakpoint_impl(n, adj, node_fn, break_v, x0=x0, tol=tol,
                                      max_iter=max_iter, use_atiken=use_atiken,
                                      use_newton=use_newton)


def compute_all_breakpoints(n, adj, node_fn, loop_nodes, x0=1.0, tol=1e-6) -> Dict:
    """Solve for each candidate breakpoint on the loop one by one, return {v: result_dict}."""
    return {v: solve_breakpoint(n, adj, node_fn, v, x0=x0, tol=tol)
            for v in loop_nodes}


# ==========================================================================
# Breakpoint residual equation r(y) = phi(y) - y (for general solvers)
# ==========================================================================
def _finite_clip(v, cap: float = 1e8) -> float:
    """Clip non-finite values to a large finite value (avoid optimizer overflow)."""
    try:
        f = float(v)
    except Exception:
        return cap
    if not np.isfinite(f):
        return cap
    return float(np.clip(f, -cap, cap))


class BreakpointEquation:
    """Breakpoint residual equation with precomputed cut-graph topology + fast evaluation.

 Each evaluation no longer rebuilds pred/topo/cyclic, greatly reducing the cost of multi-restart SA.
    """

    def __init__(self, n: int, adj: np.ndarray, node_fn, break_v: int,
                 eps: float = 1e-6, tol_in: float = 1e-8, max_in: int = 25):
        from collections import deque
        self.n, self.adj, self.node_fn, self.break_v = n, adj, node_fn, break_v
        self.eps, self.tol_in, self.max_in = eps, tol_in, max_in
        A = adj.copy()
        A[break_v, :] = 0
        self.pred = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if A[j, i]:
                    self.pred[i].append(j)
        self.pred_break = [j for j in range(n) if adj[j, break_v]]
        # Kahn topological order
        indeg = [len(self.pred[v]) for v in range(n)]
        indeg[break_v] = 0
        q = deque([v for v in range(n) if v != break_v and indeg[v] == 0])
        topo = []
        while q:
            u = q.popleft()
            topo.append(u)
            for w in range(n):
                if A[u, w] and indeg[w] > 0:
                    indeg[w] -= 1
                    if indeg[w] == 0:
                        q.append(w)
        self.topo = topo
        self.cyclic = [v for v in range(n) if v != break_v and indeg[v] > 0]
        self.acyclic = not self.cyclic

    def _eval(self, y: float) -> float:
        with _quiet():
            return self._eval_impl(y)

    def _eval_impl(self, y: float) -> float:
        import math
        n, node_fn = self.n, self.node_fn
        pred = self.pred
        x = np.full(n, y, dtype=np.float64)
        x[self.break_v] = y
        pred_break = self.pred_break

        def safe(u, vals):
            try:
                v = node_fn(u, vals)
                return v if math.isfinite(float(v)) else (x[u] if x[u] != 0 else y)
            except Exception:
                return x[u] if x[u] != 0 else y

        if self.acyclic:
            for u in self.topo:
                if pred[u]:
                    x[u] = safe(u, {j: x[j] for j in pred[u]})
            return float(node_fn(self.break_v, {j: x[j] for j in pred_break}))
        # Residual cycles: in-place Gauss-Seidel sweep over the DAG part + cyclic part
        y_cur = y
        for _ in range(self.max_in):
            for u in self.topo:
                if pred[u] and u not in self.cyclic:
                    x[u] = safe(u, {j: x[j] for j in pred[u]})
            for u in self.cyclic:
                x[u] = safe(u, {j: x[j] for j in pred[u]})
            y_new = float(node_fn(self.break_v, {j: x[j] for j in pred_break}))
            if not math.isfinite(y_new):
                return y_cur
            if abs(y_new - y_cur) < self.tol_in:
                return y_new
            y_cur = y_new
        return y_cur

    def phi(self, y) -> float:
        return self._eval(float(np.asarray(y).reshape(-1)[0]))

    def r(self, y) -> np.ndarray:
        yy = float(np.asarray(y).reshape(-1)[0])
        y_new = self._eval(yy)
        return np.array([_finite_clip(y_new - yy)])

    def J(self, y) -> np.ndarray:
        yy = float(np.asarray(y).reshape(-1)[0])
        rp = self._eval(yy + self.eps) - (yy + self.eps)
        rm = self._eval(yy - self.eps) - (yy - self.eps)
        if not _isfin(rp):
            rp = 0.0
        if not _isfin(rm):
            rm = 0.0
        return np.array([[_finite_clip((rp - rm) / (2 * self.eps), 1e6)]])

    def func(self, x) -> float:
        rr = self.r(x)
        return float(0.5 * float(rr[0]) ** 2)

    def grad(self, x) -> np.ndarray:
        return (self.J(x).T @ self.r(x)).ravel()

    def hess(self, x) -> np.ndarray:
        jj = self.J(x)
        return jj.T @ jj

    def interfaces(self):
        return self.r, self.J, self.func, self.grad, self.hess


def build_breakpoint_residual(n: int, adj: np.ndarray, node_fn, break_v: int,
                              eps: float = 1e-6):
    """Return (r, J, func, grad, hess), internally using BreakpointEquation with precomputed topology for acceleration."""
    eq = BreakpointEquation(n, adj, node_fn, break_v, eps=eps)
    return eq.r, eq.J, eq.func, eq.grad, eq.hess


# ==========================================================================
# Full-loop direct system r(x) (no loop breaking, dim=d)
# ==========================================================================
def build_system_residual(loop_nodes: List[int], adj: np.ndarray, node_fn, n: int,
                          eps: float = 1e-6):
    """Return (r, J, func, grad, hess):

        r_v(x) = x_v - node_fn(v, {preds})   for v in loop_nodes
    """
    idx = {v: k for k, v in enumerate(loop_nodes)}
    d = len(loop_nodes)

    def r(x):
        xx = np.asarray(x, dtype=float).reshape(-1)
        # Build the full-graph value table
        full = np.zeros(n, dtype=float)
        for k, v in enumerate(loop_nodes):
            full[v] = xx[k]
        rr = np.zeros(d)
        for k, v in enumerate(loop_nodes):
            pred_vals = {j: full[j] for j in range(n) if adj[j, v]}
            fv = node_fn(v, pred_vals)
            fv = fv if _isfin(fv) else xx[k]
            rr[k] = xx[k] - fv
        return rr

    def J(x):
        Jm = np.zeros((d, d))
        r0 = r(x)
        for k in range(d):
            xp = np.asarray(x, dtype=float).copy()
            xp[k] += eps
            Jm[:, k] = (r(xp) - r0) / eps
        return Jm

    def func(x):
        rr = r(x)
        return float(0.5 * float(np.dot(rr, rr)))

    def grad(x):
        return (J(x).T @ r(x)).ravel()

    def hess(x):
        return J(x).T @ J(x)

    return r, J, func, grad, hess
