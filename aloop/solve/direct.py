"""Robust solving layer: direct solution of linear algebraic loops (v2.3.3 P0-3).

The residual system of an algebraic loop is r(y) = y - phi(y). When all blocks are linear (Gain/Sum/Product with constant coefficients,
Constant sources), J is a constant matrix, and the system degenerates to the linear equation J.y = -c (c = r(0)),
which can be solved in one step without iteration. Compared with TR-Newton iteration (each round O(d^3) dense Cholesky),
linear direct solve uses sparse LU decomposition of J (cascaded model J sparsity ~0.6%, small bandwidth),
reducing large-scale (2000/5000 blocks) solving from seconds to milliseconds.

Detection: compare the Jacobian at two different x (J(0) and J(1)); if equal, the system is linear.
For linear systems, J(x) is constant; this detection costs 2 _evaluate calls (analytical AD already cached).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import scipy.sparse as _sp
    from scipy.sparse.linalg import splu as _splu
    from scipy.linalg import solve_triangular as _solve_tri
    _HAS_SPARSE = True
except Exception:
    _HAS_SPARSE = False


def _is_lower_plus_corner(J0, tol=1e-12):
    """Detect whether J is lower triangular plus a single corner element J[0,d-1] (typical structure of cascaded feedback models).

    In a cascaded model, each breakpoint depends only on its predecessors (lower triangular); the unique feedback connection makes J[0,d-1] nonzero.
Such matrices can be solved in O(d) using the Sherman-Morrison formula, 5-10x faster than sparse LU.
    """
    d = J0.shape[0]
    if d < 3:
        return False
    upper = J0 - np.tril(J0)
    if abs(upper[0, d - 1]) < tol:
        return False
    upper[0, d - 1] = 0.0
    return bool(np.all(np.abs(upper) < tol))


def _solve_sm_corner(J0, b):
    """Sherman-Morrison solve (L + u v^T) y = b, L lower triangular, u=e_0, v=corner*e_{d-1}."""
    d = J0.shape[0]
    L = np.tril(J0).copy()
    corner = float(L[0, d - 1])
    L[0, d - 1] = 0.0
    u = np.zeros(d)
    u[0] = 1.0
    v = np.zeros(d)
    v[d - 1] = corner
    z = _solve_tri(L, b, lower=True)
    q = _solve_tri(L, u, lower=True)
    denom = 1.0 + float(np.dot(v, q))
    y = z - q * (float(np.dot(v, z)) / denom)
    # Iterative refinement: r = b - J y, delta = solve(J, r) (reusing q/denom), one pass reaches machine precision.
    # Reason: Sherman-Morrison suffers cancellation error when corner feedback is strong (measured ~1e-8,
    # dense solve ~1e-16); one refinement eliminates it.
    r = b - J0.dot(y)
    z2 = _solve_tri(L, r, lower=True)
    delta = z2 - q * (float(np.dot(v, z2)) / denom)
    return y + delta


def is_linear_system(J: Callable, d: int, atol: float = 1e-6) -> bool:
    """Detect whether the system is linear: J(0) and J(1) (two different points) are identical.

    v2.3.3: default tolerance relaxed to 1e-6 - J is numerical difference; when the residual magnitude at x=1 is large, floating-point
    error can exceed 1e-9 (measured J(0) vs J(1) difference ~1e-7 for 2000blk cascaded model).
    Linear false positives are guarded by the residual fallback in the caller solve_linear_direct.

    Args:
        J: analytical Jacobian function J(x) -> (d,d) matrix
        d: breakpoint dimension
        atol: absolute tolerance
    """
    try:
        J0 = np.asarray(J(np.zeros(d)), dtype=float)
        J1 = np.asarray(J(np.ones(d)), dtype=float)
        if J0.shape != J1.shape:
            return False
        scale = max(1.0, float(np.max(np.abs(J0))))
        return np.allclose(J0, J1, rtol=1e-5, atol=max(atol, scale * 1e-7))
    except Exception:
        return False


def solve_linear_direct(r: Callable, J: Callable, d: int,
                        tol: float = 1e-8) -> Dict:
    """Direct solution of linear system: J.y = -c (c = r(0)).

    Prefer sparse LU (J sparse, 0.6% nonzeros for cascaded model); fall back to dense np.linalg.solve on failure.

    Args:
        r: residual function r(x) -> (d,) (linear system r(y) = J y + c)
        J: analytical Jacobian J(x) -> (d,d)
        d: breakpoint dimension
        tol: convergence tolerance

    Returns:
        {"x","res","success","stage","nit"}; stage="linear_direct"
    """
    J0 = np.asarray(J(np.zeros(d)), dtype=float)
    c0 = np.asarray(r(np.zeros(d)), dtype=float).ravel()
    try:
        if _HAS_SPARSE:
            if _is_lower_plus_corner(J0):
                y = _solve_sm_corner(J0, -c0)
            elif np.allclose(J0, np.tril(J0)):
                y = _solve_tri(J0, -c0, lower=True)
            elif np.allclose(J0, np.triu(J0)):
                y = _solve_tri(J0, -c0, lower=False)
            else:
                lu = _splu(_sp.csc_matrix(J0))
                y = lu.solve(-c0)
        else:
            y = np.linalg.solve(J0, -c0)
    except Exception:
        y = np.linalg.solve(J0, -c0)
    y = np.asarray(y, dtype=float).ravel()
    res = float(np.linalg.norm(np.asarray(r(y), dtype=float).ravel()))
    success = bool(np.isfinite(res) and res < max(tol, 1e-8))
    return {"x": y, "res": res, "success": success,
            "stage": "linear_direct", "nit": 1}


def solve_multibreak_linear_ad(model, breaks, tol: float = 1e-8) -> Dict:
    """One-stop entry for analytical AD + linear direct solve (v2.3.3).

    For linear systems (cascaded etc.): one-step sparse LU direct solve, no iteration needed;
    for nonlinear systems: return None and let the caller go through hast_n/hast_cma.

    Args:
        model: SlxModel (already parsed)
        breaks: list of breakpoint SIDs (should be FVS)
        tol: convergence tolerance

    Returns:
        linear system: {"x","res","success","stage","nit","func","grad","hess","r","J"}
        nonlinear system: None (caller falls back to iterative solve)
    """
    from ..simulink.parse_slx import build_multibreak_residual_ad
    r, J, func, grad, hess = build_multibreak_residual_ad(model, breaks)
    d = len(breaks)
    if not is_linear_system(J, d):
        return None
    result = solve_linear_direct(r, J, d, tol=tol)
    if not result["success"]:
        # The linear assumption is falsified: piecewise-linear blocks (DeadZone/Relay) have the same J at the detection points but are
        # overall nonlinear; the direct solve does not satisfy r(y)=0. Fall back to iterative solve.
        return None
    result.update({"func": func, "grad": grad, "hess": hess, "r": r, "J": J})
    return result
