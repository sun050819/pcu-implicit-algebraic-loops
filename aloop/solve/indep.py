"""Independent third-party solver (cross-solver robustness validation 9.1.5).

Source: "Independent Solver Validation Notes.md" - "root_scalar 'brentq' bisection root-finding +
geometric bracket expansion, without using fixed-point iteration/Aitken/Newton fallback".
Used to validate: when point selection output is fed to a completely independent solver, the conclusion that "learning-based point selection outperforms heuristics"
remains robust (proving that the point selection strategy does not depend on a specific solver).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np


def _expand_bracket(phi: Callable[[float], float], center: float,
                    lo0: float = -1e3, hi0: float = 1e3,
                    max_expand: int = 60, factor: float = 1.6) -> Optional[Tuple[float, float]]:
    """Geometric bracket expansion: find a sign-changing interval [a,b] for phi(y)-y."""
    g = lambda y: float(phi(y)) - y
    a, b = lo0, hi0
    fa, fb = g(a), g(b)
    if not np.isfinite(fa) or not np.isfinite(fb):
        return None
    if fa * fb < 0:
        return a, b
    # Expand geometrically level by level with center as the center to find a sign-changing interval
    span = 1.0
    for _ in range(max_expand):
        a, b = center - span, center + span
        fa, fb = g(a), g(b)
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
            return a, b
        span *= factor
    return None


def brentq_solve(phi: Callable[[float], float], center: float = 1.0,
                 tol: float = 1e-10, max_iter: int = 300,
                 bracket: Optional[Tuple[float, float]] = None) -> Dict:
    """Use brentq to find the real root of phi(y) - y = 0.

    Returns:
        {"x", "nit", "residual", "converged", "stage"}
    """
    from scipy.optimize import root_scalar
    g = lambda y: float(phi(y)) - y
    nit = 0
    if bracket is None:
        br = _expand_bracket(phi, center)
        if br is None:
            return {"x": float("nan"), "nit": 0, "residual": float("nan"),
                    "converged": False, "stage": "no_bracket"}
        a, b = br
    else:
        a, b = bracket
    try:
        sol = root_scalar(g, method="brentq", bracket=(a, b),
                          xtol=tol, maxiter=max_iter)
        nit = int(sol.iterations)
        x = float(sol.root)
        r = abs(g(x))
        return {"x": x, "nit": nit, "residual": r,
                "converged": bool(sol.converged and r < 1e-6),
                "stage": "brentq"}
    except Exception:
        return {"x": float("nan"), "nit": nit, "residual": float("nan"),
                "converged": False, "stage": "brentq_fail"}
