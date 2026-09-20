"""CI (Composite Complexity Index) complexity metric.

Definition (implementation consistent with the legacy source):
    D     = min(dim/10, 1)                          # dimension component
    Delta = min(log10(1 + ||grad f(x0)||) / L, 1)   # initial residual component (L=8)
    S     = min(cond_est/100, 1)                    # Jacobian ill-conditioning component
    CI    = w0*D + w1*Delta + w2*S,  w=(0.4, 0.3, 0.3)

The operation manual additionally gives a log1p variant: CI = w1*log1p(dim) + w2*log1p(init_residual) + w3*log1p(cond(J)),
provided at implementation time as `compute_ci_log1p` (for consistency verification).

Tiering (used for the three-stage configuration adaptation):
    CI < 0.3        -> tier 'low' (fast path converges directly, zero overhead)
    CI in [0.3,0.7) -> tier 'medium'
    CI >= 0.7       -> tier 'high' (enable multi-restart SA + multi-elite refinement)
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np


def ci_components(dim: int, x0: np.ndarray, grad: Callable,
                  condition_est: float = 1.0, L: float = 8.0) -> Tuple[float, float, float]:
    """Three normalized CI components (can be computed online before solving, without knowing the analytical solution)."""
    D = min(dim / 10.0, 1.0)
    g = np.asarray(grad(x0), dtype=float).flatten()
    Delta = min(np.log10(1.0 + float(np.linalg.norm(g))) / L, 1.0)
    S = min(float(condition_est) / 100.0, 1.0)
    return float(D), float(Delta), float(S)


def compute_ci(dim: int, x0: np.ndarray, grad: Callable,
               condition_est: float = 1.0, w: Tuple[float, float, float] = (0.4, 0.3, 0.3),
               L: float = 8.0) -> float:
    D, Delta, S = ci_components(dim, x0, grad, condition_est, L)
    ci = w[0] * D + w[1] * Delta + w[2] * S
    return round(float(ci), 4)


def compute_ci_log1p(dim: int, init_residual: float, cond_J: float,
                     w: Tuple[float, float, float] = (0.4, 0.3, 0.3)) -> float:
    """The operation manual's log1p variant (used for cross-validating CI definition consistency)."""
    ci = (w[0] * np.log1p(dim) / np.log1p(10.0)
          + w[1] * np.log1p(init_residual) / np.log1p(10.0)
          + w[2] * np.log1p(cond_J) / np.log1p(100.0))
    return round(float(ci), 4)


def get_ci_config(ci: float) -> Dict:
    """CI adaptive three-stage configuration."""
    if ci < 0.3:
        return {'T0': 500, 'alpha': 0.98, 'sigma': 1.0, 'n_restarts': 2,
                'sa_iter_per_restart': 300, 'tier': 'low'}
    elif ci < 0.7:
        return {'T0': 1500, 'alpha': 0.98, 'sigma': 2.0, 'n_restarts': 6,
                'sa_iter_per_restart': 267, 'tier': 'medium'}
    else:
        return {'T0': 1500, 'alpha': 0.98, 'sigma': 1.5, 'n_restarts': 6,
                'sa_iter_per_restart': 267, 'tier': 'high'}
