"""Robust solving layer: simulated-annealing-like methods."""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np


def _safe_func(func, x) -> float:
    try:
        return float(func(x))
    except Exception:
        return 1e30


def simple_sa(func, grad, x0, tol=1e-8, domain_tol=1.0, max_iter=2000,
              T0=500.0, alpha=0.96, sigma=0.5) -> Dict:
    x = x0.copy().astype(float)
    best_x = x.copy()
    best_f = _safe_func(func, x)
    T = T0
    for k in range(max_iter):
        x_new = x + sigma * np.random.randn(len(x)) * (T / T0 + 1e-2)
        f_new = _safe_func(func, x_new)
        delta_f = f_new - best_f
        if delta_f < 0 or np.random.rand() < np.exp(-delta_f / max(T, 1e-10)):
            x = x_new
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
        T *= alpha
    final_res = np.linalg.norm(grad(best_x))
    return {"x": best_x, "nit": max_iter, "res": final_res,
            "success": final_res < tol}


def adaptive_sa(func, grad, x0, tol=1e-8, domain_tol=1.0, max_iter=2000,
                T0=500.0, target_accept=0.44, sigma=0.5) -> Dict:
    x = x0.copy().astype(float)
    best_x = x.copy()
    best_f = _safe_func(func, x)
    T = T0
    accept_count = 0
    window_size = 50
    for k in range(max_iter):
        x_new = x + sigma * np.random.randn(len(x)) * (T / T0 + 1e-2)
        f_new = _safe_func(func, x_new)
        delta_f = f_new - best_f
        if delta_f < 0 or np.random.rand() < np.exp(-delta_f / max(T, 1e-10)):
            x = x_new
            accept_count += 1
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
        if (k + 1) % window_size == 0:
            accept_rate = accept_count / window_size
            if accept_rate > target_accept:
                T *= 0.95
            else:
                T *= 1.02
            accept_count = 0
    final_res = np.linalg.norm(grad(best_x))
    return {"x": best_x, "nit": max_iter, "res": final_res,
            "success": final_res < tol}


SA_SOLVERS = {
    "simple_sa": simple_sa,
    "adaptive_sa": adaptive_sa,
}
