"""Robust solving layer: Newton-type solvers.

Unified interface: (func, grad, hess, x0, tol=1e-8, max_iter) -> {"x","nit","res","success"}
where func(x)=0.5*||r(x)||^2, grad=J^T r, hess=J^T J (Gauss-Newton).
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

try:
    from scipy.linalg import cho_factor, cho_solve
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _safe_func(func, x) -> float:
    try:
        return float(func(x))
    except Exception:
        return 1e30


# ==========================================================================
# 1. Vanilla Newton
# ==========================================================================
def vanilla_newton(func, grad, hess, x0, tol=1e-8, max_iter=100) -> Dict:
    x = x0.copy().astype(float)
    for k in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            return {"x": x, "nit": k + 1, "res": np.linalg.norm(g), "success": True}
        try:
            H = hess(x)
            dx = np.linalg.solve(H, -g)
        except Exception:
            return {"x": x, "nit": k + 1, "res": np.linalg.norm(g), "success": False}
        x = x + dx
    residual = np.linalg.norm(grad(x))
    return {"x": x, "nit": max_iter, "res": residual, "success": residual < tol}


# ==========================================================================
# 2. Damped Newton (Armijo backtracking)
# ==========================================================================
def damped_newton(func, grad, hess, x0, tol=1e-8, max_iter=2000, c1=1e-4, beta=0.5) -> Dict:
    x = x0.copy().astype(float)
    for k in range(max_iter):
        g = grad(x)
        gn = np.linalg.norm(g)
        if gn < tol:
            return {"x": x, "nit": k + 1, "res": gn, "success": True}
        try:
            H = hess(x)
            d = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            return {"x": x, "nit": k + 1, "res": gn, "success": False}
        gd = np.dot(g, d)
        if gd >= 0:
            d = -g
            gd = -np.dot(g, g)
        lam = 1.0
        f0 = _safe_func(func, x)
        for _ in range(60):
            if _safe_func(func, x + lam * d) <= f0 + c1 * lam * gd:
                break
            lam *= beta
        else:
            return {"x": x, "nit": k + 1, "res": gn, "success": False}
        x = x + lam * d
    residual = np.linalg.norm(grad(x))
    return {"x": x, "nit": max_iter, "res": residual, "success": residual < tol}


# ==========================================================================
# 3. Levenberg-Marquardt
# ==========================================================================
def levenberg_marquardt(func, grad, hess, x0, tol=1e-8, max_iter=100, mu0=1e-3) -> Dict:
    x = x0.copy().astype(float)
    mu = mu0
    for k in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            return {"x": x, "nit": k + 1, "res": np.linalg.norm(g), "success": True}
        H = hess(x)
        n = len(x)
        try:
            dx = np.linalg.solve(H + mu * np.eye(n), -g)
        except Exception:
            return {"x": x, "nit": k + 1, "res": np.linalg.norm(g), "success": False}
        x_new = x + dx
        if func(x_new) < func(x):
            x = x_new
            mu *= 0.5
        else:
            mu *= 2
    residual = np.linalg.norm(grad(x))
    return {"x": x, "nit": max_iter, "res": residual, "success": residual < tol}


# ==========================================================================
# 4. Trust-Region Newton
# ==========================================================================
def trust_region(func, grad, hess, x0, tol=1e-8, max_iter=100, delta0=2.0,
                metric=None) -> Dict:
    x = x0.copy().astype(float)
    delta = delta0
    # CMTR: if a covariance metric matrix M is provided (the C learned by CMA-ES), via Cholesky transformation
    # M=L L^T, let q=L^T p, transforming the ellipsoidal constraint ||p||_M<=delta into the spherical constraint ||q||_2<=delta.
    Lm = None
    if metric is not None:
        M = np.asarray(metric, dtype=float)
        if M.shape == (len(x), len(x)):
            try:
                Lm = np.linalg.cholesky(M)
            except Exception:
                Lm = None
    for k in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            return {"x": x, "nit": k + 1, "res": np.linalg.norm(g), "success": True}
        H = hess(x)
        n = len(x)
        Hreg = H + 1e-6 * np.eye(n)
        try:
            if Lm is not None:
                # Transformed subproblem: min (L^-1 g)^T q + 0.5 q^T (L^-1 H L^-T) q, ||q||<=delta
                Linv = np.linalg.inv(Lm)
                g_t = Linv @ g
                H_t = Linv @ Hreg @ Linv.T
                if _HAS_SCIPY:
                    ct, lowt = cho_factor(H_t)
                    q = cho_solve((ct, lowt), -g_t)
                else:
                    q = np.linalg.solve(H_t, -g_t)
                q_norm = np.linalg.norm(q)
                if q_norm > delta:
                    q = q * (delta / q_norm)
                p = Linv.T @ q  # transform back to the original space
                p_norm = np.linalg.norm(p)
            else:
                if _HAS_SCIPY:
                    c, low = cho_factor(Hreg)
                    p = cho_solve((c, low), -g)
                else:
                    p = np.linalg.solve(Hreg, -g)
                p_norm = np.linalg.norm(p)
                if p_norm > delta:
                    p = p * (delta / p_norm)
        except Exception:
            return {"x": x, "nit": k + 1, "res": np.linalg.norm(g), "success": False}
        actual_reduction = func(x) - func(x + p)
        predicted_reduction = -np.dot(g, p) - 0.5 * np.dot(p, H @ p)
        rho = actual_reduction / (predicted_reduction + 1e-30)
        if rho < 0.25:
            delta *= 0.5
        elif rho > 0.75 and p_norm > delta * 0.9:
            delta *= 2
        if rho > 1e-4:
            x = x + p
    residual = np.linalg.norm(grad(x))
    return {"x": x, "nit": max_iter, "res": residual, "success": residual < tol}

# ==========================================================================
# 5. CATRI Trust-Region (Covariance-Aware Trust-Region)
# Principle: use CMA-ES covariance inverse C^-1 as Hessian proxy.
# Theory: in CMA-ES convergence, C proportional to H^-1 (under sigma^2 scaling).
# Efficiency: TR per-iteration cost drops from O(d^2) func evals (FD Hessian) to O(d).
# ==========================================================================
def trust_region_catri(func, grad, hess_proxy, x0, tol=1e-8, max_iter=100,
                       delta0=2.0, metric=None):
    x = x0.copy().astype(float)
    delta = delta0
    H = np.asarray(hess_proxy, dtype=float).copy()
    n = len(x)
    H += 1e-6 * np.eye(n)
    Lm = None
    if metric is not None:
        M = np.asarray(metric, dtype=float)
        if M.shape == (n, n):
            try:
                Lm = np.linalg.cholesky(M)
            except Exception:
                Lm = None
    for k in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            return {"x": x, "nit": k + 1, "res": np.linalg.norm(g), "success": True}
        try:
            if Lm is not None:
                Linv = np.linalg.inv(Lm)
                g_t = Linv @ g
                H_t = Linv @ H @ Linv.T
                if _HAS_SCIPY:
                    ct, lowt = cho_factor(H_t)
                    q = cho_solve((ct, lowt), -g_t)
                else:
                    q = np.linalg.solve(H_t, -g_t)
                q_norm = np.linalg.norm(q)
                if q_norm > delta:
                    q = q * (delta / q_norm)
                p = Linv.T @ q
                p_norm = np.linalg.norm(p)
            else:
                g_norm_sq = float(np.dot(g, g))
                alpha_sd = g_norm_sq / float(g @ H @ g + 1e-30)
                p_sd = -alpha_sd * g
                if _HAS_SCIPY:
                    c, low = cho_factor(H)
                    p_nt = cho_solve((c, low), -g)
                else:
                    p_nt = np.linalg.solve(H, -g)
                p_nt_norm = np.linalg.norm(p_nt)
                if p_nt_norm <= delta:
                    p = p_nt
                elif np.linalg.norm(p_sd) >= delta:
                    p = p_sd * (delta / np.linalg.norm(p_sd))
                else:
                    diff = p_nt - p_sd
                    a = float(np.dot(diff, diff))
                    b = 2.0 * float(np.dot(p_sd, diff))
                    c_val = float(np.dot(p_sd, p_sd)) - delta * delta
                    tau = (-b + np.sqrt(max(b * b - 4 * a * c_val, 0.0))) / (2 * a + 1e-30)
                    tau = min(max(tau, 0.0), 1.0)
                    p = p_sd + tau * diff
                p_norm = np.linalg.norm(p)
        except Exception:
            return {"x": x, "nit": k + 1, "res": np.linalg.norm(g), "success": False}
        actual_reduction = func(x) - func(x + p)
        predicted_reduction = -np.dot(g, p) - 0.5 * np.dot(p, H @ p)
        rho = actual_reduction / (predicted_reduction + 1e-30)
        if rho < 0.25:
            delta *= 0.5
        elif rho > 0.75 and p_norm > delta * 0.9:
            delta *= 2
        if rho > 1e-4:
            x = x + p
    residual = np.linalg.norm(grad(x))
    return {"x": x, "nit": max_iter, "res": residual, "success": residual < tol}


NEWTON_SOLVERS = {
    "vanilla": vanilla_newton,
    "damped": damped_newton,
    "lm": levenberg_marquardt,
    "trust_region": trust_region,
    "trust_region_catri": trust_region_catri,
}
