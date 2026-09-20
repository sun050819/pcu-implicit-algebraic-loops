"""CMA-ES (Covariance Matrix Adaptation Evolution Strategy) global optimization (Hansen 2006 standard algorithm, pure NumPy).

Used as the global phase of the hybrid solver (the historical configuration): compared to simulated annealing, CMA-ES adaptively models variable coupling through the covariance matrix,
and is recognized as one of the strongest global methods for highly multimodal/non-separable/high-dimensional continuous functions
(Rastrigin, Schaffer, Rotated series, real nonlinear rings, etc. - scenarios where the legacy solver fails are its main strengths).

Implementation highlights (standard CMA-ES 1.2 parameter conventions):
    lambda = 4 + floor(3*log(n))   population size
    mu = lambda // 2               number of parents
    weights = log(mu+0.5)-log(1..mu), normalized
    mueff = 1/sum(w^2)             effective population size
    cc/cs/c1/cmu/damps             default recommended formulas
    p_sigma/p_c                    path cumulation
    C covariance rank-1 + rank-mu update + hsig correction

v5 conditional step-size filtering (use_v5_filter=True):
    When the covariance condition number kappa = lambda_max/lambda_min > kappa_thresh,
    scale the step size to sigma_smoothed = sigma x min(1, sqrt(kappa_thresh/kappa)),
    preventing step-size overshoot under ill-conditioned conditions. This is the implementation of the v5 conditional filtering of the HGCA solver.

Adaptive warm start (warm_start / return_state):
    Supports warm-starting from a previous CMA-ES run state, used for exploration-refinement alternating cycles.
    The warm_start dictionary contains m, sigma, C, p_sigma, p_c;
    When return_state=True, returns (best_x, best_f, n_evals, state, stagnated).
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple, Dict, Any

import numpy as np


def _safe_f(func, x) -> float:
    try:
        return float(func(x))
    except Exception:
        return 1e30


def cma_es(func: Callable[[np.ndarray], float],
           x0: np.ndarray,
           sigma0: Optional[float] = None,
           max_evals: int = 3000,
           f_target: float = 1e-10,
           seed: int = 0,
           bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
           use_v5_filter: bool = False,
           v5_kappa_thresh: float = 100.0,
           warm_start: Optional[Dict[str, Any]] = None,
           return_state: bool = False):
    """CMA-ES global minimization.

    By default returns (best_x, best_f, n_evals).
    When return_state=True, returns (best_x, best_f, n_evals, state_dict, stagnated_flag).

    Args:
        use_v5_filter: Enable v5 conditional step-size filtering (attenuate step size under ill-conditioned conditions).
        v5_kappa_thresh: Condition number threshold; when exceeded, triggers step-size attenuation (default 100).
        warm_start: Warm-start state dictionary containing m, sigma, C, p_sigma, p_c.
        return_state: Whether to return the internal state and stagnation flag.
    """
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    n = len(x0)
    rng = np.random.RandomState(seed)

    # ---- Parameters ----
    lam = max(8, 4 + int(3.0 * np.log(n)))          # population
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mueff = 1.0 / np.sum(weights ** 2)
    cc = 4.0 / (n + 4.0)
    cs = (mueff + 2.0) / (n + mueff + 5.0)
    c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs
    chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

    # ---- Initial values (warm start supported) ----
    if warm_start is not None:
        m = np.asarray(warm_start['m'], dtype=float).copy()
        sigma = float(warm_start['sigma'])
        C = np.asarray(warm_start['C'], dtype=float).copy()
        p_sigma = np.asarray(warm_start['p_sigma'], dtype=float).copy()
        p_c = np.asarray(warm_start['p_c'], dtype=float).copy()
    else:
        if sigma0 is None:
            nrm = float(np.linalg.norm(x0))
            sigma0 = max(0.5, 0.3 * nrm) if nrm > 0 else 1.0
        sigma = float(sigma0)
        m = x0.copy()
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

    best_x = m.copy()
    best_f = _safe_f(func, m)
    n_evals = 1
    if best_f < f_target:
        if return_state:
            return best_x, best_f, n_evals, {'m': m, 'sigma': sigma, 'C': C, 'p_sigma': p_sigma, 'p_c': p_c}, False
        return best_x, best_f, n_evals

    # Adaptive early stopping: terminate early when there is no significant improvement for consecutive patience generations and a convergence basin has been entered
    # (CMA-ES only needs to locate the basin; precise convergence is completed by subsequent TR refinement)
    patience = max(20, 3 * lam)
    stagnation = 0
    prev_best = best_f
    stagnated = False

    # Initial eigendecomposition
    evals_cum = 1
    # Budget upper-bound protection (against ill-conditioning)
    max_evals = int(max_evals)
    while n_evals < max_evals:
        # ---- Eigendecomposition ----
        D2, B = np.linalg.eigh(C)
        D2 = np.maximum(D2, 1e-30)
        D = np.sqrt(D2)
        BD = B * D[None, :]                         # [n,n] each column = B_i * D_i

        # v5 conditional step-size filtering: compute the current covariance condition number
        if use_v5_filter:
            kappa = float(D[-1] / max(D[0], 1e-30))
            v5_factor = min(1.0, float(np.sqrt(v5_kappa_thresh / max(kappa, 1e-30))))
        else:
            v5_factor = 1.0

        # ---- Sample lambda candidates ----
        Z = rng.randn(lam, n)
        X = m + sigma * v5_factor * (Z @ BD.T)       # [lam, n], v5 filtering applies to the sampling step size
        if bounds is not None:
            lo, hi = bounds
            X = np.clip(X, lo, hi)
        F = np.empty(lam)
        for i in range(lam):
            F[i] = _safe_f(func, X[i])
        n_evals += lam
        evals_cum += lam

        # ---- Selection ----
        order = np.argsort(F)
        X_sel = X[order[:mu]]
        F_sel = F[order[:mu]]
        if F_sel[0] < best_f:
            best_f = F_sel[0]
            best_x = X_sel[0].copy()
            if best_f < f_target:
                break
        # Adaptive early stopping: no significant improvement for consecutive patience generations and a convergence basin has been entered
        improvement = prev_best - best_f
        if improvement < max(1e-12, 1e-8 * max(1.0, abs(prev_best))):
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best_f
        if stagnation >= patience and best_f < max(f_target * 100.0, 1e-4):
            stagnated = True
            break

        # ---- Update mean ----
        m_old = m.copy()
        m = np.sum(weights[:, None] * X_sel, axis=0)

        # ---- Update paths (rank-1 + rank-mu) ----
        # Note: path updates use the effective step size sigma_eff = sigma * v5_factor
        sigma_eff = sigma * v5_factor
        y_w = (m - m_old) / sigma_eff
        # y_i of each selected individual (used for rank-mu)
        y_i = (X_sel - m_old) / sigma_eff
        p_sigma = (1.0 - cs) * p_sigma + np.sqrt(cs * (2.0 - cs) * mueff) * (B @ (D ** -1 * (B.T @ y_w)))
        norm_p = float(np.linalg.norm(p_sigma))
        denom = np.sqrt(1.0 - (1.0 - cs) ** (2.0 * evals_cum / lam))
        hsig = 1.0 if (norm_p / denom < 1.4 + 2.0 / (n + 1.0)) else 0.0
        p_c = (1.0 - cc) * p_c + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * y_w

        C = ((1.0 - c1 - cmu) * C
             + c1 * (np.outer(p_c, p_c) + (1.0 - hsig) * cc * (2.0 - cc) * C)
             + cmu * np.einsum('i,ij,ik->jk', weights, y_i, y_i))
        # Keep symmetric
        C = (C + C.T) / 2.0

        # ---- Update step size (standard CMA-ES step-size update) ----
        sigma = sigma * np.exp((cs / damps) * (norm_p / chi_n - 1.0))
        if not (np.isfinite(sigma) and sigma > 1e-30):
            sigma = 1e-2

    state = {'m': m, 'sigma': sigma, 'C': C, 'p_sigma': p_sigma, 'p_c': p_c}
    if return_state:
        return best_x, best_f, n_evals, state, stagnated
    return best_x, best_f, n_evals
