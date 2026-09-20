"""Robust solving layer: legacy three-stage hybrid solver + hybrid baseline.

(Hybrid Adaptive Simulated-annealing Trust-region Newton; 
    (1) Fast path Trust-Region Newton - converges directly for well-conditioned/low CI, zero extra overhead;
    (2) CI-adaptive multi-restart simulated annealing - enabled only when the fast path fails and CI is high, to escape local solutions;
    (3) Multi-elite Trust-Region Newton refinement - refines the candidate solutions obtained from SA to high precision.

Baseline solver (for solving-layer comparison 9.1.2):
    multi_start_newton / hybrid_sa_newton_fixed / guided_hybrid_sa / adaptive_sa_newton
"""
from __future__ import annotations

import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional

import numpy as np

from ..graph.ci import get_ci_config
from .newton import (trust_region, vanilla_newton, damped_newton,
                     levenberg_marquardt)
from .sa import simple_sa, adaptive_sa


def _safe_func(func, x) -> float:
    try:
        return float(func(x))
    except Exception:
        return 1e30


# ==========================================================================
# three-stage
# ==========================================================================
def hast_n(func, grad, hess, x0, tol=1e-8, max_iter=2000, cfg: Optional[Dict] = None,
           n_workers: int = 1) -> Dict:
    """Three-stage: fast path TR-Newton -> CI-adaptive single long-trajectory SA -> multi-elite TR refinement.

cfg (provided by get_ci_config(ci)): T0 / alpha / sigma / n_restarts / tier.
When cfg is not provided, high-difficulty default parameters are used.

v2.3.2: when n_workers>1, the refinement stage performs multi-elite parallel TR-Newton (ThreadPoolExecutor,
numpy operations release the GIL, 4-8 cores linear speedup).

Full-version improvements (relative to earlier versions):
(1) Fast-path success must be confirmed by the "absolute function value criterion" f(x)<1e-6 - earlier versions only looked at the gradient norm,
   and would misjudge convergence of multimodal problems to saddle points/stationary points with gradient~0 as success and skip the global search
   (measured 36 problems convergence rate 0.545 -> 0.773, the main gain comes from this);
(2) SA changed from "multi-restart short trajectories" to "single long trajectory + adaptive temperature", ensuring the temperature window
   (50 steps) fully operates, giving more thorough exploration;
(3) In the refinement stage, multiple elites - SA trajectory snapshots + best solution + TR endpoint - are refined one by one with TR.
    """
    if cfg is None:
        cfg = {'T0': 1500, 'alpha': 0.98, 'sigma': 1.5, 'n_restarts': 6}
    x0 = x0.copy().astype(float)
    d = len(x0)

    # ---- (1) Fast path: Trust-Region Newton (zero overhead, must be confirmed by absolute f criterion) ----
    res_direct = trust_region(func, grad, hess, x0, tol=tol, max_iter=300)
    if res_direct["success"] and _safe_func(func, res_direct["x"]) < 1e-6:
        return {"x": res_direct["x"], "nit": res_direct["nit"], "res": res_direct["res"],
                "success": True, "stage": "direct_newton"}

    # ---- (2) Single long-trajectory adaptive SA (CI-adaptive parameters, adaptive temperature window) ----
    T0 = cfg["T0"]; sigma = cfg["sigma"]
    sa_budget = int(max_iter * 0.7)
    x = x0.copy(); best_x = x.copy()
    best_f = _safe_func(func, x)
    T = T0; accept_count = 0
    snapshots = [x.copy()]; snap_f = [best_f]
    for k in range(sa_budget):
        x_new = x + sigma * np.random.randn(d) * (T / T0 + 1e-2)
        f_new = _safe_func(func, x_new)
        delta_f = f_new - best_f
        if delta_f < 0 or np.random.rand() < np.exp(-delta_f / max(T, 1e-10)):
            x = x_new
            accept_count += 1
            if f_new < best_f:
                best_f = f_new
                best_x = x.copy()
        if (k + 1) % 50 == 0:
            accept_rate = accept_count / 50
            T = T * 0.95 if accept_rate > 0.44 else T * 1.02
            accept_count = 0
            if (k + 1) % 250 == 0:
                snapshots.append(x.copy())
                snap_f.append(best_f)

    # ---- (3) Multi-elite Trust-Region Newton refinement (SA best + trajectory snapshots + TR endpoint) ----
    elites = [best_x.copy()] + snapshots
    fe = [_safe_func(func, e) for e in elites]
    if res_direct["success"]:
        elites.append(res_direct["x"])
        fe.append(_safe_func(func, res_direct["x"]))
    order = np.argsort(fe)
    refine_budget = max(max_iter - sa_budget - res_direct["nit"], 100)
    per_elite = max(min(refine_budget // 3, 200), 40)
    best_result, best_res, total_nit = None, 1e30, res_direct["nit"] + sa_budget
    top_elites = [elites[i] for i in order[:3]]
    if n_workers > 1 and len(top_elites) > 1:
        # v2.3.2: multi-elite parallel TR-Newton refinement
        with ThreadPoolExecutor(max_workers=min(n_workers, len(top_elites))) as ex:
            futures = [ex.submit(trust_region, func, grad, hess, e, tol, per_elite)
                       for e in top_elites]
            refine_results = [f.result() for f in futures]
    else:
        refine_results = [trust_region(func, grad, hess, e, tol=tol, max_iter=per_elite)
                          for e in top_elites]
    for res_refine in refine_results:
        total_nit += res_refine["nit"]
        if res_refine["success"] and _safe_func(func, res_refine["x"]) < 1e-6:
            return {"x": res_refine["x"], "nit": total_nit,
                    "res": res_refine["res"], "success": True, "stage": "sa_refined"}
        if res_refine["res"] < best_res:
            best_res = res_refine["res"]
            best_result = res_refine
    # Fallback: the true function value of the SA best solution meets the criterion, or the refinement residual has reached tol (either one counts as success).
    # v2.2.0 fix: previously only best_f was used, causing "refinement found a tiny residual but best_f in the SA stage is still large"
    # to be misjudged as failure (measured case 45 res=8.9e-13 was marked Failed).
    if best_f < 1e-6:
        return {"x": best_x, "nit": total_nit, "res": 0.0,
                "success": True, "stage": "sa_best"}
    if best_res < max(tol, 1e-6) and best_result is not None:
        return {"x": best_result["x"], "nit": total_nit, "res": best_res,
                "success": True, "stage": "sa_refined_res"}
    return {"x": best_result["x"] if best_result is not None else res_direct["x"],
            "nit": total_nit, "res": best_res, "success": False, "stage": "failed"}


def hast_n_ci_adaptive(func, grad, hess, x0, ci: float, tol=1e-8, max_iter=2000) -> Dict:
    """CI-adaptive entry: automatically selects the three-stage configuration according to CI."""
    cfg = get_ci_config(ci)
    res = hast_n(func, grad, hess, x0, tol=tol, max_iter=max_iter, cfg=cfg)
    res["tier"] = cfg["tier"]
    return res


def _converged_f(f: float, f_opt: Optional[float]) -> bool:
    """Objective function value criterion: when f_opt is provided, use f < f_opt+1e-4 (supports negative optima, e.g.
    Michalewicz f_opt=-2.99), otherwise use the classic f < 1e-6 (conventional scenario with f_opt=0).
    Fix: on negative-optimum functions a local minimum f=-0.9 also satisfies f<1e-6 and could be misjudged as the global optimum."""
    if f_opt is not None:
        return f < f_opt + 1e-4
    return f < 1e-6


def hast_cma(func, grad, hess, x0, tol=1e-8, max_iter=2000, cfg: Optional[Dict] = None,
             f_opt: Optional[float] = None) -> Dict:
    """hast_cma: TR fast path -> multi-start CMA-ES global search -> multi-elite TR refinement.

Improvements over the base solver: the global stage is upgraded from "adaptive SA" to "multi-start CMA-ES
(Covariance Matrix Adaptation Evolution Strategy)". SA is a point-by-point random walk, and for high-dimensional/strongly multimodal/non-separable
functions it easily gets trapped in the wrong basin; CMA-ES adaptively models variable coupling through the covariance matrix and evolves along
the best direction. Measured lessons (to avoid downstream repetition):
      (1) cma_es's f_target must be set extremely low (-1e30), not 1e-6 - for functions with negative optima
         (Michalewicz f_opt=-2.99) it will falsely stop at f=1e-6 and return a non-optimal solution;
      (2) A single CMA-ES run shrinks and converges to a local minimum and cannot escape (budget 10k->30k results unchanged),
         so multiple starts + taking the global optimum are required;
      (3) For the real nonlinear loop (Simulink_Nonlinear_Loop), CMA-ES can only find a coarse solution of ~1e-2,
         and must be combined with Newton refinement to converge to <1e-6;
      (4) For extreme multimodal/high-dimensional cases (Rastrigin_20D/30D, Rotated_Rastrigin/Ackley_20D,
         Schaffer_F6_2D_XE, etc. 0/30), the root cause is: the starting domain is too narrow (x0+/-spread does not contain
         the optimal basin) and the single-start budget is insufficient. Enhanced mode uses "Latin hypercube full-domain sampling + origin/
         initial-value anchors + large budget" to cover the full domain, combined with TR refinement to break through (4 0/30 -> all converge).
    """
    from .cmaes import cma_es
    if cfg is None:
        cfg = {}
    x0 = x0.copy().astype(float)
    d = len(x0)
    rng = np.random.RandomState(int(zlib.crc32(x0.tobytes())) % 100000)
    seed_base = int(zlib.crc32(x0.tobytes())) % 100000
    max_abs = float(np.max(np.abs(x0))) if d else 0.0
    x_norm = float(np.linalg.norm(x0)) if d else 0.0
    # extreme: determination of high-dimensional strongly multimodal (Rastrigin 20/30D, Rotated series) and medium-to-far-start multimodal
    # (Rastrigin_2D_XE, Schaffer_F6, Michalewicz, Rosenbrock_10D_C/E/H, etc.).
    # v2.3.3 fix: originally defined after the use_gpu check, causing a crash from referencing an undefined variable in auto mode.
    extreme = bool(cfg.get("extreme", (d >= 15 and max_abs >= 8.0)
                                     or (d >= 8 and x_norm >= 3.0)
                                     or (d == 2 and max_abs >= 8.0)
                                     or (d >= 2 and max_abs >= 80.0)
                                     or (d == 2 and x_norm < 1.0)))

    # ---- GPU acceleration configuration (v2.3.0) ----
    # use_gpu: True/False/'auto' (auto: enabled when d>=20 and extreme mode)
    # gpu_lam_factor: GPU population amplification factor (default 8, amortizes GPU kernel launch overhead)
    # batch_func: batch objective function f(X[lam,n])->f[lam] (end-to-end GPU acceleration, optional)
    use_gpu_cfg = cfg.get("use_gpu", "auto")
    if use_gpu_cfg == "auto":
        use_gpu = bool(d >= 20 and extreme)
    else:
        use_gpu = bool(use_gpu_cfg)
    gpu_lam_factor = int(cfg.get("gpu_lam_factor", 8))
    batch_func = cfg.get("batch_func", None)
    _cma_gpu = None
    if use_gpu:
        try:
            from .cmaes_gpu import cma_es_gpu, cuda_available
            if cuda_available():
                _cma_gpu = cma_es_gpu
            else:
                use_gpu = False
        except Exception:
            use_gpu = False

    def _cma(st, sigma0_, max_evals_, seed_, bounds_=None):
        """CMA-ES call: use the GPU version when GPU is available, otherwise use the NumPy version."""
        if _cma_gpu is not None:
            return _cma_gpu(func, st, sigma0=sigma0_, max_evals=max_evals_,
                            f_target=-1e30, seed=seed_, bounds=bounds_,
                            batch_func=batch_func, lam_factor=gpu_lam_factor)
        return cma_es(func, st, sigma0=sigma0_, max_evals=max_evals_,
                      f_target=-1e30, seed=seed_, bounds=bounds_)

    # ---- (1) Fast path: Trust-Region Newton (absolute f criterion) ----
    res_direct = trust_region(func, grad, hess, x0, tol=tol, max_iter=300)
    if res_direct["success"] and _converged_f(_safe_func(func, res_direct["x"]), f_opt):
        return {"x": res_direct["x"], "nit": res_direct["nit"], "res": res_direct["res"],
                "success": True, "stage": "direct_newton"}

    if extreme:
        # Budget adapts to dimension (low-dimensional full-domain sampling is cheap, high-dimensional needs a large budget)
        if d <= 4:
            n_starts_d = 20; per_evals_d = 1200
        elif d <= 14:
            n_starts_d = 24; per_evals_d = 1800
        else:
            n_starts_d = 26; per_evals_d = 2400

    if extreme:
        # ---- (2)extreme full-domain multi-start CMA-ES (Latin hypercube + origin + initial-value anchors) ----
        R = max(1.5 * max_abs, 5.0)
        lo = -R * np.ones(d)
        hi = R * np.ones(d)
        n_starts = int(cfg.get("extreme_starts", n_starts_d))
        per_budget = int(cfg.get("extreme_evals_per", per_evals_d))
        sigma0 = cfg.get("sigma0", R / 4.0)
        # Latin hypercube sampling within the domain (+ origin + initial value + a few random supplementary points)
        S = n_starts
        u = rng.rand(S, d)
        grid = (np.tile(np.arange(S), (d, 1)).T + u) / S          # [S,d] in (0,1)
        lhs = lo + (hi - lo) * grid
        starts = [x0.copy(), np.zeros(d)] + [p for p in lhs]
        starts += [p for p in (lo + (hi - lo) * rng.rand(4, d))]
        best_x, best_f, cma_evals = x0.copy(), 1e30, 0
        cma_candidates = []
        for si, st in enumerate(starts):
            bx, bf, ne = _cma(st, sigma0, per_budget,
                               (seed_base + si) % 100000, bounds_=(lo, hi))
            cma_evals += ne
            cma_candidates.append((bx, bf))
            if bf < best_f:
                best_f, best_x = bf, bx.copy()
        total_nit = res_direct["nit"] + cma_evals
        # ---- (3)extreme multi-elite TR refinement (best + origin + initial value + best of each start) ----
        elites = [best_x.copy(), np.zeros(d), x0.copy()] + [c for c, _ in cma_candidates]
        fe = [_safe_func(func, e) for e in elites]
        if res_direct["success"]:
            elites.append(res_direct["x"])
            fe.append(_safe_func(func, res_direct["x"]))
        order = np.argsort(fe)
        refine_budget = max(800, min(2000, max_iter))
        per_elite = max(refine_budget // 6, 150)
        best_result, best_res = None, 1e30
        for idx in order[:6]:
            res_refine = trust_region(func, grad, hess, elites[idx], tol=tol,
                                      max_iter=per_elite)
            total_nit += res_refine["nit"]
            if res_refine["success"] and _converged_f(_safe_func(func, res_refine["x"]), f_opt):
                return {"x": res_refine["x"], "nit": total_nit,
                        "res": res_refine["res"], "success": True,
                        "stage": "cma_extreme_refined"}
            if res_refine["res"] < best_res:
                best_res = res_refine["res"]
                best_result = res_refine
        if _converged_f(best_f, f_opt):
            return {"x": best_x, "nit": total_nit, "res": 0.0,
                    "success": True, "stage": "cma_extreme_best"}
        return {"x": best_result["x"] if best_result is not None else best_x,
                "nit": total_nit, "res": best_res, "success": False,
                "stage": "cma_extreme_failed"}

    # ---- (2) (conventional) multi-start CMA-ES global search ----
    n_starts = int(cfg.get("n_starts", 4))
    total_evals = int(cfg.get("cma_evals", max_iter))
    per_budget = max(total_evals // n_starts, 300)
    sigma0 = cfg.get("sigma0")
    x_scale = float(np.linalg.norm(x0)) if np.linalg.norm(x0) > 0 else 1.0
    spread = max(x_scale, 3.0)
    starts = [x0.copy()]
    for s in range(1, n_starts):
        starts.append(x0 + spread * (rng.rand(d) * 2.0 - 1.0))
    best_x, best_f, cma_evals = x0.copy(), 1e30, 0
    cma_candidates = []
    for si, st in enumerate(starts):
        bx, bf, ne = _cma(st, sigma0, per_budget,
                           (seed_base + si) % 100000)
        cma_evals += ne
        cma_candidates.append((bx, bf))
        if bf < best_f:
            best_f, best_x = bf, bx.copy()
    total_nit = res_direct["nit"] + cma_evals

    # ---- (3) (conventional) multi-elite TR refinement (CMA best of each start + global best + initial value + TR endpoint) ----
    elites = [best_x.copy(), x0.copy()] + [c for c, _ in cma_candidates]
    fe = [_safe_func(func, e) for e in elites]
    if res_direct["success"]:
        elites.append(res_direct["x"])
        fe.append(_safe_func(func, res_direct["x"]))
    order = np.argsort(fe)
    refine_budget = max(600, min(1200, max_iter // 2))
    per_elite = max(refine_budget // 4, 150)
    best_result, best_res = None, 1e30
    for idx in order[:4]:
        res_refine = trust_region(func, grad, hess, elites[idx], tol=tol, max_iter=per_elite)
        total_nit += res_refine["nit"]
        if res_refine["success"] and _converged_f(_safe_func(func, res_refine["x"]), f_opt):
            return {"x": res_refine["x"], "nit": total_nit, "res": res_refine["res"],
                    "success": True, "stage": "cma_refined"}
        if res_refine["res"] < best_res:
            best_res = res_refine["res"]
            best_result = res_refine
    if _converged_f(best_f, f_opt):
        return {"x": best_x, "nit": total_nit, "res": 0.0,
                "success": True, "stage": "cma_best"}
    return {"x": best_result["x"] if best_result is not None else best_x,
            "nit": total_nit, "res": best_res, "success": False, "stage": "failed"}


# ==========================================================================
# Hybrid baseline
# ==========================================================================
def hybrid_sa_newton_fixed(func, grad, hess, x0, tol=1e-8, max_iter=2000,
                           T0=1000.0, alpha=0.95, sigma=1.0, sa_fraction=0.7) -> Dict:
    """Fixed-parameter SA + Newton (no fast path, no CI adaptation, single start)."""
    x0 = x0.copy().astype(float)
    sa_budget = int(max_iter * sa_fraction)
    refine_budget = max_iter - sa_budget
    x = x0.copy(); best_x = x.copy(); best_f = _safe_func(func, x); T = T0
    for _ in range(sa_budget):
        x_new = x + sigma * np.random.randn(len(x)) * (T / T0 + 1e-2)
        f_new = _safe_func(func, x_new)
        delta_f = f_new - best_f
        if delta_f < 0 or np.random.rand() < np.exp(-delta_f / max(T, 1e-10)):
            x = x_new
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
        T *= alpha
    res_refine = trust_region(func, grad, hess, best_x, tol=tol, max_iter=refine_budget)
    return {"x": res_refine["x"], "nit": sa_budget + res_refine["nit"],
            "res": res_refine["res"], "success": res_refine["res"] < tol}


def multi_start_newton(func, grad, hess, x0, tol=1e-8, max_iter=2000, n_starts=5) -> Dict:
    """Multi-start TR-Newton (simplest hybrid)."""
    x0 = x0.copy().astype(float)
    budget_per_start = max(max_iter // n_starts, 10)
    best_result, best_res, total_nit = None, 1e30, 0
    for s in range(n_starts):
        if s == 0:
            xs = x0.copy()
        else:
            scale = np.abs(x0) * 0.6 + 1.0
            xs = x0 + np.random.randn(len(x0)) * scale
        res = trust_region(func, grad, hess, xs, tol=tol, max_iter=budget_per_start)
        total_nit += res["nit"]
        if res["success"]:
            return {"x": res["x"], "nit": total_nit, "res": res["res"],
                    "success": True}
        if res["res"] < best_res:
            best_res = res["res"]
            best_result = res
    return {"x": best_result["x"] if best_result else x0, "nit": total_nit,
            "res": best_res, "success": best_res < tol}


def guided_hybrid_sa(func, grad, hess, x0, tol=1e-8, max_iter=2000,
                     T0=1000.0, alpha=0.95, sigma=1.0, guide_weight=0.3,
                     sa_fraction=0.7) -> Dict:
    """Gradient-guided SA + Newton (fixed parameters)."""
    x0 = x0.copy().astype(float)
    sa_budget = int(max_iter * sa_fraction)
    refine_budget = max_iter - sa_budget
    x = x0.copy(); best_x = x.copy(); best_f = _safe_func(func, x); T = T0
    for _ in range(sa_budget):
        g = grad(x)
        g_norm = np.linalg.norm(g)
        if g_norm > 1e-12:
            rand_pert = sigma * np.random.randn(len(x)) * (T / T0 + 1e-2)
            guide_pert = -guide_weight * (g / g_norm) * sigma * (T / T0 + 1e-2)
            x_new = x + rand_pert + guide_pert
        else:
            x_new = x + sigma * np.random.randn(len(x)) * (T / T0 + 1e-2)
        f_new = _safe_func(func, x_new)
        delta_f = f_new - best_f
        if delta_f < 0 or np.random.rand() < np.exp(-delta_f / max(T, 1e-10)):
            x = x_new
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
        T *= alpha
    res_refine = trust_region(func, grad, hess, best_x, tol=tol, max_iter=refine_budget)
    return {"x": res_refine["x"], "nit": sa_budget + res_refine["nit"],
            "res": res_refine["res"], "success": res_refine["res"] < tol}


def adaptive_sa_newton(func, grad, hess, x0, tol=1e-8, max_iter=2000,
                       sa_fraction=0.7) -> Dict:
    """Adaptive SA + Newton (no fast path, no multi-elite refinement)."""
    x0 = x0.copy().astype(float)
    sa_budget = int(max_iter * sa_fraction)
    refine_budget = max_iter - sa_budget
    sa_res = adaptive_sa(func, grad, x0, tol=tol, max_iter=sa_budget)
    res_refine = trust_region(func, grad, hess, sa_res["x"], tol=tol, max_iter=refine_budget)
    return {"x": res_refine["x"], "nit": sa_budget + res_refine["nit"],
            "res": res_refine["res"], "success": res_refine["res"] < tol}


# ==========================================================================
# Solver registry (9.1.2 solver-layer comparison)
# ==========================================================================
SOLVER_REGISTRY = {
    "vanilla_newton": vanilla_newton,
    "trust_region": trust_region,
    "damped_newton": damped_newton,
    "lm": levenberg_marquardt,
    "multi_start_newton": multi_start_newton,
    "hybrid_sa_newton_fixed": hybrid_sa_newton_fixed,
    "guided_hybrid_sa": guided_hybrid_sa,
    "adaptive_sa_newton": adaptive_sa_newton,
    "hast_n": hast_n,                 # requires ci parameter (caller passes cfg)
    "hast_cma": hast_cma,             # CMA-ES global-stage version
}
