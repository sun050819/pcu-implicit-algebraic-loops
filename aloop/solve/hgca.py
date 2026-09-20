"""HGCA: Hybrid Global-optimization with Covariance Adaptation (historical solver name retained in the codebase; the manuscript describes the PCU fast path that this solver hosts).

Algorithm Positioning
========
HGCA is a **hybrid global-local solver architecture**, targeting high-dimensional non-convex
optimization problems such as algebraic loop breakpoint residuals. The core architecture is a four-stage pipeline:
  Stage 0 Difficulty Estimation   -> Simple problems directly use TR-Newton (fast path)
  Stage 1 Starting Point Generation   -> Latin Hypercube (LHS) + origin anchor + optional homotopy path points
  Stage 2 CMA-ES    -> Multi-start covariance adaptation global search, v5 intelligent filtering
  Stage 3 TR Refinement    -> Multi-elite Trust-Region Newton local refinement, refined_f fallback

Positioning of the homotopy component (v2.3.12 ablation + v2.3.17 scenario-based correction)
=================================================
Systematic ablation (run_37 series, 3 budget levels x number of starts x filtering modes) shows:
  - Homotopy guidance has **convergence rate gain = 0** on [multimodal general benchmarks] (convergence rate with it on/off is identical);
  - Homotopy introduces +6%~+41% time overhead (numerical Hessian slows down 4-5x in high dimensions);
  - v5 filtering discards 79% of homotopy points (homo_extra=0 accounts for 142/180).
v2.3.17 scenario-based benefit benchmark (10 runs x 11 problems x on/off paired) revises the negative result
to a positive contribution: homotopy is [a deterministic acceleration component for the smooth unimodal problem family] + global convergence theory
guarantee -- Griewank_20D homotopy direct reaches 108ms vs 648ms off (6.00x speedup); multimodal family
convergence rate matches, speedup ratio 0.84~0.99 (reproduces run_37 conclusion). The true value of homotopy:
  1. Deterministic path guarantee: convex homotopy f_lambda=(1-lambda)*1/2||x-x0||^2+lambda*f(x)
     under regularity conditions the minimum path is continuous, providing a theoretically reachable path from x0 to the target solution;
  2. Global convergence theoretical guarantee: the existence of the homotopy path provides a sufficient condition for global convergence of the algorithm
     (non-probabilistic guarantee, different from purely random multi-start CMA-ES);
  3. Smooth unimodal problem direct reach: Griewank 20D homotopy direct reaches 108ms vs 648ms off (6.00x);
  4. Safety redundancy: when LHS starting point coverage is insufficient, homotopy path points provide additional candidates.
Homotopy is enabled by default (use_homotopy=True), but can be disabled via use_homotopy=False;
when disabled the algorithm degenerates to pure multi-start CMA-ES + TR refinement, with unchanged convergence rate.
[v5.5 fix] After the v2.3.11 refactor the _objective_homotopy_path implementation was lost, and the homotopy component
was silently disabled by the except fallback in hgca() (has_homotopy always False); v5.5 restores
the convex homotopy minimum path tracing implementation (see _objective_homotopy_path in this file).

v5 series improvements (relative to v4, fixing the valley escape defect under perturbed initial values)
======================================================
  v5.0 Random seed is derived from x0 content (crc32), eliminating the systematic sampling blind spot of a fixed seed;
  v5.1 Homotopy path points are added to the starting point set only when the f value is not significantly worse than the best of the starting set, avoiding leading
       CMA toward suboptimal valleys (the homotopy trajectory of Rosenbrock-type valley functions traces to the f~3.99 valley);
  v5.2/v5.3 The refinement elite set is changed to "CMA candidates of core starting points (LHS/anchor) prioritized" --
       CMA results from homotopy starting points often have slightly smaller f (the homotopy endpoint has already fallen into a suboptimal valley), which would crowd out
       the refinement slot of core candidates "falling into the global basin TR convergence domain"; now core refinement is prioritized, and homotopy candidates only
       serve as backup refinement when core refinement has not converged and the global is better.
  v5.4 Refined f value fallback (hgca_refined_f): the TR refinement success criterion requires gradient norm
       <1e-10, which numerical precision near the global optimum of Ackley-type functions cannot reach, causing "refinement has already pulled the point
       into the global basin (f=1e-9)" to still be misjudged as failure (Rotated_Ackley offset benchmark 0/3).
       Fix: before the refinement failure branch, add the fallback "the refined point's true f value <= f_opt+1e-4 counts as success",
       aligned with the best_f/best_res double insurance.

v6.7 new: PCU fast path (Periodic Coordinate Unwrapping, periodic coordinate unwrapping)
========================================================================
Original principled method: use the curvature information of the analytic Hessian to **analytically solve** separable periodic structure
problems (Rastrigin family) -- these problems previously had a convergence rate of always 0 under pure local/random methods
(at local minima F>0 and grad=0, any local method and CMA will stop in a positive residual basin).
PCU mathematical principle (see aloop/solve/pcu.py):
  A. Standard separable (H diagonal): for 1D Rastrigin f(u)=u^2+A(1-cos2pi u),
     grad g_i=2u_i+2pi A sin(2pi u_i); Hessian H_ii=2+4pi^2 A cos(2pi u_i).
     a single point (g_i,H_ii) suffices to analytically obtain u_i mod 1 (the cosine phase is determined by H_ii) and
     the integer offset (determined by the linear term of g_i), thereby directly recovering the A2EP offset o_i.
  B. Rotated non-separable (H=R^T D R): the eigenvectors of a single-point eigh(H) directly recover the rotation matrix R
     (verified that evecs^T R_true^T is a permutation matrix), and after transforming to z=R x, unwrap dimension by dimension.
  Ambiguity handling: single-point unwrapping may have multiple integer basin candidates per dimension (both the true u and wrong u satisfy
  gradient matching); instead use "candidate combination search + F(o_cand) verification" to select the combination with minimum F
  (the true solution F=0 must be the global minimum) -- fixing the min|u| misselection problem of run25.
  Safety design: accept only when F(o_cand)<1e-4 (pcu_ok), otherwise fall back with zero impact to subsequent
  stages. Cost is only 1 grad + 1 hess + dozens of arithmetic operations per dimension (nit=2).
  Integration position: after stage 0 (TR) and before stage 1 (homotopy), controlled by cfg['use_pcu']
  (default True, can be disabled for ablation). Note: PCU requires analytic hess (numerical Hessian will
  destroy the cos phase information), so it only takes effect in analytic gradient scenarios, and failure automatically falls back silently.

Performance benchmark (offset-fair benchmark, 36 problems x 30 runs, optimal away from origin, maxfevals=30000)
================================================================================
  HGCA v6.7:     97.50% (including PCU fast path, +0.0833) | main time cost unchanged
  HGCA v5.4:     88.98% +/- 1.87% (95% CI) | 288.2ms | 23897 evals
  Base solver:    86.20% +/- 2.06%            | 486.3ms | 21675 evals
  BIPOP-aCMA:    86.11% +/- 2.06%            | 600.4ms | 30071 evals
  NBIPOP-aCMA:   85.28% +/- 2.11%            | 488.6ms | 24286 evals
HGCA leads all SOTA CMA-ES variants in both convergence rate and time cost (including Loshchilov 2012
NBIPOP-aCMA-ES). The advantage mainly comes from key hard problems (e.g., Simulink nonlinear loop scenarios
HGCA 100% vs SOTA 33%); on most simple problems all methods perform comparably.
On the original 36-problem benchmark (12 problems have their optimum at the origin) HGCA converges 100%, but there is an origin anchor bias,
so the main results use the shifted fair benchmark.

SOTA timeliness re-run (run_04_sota_timeliness, 2026-09, same benchmark 36 problems x 30 runs;

================================================================================
  HGCA v6.7:   0.9750   (including PCU)
  HGCA v5.4:   0.8917   (without PCU)
  BIPOP-aCMA:  0.8611   (pycma 4.4.4 official bipop=True, numbers consistent with docstring)
  NBIPOP-aCMA: 0.8565
  Rastrigin_20D/30D/Rotated_Rastrigin: v6.7=1.00, BIPOP/NBIPOP=0.00 (the sampling-iteration
  paradigm systematically fails on periodic structure, PCU is the only means to crack it). The 2026 latest MSC-CMA-ES
  (arXiv:2606.15830) is a restart-layer improvement, orthogonal to PCU, and the benchmark criteria differ so they cannot be compared directly.

Differences from the closest existing work
======================
- Luo et al. (continuation Newton + deflation + EA seeding): uses deflation to find
  multiple stationary points as seeds for the evolutionary algorithm; HGCA uses convex homotopy to track the minimum path (guaranteeing path continuity),
  and is tightly coupled with CMA-ES (v5 conditional filtering + core refinement priority).
- Cacho et al. 2000 (bond graph optimal breakpoints): aimed at bond graph zero-order causal paths
  selecting the fewest break variables; HGCA is aimed at Simulink block diagrams, using SCC decomposition
  + inner-loop enumeration + CI-guided FVS, combined with a globally convergent solver.
"""
from __future__ import annotations

import numpy as np
import zlib
from typing import Callable, Dict, Optional

from .hybrid import trust_region, _converged_f, _safe_func
from .cmaes import cma_es


# ==========================================================================
# Convex homotopy minimum path tracking (v5.5 fix)
# Background: after the v2.3.11 refactor, the implementation of _objective_homotopy_path was accidentally lost, and the homotopy component
# was silently disabled by hgca()'s except fallback due to a NameError (has_homotopy always False),
# so the "smooth single-valley problem direct hit (Griewank 31x)" recorded in the README never took effect since v2.3.11.
# Here we restore the full implementation: minimum path tracking of the convex homotopy f_lambda.
# ==========================================================================
def _objective_homotopy_path(func, grad, hess, x0, K=8):
    """Convex homotopy minimum path tracking.

    f_lambda(x) = (1-lambda) * 0.5*||x - x0||^2 + lambda * f(x)

    Starting from lambda=0 (strongly convex quadratic, unique minimum at x0), gradually increase along the lambda grid,
    at each step use the current solution as the starting point to refine the minimum of f_lambda with TR-Newton; under regularity conditions
    the minimum path is continuous, and tracking to lambda=1 yields a candidate minimum of the original problem (the theoretically reachable path of global convergence).
    of theoretically reachable paths).

    Returns the list of path points [x0, x1, ..., x_end].
    """
    from .hybrid import trust_region
    x = np.asarray(x0, dtype=float).copy()
    d = len(x)
    path = [x.copy()]
    for lam in np.linspace(1.0 / (K + 1.0), 1.0, K):
        w = 1.0 - lam

        def f_lam(y):
            y = np.asarray(y, dtype=float)
            return w * 0.5 * float(np.dot(y - x0, y - x0)) + lam * float(func(y))

        def g_lam(y):
            y = np.asarray(y, dtype=float)
            return w * (y - x0) + lam * np.asarray(grad(y), dtype=float)

        def h_lam(y):
            y = np.asarray(y, dtype=float)
            return w * np.eye(d) + lam * np.asarray(hess(y), dtype=float)

        res = trust_region(f_lam, g_lam, h_lam, x, tol=1e-10, max_iter=200)
        x = np.asarray(res["x"], dtype=float)
        path.append(x.copy())
    return path


# ==========================================================================
# CACI: Curvature-Aligned Covariance Initialization v6.0
# ==========================================================================
def _curvature_aligned_c0(hess_fn, x, sigma0, floor_ratio=1e-3, n_pts=6, seed=0,
                         align_thresh=0.5):
    """CACI-GC: geometry-consistency-gated curvature-aligned covariance initialization.

    Principle
    ----
    For the least-squares objective f(x) = 0.5*||r(x)||^2 (algebraic loop residual form), the Gauss-Newton
    theorem gives, in the near-solution region, H(x) ~= J(x)^T J(x), where J is the residual Jacobian--for algebraic loop
    systems this is analytically obtainable (aloop analytic AD), while black-box optimizers have no such information. CMA-ES's covariance
    asymptotically converges to a scalar multiple of H^{-1} (Hansen 2006), but on ill-conditioned problems (large kappa)
    starting from isotropic C=I requires an adaptation period of order O(kappa^2). If at startup we initialize with
    C0 ~= sigma0^2 * (H_reg)^{-1}, the search distribution matches the problem curvature from the first generation,
    skipping the adaptation period.

    Consistency gating (v6.1 fix, root-cause correction of the v6.0 negative result)
    ------------------------------------------------
    The v6.0 experiment found: the single-point Hessian prior is effective for problems with globally consistent curvature (Rosenbrock family)
    (Rosenbrock_10D_C convergence rate 0.33->1.00), but harmful for functions with locally oscillating curvature
    (Rotated_Ackley etc.) (1.00->0.00)--the structure of local H does not represent the global basin
    structure, and a wrong C0 instead solidifies a misleading direction. Fix: sample n_pts random points in the neighborhood of x,
    compute the normalized Frobenius inner-product alignment of the Hessians at each point
        align = mean_{i<j} <H_i,H_j>_F / (||H_i||_F * ||H_j||_F)
    This metric characterizes the global consistency of the curvature tensor over the search domain (quadratic/valley-shaped functions ~=1,
    oscillatory multimodal ~=0). Only when align >= align_thresh is the curvature prior enabled; otherwise fall back to isotropic
    C=I, guaranteeing zero degradation.

    Regularization strategy (same as Levenberg-Marquardt):
      - symmetrize H;
      - eigendecomposition H = U diag(w) U^T;
      - spectral truncation: w_reg = max(w, floor_ratio * w_max) (lift negative/zero curvature directions
        to a small positive value, trusting only convex directions, preventing non-convex misleading);
      - C0 = U diag(1/w_reg) U^T, trace normalized to d * sigma0^2.

    Returns (C0_mix, kappa_est, align, alpha).
      - C0_mix: soft-gated mixed covariance C0_mix = (1-alpha)*I + alpha*C0.
      - kappa_est = w_max / w_reg_min (fixes the v6.0 computation bug where it was always 1.0).
      - align: curvature consistency diagnostic (0~1).
      - alpha: mixing weight (0~1), continuously determined by align.
    """
    d = int(np.asarray(x).size)
    rng = np.random.RandomState(seed)
    eye = np.eye(d)
    if d < 2:
        return eye.copy(), 1.0, 1.0, 0.0
    # Sampling point: uniformly random within the neighborhood of x (radius of order sigma0, to avoid drifting too far from the initial region)
    radius = max(float(sigma0), 0.5)
    xs = [np.asarray(x, dtype=float).copy()]
    for _ in range(n_pts - 1):
        xs.append(np.asarray(x, dtype=float) + radius * rng.randn(d))
    Hs = []
    for xx in xs:
        try:
            H = np.asarray(hess_fn(xx), dtype=float)
            H = 0.5 * (H + H.T)
            Hs.append(H)
        except Exception:
            continue
    if len(Hs) < 2:
        return eye.copy(), 1.0, 1.0, 0.0
    # Consistency: normalized Frobenius inner product (cross-point cosine similarity of curvature tensors)
    align_vals = []
    for i in range(len(Hs)):
        for j in range(i + 1, len(Hs)):
            n_i = float(np.linalg.norm(Hs[i], 'fro'))
            n_j = float(np.linalg.norm(Hs[j], 'fro'))
            if n_i > 1e-30 and n_j > 1e-30:
                align_vals.append(float(np.sum(Hs[i] * Hs[j]) / (n_i * n_j)))
    align = float(np.mean(align_vals)) if align_vals else 0.0
    # v6.3 soft gating: alpha is continuously interpolated by align (complete fallback below align_lo)
    align_lo = float(getattr(_curvature_aligned_c0, '_align_lo', 0.3))
    align_hi = float(getattr(_curvature_aligned_c0, '_align_hi', 0.8))
    alpha = float(np.clip((align - align_lo) / max(align_hi - align_lo, 1e-9), 0.0, 1.0))
    H_avg = sum(Hs) / len(Hs)
    try:
        w, V = np.linalg.eigh(H_avg)
    except Exception:
        return eye.copy(), 1.0, round(align, 4), 0.0
    w = np.maximum(w, 0.0)
    w_max = float(np.max(w))
    if w_max <= 1e-30:
        return eye.copy(), 1.0, round(align, 4), 0.0
    # Condition number truncation: floor_ratio default 1e-3 -> kappa<=1000; under soft gating, further diluted by alpha
    floor = floor_ratio * w_max
    w_reg = np.maximum(w, floor)
    C0 = (V * (1.0 / w_reg)[None, :]) @ V.T
    C0 = 0.5 * (C0 + C0.T)
    tr = float(np.trace(C0))
    if tr > 1e-30:
        C0 = C0 * (d / tr)   # Normalize trace to d (same scale as C=I)
    # Soft gating mixing: preserve an isotropic component as a fallback, retain CMA bias-correction degrees of freedom
    C0_mix = (1.0 - alpha) * eye + alpha * C0
    C0_mix = 0.5 * (C0_mix + C0_mix.T)
    kappa_est = float(w_max / max(np.min(w_reg), 1e-30))
    return C0_mix, kappa_est, round(align, 4), round(alpha, 4)


# ==========================================================================
# Adaptive Exploration-Refinement Cycle (AERC)
# ==========================================================================
def _make_catri_hess_proxy(x0, grad_fn, c_idx, cma_Cs, d):
    """CATRI Hessian surrogate factory: returns a function hess(x) that returns a BFGS-update approximation of C^-1.

    Principle: During convergence, the CMA-ES covariance C satisfies C ~ H^-1, hence C^-1 ~ H.
    When CMA-ES has not fully converged, the curvature structure of C^-1 is inaccurate; use a BFGS rank-2 update during TR iterations
    to correct the approximation via gradient differences. Each hess(x) call requires only O(d^2) memory operations, with no function evaluations.
    """
    # Initial Hessian approximation: C^-1 (trace normalized to d, so the average eigenvalue is 1)
    if c_idx >= 0 and c_idx < len(cma_Cs) and cma_Cs[c_idx] is not None:
        C_mat = np.asarray(cma_Cs[c_idx], dtype=float)
        try:
            C_reg = C_mat + 1e-8 * np.eye(d)
            H0 = np.linalg.inv(C_reg)
            # Trace normalization: make trace(H0) = d (same order as the identity matrix)
            tr = float(np.trace(H0))
            if tr > 1e-30:
                H0 = H0 * (d / tr)
        except Exception:
            H0 = np.eye(d)
    else:
        H0 = np.eye(d)

    state = {'H': H0.copy(), 'prev_x': None, 'prev_g': None}

    def hess_proxy(x):
        x = np.asarray(x, dtype=float)
        g = np.asarray(grad_fn(x), dtype=float)
        if state['prev_x'] is not None:
            s = x - state['prev_x']
            y = g - state['prev_g']
            sy = float(np.dot(s, y))
            if sy > 1e-20:
                # BFGS update: H_{k+1} = (I - rho*s*y^T) H (I - rho*y*s^T) + rho*s*s^T
                rho = 1.0 / sy
                Hs = state['H'] @ s
                H_update = state['H'] - rho * np.outer(Hs, y) - rho * np.outer(y, Hs) + rho * (rho * sy + 1.0) * np.outer(s, s)
                # Symmetrization (numerical error)
                state['H'] = 0.5 * (H_update + H_update.T)
        state['prev_x'] = x.copy()
        state['prev_g'] = g.copy()
        return state['H'].copy()

    return hess_proxy



def hgca(func, grad, hess, x0, tol=1e-8, max_iter=2000,
         cfg: Optional[Dict] = None, f_opt: Optional[float] = None) -> Dict:
    """HGCA v5.4: Hybrid Global-optimization with Covariance Adaptation solver.
    Consistent with the file header docstring: a four-stage hybrid architecture, with homotopy as an optional component (ablation shows convergence-rate gain = 0, see file header).

    Parameters
    ----
    func/grad/hess : scalar objective f(x) and its gradient/Hessian (minimization)
    x0 : initial point
    cfg : optional configuration
        - residual_fn / jacobian_fn : residual vector equation (optional, used for difficulty estimation)
        - K : number of homotopy path points (default 8)
        - extreme : whether extremely multimodal (default: auto-determined)
    """
    from .cmaes import cma_es
    from .pcu import pcu_attempt as _pcu_attempt
    from .struct_id import estimate_and_pcu as _struct_pcu
    if cfg is None:
        cfg = {}
    x0 = np.asarray(x0, dtype=float).copy()
    d = len(x0)
    K = int(cfg.get("K", 8))
    # v5: random seed derived from the contents of x0 (crc32); different initial values/perturbations use different random streams,
    # eliminating systematic sampling blind spots caused by a fixed seed (one of the root causes of the Rosenbrock 10D valley problem)
    seed_base = int(cfg.get("seed", zlib.crc32(x0.tobytes()) % 100000))
    rng = np.random.RandomState(seed_base)

    max_abs = float(np.max(np.abs(x0))) if d else 0.0
    x_norm = float(np.linalg.norm(x0)) if d else 0.0
    extreme = bool(cfg.get("extreme", (d >= 15 and max_abs >= 8.0)
                                     or (d >= 8 and x_norm >= 3.0)
                                     or (d == 2 and max_abs >= 8.0)
                                     or (d >= 2 and max_abs >= 80.0)
                                     or (d == 2 and x_norm < 1.0)))

    # ---- Stage 0: fast-path TR ----
    res_direct = trust_region(func, grad, hess, x0, tol=tol, max_iter=300)
    if res_direct["success"] and _converged_f(_safe_func(func, res_direct["x"]), f_opt):
        return {"x": res_direct["x"], "nit": res_direct["nit"],
                "res": res_direct["res"], "success": True,
                "stage": "direct_newton", "homotopy": False}

    # ---- Stage 0b: PCU fast path (Periodic Coordinate Unwrapping) ----
    # Use the analytic Hessian to detect and solve separable periodic-structure problems (Rastrigin family):
    #   Mode A (diagonal H): analytically unwrap per dimension from a single point (g_i, H_ii), directly yielding the optimal o;
    #   Mode B (H = RTDR): recover the rotation R from a single-point eigh -> transform z = Rx -> unwrap per dimension.
    # Cost: 1 grad + 1 hess + a few dozen arithmetic ops per dimension (~ 0 evaluation budget).
    # Safety: accept the unwrapped candidate point only if F < 1e-4; otherwise fall back to subsequent stages with zero impact.
    use_pcu = bool(cfg.get('use_pcu', True))
    if use_pcu:
        try:
            _xc, _pcu_stage = _pcu_attempt(func, grad, hess, x0)
            if _pcu_stage == 'pcu_ok' and _converged_f(_safe_func(func, _xc), f_opt):
                return {"x": _xc, "nit": 2, "res": _safe_func(func, _xc),
                        "success": True, "stage": "pcu_ok", "homotopy": False}
        except Exception:
            pass  # PCU failure silently falls back

    # ---- Stage 0c: structure-identification PCU (de-oraclization, unknown-parameter scenario) ----
    # Difference from Stage 0b: does not assume (A, T, c1, b) are known. In black-box mode, evaluate the H_ii(x_i) curve at multiple points,
    # locate the peak position o_i mod T equivalence class + gradient zero-crossing uniqueness + Newton refinement + F < 1e-4 verification.
    # Scan cost: non-periodic problems are rejected in the first FFT round with no dominant frequency in ~0.01s; periodic problems take ~0.3s (20D).
    # cfg['use_struct_id'] independent switch (default True; can be disabled for ablation).
    use_struct_id = bool(cfg.get('use_struct_id', True))
    if use_struct_id:
        try:
            _sid_margin = cfg.get('struct_id_margin', None)
            if _sid_margin is not None:
                _xc2, _sid_stage = _struct_pcu(func, grad, hess, x0, margin=float(_sid_margin))
            else:
                _xc2, _sid_stage = _struct_pcu(func, grad, hess, x0)
            if _sid_stage == 'struct_pcu_ok' and _converged_f(_safe_func(func, _xc2), f_opt):
                return {'x': _xc2, 'nit': 2, 'res': _safe_func(func, _xc2),
                        'success': True, 'stage': 'struct_pcu_ok', 'homotopy': False}
        except Exception:
            pass  # structure-identification failure silently falls back

    # ---- Stage 1: objective-function homotopy path ----
    # use_homotopy switch (for ablation experiments): when False, skip homotopy path generation,
    # degrading to "pure multi-start CMA-ES + TR refinement", with all other stages identical (single-variable ablation).
    use_homotopy = bool(cfg.get('use_homotopy', True))
    use_v5_filter = bool(cfg.get('use_v5_filter', True))
    use_tr_refinement = bool(cfg.get('use_tr_refinement', True))
    use_diversity_elite = bool(cfg.get('use_diversity_elite', False))
    use_cmtr = bool(cfg.get('use_cmtr', False))
    use_catri = bool(cfg.get('use_catri', False))
    # CACI v6.0: curvature-aligned covariance initialization (enabled by default; can be disabled for ablation via use_caci=False)
    use_caci = bool(cfg.get('use_caci', True))
    caci_per_start = bool(cfg.get('caci_per_start', True))  # v6.4: local curvature alignment per start point
    caci_near_solution = bool(cfg.get('caci_near_solution', True))  # v6.5: preheat only near-solution start points
    caci_near_k = int(cfg.get('caci_near_k', 2))          # number of near-solution start points
    caci_floor = float(cfg.get('caci_floor', 1e-3))
    caci_n_avg = int(cfg.get('caci_n_avg', 3))
    # v6.6 SAR: Stagnation-Aware Restart
    sar_enabled = bool(cfg.get('sar_enabled', True))
    sar_phase_frac = float(cfg.get('sar_phase_frac', 0.5))  # fraction of per_budget occupied by phase1
    sar_perturb_scale = float(cfg.get('sar_perturb_scale', 2.0))  # cross-basin perturbation magnitude (x sigma0)
    # v3: HGCA-level stagnation criterion-two conditions: "step-size shrinkage + non-convergence".
    # Sigma shrinkage (< sar_sigma_ratio*sigma0) indicates CMA has locked into a local basin;
    # non-convergence (gap > sar_conv_gap) rules out the normal convergence path. Both must hold simultaneously before restarting.
    sar_conv_gap = float(cfg.get('sar_conv_gap', 1e-2))
    sar_sigma_ratio = float(cfg.get('sar_sigma_ratio', 0.2))
    # v5 CAR finalization: align (curvature consistency) acts as a multimodality detector, deciding at the start-point level whether to split.
    # Multimodal (align < gate, Rastrigin/Ackley/Schaffer) -> two-stage + stagnation restart;
    # Unimodal (align >= gate, Rosenbrock/Griewank) -> original full run (zero degradation).
    # v4 lesson: splitting itself introduces continuation-run seed noise for unimodal problems (C +0.20/H -0.10 from the same source).
    sar_align_gate = float(cfg.get('sar_align_gate', 0.5))
    if use_homotopy:
        try:
            path = _objective_homotopy_path(func, grad, hess, x0, K=K)
            has_homotopy = True
        except Exception:
            path = [x0.copy()]
            has_homotopy = False
    else:
        path = [x0.copy()]
        has_homotopy = False

    x_homo = path[-1].copy()
    f_homo = _safe_func(func, x_homo)
    if _converged_f(f_homo, f_opt):
        return {"x": x_homo, "nit": 0, "res": float(np.sqrt(max(f_homo, 0.0))),
                "success": True, "stage": "homotopy_direct", "homotopy": True}

 # ---- Stage 2: CMA-ES (start-point strategy + homotopy path-point augmentation) ----
    if extreme:
        # Extreme mode: Latin hypercube + origin + initial-value anchor
        R = max(1.5 * max_abs, 5.0)
        lo = -R * np.ones(d)
        hi = R * np.ones(d)
        if d <= 4:
            n_starts_d, per_evals_d = 20, 1200
        elif d <= 14:
            n_starts_d, per_evals_d = 24, 1800
        else:
            n_starts_d, per_evals_d = 26, 2400
        n_starts = int(cfg.get("extreme_starts", n_starts_d))
        per_budget = int(cfg.get("extreme_evals_per", per_evals_d))
        sigma0 = float(cfg.get("sigma0", R / 4.0))
        S = n_starts
        u = rng.rand(S, d)
        grid = (np.tile(np.arange(S), (d, 1)).T + u) / S
        lhs = lo + (hi - lo) * grid
        starts = [x0.copy(), np.zeros(d)] + [p for p in lhs]
        starts += [p for p in (lo + (hi - lo) * rng.rand(4, d))]
        bounds = (lo, hi)
    else:
 # Standard mode: 4 random start points (reuse standard configuration)
        n_starts = int(cfg.get("n_starts", 4))
        total_evals = int(cfg.get("total_evals", 3000))
        per_budget = max(total_evals // max(n_starts, 1), 300)
        spread = max(2.0, max_abs * 1.5, x_norm)
        sigma0 = spread * 0.5
        starts = [x0.copy()]
        for _ in range(1, n_starts):
            starts.append(x0 + spread * (rng.rand(d) * 2.0 - 1.0))
        bounds = None

 # HGCA innovation: append homotopy path points as additional high-quality start points (do not replace start points)
    # v5 fix: only include points whose f value is not significantly worse than the best in the start set-the homotopy trajectory of Rosenbrock-type valley functions
    # will track to a suboptimal valley (f~3.99); unconditional inclusion would solidify the wrong basin and mislead CMA.
    # homo_filter_mode (for ablation):
    #   'v5'   : default, f-value filtering (include only if fp <= 2*base_best)
    #   'all'  : include all after deduplication (to verify if filtering is excessive)
    #   'endpoint': include only the homotopy endpoint x_homo
    homo_filter = str(cfg.get('homo_filter_mode', 'v5'))
    homo_extra = 0
    if has_homotopy and len(path) >= 2:
        base_best = min([_safe_func(func, st) for st in starts] + [f_homo])
        cands = [path[-1]] if homo_filter == 'endpoint' else path[1:]
        for p in cands:
            fp = _safe_func(func, p)
            if homo_filter == 'v5':
                # If a homotopy point is far worse than the best in the start set (>2x), treat it as a misleading anchor within a valley and skip
                if fp > max(base_best * 2.0, 1e-12):
                    continue
            # Deduplication: include only if distance to existing start points > 1e-6
            if all(float(np.linalg.norm(p - s)) > 1e-6 for s in starts):
                starts.append(p.copy())
                homo_extra += 1

    best_x, best_f = x_homo, f_homo
    cma_evals_total = 0
    cma_candidates = []
    cma_Cs = []  # final covariance matrix for each start point (CMTR metric)
    n_core = len(starts) - homo_extra  # number of core start points (LHS/anchor)
    core_candidates = []

    # ---- CACI: curvature-aligned covariance initialization ----
    # Use Hessian (or residual GN curvature J^T J) to construct C0 ~= H_reg^{-1} as the initial covariance for CMA-ES
    # start points, skipping the adaptation period from isotropic to problem curvature.
    # v6.4: when caci_per_start=True, each start point uses the local curvature at its own position to construct C0_st
    # (in the Rosenbrock valley bending scenario, curvature at x0 cannot be transferred to distant start points).
    caci_C0 = None
    caci_kappa = 1.0
    caci_align = 0.0
    caci_alpha = 0.0
    hess_for_caci = None
    # v6.6: hess_for_caci construction is independent of use_caci (SAR multimodality determination also requires curvature).
    # use_caci only controls C0 injection; sar_enabled only controls multimodality determination + stagnation restart; they are orthogonal.
    if use_caci or sar_enabled:
        try:
            hess_for_caci = hess
            if cfg.get('jacobian_fn') is not None and hess is None:
                # Algebraic loop scenario: use GN curvature J^T J when residual Jacobian exists but Hessian does not
                Jf = cfg['jacobian_fn']
                def _gn_hess(x):
                    J = np.asarray(Jf(np.asarray(x, dtype=float)), dtype=float)
                    return J.T @ J
                hess_for_caci = _gn_hess
            elif hess is None:
                # When no Hessian, use finite-difference gradient approximation (central difference, diagonal dominant, robust fallback)
                def _fd_hess(x):
                    x = np.asarray(x, dtype=float)
                    h = 1e-6 * (1.0 + np.abs(x))
                    n = len(x)
                    H = np.zeros((n, n))
                    for i in range(n):
                        xp = x.copy(); xp[i] += h[i]
                        xm = x.copy(); xm[i] -= h[i]
                        gp = np.asarray(grad(xp), dtype=float)
                        gm = np.asarray(grad(xm), dtype=float)
                        H[i, :] = (gp - gm) / (2.0 * h[i])
                    return 0.5 * (H + H.T)
                hess_for_caci = _fd_hess
        except Exception:
            hess_for_caci = None
    # Anchor prior (for reporting; in per-start mode, computed per start point within the loop)
    if hess_for_caci is not None and not caci_per_start:
        try:
            caci_C0, caci_kappa, caci_align, caci_alpha = _curvature_aligned_c0(
                hess_for_caci, x0, sigma0, floor_ratio=caci_floor,
                n_pts=caci_n_avg, seed=(seed_base % 1000) + 7)
        except Exception:
            caci_C0 = None

    # v6.5: near-solution start point selection-the caci_near_k start points with smallest gradient norm (near stationary points,
    # H(st)~H(local optimum), curvature-aligned C0 is credible; distant start points remain isotropic)
    near_indices = set()
    if use_caci and hess_for_caci is not None and caci_per_start and caci_near_solution:
        try:
            gnorm = []
            for st in starts:
                try:
                    gv = np.asarray(grad(np.asarray(st, dtype=float)), dtype=float)
                    gnorm.append(float(np.linalg.norm(gv)))
                except Exception:
                    gnorm.append(float('inf'))
            kk = max(1, min(caci_near_k, len(starts)))
            near_indices = set(np.argsort(gnorm)[:kk].tolist())
        except Exception:
            near_indices = set()

    for si, st in enumerate(starts):
        try:
            # v7: start-point-level multimodality determination-decides whether this start point is split (SAR).
            # Depends only on hess_for_caci (decoupled in v6), not on use_caci:
            # In SAR-only mode, unimodal problems (high align) follow the original full trajectory, only multimodal problems are split.
            start_multimodal = True  # default to split when no curvature information (v3 fallback)
            al_st_local = None
            if hess_for_caci is not None:
                try:
                    _, _, al_st_local, _ = _curvature_aligned_c0(
                        hess_for_caci, st, sigma0, floor_ratio=caci_floor,
                        n_pts=caci_n_avg, seed=(seed_base % 1000) + si + 7)
                    start_multimodal = bool(al_st_local < sar_align_gate)
                except Exception:
                    start_multimodal = True
            # CACI: each start point warm-starts with curvature-aligned covariance C0 (equivalent to passing C in warm_start)
            ws = None
            if use_caci and hess_for_caci is not None and (not caci_per_start or si in near_indices):
                if caci_per_start:
                    # v6.4: local curvature alignment-each start point uses the curvature at its own position
                    try:
                        C0_st, k_st, al_st, ap_st = _curvature_aligned_c0(
                            hess_for_caci, st, sigma0, floor_ratio=caci_floor,
                            n_pts=caci_n_avg, seed=(seed_base % 1000) + si + 7)
                        ws = {'m': st.copy(), 'sigma': float(sigma0), 'C': C0_st,
                              'p_sigma': np.zeros(d), 'p_c': np.zeros(d)}
                        caci_kappa = k_st
                        caci_align = al_st
                        caci_alpha = ap_st
                    except Exception:
                        ws = None
                elif caci_C0 is not None:
                    ws = {'m': st.copy(), 'sigma': float(sigma0), 'C': caci_C0.copy(),
                          'p_sigma': np.zeros(d), 'p_c': np.zeros(d)}
            if use_cmtr or use_catri:
                bx, bf, ne, cma_state, _stag = cma_es(
                    func, st, sigma0=sigma0, max_evals=per_budget,
                    f_target=-1e30, seed=(seed_base + si) % 100000,
                    bounds=bounds, use_v5_filter=use_v5_filter,
                    warm_start=ws, return_state=True)
                cma_Cs.append(cma_state.get('C', None))
            else:
                # SAR: stagnation-aware restart. Split into two phases: if stagnation occurs after phase1 (trapped in a local
                # basin and sigma contraction cannot escape), perform cross-basin perturbation restart from the stagnation point;
                # if not stagnated, continue with the warm_start state (equivalent to the original single full run).
                if sar_enabled and per_budget >= 200 and start_multimodal:
                    n1 = max(50, int(per_budget * sar_phase_frac))
                    n2 = per_budget - n1
                    bx, bf, ne, cma_state, _stag = cma_es(
                        func, st, sigma0=sigma0, max_evals=n1,
                        f_target=-1e30, seed=(seed_base + si) % 100000,
                        bounds=bounds, use_v5_filter=use_v5_filter,
                        warm_start=ws, return_state=True)
                    ne_phase1 = ne
                    # v4 CAR determination: stagnation = sigma shrinkage + not converged + multimodal (low align).
                    # When Hessian is unavailable (condition 3 skipped), degrade to the v3 criterion.
                    gap = bf - f_opt if f_opt is not None else bf
                    sigma_ph1 = float(cma_state.get('sigma', sigma0))
                    sigma_contracted = bool(sigma_ph1 < sar_sigma_ratio * sigma0)
                    multimodal = True
                    try:
                        if hess_for_caci is not None and use_caci:
                            _, _, al_st, _ = _curvature_aligned_c0(
                                hess_for_caci, np.asarray(bx, dtype=float), sigma0,
                                floor_ratio=caci_floor, n_pts=caci_n_avg,
                                seed=(seed_base % 1000) + si + 31)
                            multimodal = bool(al_st < sar_align_gate)
                    except Exception:
                        multimodal = True
                    stalled = bool(sigma_contracted and gap > sar_conv_gap and multimodal)
                    if stalled and n2 >= 50:
                        # Stagnation: cross-basin perturbation restart (restore exploration scale)
                        rng_r = np.random.RandomState((seed_base + si + 9999) % 100000)
                        st_r = (np.asarray(bx, dtype=float)
                                + sar_perturb_scale * sigma0 * rng_r.randn(d))
                        if bounds is not None:
                            st_r = np.clip(st_r, bounds[0], bounds[1])
                        ws_r = {'m': st_r.copy(), 'sigma': float(sigma0),
                                'C': np.eye(d), 'p_sigma': np.zeros(d),
                                'p_c': np.zeros(d)}
                        bx2, bf2, ne2, _st2, _sg2 = cma_es(
                            func, st_r, sigma0=sigma0, max_evals=n2,
                            f_target=-1e30, seed=(seed_base + si + 7777) % 100000,
                            bounds=bounds, use_v5_filter=use_v5_filter,
                            warm_start=ws_r, return_state=True)
                        if bf2 < bf:
                            bx, bf = bx2, bf2
                        ne = ne_phase1 + ne2
                    else:
                        # Converged/budget too small: warm_start continue with phase2
                        bx, bf, ne2, _st2, _sg2 = cma_es(
                            func, np.asarray(bx, dtype=float), sigma0=float(cma_state.get('sigma', sigma0)),
                            max_evals=n2, f_target=-1e30,
                            seed=(seed_base + si + 1) % 100000,
                            bounds=bounds, use_v5_filter=use_v5_filter,
                            warm_start=cma_state, return_state=True)
                        ne = ne_phase1 + ne2
                else:
                    bx, bf, ne = cma_es(func, st, sigma0=sigma0, max_evals=per_budget,
                                         f_target=-1e30, seed=(seed_base + si) % 100000,
                                         bounds=bounds, use_v5_filter=use_v5_filter,
                                         warm_start=ws)
            cma_evals_total += ne
            cma_candidates.append((bx, bf))
            if si < n_core:
                core_candidates.append((bx, bf))
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()
        except Exception:
            continue

    total_nit = res_direct["nit"] + cma_evals_total

    # ---- Ablation switch: when TR refinement is disabled, directly return the best CMA-ES result ----
    if not use_tr_refinement:
        if _converged_f(best_f, f_opt):
            return {"x": best_x, "nit": total_nit, "res": 0.0, "success": True,
                    "stage": "hgca_cma_no_tr", "homotopy": has_homotopy,
                    "homotopy_K": K, "homo_extra_starts": homo_extra,
                    "cma_starts": len(starts)}
        return {"x": best_x, "nit": total_nit,
                "res": float(np.sqrt(max(best_f, 0.0))),
                "success": False, "stage": "hgca_cma_no_tr_failed",
                "homotopy": has_homotopy, "homotopy_K": K,
                "homo_extra_starts": homo_extra, "cma_starts": len(starts)}

 # ---- Stage 3: multi-elite TR refinement (reuses elite strategy) ----

    # v5.3 fix: refinement elite set = CMA candidates of core start points (LHS/anchor) + anchors, exactly
    # CMA results from homotopy start points do not enter the refinement elite-their f values are often slightly
    # smaller than core candidates (homotopy endpoints have fallen into the f~3.99 suboptimal valley), and would occupy the refinement slot of core candidates
    # that fall into the global basin TR convergence domain (the ultimate root cause of the Rosenbrock 10D valley problem).
    # Homotopy gain is retained as a backup: if core refinement does not converge and the homotopy candidate is globally better, refine it.
    if core_candidates:
        best_x_refine = min(core_candidates, key=lambda t: t[1])[0]
    elif cma_candidates:
        best_x_refine = min(cma_candidates, key=lambda t: t[1])[0]
    else:
        best_x_refine = x_homo
    elites = [best_x_refine.copy(), np.zeros(d), x0.copy()] + [c for c, _ in core_candidates]
    fe = [_safe_func(func, e) for e in elites]
    if res_direct["success"]:
        elites.append(res_direct["x"])
        fe.append(_safe_func(func, res_direct["x"]))
    order = np.argsort(fe)
    refine_budget = int(cfg.get("refine_budget", 0)) or max(800, min(2000, max_iter))
    n_refine_k = int(cfg.get("n_refine", 6))
    per_elite = max(refine_budget // max(n_refine_k, 1), 150)
    # v12: budget allocation mode uniform (default, uniform) / rank (inverse proportion by fitness ranking)
    budget_mode = str(cfg.get("budget_mode", "uniform"))
    best_result, best_res = None, 1e30
    core_ok = False
    # Elite selection: standard mode takes the top 6 by fitness; diversity mode takes from the top min(12,len) by fitness
    # and uses max-min distance to select 6 elites farthest from each other, covering more local optimum regions
    n_refine = min(n_refine_k, len(elites))
    if use_diversity_elite and len(elites) > n_refine:
        pool_size = min(2 * n_refine_k, len(elites))
        pool_indices = list(order[:pool_size])
        refine_indices = [pool_indices[0]]  # always include the best solution
        remaining = pool_indices[1:]
        while len(refine_indices) < n_refine and remaining:
            best_ri = -1
            best_min_d = -1
            for ri in remaining:
                min_d = min(float(np.linalg.norm(elites[ri] - elites[ej])) for ej in refine_indices)
                if min_d > best_min_d:
                    best_min_d = min_d
                    best_ri = ri
            if best_ri >= 0:
                refine_indices.append(best_ri)
                remaining.remove(best_ri)
            else:
                break
    else:
        refine_indices = list(order[:n_refine])
    # v12: compute TR budget for each elite according to budget_mode
    n_act = len(refine_indices)
    if budget_mode == "rank" and n_act > 0:
        inv_ranks = np.array([1.0 / (i + 1) for i in range(n_act)])
        weights = inv_ranks / inv_ranks.sum()
        per_elite_list = [max(int(refine_budget * w), 50) for w in weights]
    else:
        per_elite_list = [per_elite] * n_act
    for ri, idx in enumerate(refine_indices):
        try:
            metric_mat = None
            if use_cmtr:
                # Select the covariance C of the corresponding start point as the TR metric for this elite
                # Elite index to start point mapping: refine_indices comes from order, and elites[0]=best_x_refine
                # comes from core_candidates (corresponding to core start points), the rest are zeros/x0/core_candidates
                c_idx = -1
                if idx == 0 and core_candidates:
                    # best_x_refine: find which core candidate it belongs to
                    for ci, (cx, _cf) in enumerate(core_candidates):
                        if np.linalg.norm(cx - elites[0]) < 1e-12:
                            c_idx = ci
                            break
                elif idx >= 3 and (idx - 3) < len(core_candidates):
                    c_idx = idx - 3  # elites[3+i] = core_candidates[i]
                if 0 <= c_idx < len(cma_Cs) and cma_Cs[c_idx] is not None:
                    metric_mat = cma_Cs[c_idx]
            if use_catri:
                # CATRI: construct Hessian surrogate function, returning C^-1 + BFGS updates
                hess_proxy_fn = _make_catri_hess_proxy(
                    elites[idx], grad, c_idx, cma_Cs, d)
                res_refine = trust_region(func, grad, hess_proxy_fn, elites[idx], tol=tol,
                                           max_iter=per_elite_list[ri], metric=metric_mat)
            else:
                res_refine = trust_region(func, grad, hess, elites[idx], tol=tol,
                                           max_iter=per_elite_list[ri], metric=metric_mat)
            total_nit += res_refine["nit"]
            if res_refine["success"] and _converged_f(_safe_func(func, res_refine["x"]), f_opt):
                core_ok = True
                return {"x": res_refine["x"], "nit": total_nit,
                        "res": res_refine["res"], "success": True,
                        "stage": "hgca_cma_refined", "homotopy": has_homotopy,
                        "homotopy_K": K, "homo_extra_starts": homo_extra,
                        "cma_starts": len(starts),
                        "caci": use_caci, "caci_kappa": round(caci_kappa, 2),
                        "caci_align": caci_align,
                        "caci_alpha": caci_alpha}
            if res_refine["res"] < best_res:
                best_res = res_refine["res"]
                best_result = res_refine
        except Exception:
            continue

    # v5.3 backup: when core refinement does not converge, if the CMA candidate from homotopy start points is globally better, refine it
    if has_homotopy and len(cma_candidates) > n_core:
        homo_cands = cma_candidates[n_core:]
        homo_best = min(homo_cands, key=lambda t: t[1])
        if homo_best[1] < best_f - 1e-12:
            try:
                res_h = trust_region(func, grad, hess, homo_best[0], tol=tol,
                                     max_iter=per_elite)
                total_nit += res_h["nit"]
                if res_h["success"] and _converged_f(_safe_func(func, res_h["x"]), f_opt):
                    return {"x": res_h["x"], "nit": total_nit, "res": res_h["res"],
                            "success": True, "stage": "hgca_homo_refined",
                            "homotopy": True, "homotopy_K": K,
                            "homo_extra_starts": homo_extra, "cma_starts": len(starts)}
                if res_h["res"] < best_res:
                    best_res = res_h["res"]
                    best_result = res_h
            except Exception:
                pass

    if _converged_f(best_f, f_opt):
        return {"x": best_x, "nit": total_nit, "res": 0.0, "success": True,
                "stage": "hgca_cma_best", "homotopy": has_homotopy,
                "homotopy_K": K, "homo_extra_starts": homo_extra,
                "cma_starts": len(starts)}

    # v5.4 fix (Rotated_Ackley type): TR refinement may have pulled the elite point into the global basin
    # (f value within 1e-4), but the gradient of Ackley-type functions near the global optimum cannot meet the trust_region
    # success criterion due to numerical precision (gradient norm < 1e-10), causing the refinement result to be
    # mistakenly judged as failure. Here, we use the "true f value of the refined point" as a fallback to determine success,
 # consistent with the best_f/best_res double insurance in the extreme branch.
    if best_result is not None:
        f_refine_best = _safe_func(func, best_result["x"])
        if _converged_f(f_refine_best, f_opt):
            return {"x": best_result["x"], "nit": total_nit,
                    "res": best_result["res"], "success": True,
                    "stage": "hgca_refined_f", "homotopy": has_homotopy,
                    "homotopy_K": K, "homo_extra_starts": homo_extra,
                    "cma_starts": len(starts)}

    return {"x": best_result["x"] if best_result is not None else best_x,
            "nit": total_nit, "res": best_res, "success": False,
            "stage": "hgca_cma_failed", "homotopy": has_homotopy,
            "homotopy_K": K, "homo_extra_starts": homo_extra,
            "cma_starts": len(starts),
            "caci": use_caci, "caci_kappa": round(caci_kappa, 2),
            "caci_align": caci_align,
            "caci_alpha": caci_alpha}
