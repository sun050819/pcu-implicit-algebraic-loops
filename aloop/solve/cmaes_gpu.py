"""GPU-accelerated CMA-ES (PyTorch CUDA implementation, v2.3.0 optimized version).

Key optimizations (targeting consumer-grade GPUs such as RTX 3060):
  1. **float32 precision**: On consumer-grade GPUs float64 performance is only 1/30 of float32,
     default to float32 (convergence precision still reaches 1e-6, meeting algebraic loop solving requirements).
  2. **Larger population**: Standard CMA-ES's lam=4+3log(n)~18 is too small for GPUs
     (kernel launch overhead dominates). GPU mode defaults to lam_factor=8 (lam~144),
     greatly reducing the number of iterations, with per-generation matrix operation sizes suitable for GPU parallelism.
  3. **End-to-end GPU**: When batch_func is provided, objective function evaluation is also done on the GPU,
     with no CPU-GPU data transfer overhead.
  4. **Minimize Python overhead**: All internal state is torch tensors, no Python-level numerical
     computation inside the loop.

Automatically falls back to CPU when no CUDA is available (torch device='cpu', float64 precision).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _get_device(prefer_cuda: bool = True) -> str:
    if not _HAS_TORCH:
        return "cpu"
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def cma_es_gpu(
    func: Optional[Callable[[np.ndarray], float]] = None,
    x0: Optional[np.ndarray] = None,
    sigma0: Optional[float] = None,
    max_evals: int = 3000,
    f_target: float = 1e-10,
    seed: int = 0,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    batch_func: Optional[Callable] = None,
    device: Optional[str] = None,
    lam_factor: int = 1,
    use_float32: Optional[bool] = None,
    refine_float64: bool = True,
    refine_generations: int = 20,
    explore_ratio: float = 0.15,
    refine_lbfgs: bool = True,
    refine_evals_ratio: float = 0.85,
    debug: bool = False,
) -> Tuple[np.ndarray, float, int]:
    """GPU-accelerated CMA-ES global minimization.

    Args:
        func: Scalar objective function f(x: np.ndarray) -> float (CPU mode, optional)
        x0: Initial point numpy vector
        sigma0: Initial step size
        max_evals: Maximum number of function evaluations
        f_target: Objective function value (terminate early when reached)
        seed: Random seed
        bounds: (lower, upper) bounds
        batch_func: Batch objective function f(X: torch.Tensor[lam,n]) -> torch.Tensor[lam]
                    (GPU mode, providing it enables end-to-end GPU acceleration)
        device: 'cuda' / 'cpu' / None (auto)
        lam_factor: Population scaling factor. GPU mode suggests 4-16 (default 8),
                    CPU mode suggests 1 (standard CMA-ES). lam = max(8, 4+3log(n)) * lam_factor
        use_float32: Whether to use float32. GPU defaults to True, CPU defaults to False.

    Returns:
        (best_x, best_f, n_evals)
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required for cma_es_gpu")

    if device is None:
        device = _get_device(prefer_cuda=True)
    dev = torch.device(device)
    is_cuda = (device == "cuda")

    if use_float32 is None:
        use_float32 = is_cuda  # GPU defaults to float32, CPU defaults to float64
    dtype = torch.float32 if use_float32 else torch.float64
    np_dtype = np.float32 if use_float32 else np.float64

    x0 = np.asarray(x0, dtype=np_dtype).reshape(-1)
    n = len(x0)

    # ---- Population size (enlarged in GPU mode) ----
    lam_base = max(8, 4 + int(3.0 * np.log(n)))
    lam = max(8, lam_base * lam_factor)
    mu = lam // 2

    # ---- CMA-ES parameters ----
    weights_np = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights_np /= weights_np.sum()
    mueff = 1.0 / np.sum(weights_np ** 2)
    cc = 4.0 / (n + 4.0)
    cs = (mueff + 2.0) / (n + mueff + 5.0)
    c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs
    chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

    # Convert to torch
    weights = torch.tensor(weights_np, dtype=dtype, device=dev)
    m = torch.tensor(x0, dtype=dtype, device=dev)
    C = torch.eye(n, dtype=dtype, device=dev)
    p_sigma = torch.zeros(n, dtype=dtype, device=dev)
    p_c = torch.zeros(n, dtype=dtype, device=dev)

    if sigma0 is None:
        nrm = float(torch.norm(m).item())
        sigma0 = max(0.5, 0.3 * nrm) if nrm > 0 else 1.0
    sigma = float(sigma0)

    g = torch.Generator(device=dev)
    g.manual_seed(seed)

    def _eval_scalar(x_t: torch.Tensor) -> float:
        if batch_func is not None:
            return float(batch_func(x_t.unsqueeze(0)).squeeze(0).item())
        return float(func(x_t.cpu().numpy().astype(np.float64)))

    best_x = m.clone()
    best_f = _eval_scalar(m)
    n_evals = 1
    if best_f < f_target:
        return best_x.cpu().numpy().astype(np.float64), best_f, n_evals

    patience = max(20, 3 * lam)
    stagnation = 0
    prev_best = best_f
    evals_cum = 1
    max_evals = int(max_evals)

    lo_t = hi_t = None
    if bounds is not None:
        lo_t = torch.tensor(bounds[0], dtype=dtype, device=dev)
        hi_t = torch.tensor(bounds[1], dtype=dtype, device=dev)

    # Precompute constants (avoid repeated computation inside the loop)
    cs_mueff = np.sqrt(cs * (2.0 - cs) * mueff)
    cc_mueff = np.sqrt(cc * (2.0 - cc) * mueff)
    cs_damps = cs / damps

    # Two-stage budget allocation: with refine_float64, stage 1 (large-population exploration) uses only explore_ratio budget
    phase1_budget = int(max_evals * explore_ratio) if (use_float32 and refine_float64) else max_evals

    while n_evals < phase1_budget:
        # ---- Eigendecomposition ----
        D2, B = torch.linalg.eigh(C)
        D2 = torch.clamp(D2, min=1e-30)
        D = torch.sqrt(D2)
        BD = B * D.unsqueeze(0)

        # ---- Sampling ----
        Z = torch.randn(lam, n, generator=g, dtype=dtype, device=dev)
        X = m.unsqueeze(0) + sigma * (Z @ BD.T)
        if lo_t is not None:
            X = torch.clamp(X, lo_t, hi_t)

        # ---- Objective function evaluation ----
        if batch_func is not None:
            F = batch_func(X)
        else:
            X_np = X.cpu().numpy().astype(np.float64)
            F_np = np.empty(lam)
            for i in range(lam):
                try:
                    F_np[i] = float(func(X_np[i]))
                except Exception:
                    F_np[i] = 1e30
            F = torch.tensor(F_np, dtype=dtype, device=dev)
        n_evals += lam
        evals_cum += lam

        # ---- Selection ----
        order = torch.argsort(F)
        X_sel = X[order[:mu]]
        F_sel = F[order[:mu]]
        f0 = F_sel[0].item()
        if f0 < best_f:
            best_f = f0
            best_x = X_sel[0].clone()
            if best_f < f_target:
                break

        # Adaptive early stopping
        improvement = prev_best - best_f
        if improvement < max(1e-12, 1e-8 * max(1.0, abs(prev_best))):
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best_f
        if stagnation >= patience and best_f < max(f_target * 100.0, 1e-4):
            break

        # ---- Mean update ----
        m_old = m.clone()
        m = torch.sum(weights.unsqueeze(1) * X_sel, dim=0)

        # ---- Path update ----
        y_w = (m - m_old) / sigma
        y_i = (X_sel - m_old.unsqueeze(0)) / sigma

        Bt_yw = B.T @ y_w
        p_sigma = (1.0 - cs) * p_sigma + cs_mueff * (B @ (D.reciprocal() * Bt_yw))
        norm_p = float(torch.norm(p_sigma).item())
        denom = np.sqrt(1.0 - (1.0 - cs) ** (2.0 * evals_cum / lam))
        hsig = 1.0 if (norm_p / denom < 1.4 + 2.0 / (n + 1.0)) else 0.0
        p_c = (1.0 - cc) * p_c + hsig * cc_mueff * y_w

        # ---- Covariance update ----
        wy = weights.unsqueeze(1) * y_i
        rank_mu = wy.T @ y_i
        C = ((1.0 - c1 - cmu) * C
             + c1 * (torch.outer(p_c, p_c) + (1.0 - hsig) * cc * (2.0 - cc) * C)
             + cmu * rank_mu)
        C = (C + C.T) / 2.0

        # ---- Step size update ----
        sigma = sigma * np.exp(cs_damps * (norm_p / chi_n - 1.0))
        if not (np.isfinite(sigma) and sigma > 1e-30):
            sigma = 1e-2

        # ---- Covariance condition number adaptation (v2.3.3 P1-5) ----
        # Ill-conditioned covariance (max/min eigenvalue ratio too large) means the function's long-axis direction is highly inconsistent,
        # and standard step size updates converge too slowly; enlarge the step size to aid exploration, after which CMA-ES will adaptively shrink.
        d_max = float(D2.max().item())
        d_min = float(D2.min().item())
        if d_min > 1e-30 and d_max / d_min > 1e6:
            sigma = min(sigma * 1.3, 1e3)

    # ---- float64 refinement (two-stage mixed precision + residual refinement, v2.3.11) ----
    # Stage 1: Large-population float32 exploration (GPU, quickly locate the basin)
    # Stage 2: Standard-population float64 refinement-as of v2.3.11 executed on **CPU**:
    #   . Consumer-grade GPUs (RTX 3060) have fp64 throughput only 1/30 of fp32, and GPU float64 refinement
    #     is neither sufficient nor fast, Ackley 100D on GPU used to be 0.8 (1/5 failures, f stuck at ~0.5-10);
    #   . Using the float32 exploration result best_x as the starting point, continue CMA-ES on CPU float64,
    #     i.e. "fp32 exploration + fp64 residual refinement" mixed precision; Ackley 100D converging from f~10 to
    #     1e-4 requires ~12000+ evaluations, so the refinement budget proportion is raised to 85% (refine_evals_ratio).
    #   . v2.3.8 early exit + CPU L-BFGS finishing remain unchanged: for low-dimensional/already-converged problems phase2
    #     triggers early exit within a few generations, without wasting budget.
    # Executed only when use_float32=True and refine_float64=True.
    if use_float32 and refine_float64 and best_f < 1e6 and n_evals < max_evals:
        best_x_np = best_x.cpu().numpy().astype(np.float64)
        x0_f64 = x0.astype(np.float64)
        dist = float(np.linalg.norm(best_x_np - x0_f64))
        best_f_ref = float(best_f)
        # Adaptive refinement radius (v2.3.3): when stage 1 float32 has not converged, best_x may deviate from the global optimum,
        # use the distance between best_x and x0 to ensure sufficient search range, fixing the Ackley 100D GPU 0/5 convergence problem.
        # Adaptive refinement radius: sigma*0.5 (inherits exploration step size) + dist*0.15 (deviation compensation, capped at 2.0).
        # v2.3.11 fix: In high dimensions ||best_x-x0|| ~ O(sqrt(d)) is naturally huge (Ackley 100D dist~35),
        # if not capped, sigma_r would reach 5+, and CMA-ES refinement degenerates into random search (f stuck at ~21.5 with no decrease).
        # Measured: Ackley 100D with sigma0=0.5/1/2 all converge within ~12000 evals, 5+ fails.
        sigma_r = max(sigma * 0.5, min(dist * 0.15, 2.0), 0.05)
        # Fallback cross-basin exploration radius when high-dimensional convergence is not achieved >= 0.5
        if best_f_ref > max(f_target * 10.0, 1e-3) and n > 10:
            sigma_r = max(sigma_r, 0.5)
        # Standard population (lam_factor=1)
        lam_ref = 4 + int(3 * np.log(n))
        mu_ref = lam_ref // 2
        weights_ref = np.array([np.log((lam_ref + 1) / k) for k in range(1, mu_ref + 1)])
        weights_ref = weights_ref / weights_ref.sum()
        mueff_ref = 1.0 / (weights_ref ** 2).sum()
        cs_ref = (mueff_ref + 2) / (n + mueff_ref + 5)
        damps_ref = 1 + 2 * max(0, np.sqrt((mueff_ref - 1) / (n + 1)) - 1) + cs_ref
        cc_ref = 4 / (n + 4)
        c1_ref = 2 / ((n + 1.3) ** 2 + mueff_ref)
        cmu_ref = min(1 - c1_ref, 2 * (mueff_ref - 2 + 1 / mueff_ref) / ((n + 2) ** 2 + mueff_ref))
        chi_n_ref = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

        # v2.3.11: phase2 fully on CPU (bypasses slow GPU fp64), residual refinement
        dev_ref = torch.device("cpu")
        m_r = torch.tensor(best_x_np, dtype=torch.float64, device=dev_ref)
        C_r = torch.eye(n, dtype=torch.float64, device=dev_ref) * max(sigma_r * sigma_r, 1e-4)
        p_sigma_r = torch.zeros(n, dtype=torch.float64, device=dev_ref)
        p_c_r = torch.zeros(n, dtype=torch.float64, device=dev_ref)
        g_ref = torch.Generator(device=dev_ref)
        g_ref.manual_seed(seed + 9999)
        ev_ref = 0
        max_ref_evals = int((max_evals - n_evals) * refine_evals_ratio)  # Use remaining budget (can be scaled)

        def _eval64(x_t):
            if batch_func is not None:
                # v2.3.6: phase 2 refinement uses true float64 evaluation (the original .to(float32) caused precision loss)
                return float(batch_func(x_t.unsqueeze(0)).squeeze(0).item())
            return float(func(x_t.cpu().numpy()))

        while ev_ref < max_ref_evals:
            try:
                D_r, B_r = torch.linalg.eigh(C_r)
                D_r = torch.sqrt(torch.clamp(D_r, min=1e-20))
                BD_r = B_r * D_r.unsqueeze(0)
            except Exception:
                C_r = torch.eye(n, dtype=torch.float64, device=dev_ref) * 1e-6
                BD_r = torch.eye(n, dtype=torch.float64, device=dev_ref) * 1e-3
            Z_r = torch.randn(lam_ref, n, generator=g_ref, dtype=torch.float64, device=dev_ref)
            X_r = m_r.unsqueeze(0) + sigma_r * (Z_r @ BD_r.T)
            if batch_func is not None:
                # v2.3.6: phase 2 batch evaluation uses true float64 (the original .to(float32) caused precision loss)
                F_r = batch_func(X_r).to(torch.float64)
            else:
                F_np = np.array([_eval64(X_r[i]) for i in range(lam_ref)])
                F_r = torch.tensor(F_np, dtype=torch.float64, device=dev_ref)
            ev_ref += lam_ref
            order_r = torch.argsort(F_r)
            X_sel_r = X_r[order_r[:mu_ref]]
            w_r = torch.tensor(weights_ref, dtype=torch.float64, device=dev_ref)
            m_old_r = m_r.clone()
            m_r = (w_r.unsqueeze(1) * X_sel_r).sum(dim=0)
            f0_r = float(F_r[order_r[0]].item())
            if f0_r < best_f_ref:
                best_f_ref = f0_r
                best_x = X_sel_r[0].clone()
            if best_f_ref < f_target:
                break
            # v2.3.8: float64 GPU refinement early exit-after reaching the intermediate threshold, hand off to CPU L-BFGS
            # for finishing. Consumer-grade GPU fp64 throughput is low, continuing refinement to 1e-4 is costly; L-BFGS on
            # CPU takes only a few ms to go from ~1e-3 to 1e-8. Greatly reduces GPU refinement time.
            if refine_lbfgs and best_f_ref < max(f_target * 10.0, 1e-3):
                break
            y_w_r = (m_r - m_old_r) / sigma_r
            p_sigma_r = (1 - cs_ref) * p_sigma_r + np.sqrt(cs_ref * (2 - cs_ref) * mueff_ref) * (B_r @ (D_r.reciprocal() * (B_r.T @ y_w_r)))
            norm_p_r = float(torch.norm(p_sigma_r).item())
            h_sig_r = 1.0 if (norm_p_r / np.sqrt(1 - (1 - cs_ref) ** (2 * (ev_ref / lam_ref))) < (1.4 + 2 / (n + 1)) * chi_n_ref) else 0.0
            p_c_r = (1 - cc_ref) * p_c_r + h_sig_r * np.sqrt(cc_ref * (2 - cc_ref) * mueff_ref) * y_w_r
            y_i_r = (X_sel_r - m_old_r.unsqueeze(0)) / sigma_r
            C_r = ((1 - c1_ref - cmu_ref) * C_r
                   + c1_ref * (torch.outer(p_c_r, p_c_r) + (1 - h_sig_r) * cc_ref * (2 - cc_ref) * C_r)
                   + cmu_ref * (w_r.unsqueeze(1) * y_i_r).T @ y_i_r)
            C_r = (C_r + C_r.T) / 2.0
            sigma_r = sigma_r * np.exp(cs_ref / damps_ref * (norm_p_r / chi_n_ref - 1.0))
            if not (np.isfinite(sigma_r) and sigma_r > 1e-30):
                sigma_r = 1e-4
        n_evals += ev_ref
        best_f = best_f_ref

    # v2.3.8: CPU float64 L-BFGS-B finishing refinement.
    # Consumer-grade GPU (RTX 3060) fp64 throughput is low (~1:64), float64 refinement is slow on GPU;
    # starting from best_x, use CPU L-BFGS-B for local convergence (numerical gradient), for smooth functions a few dozen iterations
    # suffice to reach 1e-8, significantly reducing reliance on expensive float64 GPU refinement.
    if refine_lbfgs and best_f > f_target and func is not None:
        try:
            from scipy.optimize import minimize as _minimize
            _bx = best_x.cpu().numpy().astype(np.float64)
            _res = _minimize(lambda x: float(func(x)), _bx, method='L-BFGS-B',
                             options={'maxiter': 300, 'ftol': 1e-14, 'maxfun': 8000})
            if float(_res.fun) < best_f:
                best_f = float(_res.fun)
                best_x = torch.tensor(np.asarray(_res.x, dtype=np.float64),
                                      dtype=torch.float64, device=dev)
        except Exception:
            pass

    return best_x.cpu().numpy().astype(np.float64), float(best_f), int(n_evals)


def cuda_available() -> bool:
    if not _HAS_TORCH:
        return False
    return torch.cuda.is_available()


def cuda_device_name() -> str:
    if not cuda_available():
        return "CPU (no CUDA)"
    return torch.cuda.get_device_name(0)


def cma_es_ipop_gpu(
    func: Optional[Callable[[np.ndarray], float]] = None,
    x0: Optional[np.ndarray] = None,
    sigma0: Optional[float] = None,
    max_evals: int = 3000,
    f_target: float = 1e-10,
    seed: int = 0,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    batch_func: Optional[Callable] = None,
    device: Optional[str] = None,
    lam_factor_init: int = 1,
    lam_factor_max: int = 64,
    n_restarts: Optional[int] = None,
    refine_float64: bool = True,
    explore_ratio: float = 0.3,
) -> Tuple[np.ndarray, float, int]:
    """IPOP-CMA-ES (Increasing Population Size CMA-ES) GPU-accelerated version, v2.3.1.

    Population-increasing restart strategy: each time convergence stalls or the budget is reached, the population size doubles and restarts,
    starting from the historical best point. Highly effective for multimodal functions (e.g., Rastrigin), able to effectively
    escape local optima.

    Algorithm flow:
      1. lam_factor = lam_factor_init (default 1, standard population ~18)
      2. Run CMA-ES (two-stage mixed precision), budget = a portion of the remaining evaluations
      3. Update the global best point
      4. If f < f_target, return
      5. lam_factor *= 2, restart from the global best point
      6. Repeat until the budget is exhausted or the maximum number of restarts is reached

    Args:
        func: scalar objective function (CPU mode)
        x0: initial point
        sigma0: initial step size
        max_evals: maximum number of evaluations (shared across all restarts)
        f_target: target function value
        seed: random seed (seed+i for each restart)
        bounds: variable bounds
        batch_func: batch objective function (end-to-end GPU)
        device: torch device
        lam_factor_init: initial population magnification factor (default 1)
        lam_factor_max: maximum population magnification factor (default 64, population ~1152)
        n_restarts: maximum number of restarts (None=unlimited, controlled by budget and lam_factor_max)
        refine_float64: whether to enable two-stage mixed precision (default True)
        explore_ratio: exploration phase budget ratio (default 0.3)

    Returns:
        (best_x, best_f, total_evals)
    """
    if x0 is None:
        raise ValueError("x0 is required for IPOP-CMA-ES")

    best_x = np.array(x0, dtype=np.float64).copy()
    best_f = float("inf")
    total_evals = 0
    lam_factor = lam_factor_init
    restart = 0

    while total_evals < max_evals:
        # Budget for this restart: 40% of the remaining budget (ensures at least 2-3 restarts)
        remaining = max_evals - total_evals
        run_budget = max(int(remaining * 0.4), 500)
        run_budget = min(run_budget, remaining)

        # Each restart resets sigma to a larger value (encourages global exploration)
        run_sigma = sigma0 if sigma0 is not None else 2.0
        if restart > 0:
            run_sigma = max(run_sigma * 0.5, 0.5)  # Subsequent restarts gradually reduce the step size

        try:
            x, f, ev = cma_es_gpu(
                func=func, x0=best_x, sigma0=run_sigma,
                max_evals=run_budget, f_target=f_target,
                seed=seed + restart, bounds=bounds,
                batch_func=batch_func, device=device,
                lam_factor=lam_factor,
                refine_float64=refine_float64,
                explore_ratio=explore_ratio,
            )
        except Exception:
            ev = 0
            f = float("inf")
            x = best_x

        total_evals += ev
        if f < best_f:
            best_f = f
            best_x = x.copy()

        if best_f < f_target:
            break

        restart += 1
        if n_restarts is not None and restart >= n_restarts:
            break
        if lam_factor >= lam_factor_max:
            # After reaching the maximum population, run the remaining budget once more with the maximum population
            if remaining > 1000:
                continue
            break
        lam_factor *= 2

    return best_x, float(best_f), int(total_evals)


def cma_es_bipop_gpu(
    func: Optional[Callable[[np.ndarray], float]] = None,
    x0: Optional[np.ndarray] = None,
    sigma0: Optional[float] = None,
    max_evals: int = 3000,
    f_target: float = 1e-10,
    seed: int = 0,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    batch_func: Optional[Callable] = None,
    device: Optional[str] = None,
    lam_factor_init: int = 1,
    lam_factor_max: int = 64,
    n_restarts: Optional[int] = None,
    refine_float64: bool = True,
    explore_ratio: float = 0.3,
    small_budget_ratio: float = 0.15,
    large_budget_ratio: float = 0.35,
) -> Tuple[np.ndarray, float, int]:
    """BIPOP-CMA-ES (Bi-population CMA-ES) GPU-accelerated version, v2.3.2.

    Two-population restart strategy: alternates between increasing population size (IPOP-style, global exploration) and small-population random
    (standard population + random initial points, local refinement + escaping local optima).
    For multimodal functions (e.g., Rastrigin) it is better than IPOP: large populations cover the global landscape, small populations quickly
    probe multiple local basins.

    Algorithm flow:
      1. Large population phase: lam_factor doubles incrementally from init, starting from the historical best point
      2. Small population phase: lam_factor=1 (standard population ~18), starting from random points (within bounds)
      3. Alternating, budget allocated by large_budget_ratio / small_budget_ratio
      4. Update the global best point each time, stop when f_target is reached or the budget is exhausted

    Args:
        func: scalar objective function (CPU mode)
        x0: initial point (used for the first time in the large population phase)
        sigma0: initial step size
        max_evals: maximum number of evaluations (shared across all restarts)
        f_target: target function value
        seed: random seed
        bounds: variable bounds (lower, upper), required for small population random restarts
        batch_func: batch objective function (end-to-end GPU)
        device: torch device
        lam_factor_init: initial magnification factor for the large population (default 1)
        lam_factor_max: maximum magnification factor for the large population (default 64)
        n_restarts: Maximum number of restarts (None=unlimited)
        refine_float64: Whether to enable two-stage mixed precision (default True)
        explore_ratio: Budget ratio for the exploration phase (default 0.3)
        small_budget_ratio: Proportion of remaining budget for the small population phase (default 0.15)
        large_budget_ratio: Proportion of remaining budget for the large population phase (default 0.35)

    Returns:
        (best_x, best_f, total_evals)
    """
    if x0 is None:
        raise ValueError("x0 is required for BIPOP-CMA-ES")
    rng = np.random.default_rng(seed)

    best_x = np.array(x0, dtype=np.float64).copy()
    best_f = float("inf")
    total_evals = 0
    large_lam = lam_factor_init
    restart = 0
    use_large = True  # Start with large population for exploration

    while total_evals < max_evals:
        remaining = max_evals - total_evals
        if use_large:
            run_budget = max(int(remaining * large_budget_ratio), 500)
            run_lam = large_lam
            run_x0 = best_x.copy()
            run_sigma = sigma0 if sigma0 is not None else 2.0
            if restart > 0:
                run_sigma = max(run_sigma * 0.5, 0.3)
        else:
            # Small population: standard population + random initial points (uniform sampling within bounds)
            run_budget = max(int(remaining * small_budget_ratio), 300)
            run_lam = 1
            if bounds is not None:
                lo, hi = bounds
                # Support scalar pairs for bounds (e.g., (-5, 5)) with automatic broadcasting to n dimensions
                if not hasattr(lo, "__len__"):
                    lo = [lo] * len(best_x)
                    hi = [hi] * len(best_x)
                run_x0 = np.asarray(lo, dtype=float) + rng.random(len(best_x)) * (np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float))
            else:
                # When no bounds, apply large perturbation near best_x
                run_x0 = best_x + rng.standard_normal(len(best_x)) * (sigma0 or 2.0) * 3
            run_sigma = (sigma0 if sigma0 is not None else 1.0) * 0.8

        run_budget = min(run_budget, remaining)

        try:
            x, f, ev = cma_es_gpu(
                func=func, x0=run_x0, sigma0=run_sigma,
                max_evals=run_budget, f_target=f_target,
                seed=seed + restart * 7 + (1000 if not use_large else 0),
                bounds=bounds, batch_func=batch_func, device=device,
                lam_factor=run_lam,
                refine_float64=refine_float64,
                explore_ratio=explore_ratio,
            )
        except Exception:
            ev = 0
            f = float("inf")
            x = best_x

        total_evals += ev
        if f < best_f:
            best_f = f
            best_x = x.copy()

        if best_f < f_target:
            break

        restart += 1
        if n_restarts is not None and restart >= n_restarts:
            break

        # Switch phase; double lam after the large population phase ends
        if use_large:
            if large_lam < lam_factor_max:
                large_lam *= 2
        use_large = not use_large

    return best_x, float(best_f), int(total_evals)


# ==========================================================================
# Adaptive parameter selection (v2.3.2)
# ==========================================================================
def auto_cma_params(n: int, ci: float = 0.0, multimodal: bool = False,
                    gpu: bool = True, cond: Optional[float] = None) -> Dict[str, float]:
    """Automatically recommend CMA-ES parameters based on problem dimension, CI (complexity index), multimodality, and covariance condition number.

    Recommendation rules (based on experimental tuning):
      - Dimension n: standard population for low dimension (n<=10), medium population for medium dimension (10<n<=50), large population for high dimension (n>50)
      - High CI (>0.5): strong nonlinearity, increase population and exploration ratio
      - Multimodal: requires larger population for global exploration, increase explore_ratio
      - GPU: higher population scaling factor (to mask kernel launch overhead)
      - Condition number cond (v2.3.3 P1-5): covariance matrix learning is slow for ill-conditioned problems,
        requiring a larger population to capture long-axis directions, increase lam_factor and explore_ratio

    Args:
        n: Problem dimension
        ci: Complexity index (0=linear, 1=extremely hard)
        multimodal: Whether the function is multimodal
        gpu: Whether to use GPU
        cond: Estimated condition number of covariance/Hessian (None=no adjustment)

    Returns:
        dict with keys: lam_factor, explore_ratio, refine_float64
    """
    if n <= 10:
        lam_factor = 1
        explore_ratio = 0.1
    elif n <= 30:
        lam_factor = 2
        explore_ratio = 0.15
    elif n <= 60:
        lam_factor = 4
        explore_ratio = 0.2
    else:
        lam_factor = 8
        # v2.3.8 fix: explore_ratio upper limit tightened to 0.15 for high dimensions.
        # Measured Ackley_100D: r=0.15 converges 3/3, r=0.20 fails 0/3 (float32 exploration budget too large
        # wasted on low-precision basin localization, insufficient float64 refinement budget cannot recover).
        explore_ratio = 0.15

    if gpu:
        lam_factor = max(lam_factor, 4)
        # v2.3.8: GPU no longer additionally amplifies explore_ratio (Ackley_100D measured r=0.3/0.2 both fail),
        # float32 exploration only needs to locate the basin, float64 refinement is key to convergence.
    if ci > 0.5:
        lam_factor *= 2
        explore_ratio += 0.1
    if multimodal:
        lam_factor *= 2
        # v2.3.8: Multimodal no longer amplifies explore_ratio (to avoid Ackley 100D type failures),
        # use lam_factor to amplify population to cover multimodality, float64 refinement ensures convergence.
    if cond is not None and cond > 100.0:
        # v2.3.11 theoretical derivation (smooth continuous, replacing original discrete thresholds):
        # The eigenvalues of the CMA-ES covariance C determine the search ellipsoid axis lengths, condition number kappa=lambda_max/lambda_min.
        # To obtain sufficient samples to estimate mean/covariance along the long axis, population size must grow with kappa.
        # Hansen empirical: lambda ~ log(kappa). Adjust only when kappa>100 (truly ill-conditioned) - run_34 statistical validation
        # shows that for well-conditioned/moderately ill-conditioned problems with kappa<=100, standard population is sufficient, and amplification instead wastes budget
        # (low float32 exploration accuracy + increased number of evaluations). Adjustment factor 1 + 0.2.log10(kappa):
        # kappa=1e2 -> x1.4 (threshold boundary), kappa=1e4 -> x1.8, kappa=1e6 -> x2.2 (capped at 32).
        log_cond = np.log10(float(cond))
        lam_factor = int(round(lam_factor * (1.0 + 0.2 * log_cond)))
        # explore_ratio: ill-conditioned problems require more exploration budget to learn covariance, but v2.3.8 has confirmed
        # upper limit 0.15 (too high leads to insufficient float64 refinement and high-dimensional Ackley failures).
        # So fine-tune on the base value by +0.004.log10(kappa) (+0.016 at kappa=1e4), still capped at 0.15.
        explore_ratio = explore_ratio + 0.004 * log_cond

    lam_factor = int(min(lam_factor, 32))
    # v2.3.8: global upper limit for explore_ratio is 0.15 (float32 exploration proportion),
    # too high leads to insufficient float64 refinement budget and high-dimensional Ackley failures.
    explore_ratio = float(min(explore_ratio, 0.15))
    refine_float64 = n > 20

    return {"lam_factor": lam_factor, "explore_ratio": explore_ratio,
            "refine_float64": refine_float64}


def estimate_cond_via_grad(grad: Optional[Callable], x0: np.ndarray,
                           eps: float = 1e-6) -> Optional[float]:
    """Estimate the condition number of the local Jacobian (gradient) using finite differences (P1-5 pre-check).

    For the gradient function grad(x), estimate J(x0) using central differences, return cond(J).
    Used for ill-conditioning detection in auto_cma_params.

    Args:
        grad: Gradient function grad(x) -> (d,) (can be None, then returns None)
        x0: Initial point
        eps: Finite difference step size

    Returns:
        Condition number estimate; returns None if unable to estimate
    """
    if grad is None:
        return None
    try:
        x0 = np.asarray(x0, dtype=float).ravel()
        d = len(x0)
        J = np.zeros((d, d))
        g0 = np.asarray(grad(x0), dtype=float)
        del g0
        for k in range(d):
            xp = x0.copy(); xp[k] += eps
            xm = x0.copy(); xm[k] -= eps
            gp = np.asarray(grad(xp), dtype=float)
            gm = np.asarray(grad(xm), dtype=float)
            J[:, k] = (gp - gm) / (2 * eps)
        if np.all(J == 0):
            return 1.0
        s = np.linalg.svd(J, compute_uv=False)
        s = s[s > 1e-30]
        if len(s) < 2:
            return float(s[0])
        return float(s[0] / s[-1])
    except Exception:
        return None




def cma_es_auto(func, x0, sigma0=None, max_evals=3000, f_target=1e-10,
                seed=0, bounds=None, batch_func=None, device=None,
                ci: float = 0.0, multimodal: bool = False,
                grad: Optional[Callable] = None,
                strategy: str = "auto") -> Tuple[np.ndarray, float, int]:
    """Adaptive CMA-ES: automatically selects parameters and strategy (standard/ipop/bipop).

    Automatically selects optimal parameters and restart strategy based on problem dimension, CI, and multimodality.
    - Unimodal/low-dimensional: standard CMA-ES (no restart overhead)
    - Multimodal/high-dimensional: BIPOP-CMA-ES (bi-population restarts, strong global exploration)
    - Medium: IPOP-CMA-ES (population increasing restarts)

    Args:
        func: Scalar objective function
        x0: Initial point
        sigma0: Initial step size
        max_evals: Maximum number of evaluations
        f_target: Objective function value
        seed: random seed
        bounds: variable bounds
        batch_func: batched objective function (GPU)
        device: torch device
        ci: complexity index (0=linear, 1=extremely hard)
        multimodal: whether multimodal
        grad: gradient function (P1-5: when provided, automatically estimates the local condition number to adjust parameters)
        strategy: "standard" / "ipop" / "bipop" / "auto" (default auto)

    Returns:
        (best_x, best_f, total_evals)
    """
    n = len(x0)
    gpu = (device == "cuda") or (device is None and torch.cuda.is_available())
    cond = estimate_cond_via_grad(grad, x0) if grad is not None else None
    params = auto_cma_params(n, ci=ci, multimodal=multimodal, gpu=gpu, cond=cond)

    if strategy == "auto":
        if multimodal or ci > 0.7 or n > 50:
            strategy = "bipop"
        elif ci > 0.4 or n > 30:
            strategy = "ipop"
        else:
            strategy = "standard"

    common = dict(func=func, x0=x0, sigma0=sigma0, max_evals=max_evals,
                  f_target=f_target, seed=seed, bounds=bounds,
                  batch_func=batch_func, device=device)

    if strategy == "bipop":
        # P1-6 fix: the large-population initial scaling factor of bipop must use auto_cma_params' lam_factor
        return cma_es_bipop_gpu(**common, refine_float64=params["refine_float64"],
                                 explore_ratio=params["explore_ratio"],
                                 lam_factor_init=max(2, params["lam_factor"]))
    elif strategy == "ipop":
        return cma_es_ipop_gpu(**common, refine_float64=params["refine_float64"],
                                explore_ratio=params["explore_ratio"],
                                lam_factor_init=max(2, params["lam_factor"]))
    else:
        return cma_es_gpu(**common, lam_factor=params["lam_factor"],
                          refine_float64=params["refine_float64"],
                          explore_ratio=params["explore_ratio"])
