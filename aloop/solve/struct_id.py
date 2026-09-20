# -*- coding: utf-8 -*-
"""struct_id.py - Periodic structure identification and oracle-free solving (v3), no import side effects.

Core idea (no longer assumes parameters are known, nor performs parameter inversion):
  1. Scan the H_ii(x_i) curve along each dimension (range covers all possible offsets around x0),
     **peak positions (refined by parabolic interpolation) give all equivalence class positions of o_i mod T**--
     the cos peak is at u=0 (mod T), i.e. x = o_i + kT.
  2. **Uniqueness of gradient zero-crossing (on the peak position set)**: the peak position set of the H_ii(x_i) curve is
     {x_i : u_i = kT, k in Z} (cos peaks). At these peak positions g_i = 2c1.kT,
     and when b=0 only k=0 (x=o_i) crosses zero. Therefore the point that is "H peak and gradient~0" is uniquely the true o_i
     -- no unwrapping, no parameter values needed. (Note: g may also accidentally cross zero at non-peak positions; the joint
     H peak condition excludes them.)
  3. Full-dimensional candidates (at most 1 per dimension) + F(o)<1e-4 verification as a fallback (safety design).

For the rotational family: a single-point eigh(H0) recovers R, scans dimension by dimension in the z=Rx space + zero-crossing verification,
then back-substitutes o = RT o_z.

Applicability boundary: periodic separable/rotationally periodic structures with b=0, and the true offset o falls within the scan
range [x0-(mag+m), x0+(mag+m)] (satisfied by the A^2EP protocol); when not satisfied, correctly
rejects (zero false-rejection fallback). Returning None means no periodic structure identified.
"""
from __future__ import annotations

import numpy as np

SCAN_MARGIN = 20.0        # scan radius lower bound: max(10, max|x0| + margin)
PTS_PER_PERIOD = 16       # sampling points per period (peak detection and parabolic refinement precision)


def _off_diag_norm(H):
    H = np.asarray(H, dtype=float)
    return float(np.sqrt(np.sum((H - np.diag(np.diag(H)))**2)))


def _parabolic_peak(xs, y, i):
    """Parabolic interpolation to refine peak position."""
    n = len(y)
    if i <= 0 or i >= n - 1:
        return xs[i]
    a, b, c = y[i - 1], y[i], y[i + 1]
    denom = a - 2.0 * b + c
    if abs(denom) < 1e-30:
        return xs[i]
    delta = 0.5 * (a - c) / denom * (xs[1] - xs[0])
    return xs[i] + delta


def _coarse_period(xs, y):
    """FFT coarse period estimate (used to set sampling points per period)."""
    n = len(y)
    if n < 16:
        return None
    yy = np.asarray(y, dtype=float) - np.mean(y)
    sp = np.abs(np.fft.rfft(yy * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=xs[1] - xs[0])
    sp[0] = 0.0
    k = int(np.argmax(sp))
    f = freqs[k]
    if f <= 0:
        return None
    T = 1.0 / f
    span = xs[-1] - xs[0]
    if T < 2.2 * (xs[1] - xs[0]) or T > 0.5 * span:
        return None
    return float(T)


def _scan_dim(hess, base, i, radius, pts_per_period, T_hint):
    """Scan along the i-th dimension (other coordinates fixed at base), sampling density adaptive to the period.

    Returns (xs, H_ii curve).
    """
    if T_hint:
        n = int(np.ceil(2.0 * radius / T_hint * pts_per_period)) + 1
        n = min(max(n, 64), 4000)
    else:
        n = 256
    xs = np.linspace(base[i] - radius, base[i] + radius, n)
    Hc = np.empty(n)
    for k in range(n):
        xk = np.asarray(base, dtype=float).copy()
        xk[i] = xs[k]
        Hc[k] = np.asarray(hess(xk), dtype=float)[i, i]
    return xs, Hc


def identify_structure(func, grad, hess, x, tol_off=1e-6, margin=SCAN_MARGIN,
                       pts_per_period=PTS_PER_PERIOD, voting=True):
    """Main entry for structure identification: returns model dict or None.

    Returns:
      {
        'T': float (cross-dimensional consensus period, median of peak spacings),
        'peaks': list[ndarray] (absolute coordinates of H peaks within the scan interval for each dimension),
        'R': None | ndarray (rotation matrix),
        'mode': 'separable' | 'rotated',
      }
    """
    x = np.asarray(x, dtype=float)
    d = len(x)
    H0 = np.asarray(hess(x), dtype=float)

    def _with_R(R):
        """Run full scan identification under the given rotation R (or None=separable)."""
        if R is None:
            hess_use = hess
            base = x
            mode = 'separable'
        else:
            base = R @ x
            hess_use = lambda zv: R @ np.asarray(hess(R.T @ zv), dtype=float) @ R.T
            mode = 'rotated'

        # Scan radius: covers all possible offsets around x0
        mag = float(np.max(np.abs(base))) if len(base) else 1.0
        radius = max(10.0, mag + margin)

        # Coarse period (radius adaptive: expand and rescan if less than 2 periods, at most 3 times)
        T_c = None
        for _attempt in range(3):
            T_ests = []
            _ok = True
            for i in range(d):
                n_rough = int(np.ceil(2.0 * radius * 12.0)) + 1
                n_rough = min(max(n_rough, 512), 4000)
                xs = np.linspace(base[i] - radius, base[i] + radius, n_rough)
                Hc = np.empty(n_rough)
                for kk in range(n_rough):
                    xk = np.asarray(base, dtype=float).copy()
                    xk[i] = xs[kk]
                    Hc[kk] = np.asarray(hess_use(xk), dtype=float)[i, i]
                Tc_i = _coarse_period(xs, Hc)
                if Tc_i is None:
                    _ok = False
                    break
                T_ests.append(Tc_i)
            if not _ok:
                radius *= 2.0
                continue
            T_c = float(np.median(T_ests))
            for Ti in T_ests:
                if abs(Ti - T_c) / T_c > 0.2:
                    return None
            if radius < 2.0 * T_c:
                radius = 2.0 * T_c + margin
                continue
            break
        if T_c is None:
            return None

        # Second round: adaptive-density scan + peak finding + peak-spacing consensus
        gap_all = []
        peaks_all = []
        for i in range(d):
            xs, Hc = _scan_dim(hess_use, base, i, radius, pts_per_period, T_c)
            n = len(Hc)
            peaks = []
            for j in range(1, n - 1):
                if Hc[j] >= Hc[j - 1] and Hc[j] >= Hc[j + 1]:
                    if peaks and (xs[j] - peaks[-1]) < 2 * (xs[1] - xs[0]):
                        continue
                    peaks.append(_parabolic_peak(xs, Hc, j))
            if len(peaks) < 2:
                return None
            peaks_all.append(np.array(peaks))
            gaps = np.diff(np.sort(peaks))
            gaps = gaps[(gaps > 0.5 * T_c) & (gaps < 1.5 * T_c)]
            if len(gaps) == 0:
                return None
            gap_all.extend(list(gaps))
        if len(gap_all) < d:
            return None
        T = float(np.median(gap_all))
        return {'T': T, 'peaks': peaks_all, 'R': R, 'mode': mode}

    if _off_diag_norm(H0) > tol_off * (1.0 + np.abs(H0).max()):
        # R multi-hypothesis tracking: K=3 point eigh + voting for the best, try full identification one by one.
        # Within an eigenvalue-degenerate subspace H~lambdaI, single-point/voting scores can both be distorted (any basis diagonalizes),
        # so all candidate hypotheses are kept; accept as soon as any hypothesis completes identification (F verification fallback zero false-rejection).
        if not voting:
            _cands = [np.linalg.eigh(np.asarray(H0, dtype=float))[1].T]
            _Hs = [np.asarray(H0, dtype=float)]
        else:
            _pert = max(1.0, 0.1 * float(np.max(np.abs(x))) if len(x) else 1.0)
            _pts = [x, x + _pert, x - _pert]
            _Hs = [np.asarray(hess(p), dtype=float) for p in _pts]
            _cands = [np.linalg.eigh(Hj)[1].T for Hj in _Hs]
        _best_R, _best_s = None, float('inf')
        for _Rj in _cands:
            _s = sum(_off_diag_norm(_Rj @ Hl @ _Rj.T) for Hl in _Hs)
            if _s < _best_s:
                _best_s, _best_R = _s, _Rj
        for _Rcand in [_best_R] + _cands:
            _mdl = _with_R(_Rcand)
            if _mdl is not None:
                return _mdl
        return None
    else:
        return _with_R(None)





def _candidates_via_sign_change(peaks_i, base, i, T_c, grad_use):
    """Gradient sign change detection in the peak position neighborhood: true zero-crossing (g=0 root) does not depend on peak position interpolation precision.

    For each H peak position p, within its neighborhood [p-w, p+w] (w=0.125*T_c, covering interpolation error)
    detect the sign change of the gradient component: the sign-change point is a true zero-crossing (positioned by linear interpolation);
    non-zero-crossing peaks (e.g. periodic peaks with k!=0) have g not crossing zero -> automatically excluded.
    Returns a list of zero-crossing candidates sorted by ascending gradient magnitude of the sign-change interval.
    """
    w = max(0.125 * float(T_c), 1e-3)
    probe = np.linspace(-w, w, 9)
    entries = []
    for p in peaks_i:
        gv_list = []
        for q in probe:
            xk = np.asarray(base, dtype=float).copy()
            xk[i] = float(p) + q
            gv_list.append(float(grad_use(xk)[i]))
        for j in range(len(probe) - 1):
            g0v, g1v = gv_list[j], gv_list[j + 1]
            if (g0v < 0.0) != (g1v < 0.0) and g0v != g1v:
                q0 = float(p) + probe[j]
                q1 = float(p) + probe[j + 1]
                zc = q0 - g0v * (q1 - q0) / (g1v - g0v)
                score = min(abs(g0v), abs(g1v))
                entries.append((score, round(float(zc), 10)))
                break
    entries.sort(key=lambda e: e[0])
    return entries


def _refine_newton(o, grad, hess, steps=3):
    """Starting from candidate o, perform damped Newton refinement (root of g=0, analytic Hessian).

    In high-dimensional near-degeneracy (spectral gap smaller than perturbation magnitude), bare Newton is ill-conditioned and diverges: when the smallest
    eigenvalue of H is about 1e-6, the H^-1 g step can reach O(1e2), flying out of the convergence basin in one step. Hence Tikhonov
    regularization (lam = 1e-6 * max|diag(H)|) and single-step truncation are added to ensure no divergence; candidates
    are usually already sufficiently accurate, damping is only a fallback.
    """
    o = np.asarray(o, dtype=float).copy()
    d = len(o)
    for _ in range(steps):
        g = np.asarray(grad(o), dtype=float)
        H = np.asarray(hess(o), dtype=float)
        lam = 1e-6 * max(float(np.max(np.abs(np.diag(H)))), 1e-12)
        try:
            delta = np.linalg.solve(H + lam * np.eye(d), g)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(delta)):
            break
        step = np.clip(delta, -1.0, 1.0)
        o = o - step
        if float(np.max(np.abs(step))) < 1e-14:
            break
    return o


def estimate_and_pcu(func, grad, hess, x, tol=1e-5, margin=SCAN_MARGIN,
                     pts_per_period=PTS_PER_PERIOD, tol_g_frac=5e-3, voting=True):
    """One-click: structure identification + gradient zero-crossing candidates + Newton refinement + F verification (oracle-free entry).

    Returns (x_cand, 'struct_pcu_ok') on success, otherwise (None, None).
    """
    from . import pcu as _pcu

    CAND_KEEP = 4  # keep the top K candidates per dimension by ascending single-dimension f (true solution f=0 must be first)
    mdl = identify_structure(func, grad, hess, x, margin=margin,
                             pts_per_period=pts_per_period, voting=voting)
    if mdl is None:
        return None, None
    peaks_all = mdl['peaks']
    R = mdl['R']
    d = len(x)

    g0 = np.asarray(grad(x), dtype=float)
    g_scale = float(np.max(np.abs(g0))) if len(g0) else 1.0
    tol_g = tol_g_frac * max(1.0, g_scale)

    if R is None:
        o_cand_list = []
        ok = True
        for i in range(d):
            entries = _candidates_via_sign_change(peaks_all[i], x, i, mdl['T'], grad)
            # Fallback: revert to |g| criterion when sign-change detection misses (compatible with smooth peak scenarios)
            if not entries:
                for p in peaks_all[i]:
                    xk = np.asarray(x, dtype=float).copy()
                    xk[i] = float(p)
                    g_val = float(np.asarray(grad(xk), dtype=float)[i])
                    if abs(g_val) < tol_g:
                        entries.append((abs(g_val), round(float(p), 10)))
                entries.sort(key=lambda e: e[0])
            if not entries:
                ok = False
                break
            # Single-dimension candidate sorting: |g| primary order (analytic/difference gradient, noise-immune) + f order.
            # Rationale: pure f sorting fails under multiplicative noise--the f value of the single-dimension scan is polluted by other-dimension
            # constants (true solution gain ~O(1) << sigma*f, flips as soon as sigma>=1e-4);
            # g is 0 at the true solution and nonzero at false zero-crossings, so sorting is stable.
            # The combination-level F(o)<tol verification still uses func as the criterion (f=0 at the true solution, multiplicative noise 0*(1+eps)=0).
            g_scored = []
            for _, p in entries:
                xk = np.asarray(x, dtype=float).copy()
                xk[i] = float(p)
                g_val = abs(float(np.asarray(grad(xk), dtype=float)[i]))
                f_val = float(np.asarray(func(xk)).item())
                g_scored.append((g_val, f_val, round(float(p), 10)))
            g_scored.sort(key=lambda e: (e[0], e[1]))
            o_cand_list.append([p for _, _, p in g_scored[:CAND_KEEP]])
        if not ok:
            return None, None
        o_best, f_best = _pcu._best_combination(func, d, o_cand_list)
        if o_best is None:
            return None, None
        # Candidates verified first (skip Newton when zero-crossing interpolation is already accurate to F<tol; under high-dimensional degeneracy Newton is ill-conditioned, so hit directly)
        if float(np.asarray(func(o_best)).item()) < tol:
            return o_best, 'struct_pcu_ok'
        # Newton refinement (damped, prevents near-degenerate ill-conditioned divergence)
        o_ref = _refine_newton(o_best, grad, hess)
        if float(np.asarray(func(o_ref)).item()) < tol:
            return o_ref, 'struct_pcu_ok'
        return None, None
    else:
        def grad_z(zv):
            return R @ np.asarray(grad(R.T @ np.asarray(zv, dtype=float)), dtype=float)

        def hess_z(zv):
            return R @ np.asarray(hess(R.T @ np.asarray(zv, dtype=float)), dtype=float) @ R.T

        z = R @ x
        g_z0 = grad_z(z)
        tol_gz = tol_g_frac * max(1.0, float(np.max(np.abs(g_z0))))
        o_cand_list = []
        ok = True
        for i in range(d):
            entries = _candidates_via_sign_change(peaks_all[i], z, i, mdl['T'], grad_z)
            if not entries:
                for p in peaks_all[i]:
                    zk = np.asarray(z, dtype=float).copy()
                    zk[i] = float(p)
                    g_val = float(grad_z(zk)[i])
                    if abs(g_val) < tol_gz:
                        entries.append((abs(g_val), round(float(p), 10)))
                entries.sort(key=lambda e: e[0])
            if not entries:
                ok = False
                break
            g_scored = []
            for _, p in entries:
                oz = np.asarray(z, dtype=float).copy()
                oz[i] = float(p)
                g_val = abs(float(grad_z(oz)[i]))
                f_val = float(np.asarray(func(R.T @ oz)).item())
                g_scored.append((g_val, f_val, round(float(p), 10)))
            g_scored.sort(key=lambda e: (e[0], e[1]))
            o_cand_list.append([p for _, _, p in g_scored[:CAND_KEEP]])
        if not ok:
            return None, None

        def _f_rot(o_z):
            return float(np.asarray(func(R.T @ np.asarray(o_z, dtype=float))).item())
        o_z, f_best = _pcu._best_combination(_f_rot, d, o_cand_list)
        if o_z is None:
            return None, None
        # Candidates verified first (skip Newton direct hit: under 500D near-degeneracy Newton is ill-conditioned but candidates are already accurate)
        o_ref0 = R.T @ np.asarray(o_z, dtype=float)
        if float(np.asarray(func(o_ref0)).item()) < tol:
            return o_ref0, 'struct_pcu_ok'
        # After damped Newton refinement in z space, back-substitute
        o_z_ref = _refine_newton(o_z, grad_z, hess_z)
        o_ref = R.T @ np.asarray(o_z_ref, dtype=float)
        if float(np.asarray(func(o_ref)).item()) < tol:
            return o_ref, 'struct_pcu_ok'
        return None, None
