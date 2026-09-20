# -*- coding: utf-8 -*-
import os
"""run_09_fd_blackbox: Finite-difference black-box version of PCU (F5 M_orth, 30D).

Setup: all gradients/Hessians of struct_id are constructed from black-box function values using central differences,
without relying on analytical derivatives - demonstrating that PCU does not require an analytical oracle (gray-box practical usability).
Implementation: reuse struct_id components (_coarse_period/_scan_dim/_parabolic_peak/
      _best_combination/_refine_newton); Hessian diagonal elements use "directional second-order differences".
"""
import io, sys, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'cec2017'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cec2017.transforms import rotations, shifts
from aloop.solve import struct_id as sid
from aloop.solve import pcu as _pcu

D = 30
N = 10
SEED0 = 20260914
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
HS = 1e-3  # Difference step size (30D f scale ~5e3: at h=1e-3 the error is ~1e-6, far better than 1e-4 at 1e-4)

o = shifts[4][:D].copy()
M_raw = rotations[D][4].copy()
u, _, vt = np.linalg.svd(M_raw)
M = u @ vt
S5 = 5.12 / 100.0

def f_black(x):
    z = S5 * (M @ (np.asarray(x, float) - o))
    return float(np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0))

def fd_grad(x):
    x = np.asarray(x, float)
    g = np.zeros(D)
    for i in range(D):
        xp = x.copy(); xm = x.copy()
        xp[i] += HS; xm[i] -= HS
        g[i] = (f_black(xp) - f_black(xm)) / (2.0 * HS)
    return g

def fd_hess_dir(x, r):
    """Second-order directional derivative along unit direction r (central difference, 3 f evaluations)."""
    x = np.asarray(x, float)
    h = HS
    return (f_black(x + h * r) - 2.0 * f_black(x) + f_black(x - h * r)) / h**2

def _fd_hess_full(x):
    """Full finite-difference Hessian (only for R recovery, K=3 point voting)."""
    x = np.asarray(x, float)
    H = np.zeros((D, D))
    for i in range(D):
        xp = x.copy(); xm = x.copy()
        xp[i] += HS; xm[i] -= HS
        H[i, i] = (f_black(xp) - 2.0*f_black(x) + f_black(xm)) / HS**2
        for j in range(i + 1, D):
            xpp = x.copy(); xpm = x.copy(); xmp = x.copy(); xmm = x.copy()
            xpp[i] += HS; xpp[j] += HS
            xpm[i] += HS; xpm[j] -= HS
            xmp[i] -= HS; xmp[j] += HS
            xmm[i] -= HS; xmm[j] -= HS
            H[i, j] = H[j, i] = (f_black(xpp) - f_black(xpm) - f_black(xmp) + f_black(xmm)) / (4.0 * HS**2)
    return H

def _off_diag(A):
    A = np.abs(A)
    np.fill_diagonal(A, 0.0)
    return float(np.sqrt(np.sum(A * A)))

def _recover_rotation(x):
    """Multi-point voting recovery of R: finite-difference H eigh from K=3 points, choose the one whose diagonalization at other points is best."""
    pts = [x, x + 1.0, x - 1.0]
    Hs = [_fd_hess_full(p) for p in pts]
    cands = []
    for Hj in Hs:
        _, evecs = np.linalg.eigh(Hj)
        cands.append(evecs.T)
    best_R, best_score = None, float('inf')
    for Rj in cands:
        score = sum(_off_diag(Rj @ Hl @ Rj.T) for Hl in Hs)
        if score < best_score:
            best_score, best_R = score, Rj
    return best_R, Hs[0]

def blackbox_identify(x, margin=100.0):
    """Black-box structure identification: H0 uses the full finite difference (K=3 point voting to recover R);
    scanning uses directional differences (3 f evaluations per dimension per point)."""
    x = np.asarray(x, float)
    R, H0 = _recover_rotation(x)
    off = _off_diag(H0)
    base = x
    if off > 1e-6 * (1.0 + np.abs(H0).max()):
        base = R @ x
    else:
        R = None
    # Scan (per dimension: directional difference Hessian diagonal element curve)
    mag = float(np.max(np.abs(base)))
    radius = max(10.0, mag + margin)
    T_ests = []
    for i in range(D):
        n_rough = min(max(int(np.ceil(2.0 * radius * 12.0)) + 1, 512), 4000)
        xs = np.linspace(base[i] - radius, base[i] + radius, n_rough)
        Hc = np.empty(n_rough)
        for k in range(n_rough):
            xk = np.asarray(base, dtype=float).copy()
            xk[i] = xs[k]
            rvec = R[i] if R is not None else np.eye(D)[i]
            Hc[k] = fd_hess_dir(R.T @ xk if R is not None else xk, rvec)
        Tc = sid._coarse_period(xs, Hc)
        if Tc is None:
            return None
        T_ests.append(Tc)
    T_c = float(np.median(T_ests))
    if any(abs(Ti - T_c) / T_c > 0.2 for Ti in T_ests):
        return None
    gap_all = []; peaks_all = []
    for i in range(D):
        # Compute directly (_scan_dim needs a hess callback; here directional differences are used instead)
        n = int(np.ceil(2.0 * radius / T_c * 16.0)) + 1
        xs = np.linspace(base[i] - radius, base[i] + radius, n)
        Hc = np.empty(n)
        for k in range(n):
            xk = np.asarray(base, dtype=float).copy()
            xk[i] = xs[k]
            rvec = R[i] if R is not None else np.eye(D)[i]
            Hc[k] = fd_hess_dir(R.T @ xk if R is not None else xk, rvec)
        peaks = []
        for j in range(1, n - 1):
            if Hc[j] >= Hc[j-1] and Hc[j] >= Hc[j+1]:
                if peaks and (xs[j] - peaks[-1]) < 2 * (xs[1] - xs[0]):
                    continue
                peaks.append(sid._parabolic_peak(xs, Hc, j))
        if len(peaks) < 2:
            return None
        peaks_all.append(np.array(peaks))
        gaps = np.diff(np.sort(peaks))
        gaps = gaps[(gaps > 0.5*T_c) & (gaps < 1.5*T_c)]
        if len(gaps) == 0:
            return None
        gap_all.extend(list(gaps))
    return {'T': float(np.median(gap_all)), 'peaks': peaks_all, 'R': R}

def blackbox_pcu(x, margin=100.0):
    mdl = blackbox_identify(x, margin=margin)
    if mdl is None:
        return None, None
    peaks_all = mdl['peaks']; R = mdl['R']
    g0 = fd_grad(x)
    tol_g = 5e-3 * max(1.0, float(np.max(np.abs(g0))))
    if R is None:
        o_cand_list = []
        for i in range(D):
            entries = []
            for p in peaks_all[i]:
                xk = np.asarray(x, dtype=float).copy()
                xk[i] = float(p)
                gv = fd_grad(xk)[i]
                if abs(gv) < tol_g:
                    entries.append((abs(gv), round(float(p), 10)))
            if not entries:
                return None, None
            entries.sort(key=lambda e: e[0])
            o_cand_list.append([p for _, p in entries])
        o_best, f_best = _pcu._best_combination(f_black, D, o_cand_list)
        if o_best is None or f_best >= 1e-4:
            return None, None
        return np.asarray(o_best, float), 'blackbox_pcu_ok'
    else:
        def grad_z(zv): return R @ fd_grad(R.T @ np.asarray(zv, float))
        z = R @ x
        g_z0 = grad_z(z)
        tol_gz = 5e-3 * max(1.0, float(np.max(np.abs(g_z0))))
        o_cand_list = []
        for i in range(D):
            entries = []
            for p in peaks_all[i]:
                zk = np.asarray(z, dtype=float).copy()
                zk[i] = float(p)
                gv = float(grad_z(zk)[i])
                if abs(gv) < tol_gz:
                    entries.append((abs(gv), round(float(p), 10)))
            if not entries:
                return None, None
            entries.sort(key=lambda e: e[0])
            o_cand_list.append([p for _, p in entries])
        def _f_rot(o_z):
            return float(np.asarray(f_black(R.T @ np.asarray(o_z, float))).item())
        o_z, f_best = _pcu._best_combination(_f_rot, D, o_cand_list)
        if o_z is None or f_best >= 1e-4:
            return None, None
        return np.asarray(R.T @ o_z, float), 'blackbox_pcu_ok'

t0 = time.time()
trig = hits = 0
period_errs = []
for k in range(N):
    rng = np.random.RandomState(SEED0 + k)
    x0 = rng.uniform(-100.0, 100.0, D)
    xc, tag = blackbox_pcu(x0, margin=100.0)
    if tag == 'blackbox_pcu_ok' and f_black(xc) < 1e-4:
        trig += 1; hits += 1
    # Period error (only when identify succeeds)
    mdl = blackbox_identify(x0, margin=100.0)
    if mdl is not None:
        period_errs.append(abs(mdl['T'] - 1.0/S5) / (1.0/S5))
    print('run%d: %s F=%.3e' % (k, tag, f_black(xc) if xc is not None else float('nan')), flush=True)

res = {'n': N, 'trigger': trig, 'hit_F<1e-4': hits,
       'hit_rate': hits / N, 'period_rel_err_mean': float(np.mean(period_errs)) if period_errs else None,
       'elapsed_s': round(time.time() - t0, 1)}
os.makedirs(RESULTS, exist_ok=True)
with open(os.path.join(RESULTS, 'cec_fd_blackbox.json'), 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(json.dumps(res, ensure_ascii=False, indent=2))
print('done')
