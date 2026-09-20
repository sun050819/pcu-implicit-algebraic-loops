# -*- coding: utf-8 -*-
import os
"""r4_noise_analytic: sigma=0 analytic-gradient PCU verification on CEC2017 f5 (M_orth, 30D).
Mirrors run_14_noise's identify pipeline but uses exact analytic Hessian (no FD noise),
to verify the paper_src claim: sigma=0 analytic -> 10/10 trigger, 10/10 hit,
median F ~ 9e-10.
"""
import io, sys, os, json, time
import numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'cec2017'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cec2017.transforms import rotations, shifts
from aloop.solve import struct_id as sid
from aloop.solve import pcu as _pcu

D = 30
N = 10
SEED0 = 20260916
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

o = shifts[4][:D].copy()
M_raw = rotations[D][4].copy()
u, _, vt = np.linalg.svd(M_raw)
M = u @ vt
S5 = 5.12 / 100.0
MT = M.T


def f_clean(x):
    z = S5 * (M @ (np.asarray(x, float) - o))
    return float(np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0))


def g_ana(x):
    z = S5 * (M @ (np.asarray(x, float) - o))
    gz = 2.0 * z + 20.0 * np.pi * np.sin(2.0 * np.pi * z)
    return S5 * (MT @ gz)


def H_ana(x):
    z = S5 * (M @ (np.asarray(x, float) - o))
    diag = 2.0 + 40.0 * np.pi * np.pi * np.cos(2.0 * np.pi * z)
    return S5 * S5 * (MT @ (M * diag[:, None]))


def _off_diag(A):
    A = np.abs(A)
    np.fill_diagonal(A, 0.0)
    return float(np.sqrt(np.sum(A * A)))


def hess_dir(x, r):
    return float(r @ (H_ana(x) @ r))


def identify(x, margin=100.0):
    x = np.asarray(x, float)
    H0 = H_ana(x)
    off = _off_diag(H0)
    base = x
    R = None
    if off > 1e-6 * (1.0 + np.abs(H0).max()):
        pts = [x, x + 1.0, x - 1.0]
        cands = [np.linalg.eigh(H_ana(p))[1].T for p in pts]
        best_R, best_s = None, float('inf')
        for Rj in cands:
            s = sum(_off_diag(Rj @ H_ana(p) @ Rj.T) for p in pts)
            if s < best_s:
                best_s, best_R = s, Rj
        R = best_R
        base = R @ x
    T_ests = []
    for i in range(D):
        n_rough = min(max(int(np.ceil(2.0 * (10.0 + float(np.max(np.abs(base))) + margin) * 12.0)) + 1, 512), 4000)
        xs = np.linspace(base[i] - (10.0 + float(np.max(np.abs(base))) + margin),
                         base[i] + (10.0 + float(np.max(np.abs(base))) + margin), n_rough)
        rvec = R[i] if R is not None else np.eye(D)[i]
        Hc = np.empty(n_rough)
        for k in range(n_rough):
            xk = np.asarray(base, dtype=float).copy()
            xk[i] = xs[k]
            Hc[k] = hess_dir(R.T @ xk if R is not None else xk, rvec)
        Tc = sid._coarse_period(xs, Hc)
        if Tc is None:
            return None
        T_ests.append(Tc)
    T_c = float(np.median(T_ests))
    if any(abs(Ti - T_c) / T_c > 0.2 for Ti in T_ests):
        return None
    gap_all = []
    peaks_all = []
    for i in range(D):
        n = int(np.ceil(2.0 * (10.0 + float(np.max(np.abs(base))) + margin) / T_c * 16.0)) + 1
        n = min(max(n, 64), 4000)
        xs = np.linspace(base[i] - (10.0 + float(np.max(np.abs(base))) + margin),
                         base[i] + (10.0 + float(np.max(np.abs(base))) + margin), n)
        rvec = R[i] if R is not None else np.eye(D)[i]
        Hc = np.empty(n)
        for k in range(n):
            xk = np.asarray(base, dtype=float).copy()
            xk[i] = xs[k]
            Hc[k] = hess_dir(R.T @ xk if R is not None else xk, rvec)
        peaks = []
        for j in range(1, n - 1):
            if Hc[j] >= Hc[j - 1] and Hc[j] >= Hc[j + 1]:
                if peaks and (xs[j] - peaks[-1]) < 2 * (xs[1] - xs[0]):
                    continue
                peaks.append(sid._parabolic_peak(xs, Hc, j))
        if len(peaks) < 2:
            return None
        peaks_all.append(np.array(peaks))
        gaps = np.diff(np.sort(peaks))
        gaps = gaps[(gaps > 0.5 * T_c) & (gaps < 1.5 * T_c)]
        if len(gaps) == 0:
            return None
        gap_all.extend(list(gaps))
    return {'T': float(np.median(gap_all)), 'peaks': peaks_all, 'R': R, 'base': base}


def candidate(x0, mdl, tol_g_factor=5e-3):
    """Replicate run_14_noise candidate logic: zero-crossing filtering in |g|,
    per-dim candidate lists, _best_combination over the rotated objective."""
    R = mdl['R']
    base = mdl['base']
    peaks_all = mdl['peaks']
    x = np.asarray(x0, float)
    if R is None:
        o_cand_list = []
        for i in range(D):
            entries = []
            for p in peaks_all[i]:
                xk = np.asarray(x, dtype=float).copy()
                xk[i] = float(p)
                gv = float(g_ana(xk)[i])
                if abs(gv) < tol_g_factor * max(1.0, float(np.max(np.abs(g_ana(x))))):
                    entries.append((abs(gv), round(float(p), 10)))
            if not entries:
                return None
            entries.sort(key=lambda e: e[0])
            o_cand_list.append([p for _, p in entries])
        if len(o_cand_list) < D:
            return None
        o_best, f_best = _pcu._best_combination(f_clean, D, o_cand_list)
        if o_best is None or f_best >= 1e-4:
            return None
        return np.asarray(o_best, float)
    else:
        def grad_z(zv):
            return R @ g_ana(R.T @ np.asarray(zv, float))
        z = R @ x
        g_z0 = grad_z(z)
        tol_gz = tol_g_factor * max(1.0, float(np.max(np.abs(g_z0))))
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
                return None
            entries.sort(key=lambda e: e[0])
            o_cand_list.append([p for _, p in entries])
        if len(o_cand_list) < D:
            return None

        def _f_rot(o_z):
            return float(np.asarray(f_clean(R.T @ np.asarray(o_z, float))).item())
        o_z, f_best = _pcu._best_combination(_f_rot, D, o_cand_list)
        if o_z is None or f_best >= 1e-4:
            return None
        return np.asarray(R.T @ o_z, float)


trig = hits = 0
F_vals = []
Ts = []
t0 = time.time()
for k in range(N):
    rng = np.random.RandomState(SEED0 + k)
    x0 = rng.uniform(-100.0, 100.0, D)
    mdl = identify(x0, margin=100.0)
    if mdl is None:
        continue
    trig += 1
    Ts.append(mdl['T'])
    xb = candidate(x0, mdl)
    if xb is None:
        continue
    F = f_clean(xb)
    F_vals.append(F)
    if F < 1e-4:
        hits += 1
res = {
    'config': 'sigma=0 analytic g/H',
    'n': N,
    'trigger': trig,
    'hit': hits,
    'hit_rate': hits / N,
    'F_median': float(np.median(F_vals)) if F_vals else None,
    'F_all': F_vals,
    'T_median': float(np.median(Ts)) if Ts else None,
    'elapsed_s': round(time.time() - t0, 1),
}
out = os.path.join(RESULTS, 'noise_sigma0_analytic.json')
json.dump(res, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
