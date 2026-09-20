# -*- coding: utf-8 -*-
import os
"""run_14_noise: Noise robustness (review item: noise).

Caliber: FD black-box PCU of run_09_fd_blackbox (F5 M_orth, 30D) + multiplicative function-value noise
      f' = f.(1 + eps), eps ~ N(0, sigma), sigma in {1e-4, 1e-3, 1e-2}.
      Difference step HS scaled up accordingly (under noise, too-small h -> difference noise dominates).
Statistics: 10 runs per level, trigger / hit / F distribution.
Output: results/noise_robustness.json
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
SIGMAS = [0.0, 1e-4, 1e-3, 1e-2]
HS_BY_SIGMA = {0.0: 1e-3, 1e-4: 1e-3, 1e-3: 3e-3, 1e-2: 1e-2}

o = shifts[4][:D].copy()
M_raw = rotations[D][4].copy()
u, _, vt = np.linalg.svd(M_raw)
M = u @ vt
S5 = 5.12 / 100.0

def make_f(sigma, seed):
    rng = np.random.RandomState(seed)
    def f_clean(x):
        z = S5 * (M @ (np.asarray(x, float) - o))
        return float(np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0))
    def f_noisy(x):
        v = f_clean(x)
        return v * (1.0 + rng.randn() * sigma)
    return f_noisy if sigma > 0 else f_clean

def _off_diag(A):
    A = np.abs(A); np.fill_diagonal(A, 0.0)
    return float(np.sqrt(np.sum(A * A)))

def run_sigma(sigma):
    HS = HS_BY_SIGMA[sigma]
    def fd_grad(fn, x):
        g = np.zeros(D)
        for i in range(D):
            xp = x.copy(); xm = x.copy()
            xp[i] += HS; xm[i] -= HS
            g[i] = (fn(xp) - fn(xm)) / (2.0 * HS)
        return g
    def fd_hess_dir(fn, x, r):
        return (fn(x + HS*r) - 2.0*fn(x) + fn(x - HS*r)) / HS**2
    def fd_hess_full(fn, x):
        H = np.zeros((D, D))
        for i in range(D):
            xp = x.copy(); xm = x.copy()
            xp[i] += HS; xm[i] -= HS
            H[i, i] = (fn(xp) - 2.0*fn(x) + fn(xm)) / HS**2
            for j in range(i + 1, D):
                xpp = x.copy(); xpm = x.copy(); xmp = x.copy(); xmm = x.copy()
                xpp[i] += HS; xpp[j] += HS
                xpm[i] += HS; xpm[j] -= HS
                xmp[i] -= HS; xmp[j] += HS
                xmm[i] -= HS; xmm[j] -= HS
                H[i, j] = H[j, i] = (fn(xpp) - fn(xpm) - fn(xmp) + fn(xmm)) / (4.0*HS**2)
        return H
    def identify(fn, x, margin=100.0):
        x = np.asarray(x, float)
        H0 = fd_hess_full(fn, x)
        off = _off_diag(H0)
        base = x; R = None
        if off > 1e-6 * (1.0 + np.abs(H0).max()):
            pts = [x, x+1.0, x-1.0]
            cands = [np.linalg.eigh(fd_hess_full(fn, p))[1].T for p in pts]
            best_R, best_s = None, float('inf')
            for Rj in cands:
                s = sum(_off_diag(Rj @ fd_hess_full(fn, p) @ Rj.T) for p in pts)
                if s < best_s:
                    best_s, best_R = s, Rj
            R = best_R; base = R @ x
        T_ests = []
        for i in range(D):
            n_rough = min(max(int(np.ceil(2.0 * (10.0 + float(np.max(np.abs(base))) + margin) * 12.0)) + 1, 512), 4000)
            xs = np.linspace(base[i] - (10.0 + float(np.max(np.abs(base))) + margin),
                             base[i] + (10.0 + float(np.max(np.abs(base))) + margin), n_rough)
            Hc = np.empty(n_rough)
            rvec = R[i] if R is not None else np.eye(D)[i]
            for k in range(n_rough):
                xk = np.asarray(base, dtype=float).copy(); xk[i] = xs[k]
                Hc[k] = fd_hess_dir(fn, R.T @ xk if R is not None else xk, rvec)
            Tc = sid._coarse_period(xs, Hc)
            if Tc is None:
                return None
            T_ests.append(Tc)
        T_c = float(np.median(T_ests))
        if any(abs(Ti - T_c)/T_c > 0.2 for Ti in T_ests):
            return None
        gap_all = []; peaks_all = []
        for i in range(D):
            n = int(np.ceil(2.0 * (10.0 + float(np.max(np.abs(base))) + margin) / T_c * 16.0)) + 1
            n = min(max(n, 64), 4000)
            xs = np.linspace(base[i] - (10.0 + float(np.max(np.abs(base))) + margin),
                             base[i] + (10.0 + float(np.max(np.abs(base))) + margin), n)
            Hc = np.empty(n)
            rvec = R[i] if R is not None else np.eye(D)[i]
            for k in range(n):
                xk = np.asarray(base, dtype=float).copy(); xk[i] = xs[k]
                Hc[k] = fd_hess_dir(fn, R.T @ xk if R is not None else xk, rvec)
            peaks = []
            for j in range(1, n-1):
                if Hc[j] >= Hc[j-1] and Hc[j] >= Hc[j+1]:
                    if peaks and (xs[j] - peaks[-1]) < 2*(xs[1]-xs[0]):
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
    trig = hits = 0
    F_vals = []
    for k in range(N):
        rng = np.random.RandomState(SEED0 + k)
        x0 = rng.uniform(-100.0, 100.0, D)
        fn = make_f(sigma, SEED0 + 1000 + k)
        mdl = identify(fn, x0, margin=100.0)
        if mdl is None:
            continue
        trig += 1
        # Black-box candidate (|g| caliber + combination)
        R = mdl['R']; peaks_all = mdl['peaks']
        x = np.asarray(x0, float)
        g0 = fd_grad(fn, x)
        tol_g = 5e-3 * max(1.0, float(np.max(np.abs(g0))))
        if R is None:
            o_cand_list = []
            for i in range(D):
                entries = []
                for p in peaks_all[i]:
                    xk = np.asarray(x, dtype=float).copy(); xk[i] = float(p)
                    gv = fd_grad(fn, xk)[i]
                    if abs(gv) < tol_g:
                        entries.append((abs(gv), round(float(p), 10)))
                if not entries:
                    break
                entries.sort(key=lambda e: e[0])
                o_cand_list.append([p for _, p in entries])
            if len(o_cand_list) < D:
                continue
            o_best, f_best = _pcu._best_combination(fn, D, o_cand_list)
            if o_best is None or f_best >= 1e-4:
                continue
            xc = np.asarray(o_best, float)
        else:
            def grad_z(zv): return R @ fd_grad(fn, R.T @ np.asarray(zv, float))
            z = R @ x
            g_z0 = grad_z(z)
            tol_gz = 5e-3 * max(1.0, float(np.max(np.abs(g_z0))))
            o_cand_list = []
            for i in range(D):
                entries = []
                for p in peaks_all[i]:
                    zk = np.asarray(z, dtype=float).copy(); zk[i] = float(p)
                    gv = float(grad_z(zk)[i])
                    if abs(gv) < tol_gz:
                        entries.append((abs(gv), round(float(p), 10)))
                if not entries:
                    break
                entries.sort(key=lambda e: e[0])
                o_cand_list.append([p for _, p in entries])
            if len(o_cand_list) < D:
                continue
            def _f_rot(o_z):
                return float(np.asarray(fn(R.T @ np.asarray(o_z, float))).item())
            o_z, f_best = _pcu._best_combination(_f_rot, D, o_cand_list)
            if o_z is None or f_best >= 1e-4:
                continue
            xc = np.asarray(R.T @ o_z, float)
        F_vals.append(float(np.asarray(fn(xc)).item()))
        if F_vals[-1] < 1e-4:
            hits += 1
    return {'sigma': sigma, 'HS': HS, 'n': N, 'trigger': trig,
            'hit': hits, 'hit_rate': hits/N,
            'F_median': float(np.median(F_vals)) if F_vals else None}

def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    t0 = time.time()
    results = [run_sigma(s) for s in SIGMAS]
    # Analytical control: f is noised but g/h analytical (common in algebraic loop residuals: objective noise vs analytical derivatives)
    results.append(run_analytic_control())
    summ = {'elapsed_s': round(time.time()-t0, 1), 'detail': results}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, 'noise_robustness.json'), 'w', encoding='utf-8') as fp:
        json.dump(summ, fp, ensure_ascii=False, indent=2)
    for r in results:
        print(json.dumps(r, ensure_ascii=False))
    print('done')


def run_analytic_control():
    """Analytical g/h + noised f: structural identification goes through the analytical channel -> should be immune to noise."""
    from aloop.solve import struct_id as sid
    def g5(x):
        x = np.asarray(x, float); z = S5*(M @ (x-o))
        gz = S5*(2.0*z + 20.0*np.pi*np.sin(2.0*np.pi*z))
        return M.T @ gz
    def h5(x):
        x = np.asarray(x, float); z = S5*(M @ (x-o))
        Hzz = S5*S5*(2.0 + 40.0*np.pi*np.pi*np.cos(2.0*np.pi*z))
        return (M.T * Hzz) @ M
    trig = hits = 0
    F_vals = []
    for k in range(N):
        rng = np.random.RandomState(SEED0 + 2000 + k)
        x0 = rng.uniform(-100.0, 100.0, D)
        fn_noisy = make_f(1e-2, SEED0 + 3000 + k)
        xc, tag = sid.estimate_and_pcu(fn_noisy, g5, h5, x0, margin=100.0)
        if tag == 'struct_pcu_ok':
            trig += 1
            Fv = float(np.asarray(fn_noisy(xc)).item())
            F_vals.append(Fv)
            if Fv < 1e-4:
                hits += 1
    return {'sigma': '1e-2 (analytic g/h)', 'HS': '-', 'n': N, 'trigger': trig,
            'hit': hits, 'hit_rate': hits/N,
            'F_median': float(np.median(F_vals)) if F_vals else None}

if __name__ == '__main__':
    main()
