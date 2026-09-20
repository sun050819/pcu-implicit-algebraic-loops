# -*- coding: utf-8 -*-
import os
"""run_11_fulltable: CEC2017 F1-F30 full-table structural identification sweep (review item: full-table of official benchmarks).

Two tracks:
  raw  : official M as-is (non-orthogonal) -- expected to have many rejections (PCU trigger conditions are strict)
  orth : official M after SVD orthogonalization -- periodic families (Rastrigin/Levy/Lunacek/Schwefel, etc.)
         expected to trigger+hit; non-periodic families (BentCigar/Elliptic/Rosenbrock, etc.) expected to be rejected.

Scope: fully black-box (official f1-f30 function values, finite-difference gradient/Hessian, directional finite-difference sweep),
      reusing the low-cost pipeline validated in run_09_fd_blackbox (R recovery with a full 3-point difference + directional finite-difference sweep).
      1 run per function (structural decision is mainly deterministic; x0 random within the domain and reproducible).
Output: results/cec_fulltable.json
"""
import io, sys, os, json, time
import numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'cec2017'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import cec2017.functions as CEC
from cec2017.transforms import rotations
from aloop.solve import struct_id as sid
from aloop.solve import pcu as _pcu

D = 30
SEED0 = 20260916
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
HS = 1e-3
FN_IDS = ['f%d' % i for i in range(1, 31)]

# official f_opt = function bias (f1=100, f2=200, ..., f30=3000)
F_OPT = {i: 100.0 * i for i in range(1, 31)}

def _off_diag(A):
    A = np.abs(A)
    np.fill_diagonal(A, 0.0)
    return float(np.sqrt(np.sum(A * A)))

def fd_grad(fn, x):
    g = np.zeros(D)
    for i in range(D):
        xp = x.copy(); xm = x.copy()
        xp[i] += HS; xm[i] -= HS
        g[i] = (fn(xp) - fn(xm)) / (2.0 * HS)
    return g

def fd_hess_dir(fn, x, r):
    h = HS
    return (fn(x + h * r) - 2.0 * fn(x) + fn(x - h * r)) / h**2

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
            H[i, j] = H[j, i] = (fn(xpp) - fn(xpm) - fn(xmp) + fn(xmm)) / (4.0 * HS**2)
    return H

def blackbox_identify(fn, x, margin=100.0):
    """Black-box structural identification (directional finite-difference sweep). Returns mdl / None.

    R multi-hypothesis: try K=3 point eigh candidates one by one for full identification (voting score within a degenerate subspace
    may be distorted -- any basis diagonalizes, but the axes are wrong; accept as soon as any of the multiple hypotheses completes).
    """
    x = np.asarray(x, float)
    try:
        H0 = fd_hess_full(fn, x)
    except Exception:
        return None
    off = _off_diag(H0)

    def _scan_with(R):
        """Run the full sweep under the given R (None=separable)."""
        if R is None:
            base = x
            rvec_i = lambda i: np.eye(D)[i]
            to_x = lambda zk: zk
        else:
            base = R @ x
            rvec_i = lambda i: R[i]
            to_x = lambda zk: R.T @ np.asarray(zk, float)
        mag = float(np.max(np.abs(base)))
        radius = max(10.0, mag + margin)
        T_ests = []
        for i in range(D):
            n_rough = min(max(int(np.ceil(2.0 * radius * 12.0)) + 1, 512), 4000)
            xs = np.linspace(base[i] - radius, base[i] + radius, n_rough)
            Hc = np.empty(n_rough)
            rvec = rvec_i(i)
            for k in range(n_rough):
                xk = np.asarray(base, dtype=float).copy()
                xk[i] = xs[k]
                Hc[k] = fd_hess_dir(fn, to_x(xk), rvec)
            Tc = sid._coarse_period(xs, Hc)
            if Tc is None:
                return None
            T_ests.append(Tc)
        T_c = float(np.median(T_ests))
        if any(abs(Ti - T_c) / T_c > 0.2 for Ti in T_ests):
            return None
        gap_all = []; peaks_all = []
        for i in range(D):
            n = int(np.ceil(2.0 * radius / T_c * 16.0)) + 1
            n = min(max(n, 64), 4000)
            xs = np.linspace(base[i] - radius, base[i] + radius, n)
            Hc = np.empty(n)
            rvec = rvec_i(i)
            for k in range(n):
                xk = np.asarray(base, dtype=float).copy()
                xk[i] = xs[k]
                Hc[k] = fd_hess_dir(fn, to_x(xk), rvec)
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

    if off <= 1e-6 * (1.0 + np.abs(H0).max()):
        return _scan_with(None)
    # R multi-hypothesis
    pts = [x, x + 1.0, x - 1.0]
    cands = []
    for pj in pts:
        try:
            Hj = fd_hess_full(fn, pj)
            _, evecs = np.linalg.eigh(Hj)
            cands.append(evecs.T)
        except Exception:
            pass
    if not cands:
        return None
    best_R, best_s = None, float('inf')
    for Rj in cands:
        s = 0.0
        for pj in pts:
            try:
                s += _off_diag(Rj @ fd_hess_full(fn, pj) @ Rj.T)
            except Exception:
                s = float('inf'); break
        if s < best_s:
            best_s, best_R = s, Rj
    for _Rcand in ([best_R] + cands):
        if _Rcand is None:
            continue
        mdl = _scan_with(_Rcand)
        if mdl is not None:
            return mdl
    return None

def blackbox_pcu(fn, x, f_opt, margin=100.0):
    mdl = blackbox_identify(fn, x, margin=margin)
    if mdl is None:
        return None, None, None
    peaks_all = mdl['peaks']; R = mdl['R']
    g0 = fd_grad(fn, x)
    tol_g = 5e-3 * max(1.0, float(np.max(np.abs(g0))))
    def _fval(xx): return float(np.asarray(fn(xx)).item()) - f_opt
    if R is None:
        o_cand_list = []
        for i in range(D):
            entries = []
            for p in peaks_all[i]:
                xk = np.asarray(x, dtype=float).copy()
                xk[i] = float(p)
                gv = fd_grad(fn, xk)[i]
                if abs(gv) < tol_g:
                    entries.append((abs(gv), round(float(p), 10)))
            if not entries:
                return None, None, mdl['T']
            entries.sort(key=lambda e: e[0])
            o_cand_list.append([p for _, p in entries])
        o_best, f_best = _pcu._best_combination(_fval, D, o_cand_list)
        if o_best is None or f_best >= 1e-4:
            return None, None, mdl['T']
        return np.asarray(o_best, float), 'blackbox_pcu_ok', mdl['T']
    else:
        def grad_z(zv): return R @ fd_grad(fn, R.T @ np.asarray(zv, float))
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
                return None, None, mdl['T']
            entries.sort(key=lambda e: e[0])
            o_cand_list.append([p for _, p in entries])
        def _f_rot(o_z):
            return float(np.asarray(fn(R.T @ np.asarray(o_z, float))).item()) - f_opt
        o_z, f_best = _pcu._best_combination(_f_rot, D, o_cand_list)
        if o_z is None or f_best >= 1e-4:
            return None, None, mdl['T']
        return np.asarray(R.T @ o_z, float), 'blackbox_pcu_ok', mdl['T']

def make_orth_M(fn_id_idx):
    """SVD orthogonalization of the official M (transpose convention consistent with f5: use rotations[D][idx])."""
    Mr = rotations[D][fn_id_idx]
    u, _, vt = np.linalg.svd(Mr)
    return u @ vt

def run_one(fn_id, orth=False, seed=0):
    idx = int(fn_id[1:]) - 1
    if idx >= 20:
        # f21-f30 composition functions: the official code uses rotations_cf (one rotation for each of the N subfunctions),
        # there is no single rotation structure -> the orth track is structurally inapplicable (N/A); the raw track uses the official default.
        if orth:
            return {'function': fn_id, 'variant': 'orth', 'trigger': False, 'hit': False,
                    'tag': 'composite_na', 'T_est': None, 'time_s': 0.0,
                    'note': 'Composition function (rotations_cf multi-rotation mix); structurally not applicable'}
        M = None
    elif orth:
        M = make_orth_M(idx)
    else:
        M = rotations[D][idx]
    fn = getattr(CEC, fn_id)
    def fwrap(x):
        xx = np.atleast_2d(np.asarray(x, float))
        if idx >= 20:
            # the f21+ composition function signature uses rotations=; the raw track uses the official default
            return float(np.asarray(fn(xx)).item())
        return float(np.asarray(fn(xx, rotation=M)).item())
    rng = np.random.RandomState(seed)
    x0 = rng.uniform(-100.0, 100.0, D)
    f_opt = F_OPT[int(fn_id[1:])]
    t0 = time.time()
    xc, tag, T_est = blackbox_pcu(fwrap, x0, f_opt, margin=100.0)
    dt = time.time() - t0
    hit = (tag == 'blackbox_pcu_ok' and xc is not None and
           (float(np.asarray(fwrap(xc)).item()) - f_opt) < 1e-4)
    return {'function': fn_id, 'variant': 'orth' if orth else 'raw', 'trigger': tag is not None,
            'hit': bool(hit), 'tag': tag, 'T_est': round(T_est, 4) if T_est else None,
            'time_s': round(dt, 1)}

def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    t_all = time.time()
    results = []
    start_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    quick = ['f4', 'f5', 'f9', 'f10', 'f1', 'f3']
    order = quick + [f for f in FN_IDS if f not in quick]
    order = [f for f in order if int(f[1:]) >= start_idx]
    for fn_id in order:
        for orth in [False, True]:
            r = run_one(fn_id, orth=orth, seed=SEED0)
            results.append(r)
            print('%s %-4s: trigger=%s hit=%s tag=%s T=%s (%.1fs)' % (
                fn_id, 'orth' if orth else 'raw', r['trigger'], r['hit'],
                r['tag'], r['T_est'], r['time_s']), flush=True)
    # incremental merge: merge with existing result files (keep the already-run parts of f1-f20)
    json_path = os.path.join(RESULTS, 'cec_fulltable.json')
    old = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, encoding='utf-8') as fp:
                old = json.load(fp)
        except Exception:
            old = {}
    detail = old.get('detail', [])
    new_keys = {(r['function'], r['variant']) for r in results}
    detail = [r for r in detail if (r['function'], r['variant']) not in new_keys] + results
    # summary (based on the merged full detail)
    summ = {'n_funcs': len(FN_IDS), 'elapsed_s': round(time.time() - t_all, 1), 'detail': detail}
    raw = [r for r in detail if r['variant'] == 'raw']
    orth = [r for r in detail if r['variant'] == 'orth']
    raw_na = [r for r in detail if r.get('tag') == 'composite_na']
    summ['raw_trigger'] = sum(1 for r in raw if r['trigger'])
    summ['raw_hit'] = sum(1 for r in raw if r['hit'])
    summ['orth_trigger'] = sum(1 for r in orth if r['trigger'])
    summ['orth_hit'] = sum(1 for r in orth if r['hit'])
    summ['n_composite_na'] = len(raw_na)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, 'cec_fulltable.json'), 'w', encoding='utf-8') as fp:
        json.dump(summ, fp, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summ.items() if k != 'detail'}, ensure_ascii=False, indent=2))
    print('done')

if __name__ == '__main__':
    main()
