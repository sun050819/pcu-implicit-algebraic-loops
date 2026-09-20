# -*- coding: utf-8 -*-
import os
"""run_15_ablation: ablation (review item: ablation).

Three groups of ablation (all reproducible statistics):
  A. CAND_KEEP in {1, 2, 4, 8}: separable A^2EP 30D T=1 (impact of the number of candidates for the greedy/enumeration path)
  B. Directional finite difference vs full Hessian (FD scan): F5 M_orth 30D (baseline comparison with run_09_fd_blackbox)
  C. Threshold sensitivity: tol_off in {1e-4, 1e-6, 1e-8}, margin in {50, 100, 200}
     (F5 M_orth analytical version; tol_off discriminated using analytical H, margin uses struct_id_margin)
Output: results/ablation.json
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

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
N = 10
SEED0 = 20260916
A = 10.0; c = 10.0

def make_sep(D=30, T=1.0, seed=0):
    rng = np.random.RandomState(seed)
    o = rng.uniform(-5.0, 5.0, D)
    w = 2.0*np.pi/T
    def f(x):
        z = np.asarray(x, float)-o
        return float(np.sum(z*z - A*np.cos(w*z) + c))
    def g(x):
        z = np.asarray(x, float)-o
        return 2.0*z + A*w*np.sin(w*z)
    def h(x):
        z = np.asarray(x, float)-o
        return np.diag(2.0 + A*w*w*np.cos(w*z))
    return f, g, h, o

# ---- A: CAND_KEEP ablation (monkeypatch struct_id.CAND_KEEP) ----
def ablation_cand_keep():
    D = 30; T = 1.0
    f, g, h, o = make_sep(D, T, SEED0)
    out = {}
    for K in [1, 2, 4, 8]:
        sid.CAND_KEEP = K
        ok = 0; trig = 0
        for k in range(N):
            rng = np.random.RandomState(SEED0 + k)
            x0 = rng.uniform(-5.0, 5.0, D)
            xc, tag = sid.estimate_and_pcu(f, g, h, x0, margin=5.0)
            if tag == 'struct_pcu_ok':
                trig += 1
                if f(xc) < 1e-4:
                    ok += 1
        out['CAND_KEEP=%d' % K] = {'trigger': trig, 'hit': ok, 'hit_rate': ok/N}
    sid.CAND_KEEP = 4
    return out

# ---- B: directional finite difference vs full Hessian (FD scan) ----
def ablation_fd_scan():
    D = 30
    o = shifts[4][:D].copy()
    M_raw = rotations[D][4].copy()
    u, _, vt = np.linalg.svd(M_raw)
    M = u @ vt
    S5 = 5.12/100.0
    HS = 1e-3
    def f_black(x):
        z = S5*(M @ (np.asarray(x, float)-o))
        return float(np.sum(z*z - 10.0*np.cos(2.0*np.pi*z) + 10.0))
    def fd_hess_dir(x, r):
        return (f_black(x+HS*r) - 2.0*f_black(x) + f_black(x-HS*r)) / HS**2
    def fd_hess_full(x):
        H = np.zeros((D, D))
        for i in range(D):
            xp = x.copy(); xm = x.copy()
            xp[i] += HS; xm[i] -= HS
            H[i,i] = (f_black(xp) - 2.0*f_black(x) + f_black(xm)) / HS**2
            for j in range(i+1, D):
                xpp=x.copy(); xpm=x.copy(); xmp=x.copy(); xmm=x.copy()
                xpp[i]+=HS; xpp[j]+=HS; xpm[i]+=HS; xpm[j]-=HS
                xmp[i]-=HS; xmp[j]+=HS; xmm[i]-=HS; xmm[j]-=HS
                H[i,j]=H[j,i]=(f_black(xpp)-f_black(xpm)-f_black(xmp)+f_black(xmm))/(4.0*HS**2)
        return H
    def _off(A):
        A = np.abs(A); np.fill_diagonal(A, 0.0)
        return float(np.sqrt(np.sum(A*A)))
    out = {}
    for mode in ['dir', 'full']:
        ok = trig = 0
        for k in range(N):
            rng = np.random.RandomState(SEED0 + k)
            x0 = rng.uniform(-100.0, 100.0, D)
            # R recovery (shared by both modes: full 3-point finite difference)
            pts = [x0, x0+1.0, x0-1.0]
            Hs = [fd_hess_full(p) for p in pts]
            cands = [np.linalg.eigh(Hj)[1].T for Hj in Hs]
            best_R, best_s = None, float('inf')
            for Rj in cands:
                s = sum(_off(Rj @ Hl @ Rj.T) for Hl in Hs)
                if s < best_s:
                    best_s, best_R = s, Rj
            R = best_R; base = R @ x0
            radius = 10.0 + float(np.max(np.abs(base))) + 100.0
            # scan
            T_ests = []
            for i in range(D):
                n_rough = min(max(int(np.ceil(2.0*radius*12.0))+1, 512), 4000)
                xs = np.linspace(base[i]-radius, base[i]+radius, n_rough)
                Hc = np.empty(n_rough)
                rvec = R[i]
                for jj in range(n_rough):
                    xk = np.asarray(base, dtype=float).copy(); xk[i] = xs[jj]
                    if mode == 'dir':
                        Hc[jj] = fd_hess_dir(R.T @ xk, rvec)
                    else:
                        Hc[jj] = fd_hess_full(R.T @ xk)[i, i]
                Tc = sid._coarse_period(xs, Hc)
                if Tc is None:
                    break
                T_ests.append(Tc)
            if len(T_ests) < D:
                continue
            T_c = float(np.median(T_ests))
            if any(abs(Ti-T_c)/T_c > 0.2 for Ti in T_ests):
                continue
            trig += 1
            # simplified hit: T recovery correct (1/S5=19.53125) is considered a successful trigger
            if abs(T_c - 19.53125)/19.53125 < 1e-2:
                ok += 1
        out['fd_%s' % mode] = {'trigger': trig, 'T_hit': ok, 'T_hit_rate': ok/N}
    return out

# ---- C: threshold sensitivity (F5 M_orth analytical version) ----
def ablation_thresholds():
    D = 30
    o = shifts[4][:D].copy()
    M_raw = rotations[D][4].copy()
    u, _, vt = np.linalg.svd(M_raw)
    M = u @ vt
    S5 = 5.12/100.0
    def f5(x):
        z = S5*(M @ (np.asarray(x, float)-o))
        return float(np.sum(z*z - 10.0*np.cos(2.0*np.pi*z) + 10.0))
    def g5(x):
        x = np.asarray(x, float); z = S5*(M @ (x-o))
        gz = S5*(2.0*z + 20.0*np.pi*np.sin(2.0*np.pi*z))
        return M.T @ gz
    def h5(x):
        x = np.asarray(x, float); z = S5*(M @ (x-o))
        Hzz = S5*S5*(2.0 + 40.0*np.pi*np.pi*np.cos(2.0*np.pi*z))
        return (M.T * Hzz) @ M
    out = {}
    for tol_off in [1e-4, 1e-6, 1e-8]:
        ok = trig = 0
        for k in range(N):
            rng = np.random.RandomState(SEED0 + k)
            x0 = rng.uniform(-100.0, 100.0, D)
            mdl = sid.identify_structure(f5, g5, h5, x0, margin=100.0, tol_off=tol_off)
            if mdl is not None:
                trig += 1
                xc, tag = sid.estimate_and_pcu(f5, g5, h5, x0, margin=100.0)
                if tag == 'struct_pcu_ok' and f5(xc) < 1e-4:
                    ok += 1
        out['tol_off=%.0e' % tol_off] = {'trigger': trig, 'hit': ok, 'hit_rate': ok/N}
    # margin ablation (struct_id_margin passed through)
    from aloop.solve import hgca
    for margin in [50.0, 100.0, 200.0]:
        ok = trig = 0
        for k in range(N):
            rng = np.random.RandomState(SEED0 + k)
            x0 = rng.uniform(-100.0, 100.0, D)
            r = hgca(f5, g5, h5, x0, tol=1e-10, max_iter=3000,
                     cfg={"residual_fn": g5, "jacobian_fn": h5, "use_caci": False,
                          "sar_enabled": False, "use_pcu": True, "use_struct_id": True,
                          "struct_id_margin": margin}, f_opt=0.0)
            xb = r.get('x')
            fb = float(f5(xb)) if xb is not None else float('nan')
            if r.get('success') and fb < 1e-4:
                ok += 1
            if 'pcu' in str(r.get('stage', '')).lower() or 'struct' in str(r.get('stage', '')).lower():
                trig += 1
        out['margin=%.0f' % margin] = {'trigger': trig, 'hit': ok, 'hit_rate': ok/N}
    return out

def main():
    t0 = time.time()
    res = {'A_cand_keep': ablation_cand_keep(),
           'B_fd_scan': ablation_fd_scan(),
           'C_thresholds': ablation_thresholds()}
    res['_meta'] = {'n_runs': N, 'seed0': SEED0, 'elapsed_s': round(time.time()-t0, 1)}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, 'ablation.json'), 'w', encoding='utf-8') as fp:
        json.dump(res, fp, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print('done')

if __name__ == '__main__':
    main()
