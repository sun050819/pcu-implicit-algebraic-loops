# -*- coding: utf-8 -*-
"""GPU version of the real identify process (replicating the full struct_id logic, hess batched on GPU):
Re-verify the true boundaries of orthogonal rotation for 250/300/400/500D (two margin regimes: 5.0 and 20.0)"""
import io, sys, time
# stdout wrapper removed (nested wrapper breaks background runs)
import numpy as np, warnings
warnings.filterwarnings('ignore')
import torch

T = 19.53125; A = 10.0; c = 10.0
import os as _os
DEV = _os.environ.get('PCU_DEV', 'cuda') if _os.environ.get('PCU_DEV') else ('cuda' if __import__('torch').cuda.is_available() else 'cpu')

def make_orth_t(Dd, seed):
    rng = np.random.RandomState(seed)
    G = rng.randn(Dd, Dd)
    u, _, vt = np.linalg.svd(G)
    M = u @ vt
    o = rng.uniform(-5.0, 5.0, Dd)
    x0 = rng.uniform(-5.0, 5.0, Dd)
    w = 2.0 * np.pi / T
    Mt = torch.tensor(M, dtype=torch.float64, device=DEV)
    ot = torch.tensor(o, dtype=torch.float64, device=DEV)
    wt = torch.tensor(w, dtype=torch.float64, device=DEV)
    return M, o, x0, Mt, ot, wt

def coarse_period(xs, y):
    n = len(y)
    yy = np.asarray(y) - np.mean(y)
    sp = np.abs(np.fft.rfft(yy * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=xs[1] - xs[0])
    sp[0] = 0.0
    k = int(np.argmax(sp)); f = freqs[k]
    if f <= 0: return None
    Tc = 1.0 / f
    span = xs[-1] - xs[0]
    if Tc < 2.2 * (xs[1] - xs[0]) or Tc > 0.5 * span: return None
    return float(Tc)

def parabolic_peak(xs, y, i):
    n = len(y)
    if i <= 0 or i >= n - 1: return xs[i]
    a, b, cc = y[i-1], y[i], y[i+1]
    denom = a - 2.0*b + cc
    if abs(denom) < 1e-30: return xs[i]
    return xs[i] + 0.5*(a-cc)/denom*(xs[1]-xs[0])

def scan_hess_diag_batch(Mt, ot, wt, R_np, base, Dd, n, radius, chunk=64):
    """GPU batching (chunking to prevent OOM): compute the diagonal elements of hess_z along scan lines of n points per dimension"""
    Rt = torch.tensor(R_np, dtype=torch.float64, device=DEV)
    RM = Rt @ Mt.T
    out = np.empty(Dd * n)
    for i0 in range(0, Dd, chunk):
        i1 = min(i0 + chunk, Dd)
        grids = []
        for i in range(i0, i1):
            xs = np.linspace(base[i] - radius, base[i] + radius, n)
            Xg = np.tile(base, (n, 1)); Xg[:, i] = xs
            grids.append(Xg)
        Xp = np.vstack(grids)  # ((i1-i0)*n, Dd)
        Xz = torch.as_tensor(Xp, dtype=torch.float64, device=DEV)
        Xx = Xz @ Rt
        Z = (Xx - ot) @ Mt.T
        Hzz = 2.0 + A*wt*wt*torch.cos(wt*Z)
        mat = Hzz @ (RM*RM).T
        blk = torch.arange(i0, i1, device=DEV).repeat_interleave(n)
        idx = torch.arange((i1-i0)*n, device=DEV)
        out[(i0*n):(i1*n)] = mat[idx, blk].cpu().numpy()
        del Xp, Xz, Xx, Z, Hzz, mat
        torch.cuda.empty_cache()
    return out

def identify_gpu(Dd, seed, margin, voting=True):
    M, o, x0, Mt, ot, wt = make_orth_t(Dd, seed)
    # H0 (center point)
    z0 = Mt @ (torch.as_tensor(x0, dtype=torch.float64, device=DEV) - ot)
    H0 = (Mt.T * (2.0 + A*wt*wt*torch.cos(wt*z0))) @ Mt
    H0n = H0.cpu().numpy()
    odiag = float(np.sqrt(np.sum((H0n - np.diag(np.diag(H0n)))**2)))
    tol = 1e-6 * (1.0 + np.abs(H0n).max())
    if odiag <= tol:
        R = None; base = np.asarray(x0, float)
        hess_diag = None
    else:
        # Three-point eigh + voting (when voting=False, use only the center point for a single decomposition)
        if voting:
            pert = max(1.0, 0.1 * float(np.max(np.abs(x0))))
            pts = [x0, x0 + pert, x0 - pert]
            Rcands = []
            for p in pts:
                zp = Mt @ (torch.as_tensor(p, dtype=torch.float64, device=DEV) - ot)
                Hp = (Mt.T * (2.0 + A*wt*wt*torch.cos(wt*zp))) @ Mt
                Rcands.append(torch.linalg.eigh(Hp)[1].T.cpu().numpy())
        else:
            pts = [x0]
            Rcands = [torch.linalg.eigh(H0)[1].T.cpu().numpy()]
        best_R, best_s = None, float('inf')
        for Rj in Rcands:
            s = 0.0
            for p in pts:
                zp = Mt @ (torch.as_tensor(p, dtype=torch.float64, device=DEV) - ot)
                Hp = (Mt.T * (2.0 + A*wt*wt*torch.cos(wt*zp))) @ Mt
                Hp_n = Hp.cpu().numpy()
                s += float(np.sqrt(np.sum((Rj @ Hp_n @ Rj.T - np.diag(np.diag(Rj @ Hp_n @ Rj.T)))**2)))
            if s < best_s:
                best_s, best_R = s, Rj
        R = best_R
        base = R @ np.asarray(x0, float)
    mag = float(np.max(np.abs(base)))
    radius = max(10.0, mag + margin)
    # Coarse scan (n_rough points per dimension, one batch)
    t0 = time.time()
    for _attempt in range(3):
        n_r = min(max(int(np.ceil(2.0*radius*12.0)) + 1, 512), 4000)
        diagH = scan_hess_diag_batch(Mt, ot, wt, R, base, Dd, n_r, radius)
        T_ests = []
        ok = True
        for i in range(Dd):
            xs = np.linspace(base[i] - radius, base[i] + radius, n_r)
            Tc_i = coarse_period(xs, diagH[i*n_r:(i+1)*n_r])
            if Tc_i is None:
                ok = False
                break
            T_ests.append(Tc_i)
        if not ok:
            radius *= 2.0
            continue
        T_c = float(np.median(T_ests))
        bad = [Ti for Ti in T_ests if abs(Ti - T_c)/T_c > 0.2]
        if bad:
            return None, {'stage': 'coarse_consensus', 'radius': radius, 'T_ests_range': (min(T_ests), max(T_ests))}
        if radius < 2.0 * T_c:
            radius = 2.0 * T_c + margin
            continue
        break
    if not T_ests:
        return None, {'stage': 'coarse_fail', 'radius': radius}
    t_coarse = time.time() - t0
    # Fine scan: n=ceil(2r/Tc.16)+1 >=64
    n = int(np.ceil(2.0*radius/T_c*16.0)) + 1
    n = min(max(n, 64), 4000)
    diagH = scan_hess_diag_batch(Mt, ot, wt, R, base, Dd, n, radius)
    gap_all = []
    for i in range(Dd):
        xs = np.linspace(base[i] - radius, base[i] + radius, n)
        Hc = diagH[i*n:(i+1)*n]
        peaks = []
        for j in range(1, n-1):
            if Hc[j] >= Hc[j-1] and Hc[j] >= Hc[j+1]:
                if peaks and (xs[j] - peaks[-1]) < 2*(xs[1]-xs[0]):
                    continue
                peaks.append(parabolic_peak(xs, Hc, j))
        if len(peaks) < 2:
            return None, {'stage': 'fine_peaks', 'dim': i, 'radius': radius, 'T_c': T_c}
        gaps = np.diff(np.sort(peaks))
        gaps = gaps[(gaps > 0.5*T_c) & (gaps < 1.5*T_c)]
        if len(gaps) == 0:
            return None, {'stage': 'fine_gap', 'dim': i, 'radius': radius, 'T_c': T_c}
        gap_all.extend(list(gaps))
    if len(gap_all) < Dd:
        return None, {'stage': 'gap_total', 'radius': radius}
    Tf = float(np.median(gap_all))
    return {'T': Tf, 'mode': 'rotated' if R is not None else 'separable',
            't_coarse': t_coarse, 'radius': radius, 'n': n, 'R': R,
            'base': base, 'H0': H0n}, None

if __name__ == '__main__':
    import json, os
    out = {}
    for Dd in [250, 300, 400, 500]:
        for margin in [5.0, 20.0]:
            key = 'D%d_m%.0f' % (Dd, margin)
            t0 = time.time()
            mdl, diag = identify_gpu(Dd, 20260916, margin)
            el = time.time() - t0
            if mdl is None:
                out[key] = {'reject': True, 'diag': diag, 'elapsed_s': round(el, 1)}
            else:
                out[key] = {'reject': False, 'T': round(mdl['T'], 4), 'mode': mdl['mode'],
                            'elapsed_s': round(el, 1), 'radius': round(mdl['radius'], 1), 'n': mdl['n']}
            print(key, json.dumps(out[key], ensure_ascii=False), flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', '_r3_gpu_identify.json'), 'w', encoding='utf-8') as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print('saved _r3_gpu_identify.json')
