# -*- coding: utf-8 -*-
"""500D orth multiple seeds full pipeline (reusing identify_gpu's R/base/radius)"""
import os, sys, time, json
import numpy as np, warnings
warnings.filterwarnings('ignore')
import torch
from r3_gpu_identify import identify_gpu, make_orth_t

T = 19.53125; A = 10.0; c = 10.0
import os as _os, torch as _torch
DEV = _os.environ.get('PCU_DEV') or ('cuda' if _torch.cuda.is_available() else 'cpu')

def full_pcu(Dd, seed, margin=5.0, voting=True):
    M, o, x0, Mt, ot, wt = make_orth_t(Dd, seed)
    mdl, diag = identify_gpu(Dd, seed, margin, voting=voting)
    if mdl is None:
        return {'trigger': False, 'reason': diag}
    Tf = mdl['T']; R = mdl['R']; base = mdl['base']; radius = mdl['radius']
    H0n = mdl['H0']
    odiag = float(np.sqrt(np.sum((H0n - np.diag(np.diag(H0n)))**2)))
    if R is None and odiag > 1e-6*(1.0+np.abs(H0n).max()):
        return {'trigger': False, 'reason': 'R None but offdiag'}
    Rt = torch.tensor(R, dtype=torch.float64, device=DEV) if R is not None else None
    n = mdl['n']
    grids = []
    for i in range(Dd):
        xs = np.linspace(base[i] - radius, base[i] + radius, n)
        Xg = np.tile(base, (n, 1)); Xg[:, i] = xs
        grids.append(Xg)
    Xp = np.vstack(grids)
    Xz = torch.as_tensor(Xp, dtype=torch.float64, device=DEV)
    Xx = Xz @ Rt
    Z = (Xx - ot) @ Mt.T
    Hzz = 2.0 + A*wt*wt*torch.cos(wt*Z)
    RM = Rt @ Mt.T
    mat = Hzz @ (RM*RM).T
    blk = torch.arange(Dd, device=DEV).repeat_interleave(n)
    diagH = mat[torch.arange(Dd*n, device=DEV), blk].cpu().numpy()
    peaks_all = []
    for i in range(Dd):
        xs = np.linspace(base[i] - radius, base[i] + radius, n)
        Hc = diagH[i*n:(i+1)*n]
        peaks = []
        for j in range(1, n-1):
            if Hc[j] >= Hc[j-1] and Hc[j] >= Hc[j+1]:
                if peaks and (xs[j] - peaks[-1]) < 2*(xs[1]-xs[0]):
                    continue
                a, b, cc = Hc[j-1], Hc[j], Hc[j+1]
                den = a - 2*b + cc
                p = xs[j] + (0.5*(a-cc)/den*(xs[1]-xs[0]) if abs(den) > 1e-30 else 0.0)
                peaks.append(p)
        peaks_all.append(peaks)
    # Zero-crossing candidates
    w = max(0.125*Tf, 1e-3)
    probe = np.linspace(-w, w, 9)
    o_cand = np.empty(Dd)
    for i in range(Dd):
        best = None
        for p in peaks_all[i]:
            zq = np.tile(base, (9, 1))
            zq[:, i] = p + probe
            xq = torch.as_tensor(zq, dtype=torch.float64, device=DEV) @ Rt
            Zq = (xq - ot) @ Mt.T
            gz = 2.0*Zq + A*wt*torch.sin(wt*Zq)
            gx = gz @ Mt
            gzv = (gx @ Rt.T).cpu().numpy()[:, i]
            for j in range(8):
                if (gzv[j] < 0.0) != (gzv[j+1] < 0.0) and gzv[j] != gzv[j+1]:
                    q0, q1 = probe[j], probe[j+1]
                    zc = q0 - gzv[j]*(q1-q0)/(gzv[j+1]-gzv[j])
                    score = min(abs(gzv[j]), abs(gzv[j+1]))
                    if best is None or score < best[0]:
                        best = (score, float(p + zc))
                    break
        if best is None:
            return {'trigger': True, 'hit': False, 'reason': 'no_zero_cross_dim_%d' % i}
        o_cand[i] = best[1]
    # Direct verification of candidates (zero-crossing candidates are already exact; F<tol means a direct hit)
    x0_c = R.T @ o_cand
    Zf0 = Mt @ (torch.as_tensor(x0_c, dtype=torch.float64, device=DEV) - ot)
    F0 = float((Zf0*Zf0 - A*torch.cos(wt*Zf0) + c).sum().item())
    if F0 < 1e-4:
        return {'trigger': True, 'hit': True, 'F': F0, 'T': round(Tf, 4), 'mode': 'cand_direct'}
    # Damped Newton refinement (500D near-degenerate ill-conditioning: Tikhonov regularization + step truncation)
    oz = torch.as_tensor(o_cand, dtype=torch.float64, device=DEV)
    for _ in range(3):
        Zc = (oz - ot) @ Mt.T
        gz = 2.0*Zc + A*wt*torch.sin(wt*Zc)
        gx = gz @ Mt
        g = (gx @ Rt.T).cpu().numpy()
        Hzzc = 2.0 + A*wt*wt*torch.cos(wt*Zc)
        H = (Mt.T * Hzzc) @ Mt
        H_z = R @ H.cpu().numpy() @ R.T
        lam = 1e-6 * max(float(np.max(np.abs(np.diag(H_z)))), 1e-12)
        try:
            delta = np.linalg.solve(H_z + lam*np.eye(Dd), g)
        except np.linalg.LinAlgError:
            break
        step = np.clip(delta, -1.0, 1.0)
        oz_n = oz.cpu().numpy() - step
        oz = torch.as_tensor(oz_n, dtype=torch.float64, device=DEV)
        if float(np.max(np.abs(step))) < 1e-14:
            break
    x_ref = oz.cpu().numpy() @ Rt.cpu().numpy()
    Zf = Mt @ (torch.as_tensor(x_ref, dtype=torch.float64, device=DEV) - ot)
    F = float((Zf*Zf - A*torch.cos(wt*Zf) + c).sum().item())
    return {'trigger': True, 'hit': F < 1e-4, 'F': F, 'T': round(Tf, 4)}

if __name__ == '__main__':
    out = {}
    for sd in [20260916, 20260917, 20260918, 20260919, 20260920,
               20260921, 20260922, 20260923, 20260924, 20260925]:
        r = full_pcu(500, sd)
        out['s%d' % sd] = r
        print('500D seed %d: %s' % (sd, json.dumps(r, ensure_ascii=False)), flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', '_r3_orth500_full.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print('saved _r3_orth500_full.json')
