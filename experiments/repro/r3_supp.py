# -*- coding: utf-8 -*-
import os
"""The third round of reinforcement:
A. Newton-free ablation: 100/250/500D candidates directly F vs candidates+Newton F
B. Three-point eigh R difference + voting vs single point (500D)
C. Same-hardware GPU identify scaling: 100/200/250/400/500D runtime
D. Spectral gap condition number ||H||/gap and vector sensitivity
"""
import sys, time, json
import numpy as np, warnings
warnings.filterwarnings('ignore')
import torch
from r3_gpu_identify import identify_gpu, make_orth_t

T = 19.53125; A = 10.0; c = 10.0
DEV = 'cuda'
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', '_r3_supp.json')
out = {}
try:
    out = json.load(open(RES, encoding='utf-8'))
except Exception:
    pass

def R_of(Dd, seed):
    M, o, x0, Mt, ot, wt = make_orth_t(Dd, seed)
    z0 = Mt @ (torch.as_tensor(x0, dtype=torch.float64, device=DEV) - ot)
    H0 = (Mt.T * (2.0 + A*wt*wt*torch.cos(wt*z0))) @ Mt
    H0n = H0.cpu().numpy()
    lam = np.linalg.eigvalsh(H0n)
    gaps = np.diff(lam)
    cond = float(np.abs(lam).max() / gaps.min())
    pert = max(1.0, 0.1 * float(np.max(np.abs(x0))))
    pts = [x0, x0 + pert, x0 - pert]
    Rs = []
    for p in pts:
        zp = Mt @ (torch.as_tensor(p, dtype=torch.float64, device=DEV) - ot)
        Hp = (Mt.T * (2.0 + A*wt*wt*torch.cos(wt*zp))) @ Mt
        Rs.append(torch.linalg.eigh(Hp)[1].T.cpu().numpy())
    # Three-point pairwise column distances (min sign)
    pair = []
    for a in range(3):
        for b in range(a+1, 3):
            pair.append(float(min(np.linalg.norm(Rs[a]-Rs[b]), np.linalg.norm(Rs[a]+Rs[b])) / np.sqrt(Dd)))
    return Rs, pts, H0n, gaps, cond

def cand_f_and_newton_f(Dd, seed, margin=5.0):
    """Candidate direct F and candidate+damped Newton F (a simplification reusing the full logic)"""
    from r3_gpu_full import full_pcu
    M, o, x0, Mt, ot, wt = make_orth_t(Dd, seed)
    mdl, diag = identify_gpu(Dd, seed, margin)
    if mdl is None:
        return {'trigger': False}
    Tf = mdl['T']; R = mdl['R']; base = mdl['base']; radius = mdl['radius']; n = mdl['n']
    Rt = torch.tensor(R, dtype=torch.float64, device=DEV)
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
    del mat, Hzz, Z, Xx, Xz, Xp; torch.cuda.empty_cache()
    peaks_all = []
    for i in range(Dd):
        xs = np.linspace(base[i] - radius, base[i] + radius, n)
        Hc = diagH[i*n:(i+1)*n]
        peaks = []
        for j in range(1, n-1):
            if Hc[j] >= Hc[j-1] and Hc[j] >= Hc[j+1]:
                if peaks and (xs[j] - peaks[-1]) < 2*(xs[1]-xs[0]):
                    continue
                a2, b2, cc = Hc[j-1], Hc[j], Hc[j+1]
                den = a2 - 2*b2 + cc
                p = xs[j] + (0.5*(a2-cc)/den*(xs[1]-xs[0]) if abs(den) > 1e-30 else 0.0)
                peaks.append(p)
        peaks_all.append(peaks)
    w = max(0.125*Tf, 1e-3)
    probe = np.linspace(-w, w, 9)
    o_cand = np.empty(Dd)
    for i in range(Dd):
        best = None
        for p in peaks_all[i]:
            zq = np.tile(base, (9, 1)); zq[:, i] = p + probe
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
            return {'trigger': True, 'cand_f': None}
        o_cand[i] = best[1]
    def F_of(oz_np):
        xr = oz_np @ Rt.cpu().numpy()
        Zf = Mt @ (torch.as_tensor(xr, dtype=torch.float64, device=DEV) - ot)
        return float((Zf*Zf - A*torch.cos(wt*Zf) + c).sum().item())
    F0 = F_of(o_cand)
    # Damped Newton 3 steps
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
        oz = torch.as_tensor(oz.cpu().numpy() - step, dtype=torch.float64, device=DEV)
        if float(np.max(np.abs(step))) < 1e-14:
            break
    Fn = F_of(oz.cpu().numpy())
    return {'trigger': True, 'cand_f': F0, 'newton_f': Fn}

# A. Newton-free ablation
for Dd in [100, 250, 500]:
    key = 'ablation_D%d' % Dd
    if key not in out:
        r = cand_f_and_newton_f(Dd, 20260916)
        out[key] = r
        print(key, json.dumps(r, ensure_ascii=False), flush=True)
        json.dump(out, open(RES, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

# B. Three-point R difference + voting vs single point (500D)
for Dd in [100, 500]:
    key = 'threeR_D%d' % Dd
    if key not in out:
        Rs, pts, H0n, gaps, cond = R_of(Dd, 20260916)
        # Whether single-point R (Rs[0]) via identify succeeds vs voting R
        single_ok = identify_gpu(Dd, 20260916, 5.0)[0] is not None
        out[key] = {'single_ok': single_ok}
        out[key]['pair_col_dist'] = [round(float(min(np.linalg.norm(Rs[a]-Rs[b]), np.linalg.norm(Rs[a]+Rs[b])) / np.sqrt(Dd)), 3)
                                     for a in range(3) for b in range(a+1, 3)]
        out[key]['cond_H0_gap'] = round(cond, 1)
        out[key]['min_gap'] = float(gaps.min())
        print(key, json.dumps(out[key], ensure_ascii=False), flush=True)
        json.dump(out, open(RES, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

# C. Same-hardware GPU scaling
for Dd in [100, 200, 250, 400, 500]:
    key = 'scaling_D%d' % Dd
    if key not in out:
        t0 = time.time()
        mdl, diag = identify_gpu(Dd, 20260916, 5.0)
        el = time.time() - t0
        out[key] = {'elapsed_s': round(el, 2), 'success': mdl is not None}
        print(key, json.dumps(out[key], ensure_ascii=False), flush=True)
        json.dump(out, open(RES, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

print('ALL DONE -> ' + RES)
