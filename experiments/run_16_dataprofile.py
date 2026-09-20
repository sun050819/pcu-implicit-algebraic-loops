# -*- coding: utf-8 -*-
import os
"""run_16_dataprofile: data profile (review item: FE comparison curve).

Problem: CEC2017 F5 M_orth, 30D (same setting as run_07_cec).
Two solvers:
  PCU    : black-box FD-PCU (run_09_fd_blackbox setting), records total number of f calls FE_pcu (single-point F=0 vertical curve)
  CMA-ES : minimal CMA-ES implementation (same family as hgca), records the (FE, best F) trajectory
          -- the progress of CMA-ES when PCU has already converged within the same budget.
Output: results/dataprofile.json (scatter + CMA trajectory array)
"""
import io, sys, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'cec2017'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
D = 30
SEED0 = 20260916

from cec2017.transforms import rotations, shifts
o = shifts[4][:D].copy()
M_raw = rotations[D][4].copy()
u, _, vt = np.linalg.svd(M_raw)
M = u @ vt
S5 = 5.12 / 100.0

def f5(x):
    z = S5 * (M @ (np.asarray(x, float) - o))
    return float(np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0))

def cma_es_track(fn, x0, sigma0=1.0, popsize=30, max_fe=200000, seed=0):
    """Minimal CMA-ES (rank-mu update) returns (fe_list, f_list)."""
    rng = np.random.RandomState(seed)
    d = len(x0)
    lam = popsize
    mu = lam // 2
    m = np.asarray(x0, float).copy()
    sigma = sigma0
    C = np.eye(d)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mueff = 1.0 / np.sum(weights**2)
    cc = (4.0 + mueff / d) / (d + 4.0 + 2.0 * mueff / d)
    cs = (mueff + 2.0) / (d + mueff + 5.0)
    c1 = 2.0 / ((d + 1.3)**2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((d + 2.0)**2 + mueff))
    dams = np.sqrt(cs * (2.0 - cs))
    pc = np.zeros(d); ps = np.zeros(d)
    B = np.eye(d); Dv = np.ones(d)
    fe = 0; fe_list = []; f_list = []
    best_f = float('inf')
    while fe < max_fe:
        zs = rng.randn(lam, d)
        ys = zs @ (B * Dv).T
        xs = m + sigma * ys
        fs = np.array([fn(x) for x in xs])
        fe += lam
        order = np.argsort(fs)
        best_f = min(best_f, float(fs[order[0]]))
        fe_list.append(fe); f_list.append(best_f)
        # Update
        y_w = ys[order[:mu]].T @ weights
        m_new = m + sigma * y_w
        ps = (1.0 - cs) * ps + dams * np.sqrt(cs * (2.0 - cs) * mueff) * (m_new - m) / sigma
        hsig = float(np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs)**(2.0 * fe / lam))) < (1.4 + 2.0 / (d + 1.0))
        pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * y_w
        C = (1.0 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1.0 - hsig) * cc * (2.0 - cc) * C) + \
            cmu * ((ys[order[:mu]].T * weights) @ ys[order[:mu]])
        C = (C + C.T) / 2.0
        # Eigendecomposition
        evals, B = np.linalg.eigh(C)
        Dv = np.sqrt(np.clip(evals, 1e-20, None))
        sigma *= np.exp((cs / dams) * (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs)**(2.0 * fe / lam)) - 1.0))
        sigma = max(sigma, 1e-12)
        m = m_new
        # Sampling-based convergence criterion
        if np.max(Dv) * sigma < 1e-12:
            break
    return fe_list, f_list, best_f

def pcu_fe_count():
    """f-call count for FD-PCU (run_09_fd_blackbox setting, F5 M_orth 30D)."""
    from aloop.solve import struct_id as sid
    from aloop.solve import pcu as _pcu
    HS = 1e-3
    def fd_grad(x):
        x = np.asarray(x, float); g = np.zeros(D)
        for i in range(D):
            xp = x.copy(); xm = x.copy(); xp[i] += HS; xm[i] -= HS
            g[i] = (f5(xp) - f5(xm)) / (2.0 * HS)
        return g
    def fd_hess_dir(x, r):
        return (f5(x + HS*r) - 2.0*f5(x) + f5(x - HS*r)) / HS**2
    def fd_hess_full(x):
        H = np.zeros((D, D))
        for i in range(D):
            xp = x.copy(); xm = x.copy(); xp[i] += HS; xm[i] -= HS
            H[i,i] = (f5(xp) - 2.0*f5(x) + f5(xm)) / HS**2
            for j in range(i+1, D):
                xpp=x.copy(); xpm=x.copy(); xmp=x.copy(); xmm=x.copy()
                xpp[i]+=HS; xpp[j]+=HS; xpm[i]+=HS; xpm[j]-=HS
                xmp[i]-=HS; xmp[j]+=HS; xmm[i]-=HS; xmm[j]-=HS
                H[i,j]=H[j,i]=(f5(xpp)-f5(xpm)-f5(xmp)+f5(xmm))/(4.0*HS**2)
        return H
    def _off(A):
        A = np.abs(A); np.fill_diagonal(A, 0.0)
        return float(np.sqrt(np.sum(A*A)))
    rng = np.random.RandomState(SEED0)
    x0 = rng.uniform(-100.0, 100.0, D)
    # R recovery: full 3-point difference
    n_fe = 0
    pts = [x0, x0+1.0, x0-1.0]
    Hs = []
    for p in pts:
        Hs.append(fd_hess_full(p)); n_fe += D*D  # finite-difference Hessian approximation uses D^2 f calls (actually (D^2+D)/2*2+1)
    # Actual number of f calls for the full finite-difference Hessian: (D^2+D)/2 * 2 + 1
    n_fe = 3 * ((D*D + D) // 2 * 2 + 1)
    cands = [np.linalg.eigh(Hj)[1].T for Hj in Hs]
    best_R, best_s = None, float('inf')
    for Rj in cands:
        s = sum(_off(Rj @ Hl @ Rj.T) for Hl in Hs)
        if s < best_s:
            best_s, best_R = s, Rj
    R = best_R; base = R @ x0
    radius = 10.0 + float(np.max(np.abs(base))) + 100.0
    # Coarse sweep + fine sweep (directional difference 3 f/point)
    for i in range(D):
        n_rough = min(max(int(np.ceil(2.0*radius*12.0))+1, 512), 4000)
        n_fine = int(np.ceil(2.0*radius/19.53125*16.0))+1
        n_fe += (n_rough + n_fine) * 3
    n_fe += D * D  # candidate evaluation fd_grad approximation
    return n_fe

def main():
    t0 = time.time()
    rng = np.random.RandomState(SEED0)
    x0 = rng.uniform(-100.0, 100.0, D)
    # PCU budget (run_09_fd_blackbox measured setting: coarse sweep + fine sweep directional difference 3 f/point + R recovery 3x full finite-difference Hessian)
    radius = 10.0 + float(np.max(np.abs(R := None))) if False else 0.0
    fe_pcu = pcu_fe_count()
    # CMA-ES trajectory (same PCU budget)
    fe_list, f_list, best_f = cma_es_track(f5, x0, sigma0=50.0, popsize=40, max_fe=fe_pcu, seed=SEED0)
    res = {
        'problem': 'CEC2017 F5 M_orth 30D',
        'pcu': {'fe_estimate': fe_pcu, 'f': 0.0, 'note': 'run_09_fd_blackbox measured 10/10 hits with F<1e-4'},
        'cma_es': {'max_fe': fe_pcu, 'best_f_after_budget': round(float(best_f), 4),
                   'trajectory': [[int(a), round(float(b), 3)] for a, b in
                                  zip(fe_list[::max(1, len(fe_list)//40)], f_list[::max(1, len(f_list)//40)])]},
        'elapsed_s': round(time.time()-t0, 1),
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, 'dataprofile.json'), 'w', encoding='utf-8') as fp:
        json.dump(res, fp, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False)[:3000])
    print('done')

if __name__ == '__main__':
    main()
