# -*- coding: utf-8 -*-
import os
"""run_17_sota_compare.py - Cross-family comparison of SOTA sampling-iterative algorithms (v2, revised per review).

Revision points (review comments):
  1) 1 run -> 5 runs (median + quartiles), 5 seeds per algorithm;
  2) BO changed to controlled kernel comparison: GP-RBF (smooth kernel) vs GP-ExpSineSquared (periodic kernel, periodicity=19.53)
     -- to argue "smooth prior + few samples mismatched with periodic structure", not "BO itself doesn't work";
  3) Added GWO (non-DE/PSO family);
  4) Wording: report "systematic miss under this setting", not "general failure".

Problem: CEC2017 F5 M_orth (30D main comparison / 10D BO demo), f_opt=500 calibration, T=19.53.
Output: results/sota_compare.json
"""
import io, sys, os, json, time
import numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'cec2017'))
from cec2017.transforms import rotations, shifts

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
BUDGET = 396993
S5 = 5.12 / 100.0
N_RUNS = 5


def make_f5(D, idx=4):
    Mr = rotations[D][idx]
    u, _, vt = np.linalg.svd(Mr)
    M = u @ vt
    o = shifts[idx][:D].copy()
    oM = o @ M.T
    def f(X):
        Xa = np.atleast_2d(np.asarray(X, float))
        z = S5 * (Xa @ M.T - oM)
        # Hand-written bare Rastrigin = official f5 - f_opt (error value, true solution=0), consistent with PCU F<1e-4 criterion
        return np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0, axis=1)
    return f, M, o


def _traj(traj, fe, best_f, pts):
    if len(traj) < pts:
        traj.append([int(fe), round(float(best_f), 4)])


def shade(f, D, budget=BUDGET, pop=100, seed=0, traj_pts=40):
    rng = np.random.RandomState(seed)
    NP = pop; H = 100
    M_CR = np.full(H, 0.5); M_F = np.full(H, 0.5); k_mem = 0
    X = rng.uniform(-100.0, 100.0, (NP, D)); fx = f(X)
    fe = NP
    archive = np.empty((0, D))
    best_f = float(np.min(fx)); traj = [[fe, round(best_f, 4)]]
    while fe < budget:
        rem = budget - fe
        n_new = min(NP, rem)
        r_i = rng.randint(H, size=NP)
        CR_i = np.clip(rng.normal(M_CR[r_i], 0.1), 0.0, 1.0)
        F_i = np.clip(rng.standard_cauchy(size=NP) * 0.1 + M_F[r_i], 0.0, 1.0)
        n_pbest = max(2, int(0.1 * NP))
        order = np.argsort(fx)
        S_CR, S_F, S_dF = [], [], []
        U = X.copy()
        for i in range(NP):
            pb = order[rng.randint(n_pbest)]
            r1 = rng.randint(NP); r2 = rng.randint(NP)
            g = 0
            while (r1 == i or r1 == pb) and g < 10: r1 = rng.randint(NP); g += 1
            g = 0
            while (r2 == i or r2 == pb or r2 == r1) and g < 10: r2 = rng.randint(NP); g += 1
            if len(archive) > 0 and rng.rand() < 0.5:
                xr2 = archive[rng.randint(len(archive))]
            else:
                xr2 = X[r2]
            v = X[i] + F_i[i] * (X[pb] - X[i]) + F_i[i] * (X[r1] - xr2)
            jr = rng.randint(D)
            mask = rng.rand(D) < CR_i[i]; mask[jr] = True
            U[i] = np.clip(np.where(mask, v, X[i]), -100.0, 100.0)
        fu = f(U[:n_new])
        for i in range(n_new):
            if fu[i] < fx[i]:
                S_CR.append(CR_i[i]); S_F.append(F_i[i]); S_dF.append(abs(fx[i] - fu[i]))
                archive = np.vstack([archive, X[i].copy()]) if len(archive) else X[i:i+1].copy()
                X[i] = U[i]; fx[i] = fu[i]
        if len(archive) > NP:
            archive = archive[rng.choice(len(archive), NP, replace=False)]
        if S_dF:
            w = np.array(S_dF) / np.sum(S_dF)
            M_CR[k_mem] = np.sum(w * np.array(S_CR))
            M_F[k_mem] = np.sum(w * np.array(S_F) ** 2) / np.sum(w * np.array(S_F))
            k_mem = (k_mem + 1) % H
        fe += n_new
        if float(np.min(fx)) < best_f: best_f = float(np.min(fx))
        _traj(traj, fe, best_f, traj_pts)
    return best_f, traj


def lshade(f, D, budget=BUDGET, pop=100, seed=0, traj_pts=40):
    rng = np.random.RandomState(seed)
    NP = pop; NP_min = 4; H = 100
    M_CR = np.full(H, 0.5); M_F = np.full(H, 0.5); k_mem = 0
    X = rng.uniform(-100.0, 100.0, (NP, D)); fx = f(X)
    fe = NP
    archive = np.empty((0, D))
    best_f = float(np.min(fx)); traj = [[fe, round(best_f, 4)]]
    while fe < budget:
        rem = budget - fe
        n_new = min(NP, rem)
        r_i = rng.randint(H, size=NP)
        CR_i = np.clip(rng.normal(M_CR[r_i], 0.1), 0.0, 1.0)
        F_i = np.clip(rng.standard_cauchy(size=NP) * 0.1 + M_F[r_i], 0.0, 1.0)
        n_pbest = max(2, int(0.1 * NP))
        order = np.argsort(fx)
        S_CR, S_F, S_dF = [], [], []
        U = X.copy()
        for i in range(NP):
            pb = order[rng.randint(n_pbest)]
            r1 = rng.randint(NP); r2 = rng.randint(NP)
            g = 0
            while (r1 == i or r1 == pb) and g < 10: r1 = rng.randint(NP); g += 1
            g = 0
            while (r2 == i or r2 == pb or r2 == r1) and g < 10: r2 = rng.randint(NP); g += 1
            if len(archive) > 0 and rng.rand() < 0.5:
                xr2 = archive[rng.randint(len(archive))]
            else:
                xr2 = X[r2]
            v = X[i] + F_i[i] * (X[pb] - X[i]) + F_i[i] * (X[r1] - xr2)
            jr = rng.randint(D)
            mask = rng.rand(D) < CR_i[i]; mask[jr] = True
            U[i] = np.clip(np.where(mask, v, X[i]), -100.0, 100.0)
        fu = f(U[:n_new])
        for i in range(n_new):
            if fu[i] < fx[i]:
                S_CR.append(CR_i[i]); S_F.append(F_i[i]); S_dF.append(abs(fx[i] - fu[i]))
                archive = np.vstack([archive, X[i].copy()]) if len(archive) else X[i:i+1].copy()
                X[i] = U[i]; fx[i] = fu[i]
        if len(archive) > NP:
            archive = archive[rng.choice(len(archive), NP, replace=False)]
        if S_dF:
            w = np.array(S_dF) / np.sum(S_dF)
            M_CR[k_mem] = np.sum(w * np.array(S_CR))
            M_F[k_mem] = np.sum(w * np.array(S_F) ** 2) / np.sum(w * np.array(S_F))
            k_mem = (k_mem + 1) % H
        fe += n_new
        if float(np.min(fx)) < best_f: best_f = float(np.min(fx))
        _traj(traj, fe, best_f, traj_pts)
        if fe < budget:
            NP = max(NP_min, int(pop + (NP_min - pop) * (fe / budget)))
            if NP < len(X):
                keep = np.argsort(fx)[:NP]
                X = X[keep]; fx = fx[keep]
    return best_f, traj


def jade(f, D, budget=BUDGET, pop=100, seed=0, traj_pts=40):
    rng = np.random.RandomState(seed)
    NP = pop
    mu_CR = 0.5; mu_F = 0.5; c = 0.1
    X = rng.uniform(-100.0, 100.0, (NP, D)); fx = f(X)
    fe = NP
    archive = np.empty((0, D))
    best_f = float(np.min(fx)); traj = [[fe, round(best_f, 4)]]
    while fe < budget:
        rem = budget - fe
        n_new = min(NP, rem)
        CR_i = np.clip(rng.normal(mu_CR, 0.1, NP), 0.0, 1.0)
        F_i = np.clip(rng.standard_cauchy(size=NP) * 0.1 + mu_F, 0.0, 1.0)
        n_pbest = max(2, int(0.1 * NP))
        order = np.argsort(fx)
        S_CR, S_F = [], []
        U = X.copy()
        for i in range(NP):
            pb = order[rng.randint(n_pbest)]
            r1 = rng.randint(NP); r2 = rng.randint(NP)
            g = 0
            while (r1 == i or r1 == pb) and g < 10: r1 = rng.randint(NP); g += 1
            g = 0
            while (r2 == i or r2 == pb or r2 == r1) and g < 10: r2 = rng.randint(NP); g += 1
            if len(archive) > 0 and rng.rand() < 0.5:
                xr2 = archive[rng.randint(len(archive))]
            else:
                xr2 = X[r2]
            v = X[i] + F_i[i] * (X[pb] - X[i]) + F_i[i] * (X[r1] - xr2)
            jr = rng.randint(D)
            mask = rng.rand(D) < CR_i[i]; mask[jr] = True
            U[i] = np.clip(np.where(mask, v, X[i]), -100.0, 100.0)
        fu = f(U[:n_new])
        for i in range(n_new):
            if fu[i] < fx[i]:
                S_CR.append(CR_i[i]); S_F.append(F_i[i])
                archive = np.vstack([archive, X[i].copy()]) if len(archive) else X[i:i+1].copy()
                X[i] = U[i]; fx[i] = fu[i]
        if len(archive) > NP:
            archive = archive[rng.choice(len(archive), NP, replace=False)]
        if S_CR:
            mu_CR = (1 - c) * mu_CR + c * float(np.mean(S_CR))
            mu_F = (1 - c) * mu_F + c * (float(np.sum(np.square(S_F))) / float(np.sum(S_F)))
        fe += n_new
        if float(np.min(fx)) < best_f: best_f = float(np.min(fx))
        _traj(traj, fe, best_f, traj_pts)
    return best_f, traj


def pso(f, D, budget=BUDGET, pop=100, seed=0, traj_pts=40):
    rng = np.random.RandomState(seed)
    NP = pop; w = 0.72; c1 = c2 = 1.49
    X = rng.uniform(-100.0, 100.0, (NP, D))
    V = rng.uniform(-100.0, 100.0, (NP, D)) * 0.1
    fx = f(X)
    pbest = X.copy(); pbest_f = fx.copy()
    g = int(np.argmin(fx)); gbest = X[g].copy(); gbest_f = float(fx[g])
    fe = NP
    traj = [[fe, round(gbest_f, 4)]]
    while fe < budget:
        rem = budget - fe
        n_new = min(NP, rem)
        r1 = rng.rand(NP, D); r2 = rng.rand(NP, D)
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        V = np.clip(V, -100.0, 100.0)
        X = np.clip(X + V, -100.0, 100.0)
        fu = f(X)
        better = fu < pbest_f
        pbest[better] = X[better]; pbest_f[better] = fu[better]
        gi = int(np.argmin(fu))
        if fu[gi] < gbest_f: gbest_f = float(fu[gi])
        fe += n_new
        _traj(traj, fe, gbest_f, traj_pts)
    return gbest_f, traj


def gwo(f, D, budget=BUDGET, pop=30, seed=0, traj_pts=40):
    """Grey Wolf Optimizer (Mirjalili 2014): alpha/beta/delta guidance, a linearly 2->0."""
    rng = np.random.RandomState(seed)
    NP = pop
    X = rng.uniform(-100.0, 100.0, (NP, D)); fx = f(X)
    fe = NP
    order = np.argsort(fx)
    alpha, beta, delta = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()
    best_f = float(fx[order[0]])
    traj = [[fe, round(best_f, 4)]]
    gen = 0
    max_gen = int(budget / NP)
    while fe < budget:
        rem = budget - fe
        n_new = min(NP, rem)
        a = 2.0 * (1.0 - gen / max_gen)
        for i in range(n_new):
            for leader, label in [(alpha, 1), (beta, 2), (delta, 3)]:
                r1 = rng.rand(D); r2 = rng.rand(D)
                A = 2 * a * r1 - a
                C = 2 * r2
                Dv = np.abs(C * leader - X[i])
                X[i] = X[i] + (leader - A * Dv) / 3.0
            X[i] = np.clip(X[i], -100.0, 100.0)
        fu = f(X[:n_new])
        for i in range(n_new):
            if fu[i] < fx[i]: fx[i] = fu[i]
        order = np.argsort(fx)
        alpha, beta, delta = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()
        fe += n_new; gen += 1
        if float(fx[order[0]]) < best_f: best_f = float(fx[order[0]])
        _traj(traj, fe, best_f, traj_pts)
    return best_f, traj


def bayes_opt_10d(f10, kernel, budget=150, seed=0, traj_pts=40, n_init=20):
    """BO (10D official rotated F5): given GP kernel + EI, n_init random + BO steps."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from scipy.optimize import minimize
    from scipy.stats import norm
    rng = np.random.RandomState(seed)
    D = 10
    X = rng.uniform(-100.0, 100.0, (n_init, D))
    y = f10(X)
    fe = n_init
    best_f = float(np.min(y)); traj = [[fe, round(best_f, 4)]]
    bounds = [(-100.0, 100.0)] * D
    while fe < budget:
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True,
                                      n_restarts_optimizer=1, random_state=seed)
        gp.fit(X, y)
        yb = float(np.min(y)); xb = X[int(np.argmin(y))]
        def neg_ei(x):
            x = np.asarray(x)
            mu, sd = gp.predict(x.reshape(1, -1), return_std=True)
            sd = float(sd[0])
            if sd < 1e-12:
                return 0.0
            imp = yb - float(mu[0])
            Z = imp / sd
            return -(imp * norm.cdf(Z) + sd * norm.pdf(Z))
        best_x = None; best_ei = -1e18
        starts = np.vstack([xb + rng.normal(0, 20, (1, D)), rng.uniform(-100, 100, (9, D))])
        for s in starts:
            res = minimize(neg_ei, s, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 30, 'ftol': 1e-6})
            ei = -res.fun
            if ei > best_ei: best_ei = ei; best_x = res.x
        best_x = np.clip(best_x, -100, 100)
        X = np.vstack([X, best_x.reshape(1, -1)])
        y = np.append(y, float(f10(best_x.reshape(1, -1))[0]))
        fe += 1
        if float(np.min(y)) < best_f: best_f = float(np.min(y))
        _traj(traj, fe, best_f, traj_pts)
    return best_f, traj


def stats(vals):
    v = sorted(vals)
    return {'median': round(float(np.median(v)), 4),
            'q1': round(float(np.percentile(v, 25)), 4),
            'q3': round(float(np.percentile(v, 75)), 4),
            'min': round(float(min(v)), 4), 'max': round(float(max(v)), 4),
            'all_hit': all(x < 1e-4 for x in v), 'n_hit': sum(1 for x in v if x < 1e-4)}


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    t0 = time.time()
    f30, _, _ = make_f5(30)
    f10, _, _ = make_f5(10)
    res = {'problem': 'CEC2017 F5 M_orth (30D main comparison / 10D BO demo), f_opt=500 calibration, T=19.53',
           'budget_fe': BUDGET, 'n_runs': N_RUNS,
           'pcu_reference': {'fe_estimate': BUDGET, 'f': 0.0, 'hit_10of10': True,
                             'note': 'run_09_fd_blackbox measured 10/10 F<1e-4 (30D)'}}
    algos = [('SHADE', shade), ('LSHADE', lshade), ('JADE', jade), ('PSO', pso), ('GWO', gwo)]
    for name, fn in algos:
        finals = []; trajs = []
        for k in range(N_RUNS):
            bf, tr = fn(f30, 30, seed=k)
            finals.append(bf); trajs.append(tr)
        res[name] = {'final_f': stats(finals), 'trajectory_median': trajs[2]}
        print('%s: %s (%.1fs)' % (name, res[name]['final_f'], time.time() - t0), flush=True)
    # BO kernel comparison: RBF (smooth) vs ExpSineSquared (period 19.53)
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, ExpSineSquared
    kern_rbf = ConstantKernel(1.0) * RBF(length_scale=50.0) + WhiteKernel(1e-3)
    kern_per = (ConstantKernel(1.0) * RBF(length_scale=50.0)
                * ExpSineSquared(length_scale=50.0, periodicity=19.53125) + WhiteKernel(1e-3))
    for kname, kern in [('BO_RBF_smooth', kern_rbf), ('BO_periodic_kernel', kern_per)]:
        finals = []; trajs = []
        for k in range(N_RUNS):
            bf, tr = bayes_opt_10d(f10, kern, seed=k)
            finals.append(bf); trajs.append(tr)
        res[kname] = {'final_f': stats(finals), 'trajectory_median': trajs[2],
                      'note': '10D official rotation, 150 FE; RBF=smooth kernel, ExpSineSquared=periodic kernel (periodicity=19.53)'}
        print('%s: %s (%.1fs)' % (kname, res[kname]['final_f'], time.time() - t0), flush=True)
    res['elapsed_s'] = round(time.time() - t0, 1)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, 'sota_compare.json'), 'w', encoding='utf-8') as fp:
        json.dump(res, fp, ensure_ascii=False, indent=2)
    print('done', round(time.time() - t0, 1))


if __name__ == '__main__':
    main()
