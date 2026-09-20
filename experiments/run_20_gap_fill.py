# -*- coding: utf-8 -*-
import os
"""run_20_gap_fill.py - Pre-submission gap-filling experiments (TEVC GAP ANALYSIS G2/G3).

Gap-filling items:
  A) Fourier feature surrogate (RFF, Rahimi & Recht 2008): Gaussian spectral RFF surrogate + iterative sampling,
     same budget 396,993 FE -- argue that "generic surrogates (smooth kernels) systematically cannot represent periodic structure".
  B) Multi-start L-BFGS (scipy): random restarts within budget for local refinement -- argue that "multi-start alone is not enough;
     PCU's advantage comes from structure identification rather than multi-start".
  C) basinhopping (scipy): classic global-local hybrid -- same as above.
  D) PCU FE dual accounting: implemented separately in run_21_fd_dual.py (FE count of "scan + candidates only",
     excluding the O(D^2) Hessian finite differences, vs the black-box full pipeline's 396,993); not part of this script.

Problem: CEC2017 F5 M_orth 30D (same setup as run_17_sota_compare, f_opt calibrated, T=19.53, F<1e-4 hit).
Output: results/gap_fill.json
"""
import io, sys, os, json, time
import numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_17_sota_compare import make_f5, BUDGET, stats

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
N_RUNS = 5


def rff_surrogate(f, D, budget=BUDGET, seed=0, n_feat=512, n_init=100,
                  sigma=50.0, batch=4000, n_restarts=60):
    """Standard RFF surrogate optimization (full budget): Gaussian spectral random features -> ridge regression surrogate ->
    multi-start L-BFGS on the surrogate to find the best candidate -> batch true-function evaluations -> augmented refitting,
    until the budget of 396,993 FE is exhausted (same budget and 5-run protocol as SHADE)."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-100.0, 100.0, (n_init, D)); y = np.array([float(np.asarray(f(np.atleast_2d(x))).reshape(-1)[0]) for x in X], float)
    fe = n_init
    best_f = float(y.min()); traj = [[fe, round(best_f, 4)]]
    W = rng.normal(0.0, 1.0 / sigma, (D, n_feat))
    b = rng.uniform(0.0, 2.0 * np.pi, n_feat)

    def phi(Xm):
        Xm = np.atleast_2d(Xm)
        return np.sqrt(2.0 / n_feat) * np.cos(Xm @ W + b)

    from scipy.optimize import minimize
    lam = 1e-6
    while fe < budget:
        P = phi(X)
        A = P.T @ P + lam * np.eye(n_feat)
        coef = np.linalg.solve(A, P.T @ y)
        def surr(x):
            return float((phi(np.atleast_2d(x)) @ coef).item())
        # Surrogate global candidates: uniform sampling + local refinement (0 FE, evaluated on surrogate only)
        Xs = rng.uniform(-100.0, 100.0, (3000, D))
        sv = phi(Xs) @ coef
        order = np.argsort(sv)[:n_restarts]
        best_x = None; best_s = float('inf')
        for ix in order:
            r = minimize(surr, np.asarray(Xs[ix]).reshape(-1), method='L-BFGS-B',
                         bounds=[(-100.0, 100.0)] * D,
                         options={'maxiter': 40, 'ftol': 1e-10})
            if r.fun < best_s:
                best_s = r.fun; best_x = r.x
        best_x = np.clip(best_x, -100, 100)
        # Batch candidates: surrogate optimum + neighborhood perturbation, true-function evaluation (counts FE)
        cands = [best_x]
        while len(cands) < batch:
            cands.append(np.clip(best_x + rng.normal(0, 3.0, D), -100, 100))
        rem = min(batch, budget - fe)
        vals = [float(np.asarray(f(np.atleast_2d(xc))).reshape(-1)[0]) for xc in cands[:rem]]
        X = np.vstack([X, np.array(cands[:rem])]); y = np.append(y, vals)
        fe += rem
        bf = float(min(vals))
        if bf < best_f: best_f = bf
        traj.append([fe, round(best_f, 4)])
    return best_f, traj


def multistart_lbfgs(f, D, budget=BUDGET, seed=0, n_restarts=400):
    """Multi-start L-BFGS: random-start local refinement within budget (actual scipy implementation)."""
    from scipy.optimize import minimize
    rng = np.random.RandomState(seed)
    per = max(1, budget // n_restarts)
    best_f = float('inf'); traj = []
    fe = 0
    for k in range(n_restarts):
        x0 = rng.uniform(-100.0, 100.0, D)
        cnt = [0]
        def ff(x):
            cnt[0] += 1
            return float(f(x))
        r = minimize(ff, x0, method='L-BFGS-B', bounds=[(-100.0, 100.0)] * D,
                     options={'maxiter': max(50, per // 2), 'ftol': 1e-12})
        fe += cnt[0]
        if r.fun < best_f: best_f = float(r.fun)
        if fe >= budget: break
        traj.append([fe, round(best_f, 4)])
    return best_f, traj


def basinhop(f, D, budget=BUDGET, seed=0, niter=1200):
    """basinhopping: random perturbation + local L-BFGS hybrid (actual scipy implementation)."""
    from scipy.optimize import basinhopping
    rng = np.random.RandomState(seed)
    x0 = rng.uniform(-100.0, 100.0, D)
    cnt = [0]
    def ff(x):
        cnt[0] += 1
        return float(f(x))
    def budget_check(f_new, x_new, accept):
        return cnt[0] < budget
    r = basinhopping(ff, x0, niter=niter, T=1.0, stepsize=20.0,
                     minimizer_kwargs={'method': 'L-BFGS-B',
                                       'bounds': [(-100.0, 100.0)] * D,
                                       'options': {'maxiter': 60, 'ftol': 1e-12}},
                     take_step=None, accept_test=budget_check,
                     seed=seed, niter_success=None)
    return float(r.fun), None


def main():
    t0 = time.time()
    f30, _, _ = make_f5(30)
    def f1(x):
        return float(np.asarray(f30(np.atleast_2d(x))).reshape(-1)[0])
    res = {'problem': 'CEC2017 F5 M_orth 30D, f_opt=500 calibrated, T=19.53, budget=%d' % BUDGET,
           'budget_fe': BUDGET, 'n_runs': N_RUNS}

    # If json already exists (RFF/L-BFGS done), only add basinhopping
    gap_path = os.path.join(RESULTS, 'gap_fill.json')
    if os.path.exists(gap_path):
        with open(gap_path, encoding='utf-8') as fp:
            res = json.load(fp)

    if 'RFF_surrogate' not in res:
        for name, fn in [('RFF_surrogate', rff_surrogate), ('Multi-start L-BFGS', multistart_lbfgs)]:
            finals = []; trajs = []
            for k in range(N_RUNS):
                bf, tr = fn(f1, 30, seed=k)
                finals.append(bf)
                if tr: trajs.append(tr)
            res[name] = {'final_f': stats(finals),
                         'trajectory_median': (trajs[2] if len(trajs) >= 3 else trajs[0])}

    # basinhopping 1 run (slower, 1 run for reference)
    if 'Basinhopping' not in res:
        bf, _ = basinhop(f1, 30, seed=0, niter=600)
        res['Basinhopping'] = {'final_f': {'median': round(float(bf), 4), 'n_hit': int(bf < 1e-4)}}

    res['elapsed_s'] = round(time.time() - t0, 1)
    os.makedirs(RESULTS, exist_ok=True)
    with open(gap_path, 'w', encoding='utf-8') as fp:
        json.dump(res, fp, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
