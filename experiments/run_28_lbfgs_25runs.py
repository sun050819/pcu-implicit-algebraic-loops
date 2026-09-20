# -*- coding: utf-8 -*-
import os
"""run_28_lbfgs_25runs.py
Hedging against information asymmetry risk: giving the sampling-iteration baseline second-order information (Multi-start L-BFGS, quasi-Newton),
compared with PCU under the same 396,993 FE budget. If it still has 0 hits, this proves a "paradigm problem" rather than an "information problem".
25 seeds aligned with sota_baselines_25runs.json. Output results/lbfgs_25runs.json
"""
import io, sys, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'cec2017'))
from cec2017.transforms import rotations, shifts

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
BUDGET = 396993
S5 = 5.12 / 100.0
D = 30
SEEDS = list(range(20260912, 20260937))  # same 25 seeds as run_26_pcu_vs_baselines

Mr = rotations[D][4]
u, _, vt = np.linalg.svd(Mr)
M = u @ vt
o = shifts[4][:D].copy()
oM = o @ M.T


def f5(X):
    Xa = np.atleast_2d(np.asarray(X, float))
    z = S5 * (Xa @ M.T - oM)
    return np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0, axis=1)


def multistart_lbfgs(f, budget=BUDGET, seed=0, n_restarts=100000):
    """Multi-start L-BFGS: random-start local refinement within budget (quasi-Newton = second-order information), same FE convention."""
    from scipy.optimize import minimize
    rng = np.random.RandomState(seed)
    best_f = float('inf')
    fe = 0
    n_start = 0
    while fe < budget:
        x0 = rng.uniform(-100.0, 100.0, D)
        cnt = [0]
        def ff(x):
            cnt[0] += 1
            return float(np.asarray(f(np.atleast_2d(x))).reshape(-1)[0])
        per = max(50, (budget - fe) // 4)
        r = minimize(ff, x0, method='L-BFGS-B', bounds=[(-100.0, 100.0)] * D,
                     options={'maxiter': per, 'ftol': 1e-12, 'maxfun': budget - fe})
        fe += cnt[0]
        n_start += 1
        if r.fun < best_f:
            best_f = float(r.fun)
        if best_f < 1e-4:
            break
    return best_f, fe, n_start


def main():
    t0 = time.time()
    rows = {}
    for seed in SEEDS:
        best_f, fe, n_start = multistart_lbfgs(f5, budget=BUDGET, seed=seed)
        rows[str(seed)] = {'best_F': best_f, 'fe': fe, 'n_restarts': n_start}
        print(f'seed {seed}: F={best_f:.4g} fe={fe} restarts={n_start}', flush=True)
    Fs = [r['best_F'] for r in rows.values()]
    hits = sum(1 for v in Fs if v < 1e-4)
    out = {
        'problem': 'CEC2017 F5 M_orth 30D, f_opt=500 calibrated, T=19.53, budget=396993',
        'budget_fe': BUDGET,
        'n_runs': len(SEEDS),
        'seeds': SEEDS,
        'n_hit': hits,
        'final_f': {
            'median': float(np.median(Fs)),
            'min': float(np.min(Fs)),
            'max': float(np.max(Fs)),
        },
        'per_run': rows,
        'elapsed_s': round(time.time() - t0, 1),
    }
    json.dump(out, io.open(os.path.join(RESULTS, 'lbfgs_25runs.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('SAVED lbfgs_25runs.json | n_hit:', hits, '/', len(SEEDS),
          '| median F:', round(float(np.median(Fs)), 3), flush=True)


if __name__ == '__main__':
    main()
