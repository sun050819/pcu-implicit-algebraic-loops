# -*- coding: utf-8 -*-
import os
"""run_bipop: BIPOP-CMA-ES on CEC2017 f5 (M_orth, 30D), same budget 396,993 FE,
5 runs (exploratory median), consistent with sota_compare.json."""
import io, sys, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'cec2017'))
from cec2017.transforms import rotations, shifts
import cma

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
        v = np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0, axis=1)
        return float(v[0]) if v.size == 1 else v
    return f


def run_one(seed):
    f = make_f5(30)
    rng = np.random.RandomState(seed)
    x0 = rng.uniform(-100.0, 100.0, 30)
    opts = {'maxfevals': BUDGET, 'bounds': [-100.0, 100.0], 'seed': seed,
            'verbose': -1, 'verb_disp': 0, 'tolx': 0.0, 'tolfun': 0.0}
    t0 = time.time()
    try:
        res = cma.fmin(f, x0, 20.0, options=opts, restarts=9, bipop=True)
        xopt, fopt, evals = res[0], res[1], res[2]
    except Exception as e:
        return {'seed': seed, 'error': str(e), 'elapsed_s': round(time.time()-t0, 1)}
    hit = float(fopt) < 1e-4
    return {'seed': seed, 'best_f': float(fopt), 'fe': int(evals), 'hit': hit,
            'elapsed_s': round(time.time()-t0, 1)}


t0 = time.time()
rows = [run_one(20260901 + k) for k in range(N_RUNS)]
med = float(np.median([r['best_f'] for r in rows if 'best_f' in r]))
q1 = float(np.percentile([r['best_f'] for r in rows if 'best_f' in r], 25))
q3 = float(np.percentile([r['best_f'] for r in rows if 'best_f' in r], 75))
hits = sum(1 for r in rows if r.get('hit'))
out = {'problem': 'CEC2017 f5 M_orth 30D', 'budget_fe': BUDGET, 'n_runs': N_RUNS,
       'median_best_f': med, 'q1': q1, 'q3': q3, 'hit': hits,
       'elapsed_s': round(time.time()-t0, 1), 'rows': rows}
with open(os.path.join(RESULTS, 'bipop_cma.json'), 'w', encoding='utf-8') as fp:
    json.dump(out, fp, ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=2))
print('done -> bipop_cma.json')
