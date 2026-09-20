# -*- coding: utf-8 -*-
import os
"""run_26_pcu_vs_baselines.py
#5 Reviewer revision: Statistical strengthening of the core comparison in Table III (reviewer revision comment #5).
Fix points:
  1) run_24_cmaes_5runs's CMA-ES stops early due to default tolfun (measured only 5.7k FE / 0.35s),
     did not run the full 396,993 FE budget -> disable tolfun/tolx so CMA-ES actually runs the full same budget;
  2) CMA-ES and SHADE each 25 runs (seeds 20260912-20260936), same protocol as run_17_sota_compare:
     CEC2017 f5 M_orth 30D, f_opt calibrated, F<1e-4 hit;
  3) Output 25-run hit rate + Fisher exact test (vs PCU 10/10 hit).
Output: results/sota_baselines_25runs.json
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
SEEDS = list(range(20260912, 20260937))  # 25 seeds

Mr = rotations[D][4]
u, _, vt = np.linalg.svd(Mr)
M = u @ vt
o = shifts[4][:D].copy()
oM = o @ M.T


def f5(X):
    Xa = np.atleast_2d(np.asarray(X, float))
    z = S5 * (Xa @ M.T - oM)
    return np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0, axis=1)


def cmaes_full(f, Dd=D, budget=BUDGET, seed=0):
    """CMA-ES runs the full budget: manual loop controlling number of evaluations (cma 4.4.4's tolflatfitness
    cannot be disabled via a threshold, and on rotated Rastrigin it stops early at 6k-9k FE)."""
    import cma
    rng = np.random.RandomState(seed)
    x0 = rng.uniform(-100.0, 100.0, Dd)
    opts = {'maxfevals': budget, 'seed': seed, 'verbose': -1, 'CMA_diagonal': False}
    es = cma.CMAEvolutionStrategy(x0, 20.0, opts)
    while es.countevals < budget:
        X = es.ask()
        es.tell(X, f(X))
    return float(es.result.fbest)


def shade(f, Dd=D, budget=BUDGET, pop=100, seed=0):
    """SHADE: reuse the already-debugged implementation of run_17_sota_compare.py (runs the full budget)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_17_sota_compare import shade as _shade
    best_f, _ = _shade(f, Dd, budget=budget, pop=pop, seed=seed)
    return best_f


def fisher_exact(a, b, c, dd):
    """Fisher exact test for the 2x2 table [[a,b],[c,d]] (one-sided, in the direction where PCU has a higher hit rate)."""
    from scipy.stats import fisher_exact as _fe
    _, pv = _fe([[a, b], [c, dd]], alternative='greater')
    return float(pv)


if __name__ == '__main__':
    out = {'problem': 'CEC2017 f5 M_orth 30D', 'budget_fe': BUDGET,
           'n_runs': len(SEEDS), 'algorithms': {}}
    for name, fn in [('CMA-ES', cmaes_full), ('SHADE', shade)]:
        rows = []
        for sd in SEEDS:
            t0 = time.time()
            fb = fn(f5, seed=sd)
            rows.append({'seed': sd, 'f': fb, 'hit': fb < 1e-4,
                         'elapsed_s': round(time.time() - t0, 2)})
            print(f'{name} seed {sd}: f={fb:.4e} hit={fb < 1e-4} ({time.time()-t0:.1f}s)', flush=True)
        fs = np.array([r['f'] for r in rows])
        hits = sum(r['hit'] for r in rows)
        out['algorithms'][name] = {
            'hits': f'{hits}/{len(rows)}', 'hit_rate': hits / len(rows),
            'median_f': float(np.median(fs)), 'min_f': float(fs.min()),
            'max_f': float(fs.max()), 'rows': rows}
        print(f'{name}: hits {hits}/{len(rows)}, median F={np.median(fs):.4e}', flush=True)
    # Fisher exact test: PCU 10/10 vs baseline hits/25 (one-sided: PCU has a higher hit rate)
    for name in ['CMA-ES', 'SHADE']:
        h = out['algorithms'][name]['hits'].split('/')
        a, b = 10, 0          # PCU 10/10
        c, dd = int(h[0]), int(h[1]) - int(h[0])  # baseline hits / misses
        pv = fisher_exact(a, b, c, dd)
        out['algorithms'][name]['fisher_p_vs_pcu_1sided'] = round(pv, 6)
        print(f'Fisher (PCU 10/10 vs {name} {h[0]}/{h[1]}): p={pv:.6f}', flush=True)
    with open(os.path.join(RESULTS, 'sota_baselines_25runs.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('saved sota_baselines_25runs.json')
