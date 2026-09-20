# -*- coding: utf-8 -*-
"""run_40_cmaes_pcu_init.py
PCU candidates as EA initialization (fair protocol):
same start pool (CEC2017 f5 M_orth 30D, x0 = shift + U(-5,5), the PCU
trigger window of run_38) for both
  (a) CMA-ES from the raw neighbor point x0, and
  (b) CMA-ES from PCU's candidate x_cand (estimate_and_pcu full pipeline).
Full budget 396,993 FE each; also records a 10% budget run to show
PCU-initialized EA reaches the tolerance with a fraction of the budget.
Directly answers "how does PCU relate to evolutionary computation?":
identification supplies a precise basin point; whether the raw neighbor
point already suffices is measured, not assumed.
Output: results/cmaes_pcu_init.json
"""
import io, sys, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'cec2017'))
from aloop.solve.struct_id import estimate_and_pcu
from cec2017.transforms import rotations, shifts
import cma

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
BUDGET = 396993
S5 = 5.12 / 100.0
D = 30
SEEDS = [20260912, 20260913, 20260914, 20260915, 20260916]

Mr = rotations[D][4]
u, _, vt = np.linalg.svd(Mr)
M = u @ vt
o = shifts[4][:D].copy()


def f5(X):
    Xa = np.atleast_2d(np.asarray(X, float))
    z = S5 * (Xa @ M.T - o @ M.T)
    return np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0, axis=1)


def grad5(x):
    x = np.atleast_1d(np.asarray(x, float))
    z = S5 * (M @ (x - o))
    gz = S5 * (2.0 * z + 20.0 * np.pi * np.sin(2.0 * np.pi * z))
    return M.T @ gz


def hess5(x):
    x = np.atleast_1d(np.asarray(x, float))
    z = S5 * (M @ (x - o))
    Hzz = S5 * S5 * (2.0 + 40.0 * np.pi * np.pi * np.cos(2.0 * np.pi * z))
    return (M.T * Hzz) @ M


def cmaes_from(x0, budget, seed):
    opts = {'maxfevals': budget, 'seed': seed, 'verbose': -1, 'CMA_diagonal': False}
    es = cma.CMAEvolutionStrategy(np.asarray(x0, float), 2.0, opts)
    first_hit = None
    best = float('inf')
    while es.countevals < budget:
        X = es.ask()
        F = f5(X)
        es.tell(X, F)
        fb = float(es.result.fbest)
        if fb < best:
            best = fb
        if first_hit is None and best < 1e-4:
            first_hit = es.countevals
    return best, first_hit


if __name__ == '__main__':
    rows = []
    for sd in SEEDS:
        rng = np.random.RandomState(sd)
        x0 = np.asarray(o, float) + rng.uniform(-5.0, 5.0, D)
        # (a) CMA-ES from the raw neighbor point
        fb_raw, hit_fe_raw = cmaes_from(x0, BUDGET, sd)
        # PCU full pipeline from the same point
        t0 = time.time()
        x_cand, status = estimate_and_pcu(f5, grad5, hess5, x0, tol=1e-4)
        pcu_s = time.time() - t0
        if status is None:
            print(f'seed {sd}: PCU no-trigger (CMA raw f={fb_raw:.2e})', flush=True)
            rows.append({'seed': sd, 'pcu_trigger': False, 'cma_raw_f': fb_raw,
                         'cma_raw_hit': fb_raw < 1e-4})
            continue
        f_at_cand = float(np.asarray(f5(x_cand)).item())
        # (b) CMA-ES from PCU candidate
        fb_pcu, hit_fe_pcu = cmaes_from(x_cand, BUDGET, sd)
        fb_10, hit_fe_10 = cmaes_from(x_cand, BUDGET // 10, sd)
        rows.append({
            'seed': sd, 'pcu_trigger': True, 'pcu_candidate_f': f_at_cand,
            'pcu_elapsed_s': round(pcu_s, 3),
            'cma_raw_f': fb_raw, 'cma_raw_hit': fb_raw < 1e-4,
            'cma_raw_first_hit_fe': hit_fe_raw,
            'cma_pcu_f': fb_pcu, 'cma_pcu_hit': fb_pcu < 1e-4,
            'cma_pcu_first_hit_fe': hit_fe_pcu,
            'cma_10pct_f': fb_10, 'cma_10pct_hit': fb_10 < 1e-4,
            'cma_10pct_first_hit_fe': hit_fe_10,
        })
        print(f'seed {sd}: PCU F(cand)={f_at_cand:.2e} | CMA(raw) f={fb_raw:.2e} hit={fb_raw<1e-4} '
              f'| CMA(PCU) f={fb_pcu:.2e} hit={fb_pcu<1e-4} first_hit={hit_fe_pcu} '
              f'| CMA-10% f={fb_10:.2e} hit={fb_10<1e-4}', flush=True)

    with open(os.path.join(RESULTS, 'cmaes_pcu_init.json'), 'w', encoding='utf-8') as fp:
        json.dump({'problem': 'CEC2017 f5 M_orth 30D',
                   'start': 'shift + U(-5,5) (PCU trigger window)',
                   'budget_fe': BUDGET, 'budget_10pct_fe': BUDGET // 10,
                   'init': 'PCU estimate_and_pcu full pipeline + CMA-ES sigma=2.0',
                   'n_runs': len(SEEDS), 'rows': rows}, fp, indent=2)
    trig = [r for r in rows if r.get('pcu_trigger')]
    raw_hits = sum(r['cma_raw_hit'] for r in rows)
    if trig:
        pcu_hits = sum(r['cma_pcu_hit'] for r in trig)
        ten_hits = sum(r['cma_10pct_hit'] for r in trig)
        print(f'SUMMARY: triggered {len(trig)}/{len(rows)} | CMA(raw) hits {raw_hits}/{len(rows)} '
              f'| CMA(PCU) hits {pcu_hits}/{len(trig)} | CMA(PCU,10%) hits {ten_hits}/{len(trig)}', flush=True)
