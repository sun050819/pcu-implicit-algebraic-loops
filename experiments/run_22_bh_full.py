# -*- coding: utf-8 -*-
import os
"""run_22_bh_full.py - Run Basinhopping to the full 396,993 FE (5 runs, same-budget statistical protocol).

Reuse the f5 definition and basinhop implementation from run_20_gap_fill; the budget guard is precise to 396,993 FE,
and each of the 5 seeds records final F / FE / hit (F<1e-4) / wall time.
Output: results/bh_full.json + results/bh_full.log
"""
import io, sys, os, json, time
import numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_17_sota_compare import make_f5, BUDGET, stats

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
LOG = os.path.join(RESULTS, 'bh_full.log')
N_RUNS = 5
HIT_TOL = 1e-4


def log(msg):
    with io.open(LOG, 'a', encoding='utf-8') as fp:
        fp.write('%s %s\n' % (time.strftime('%H:%M:%S'), msg))


def basinhop_full(f, D, budget=BUDGET, seed=0, niter=3000):
    from scipy.optimize import basinhopping
    rng = np.random.RandomState(seed)
    x0 = rng.uniform(-100.0, 100.0, D)
    cnt = [0]

    def ff(x):
        cnt[0] += 1
        return float(f(x))

    def budget_check(f_new=None, x_new=None, **kw):
        return cnt[0] < budget

    t0 = time.time()
    r = basinhopping(ff, x0, niter=niter, T=1.0, stepsize=20.0,
                     minimizer_kwargs={'method': 'L-BFGS-B',
                                       'bounds': [(-100.0, 100.0)] * D,
                                       'options': {'maxiter': 60, 'ftol': 1e-12}},
                     take_step=None, accept_test=budget_check,
                     seed=seed, niter_success=None)
    dt = time.time() - t0
    return float(r.fun), cnt[0], dt


def main():
    t0 = time.time()
    f30, _, _ = make_f5(30)

    def f1(x):
        return float(np.asarray(f30(np.atleast_2d(x))).reshape(-1)[0])

    log('start BH full-budget x%d, budget=%d' % (N_RUNS, BUDGET))
    runs = []
    for k in range(N_RUNS):
        bf, fe, dt = basinhop_full(f1, 30, seed=k)
        hit = bf < HIT_TOL
        runs.append({'seed': k, 'final_f': round(float(bf), 4),
                     'fe': int(fe), 'hit': bool(hit), 'wall_s': round(dt, 1)})
        log('seed=%d F=%.4f fe=%d hit=%s %.1fs' % (k, bf, fe, hit, dt))
    finals = [r['final_f'] for r in runs]
    res = {
        'problem': 'CEC2017 F5 M_orth 30D, f_opt=500 calibration, T=19.53',
        'budget_fe': BUDGET, 'n_runs': N_RUNS, 'hit_tol': HIT_TOL,
        'median_f': round(float(np.median(finals)), 4),
        'q1': round(float(np.percentile(finals, 25)), 4),
        'q3': round(float(np.percentile(finals, 75)), 4),
        'n_hit': int(sum(1 for r in runs if r['hit'])),
        'fe_consumed': [r['fe'] for r in runs],
        'runs': runs,
        'elapsed_s': round(time.time() - t0, 1),
    }
    with io.open(os.path.join(RESULTS, 'bh_full.json'), 'w', encoding='utf-8') as fp:
        json.dump(res, fp, ensure_ascii=False, indent=2)
    log('DONE median=%.4f hit=%d/%d elapsed=%.1fs'
        % (res['median_f'], res['n_hit'], N_RUNS, res['elapsed_s']))


if __name__ == '__main__':
    main()
