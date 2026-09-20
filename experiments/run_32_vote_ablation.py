# -*- coding: utf-8 -*-
import os
"""run_32_vote_ablation.py - Rotation voting on/off ablation (M8).
Voting off = only use the single-point eigh at the initial point to recover R (no 3-point voting); voting on = default 3-point voting.
CEC2017 f5 (M_orth, 30D), 10 seeds. Output results/vote_ablation.json.
"""
import io, sys, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'cec2017'))
from cec2017.transforms import rotations
from cec2017.functions import f5
from aloop.solve.struct_id import estimate_and_pcu

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
D = 30
SEEDS = list(range(20260912, 20260922))
Msyn = rotations[D][4]
u, _, vt = np.linalg.svd(Msyn)
Mrot = u @ vt


def make_rastrigin():
    M = Mrot
    o = np.zeros(D)
    def f(X):
        Xa = np.atleast_2d(np.asarray(X, float))
        Z = (Xa - o) @ M.T
        return np.sum(Z * Z - 10.0 * np.cos(2.0 * np.pi * Z) + 10.0, axis=1)
    def grad(x):
        z = M @ (np.asarray(x, float) - o)
        return M.T @ (2.0 * z + 20.0 * np.pi * np.sin(2.0 * np.pi * z))
    def hess(x):
        z = M @ (np.asarray(x, float) - o)
        return M.T @ np.diag(2.0 + 40.0 * np.pi * np.pi * np.cos(2.0 * np.pi * z)) @ M
    return f, grad, hess


def run_vote(voting, n_seeds=10):
    f, grad, hess = make_rastrigin()
    rows = []
    for seed in SEEDS[:n_seeds]:
        rng = np.random.RandomState(seed)
        x0 = rng.uniform(-25.0, 25.0, D)
        try:
            xc, tag = estimate_and_pcu(f, grad, hess, x0, tol=1e-5, margin=60.0, voting=voting)
        except Exception as e:
            rows.append({'seed': seed, 'trigger': False, 'error': str(e)[:60]})
            continue
        if tag is None:
            rows.append({'seed': seed, 'trigger': False})
        else:
            F = float(np.asarray(f(np.atleast_2d(xc))).reshape(-1)[0])
            rows.append({'seed': seed, 'trigger': True, 'hit': bool(F < 1e-4), 'F': F})
    return {'n_runs': len(rows), 'n_trigger': sum(1 for r in rows if r.get('trigger')),
            'n_hit': sum(1 for r in rows if r.get('hit')), 'rows': rows}


def main():
    t0 = time.time()
    out = {}
    print('voting ON (default 3-point)', flush=True)
    out['vote_on'] = run_vote(True)
    print('voting OFF (single-point eigh)', flush=True)
    out['vote_off'] = run_vote(False)
    out['elapsed_s'] = round(time.time() - t0, 1)
    json.dump(out, io.open(os.path.join(RESULTS, 'vote_ablation.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    for k in ('vote_on', 'vote_off'):
        v = out[k]
        print('  %s: %d/%d trigger, %d hit' % (k, v['n_trigger'], v['n_runs'], v['n_hit']), flush=True)
    print('SAVED vote_ablation.json | %.1fs' % out['elapsed_s'], flush=True)


if __name__ == '__main__':
    main()
