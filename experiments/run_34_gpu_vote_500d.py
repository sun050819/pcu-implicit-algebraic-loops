# -*- coding: utf-8 -*-
import os
"""run_34_gpu_vote_500d.py - 500D rotational voting on/off ablation (GPU, FP64).
Observation 1 claims that three-point voting is key to 500D success; here we directly verify: voting on/off, 5 seeds each.
Output results/vote_ablation_500d.json
"""
import io, sys, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'repro'))
from r3_gpu_full import full_pcu

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
SEEDS = [20260912, 20260913, 20260914, 20260915, 20260916]
D = 500


def run(mode, voting):
    rows = []
    for sd in SEEDS:
        t0 = time.time()
        r = full_pcu(D, sd, voting=voting)
        r['seed'] = sd
        r['elapsed_s'] = round(time.time() - t0, 1)
        rows.append(r)
        print('%s seed %d: trigger=%s hit=%s F=%s (%.1fs)' % (
            mode, sd, r.get('trigger'), r.get('hit'), r.get('F'), r['elapsed_s']), flush=True)
    return {'n_runs': len(rows), 'n_trigger': sum(1 for r in rows if r.get('trigger')),
            'n_hit': sum(1 for r in rows if r.get('hit')), 'rows': rows}


def main():
    t0 = time.time()
    out = {'D': D, 'seeds': SEEDS}
    print('500D voting ON', flush=True)
    out['vote_on'] = run('ON ', True)
    print('500D voting OFF', flush=True)
    out['vote_off'] = run('OFF', False)
    out['elapsed_s'] = round(time.time() - t0, 1)
    json.dump(out, io.open(os.path.join(RESULTS, 'vote_ablation_500d.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    for k in ('vote_on', 'vote_off'):
        v = out[k]
        print('  %s: %d/%d trigger, %d/%d hit' % (k, v['n_trigger'], v['n_runs'], v['n_hit'], v['n_runs']), flush=True)
    print('SAVED vote_ablation_500d.json | %.1fs' % out['elapsed_s'], flush=True)


if __name__ == '__main__':
    main()
