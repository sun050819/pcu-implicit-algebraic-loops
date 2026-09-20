# -*- coding: utf-8 -*-
import os
"""_r3_gpu_gap.py - GPU catch-up run: full pipeline for 300/400D (identify+candidate+verify) + 500D strengthened to 10 seeds.
Reuses _r3_gpu_full.full_pcu (full pipeline implementation), with the same seeds as _r3_boundary to ensure reproducibility.
"""
import io, os, sys, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r3_gpu_full import full_pcu

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
JOBS = [
    (300, [20260917, 20260918, 20260919]),
    (400, [20260916, 20260917, 20260918]),
    (500, [20260916, 20260917, 20260918, 20260919, 20260920,
           20260921, 20260922, 20260923, 20260924, 20260925]),
]

if __name__ == '__main__':
    t0 = time.time()
    out = {}
    log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', '_r3_gpu_gap.log'), 'w', encoding='utf-8')
    for Dd, seeds in JOBS:
        for sd in seeds:
            key = 'D%d_s%d' % (Dd, sd)
            try:
                r = full_pcu(Dd, sd)
                out[key] = r
            except Exception as e:
                out[key] = {'error': str(e)}
            log.write('%s: %s (%.0fs)\n' % (key, json.dumps(out[key], ensure_ascii=False), time.time() - t0))
            log.flush()
    out['_meta'] = {'note': 'GPU RTX3060 FP64 full pipeline (identify+zero-crossing+verify), '
                            'seeds aligned with _r3_boundary where available',
                    'elapsed_s': round(time.time() - t0, 1)}
    with open(os.path.join(RESULTS, '_r3_gpu_gap.json'), 'w', encoding='utf-8') as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    log.close()
    print('saved _r3_gpu_gap.json', round(time.time() - t0, 1))
