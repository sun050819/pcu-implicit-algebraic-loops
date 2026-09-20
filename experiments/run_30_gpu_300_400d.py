# -*- coding: utf-8 -*-
import os
"""run_30_gpu_300_400d.py
E5 Fix: 300/400D supplementary runs to 10 seeds (original 3 seeds were questioned by reviewers).
Add 7 seeds (20260926-20260932), merge with existing 3 seeds (20260919-20260921 if present).
Reuse repro/r3_gpu_full.full_pcu, GPU RTX 3060.
Output: results/gpu_300400d_10seeds.json
"""
import io, sys, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'repro'))
from r3_gpu_full import full_pcu

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
NEW_SEEDS = list(range(20260926, 20260933))  # 7 new seeds

if __name__ == '__main__':
    out = {}
    for D in (300, 400):
        rows = []
        for sd in NEW_SEEDS:
            t0 = time.time()
            r = full_pcu(D, sd)
            r['seed'] = sd
            r['elapsed_s'] = round(time.time() - t0, 1)
            rows.append(r)
            print(f'{D}D seed {sd}: {json.dumps(r, ensure_ascii=False)} ({time.time()-t0:.1f}s)', flush=True)
        out[str(D)] = rows
    json.dump(out, io.open(os.path.join(RESULTS, 'gpu_300400d_10seeds.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    for D in (300, 400):
        hits = sum(1 for r in out[str(D)] if r.get('hit'))
        print(f'{D}D new 7 seeds: {hits}/7 hit', flush=True)
