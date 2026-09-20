# -*- coding: utf-8 -*-
import os
"""run_27_gpu_250d.py
#6 Review revision: add 250D full pipeline (reviewer comment: 250D only has identification 3/3,
no full pipeline hit data). Reuse repro/r3_gpu_full.py's GPU full_pcu (RTX 3060, FP64),
run 250D x 10 seeds (20260916-20260925), same protocol as 300/400/500D.
Output: results/gpu_250d_full.json
"""
import io, sys, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'repro'))
from r3_gpu_full import full_pcu

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
SEEDS = list(range(20260916, 20260926))  # 10 seeds


if __name__ == '__main__':
    out = {'problem': 'orthogonalized A^2EP 250D', 'T': 19.53125, 'n_runs': len(SEEDS)}
    rows = []
    for sd in SEEDS:
        t0 = time.time()
        r = full_pcu(250, sd)
        r['seed'] = sd
        r['elapsed_s'] = round(time.time() - t0, 1)
        rows.append(r)
        print(f'250D seed {sd}: {json.dumps(r, ensure_ascii=False)} ({time.time()-t0:.1f}s)', flush=True)
    out['rows'] = rows
    trig = sum(1 for r in rows if r.get('trigger'))
    hits = sum(1 for r in rows if r.get('hit'))
    out['trigger'] = f'{trig}/{len(rows)}'
    out['hit'] = f'{hits}/{len(rows)}'
    if hits:
        out['F_min'] = min(r['F'] for r in rows if r.get('hit'))
        out['F_max'] = max(r['F'] for r in rows if r.get('hit'))
    print(f'250D full: trigger {trig}/{len(rows)}, hit {hits}/{len(rows)}', flush=True)
    with open(os.path.join(RESULTS, 'gpu_250d_full.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('saved gpu_250d_full.json')
