# -*- coding: utf-8 -*-
import os
"""Focused re-verification: whether boundaries exist.
500D/400D single instance (seed 20260916, margin=5) + 250D/300D multiple seeds (GPU replicates the real pipeline)
Optimization: coarse scan uses large density only when necessary; reuse GPU batching.
"""
import sys, time, json
import numpy as np, warnings
warnings.filterwarnings('ignore')
import torch
from r3_gpu_identify import identify_gpu, make_orth_t

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', '_r3_boundary.json')
out = {}
try:
    out = json.load(open(RES, encoding='utf-8'))
except Exception:
    pass

def run(key, Dd, seed, margin):
    if key in out:
        print(key, 'cached', json.dumps(out[key], ensure_ascii=False), flush=True)
        return
    t0 = time.time()
    mdl, diag = identify_gpu(Dd, seed, margin)
    el = time.time() - t0
    if mdl is None:
        out[key] = {'reject': True, 'diag': diag, 'elapsed_s': round(el, 1)}
    else:
        out[key] = {'reject': False, 'T': round(mdl['T'], 4), 'mode': mdl['mode'],
                    'elapsed_s': round(el, 1), 'radius': round(mdl['radius'], 1), 'n': mdl['n']}
    print(key, json.dumps(out[key], ensure_ascii=False), flush=True)
    json.dump(out, open(RES, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

run('D500_m5_s20260916', 500, 20260916, 5.0)
run('D400_m5_s20260916', 400, 20260916, 5.0)
for sd in [20260917, 20260918, 20260919]:
    run('D250_m5_s%d' % sd, 250, sd, 5.0)
    run('D300_m5_s%d' % sd, 300, sd, 5.0)
print('ALL DONE -> ' + RES)
