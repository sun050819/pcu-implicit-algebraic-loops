# -*- coding: utf-8 -*-
import os
"""run_07_cec: CEC2017 F5 (Shifted+Rotated Rastrigin, 30D) generalization validation.

Three-tier protocol (50 runs x 3):
  base      : solver without PCU (control group)
  auto_raw  : struct_id + PCU, official data M (non-orthogonal) -- expected to correctly reject (boundary behavior, success rate approx base)
  auto_orth : struct_id + PCU, official M orthogonalized via SVD -- expected to hit (structural generalization)

Convergence criterion: (success and (fb-f_opt)<1e-4), f_opt=0 (after normalization).
x0 protocol: uniform random in domain [-100,100]^30, fixed seed, reproducible per run.
"""
import io, sys, os, json, time, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'cec2017'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cec2017.transforms import rotations, shifts
from aloop.solve.hgca import hgca

D = 30
N_RUNS = 50
SEED0 = 20260912
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

o = shifts[4][:D].copy()
M_raw = rotations[D][4].copy()
u, _, vt = np.linalg.svd(M_raw)
M_orth = u @ vt
S = 0.0512

def make_funcs(Mm):
    def f5(x):
        z = S * (Mm @ (np.asarray(x, float) - o))
        return float(np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0))
    def g5(x):
        x = np.asarray(x, float)
        z = S * (Mm @ (x - o))
        gz = S * (2.0 * z + 20.0 * np.pi * np.sin(2.0 * np.pi * z))
        return Mm.T @ gz
    def h5(x):
        x = np.asarray(x, float)
        z = S * (Mm @ (x - o))
        Hzz = S * S * (2.0 + 40.0 * np.pi * np.pi * np.cos(2.0 * np.pi * z))
        return (Mm.T * Hzz) @ Mm
    return f5, g5, h5

f_raw, g_raw, h_raw = make_funcs(M_raw)
f_ort, g_ort, h_ort = make_funcs(M_orth)

def run_once(fn, gn, hn, x0, cfg):
    r = hgca(fn, gn, hn, x0, tol=1e-10, max_iter=3000,
             cfg=cfg, f_opt=0.0)
    xb = r.get('x')
    fb = float(fn(xb)) if xb is not None else float('nan')
    ok = bool(r.get('success')) and (fb - 0.0) < 1e-4
    return ok, fb, r.get('stage', ''), r.get('nit', -1)

rows = []
for run_idx in range(N_RUNS):
    rng = np.random.RandomState(SEED0 + run_idx)
    x0 = rng.uniform(-100.0, 100.0, D)

    # base
    ok, fb, stage, nit = run_once(f_raw, g_raw, h_raw, x0,
        {"residual_fn": g_raw, "jacobian_fn": h_raw, "use_caci": False,
         "sar_enabled": False, "use_pcu": False, "use_struct_id": False})
    rows.append(dict(run=run_idx, variant='base', ok=ok, fb=fb, stage=stage, n_iter=nit))

    # auto_raw
    ok, fb, stage, nit = run_once(f_raw, g_raw, h_raw, x0,
        {"residual_fn": g_raw, "jacobian_fn": h_raw, "use_caci": False,
         "sar_enabled": False, "use_pcu": True, "use_struct_id": True})
    rows.append(dict(run=run_idx, variant='auto_raw', ok=ok, fb=fb, stage=stage, n_iter=nit))

    # auto_orth
    ok, fb, stage, nit = run_once(f_ort, g_ort, h_ort, x0,
        {"residual_fn": g_ort, "jacobian_fn": h_ort, "use_caci": False,
         "sar_enabled": False, "use_pcu": True, "use_struct_id": True,
         "struct_id_margin": 100.0})
    rows.append(dict(run=run_idx, variant='auto_orth', ok=ok, fb=fb, stage=stage, n_iter=nit))

    if (run_idx + 1) % 10 == 0:
        print('run %d/%d done' % (run_idx + 1, N_RUNS), flush=True)

# Summary
os.makedirs(RESULTS, exist_ok=True)
csv_path = os.path.join(RESULTS, 'cec_generalize.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['run', 'variant', 'ok', 'fb', 'stage', 'n_iter'])
    w.writeheader()
    for r in rows:
        w.writerow(r)

summary = {}
for v in ['base', 'auto_raw', 'auto_orth']:
    sub = [r for r in rows if r['variant'] == v]
    n_ok = sum(r['ok'] for r in sub)
    fb = np.array([r['fb'] for r in sub])
    summary[v] = {
        'n_runs': len(sub), 'n_ok': int(n_ok),
        'success_rate': float(n_ok) / len(sub),
        'median_fb': float(np.median(fb)),
        'pcu_stage': int(sum(1 for r in sub if 'pcu' in r['stage'].lower() or 'struct' in r['stage'].lower())),
    }
with open(os.path.join(RESULTS, 'cec_generalize_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print('done ->', csv_path)
