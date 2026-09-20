# -*- coding: utf-8 -*-
import os
"""CMA-ES extra runs: CEC2017 f5 M_orth 30D, 396,993 FE, 5 seeds (to make up the 5 runs for Table II)."""
import io, sys, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'cec2017'))
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
oM = o @ M.T

def f5(X):
    Xa = np.atleast_2d(np.asarray(X, float))
    z = S5 * (Xa @ M.T - o)
    return np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0, axis=1)

rows = []
for seed in SEEDS:
    rng = np.random.RandomState(seed)
    x0 = rng.uniform(-100.0, 100.0, D)
    t0 = time.time()
    es = cma.CMAEvolutionStrategy(x0, 2.0, {"maxfevals": BUDGET, "seed": seed,
                                            "verbose": -1, "CMA_diagonal": False})
    best = None
    while not es.stop():
        X = es.ask()
        es.tell(X, f5(X))
        if best is None or es.result.fbest < best:
            best = es.result.fbest
    fb = float(es.result.fbest)
    rows.append({"seed": seed, "f": fb, "hit": fb < 1e-4, "elapsed_s": round(time.time() - t0, 2)})
    print(f"seed {seed}: f={fb:.4e} hit={fb < 1e-4}", flush=True)

with open(os.path.join(RESULTS, "sota_cmaes_5runs.json"), "w", encoding="utf-8") as f:
    json.dump({"problem": "CEC2017 f5 M_orth 30D", "budget_fe": BUDGET, "n_runs": len(rows), "rows": rows}, f, indent=2)
fs = np.array([r["f"] for r in rows])
print("SUMMARY median:", float(np.median(fs)), "hits:", sum(r["hit"] for r in rows), "/", len(rows))
