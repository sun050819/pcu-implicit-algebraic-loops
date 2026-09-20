# -*- coding: utf-8 -*-
import os
"""CMA-ES 5-runs with median trajectory: CEC2017 f5 M_orth 30D, 396,993 FE, seeds 20260912-16.
Output: results/dataprofile.json updated (cma_es: 5-run median best + median trajectory)
"""
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
SNAP = 4000  # trajectory snapshot interval FE

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
traj_all = []
for seed in SEEDS:
    rng = np.random.RandomState(seed)
    x0 = rng.uniform(-100.0, 100.0, D)
    t0 = time.time()
    es = cma.CMAEvolutionStrategy(x0, 2.0, {"maxfevals": BUDGET, "seed": seed,
                                            "verbose": -1, "CMA_diagonal": False})
    traj = []
    while not es.stop():
        X = es.ask()
        es.tell(X, f5(X))
        fe = es.countiter * es.popsize
        traj.append([fe, float(es.result.fbest)])
    fb = float(es.result.fbest)
    # Align to snapshot grid: take best at each SNAP FE
    grid = np.arange(0, BUDGET + 1, SNAP)
    vals = []
    tj = np.array(traj)
    for g in grid:
        m = tj[tj[:, 0] <= g]
        vals.append(float(m[-1, 1]) if len(m) else float("nan"))
    traj_all.append(vals)
    rows.append({"seed": seed, "f": fb, "hit": fb < 1e-4, "elapsed_s": round(time.time() - t0, 2)})
    print(f"seed {seed}: f={fb:.4e} hit={fb < 1e-4}", flush=True)

grid = np.arange(0, BUDGET + 1, SNAP)
traj_med = [[int(g), float(np.nanmedian([t[i] for t in traj_all]))] for i, g in enumerate(grid)]

fs = np.array([r["f"] for r in rows])
print("SUMMARY median:", float(np.median(fs)), "hits:", sum(r["hit"] for r in rows), "/", len(rows))

# Update dataprofile.json
dp_path = os.path.join(RESULTS, "dataprofile.json")
dp = json.load(open(dp_path, encoding="utf-8"))
dp["cma_es"] = {
    "max_fe": BUDGET,
    "best_f_after_budget": float(np.median(fs)),
    "n_runs": len(rows),
    "seeds": SEEDS,
    "median_of_5": True,
    "trajectory": traj_med,
    "runs": rows,
}
with open(dp_path, "w", encoding="utf-8") as f:
    json.dump(dp, f, indent=2)
print("dataprofile.json updated")
