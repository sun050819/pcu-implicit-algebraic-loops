# -*- coding: utf-8 -*-
"""run_39_bipop_cmaes: BIPOP-CMA-ES (CEC-2013 champion SOTA variant) on CEC2017 f5 M_orth 30D.

Same budget (396,993 FE) and seed convention as Table II. Purpose: close the
"SOTA-variant baseline missing" gap for TEVC review -- show that even the
restart-based champion CMA variant (BIPOP, CEC-2013 competition winner) cannot
escape the periodic stagnation on the consensus-periodic landscape, i.e., the
sample-and-iterate paradigm failure is not specific to plain CMA-ES.

Output: results/bipop_cmaes.json
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

Mr = rotations[D][4]
u, _, vt = np.linalg.svd(Mr)
M = u @ vt
o = shifts[4][:D].copy()
oM = o @ M.T

def f5(X):
    Xa = np.atleast_2d(np.asarray(X, float))
    z = S5 * (Xa @ M.T - o)
    v = np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0, axis=1)
    return float(v[0]) if v.shape[0] == 1 else v

rows = []
for seed in SEEDS:
    rng = np.random.RandomState(seed)
    x0 = rng.uniform(-100.0, 100.0, D)
    t0 = time.time()
    # BIPOP restart scheme, up to 30 restarts; budget-capped overall
    xopt, es = cma.fmin2(f5, x0, 2.0,
                         {"maxfevals": BUDGET, "seed": seed, "verbose": -1,
                          "CMA_diagonal": False},
                         restarts=30, bipop=True, eval_initial_x=True)
    fb = float(es.result.fbest)
    fe = int(es.result.evaluations)
    hits_bipop = 0
    rows.append({"seed": seed, "f": fb, "fe_used": fe, "hit": fb < 1e-4,
                 "elapsed_s": round(time.time() - t0, 2)})
    print(f"seed {seed}: f={fb:.4e} hit={fb < 1e-4} fe={fe}", flush=True)

fs = np.array([r["f"] for r in rows])
print("SUMMARY median:", float(np.median(fs)), "hits:", sum(r["hit"] for r in rows), "/", len(rows))

out = {
    "problem": "CEC2017 F5 M_orth (30D)",
    "algorithm": "BIPOP-CMA-ES (cma 4.5.0, bipop=True, restarts<=30)",
    "budget_fe": BUDGET,
    "n_runs": len(rows),
    "seeds": SEEDS,
    "hit_threshold": "F<1e-4",
    "median_f": float(np.median(fs)),
    "hits": sum(r["hit"] for r in rows),
    "runs": rows,
    "note": "Same budget and seeds as Table II; BIPOP is the CEC-2013 champion restart variant, included to rule out the 'plain-CMA-ES artifact' reading of the main comparison.",
}
with open(os.path.join(RESULTS, "bipop_cmaes.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("bipop_cmaes.json written")
