# -*- coding: utf-8 -*-
import os
"""run_02_pcu_integrated.py - Full 30-run validation after integrating PCU into the solver.

Comparison: v54 (use_pcu=False) vs v71 (use_pcu=True, all else default).
"""
import io, sys, zlib, time
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import save_json, save_csv
from aloop.data.functions_36 import REGISTRY_36
from aloop.solve.hgca import hgca
from run_00_benchmark import make_shifted, _perturb_x0


def main(n_runs=30, seed=42):
    shifts = {}
    for pname, p in REGISTRY_36.items():
        d = p.dim
        x0 = np.asarray(p.x0, dtype=float)
        mag = max(float(np.max(np.abs(x0))), 1.0)
        rng = np.random.RandomState(20240817 + zlib.crc32(pname.encode("utf-8")) % 1000)
        o = mag * (rng.rand(d) * 2.0 - 1.0) * 0.5
        shifts[pname] = o

    probs = sorted(REGISTRY_36.keys())
    rows = []
    total = len(probs) * n_runs * 2
    cnt = 0
    for pname in probs:
        p = REGISTRY_36[pname]
        sp = make_shifted(p, shifts[pname])
        x0 = np.asarray(sp.x0, dtype=float)
        for r in range(n_runs):
            x0r = _perturb_x0(x0, seed=seed + r * 7 + zlib.crc32(pname.encode("utf-8")) % 1000)
            for ver, use_pcu in (("v54", False), ("v71", True)):
                cnt += 1
                t0 = time.time()
                res = hgca(sp.func, sp.grad, sp.hess, x0r, tol=1e-10, max_iter=3000,
                           cfg={"residual_fn": sp.grad, "jacobian_fn": sp.hess,
                                "use_caci": False, "sar_enabled": False,
                                "use_pcu": use_pcu},
                           f_opt=sp.f_opt)
                fb = float(np.asarray(sp.func(np.asarray(res["x"], dtype=float))).item())
                rows.append({"problem": pname, "solver": ver, "run": r,
                             "conv": int(res["success"] and (fb - sp.f_opt) < 1e-4),
                             "time_ms": round((time.time() - t0) * 1000, 1),
                             "nit": int(res.get("nit", 0)),
                             "stage": res.get("stage", "")})
            if cnt % 100 == 0:
                print(f"  [{cnt}/{total}]", flush=True)
    save_csv("pcu_integrated", rows)
    print("\n%-26s %8s %8s | %s" % ("problem", "v54", "v71", "diff"))
    summ = {}
    for pname in probs:
        acc = {}
        for ver in ("v54", "v71"):
            v = [x["conv"] for x in rows if x["problem"] == pname and x["solver"] == ver]
            acc[ver] = float(np.mean(v))
        summ[pname] = acc
        if abs(acc["v71"] - acc["v54"]) > 1e-9:
            print("%-26s %8.2f %8.2f | %+6.2f" % (pname, acc["v54"], acc["v71"],
                                                  acc["v71"] - acc["v54"]))
    print()
    for ver in ("v54", "v71"):
        t = float(np.mean([x["conv"] for x in rows if x["solver"] == ver]))
        print("TOTAL %-4s = %.4f" % (ver, t))
    # stage distribution (v71)
    from collections import Counter
    st = Counter(x["stage"] for x in rows if x["solver"] == "v71")
    print("v71 stage distribution:", dict(st))
    save_json("pcu_integrated_summary", {"n_runs": n_runs, "per_problem": summ,
                                         "stage_dist": dict(st)})
    print("saved pcu_integrated*")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_runs", type=int, default=30)
    args = ap.parse_args()
    main(n_runs=args.n_runs)
