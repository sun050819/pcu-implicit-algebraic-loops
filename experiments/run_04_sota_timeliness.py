# -*- coding: utf-8 -*-
import os
"""run_04_sota_timeliness.py - SOTA timeliness comparison (same A^2EP shift fair benchmark)

Compare 4 methods (36 problems x 30 runs x same budget 30000):
 - PCU solver (with structure-aware fast path)
 - baseline solver (without PCU, pure baseline)
  BIPOP-aCMA-ES (official pycma)
  NBIPOP-aCMA-ES (Loshchilov 2012 dual-population restart enhanced version)

The setup is exactly the same as run_00_benchmark (make_shifted / _perturb_x0 / A^2EP shift),
ensuring the "convergence rate comparison" is fair on the same benchmark.
"""
import io, sys, zlib, time
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import save_json, save_csv
from aloop.data.functions_36 import REGISTRY_36
from aloop.solve.hgca import hgca
from run_00_benchmark import make_shifted, _perturb_x0, _run_bipop_acma, _run_nbipop_acma

MAXFEVALS = 30000


def _hgca_run(sp, x0r, use_pcu):
    t0 = time.time()
    res = hgca(sp.func, sp.grad, sp.hess, x0r, tol=1e-10, max_iter=3000,
               cfg={"residual_fn": sp.grad, "jacobian_fn": sp.hess,
                    "use_caci": False, "sar_enabled": False,
                    "use_pcu": use_pcu},
               f_opt=sp.f_opt)
    dt = (time.time() - t0) * 1000
    return res, dt


def main(n_runs=30, seed=42, maxfevals=MAXFEVALS):
    shifts = {}
    for pname, p in REGISTRY_36.items():
        d = p.dim
        x0 = np.asarray(p.x0, dtype=float)
        mag = max(float(np.max(np.abs(x0))), 1.0)
        rng = np.random.RandomState(20240817 + zlib.crc32(pname.encode("utf-8")) % 1000)
        shifts[pname] = mag * (rng.rand(d) * 2.0 - 1.0) * 0.5

    probs = sorted(REGISTRY_36.keys())
    methods = ["hgca_v67", "hgca_v54", "bipop", "nbipop"]
    rows = []
    total = len(probs) * n_runs * len(methods)
    cnt = 0
    for pname in probs:
        p = REGISTRY_36[pname]
        sp = make_shifted(p, shifts[pname])
        x0 = np.asarray(sp.x0, dtype=float)
        for r in range(n_runs):
            x0r = _perturb_x0(x0, seed=seed + r * 7 + zlib.crc32(pname.encode("utf-8")) % 1000)
            # PCU solver / baseline solver
            for m, use_pcu in (("hgca_v67", True), ("hgca_v54", False)):
                cnt += 1
                res, dt = _hgca_run(sp, x0r, use_pcu)
                fb = float(np.asarray(sp.func(np.asarray(res["x"], dtype=float))).item())
                rows.append({"problem": pname, "solver": m, "run": r,
                             "conv": int(res["success"] and (fb - sp.f_opt) < 1e-4),
                             "time_ms": round(dt, 1), "nit": int(res.get("nit", 0)),
                             "stage": res.get("stage", "")})
            # BIPOP / NBIPOP (same budget)
            for m, runner in (("bipop", _run_bipop_acma), ("nbipop", _run_nbipop_acma)):
                cnt += 1
                r2 = runner(sp.func, x0r, maxfevals)
                fb = float(np.asarray(sp.func(np.asarray(r2["x"], dtype=float))).item())
                rows.append({"problem": pname, "solver": m, "run": r,
                             "conv": int(fb - sp.f_opt < 1e-4),
                             "time_ms": round(r2["dt"], 1),
                             "nit": int(r2["nit"]), "stage": ""})
            if cnt % 100 == 0:
                print(f"  [{cnt}/{total}]", flush=True)
    save_csv("sota_compare", rows)
    print("\n%-26s %8s %8s %8s %8s" % ("problem", "v67", "v54", "bipop", "nbipop"))
    summ = {}
    for pname in probs:
        acc = {}
        for m in methods:
            v = [x["conv"] for x in rows if x["problem"] == pname and x["solver"] == m]
            acc[m] = float(np.mean(v))
        summ[pname] = acc
        print("%-26s %8.2f %8.2f %8.2f %8.2f" % (pname, acc["hgca_v67"], acc["hgca_v54"],
                                                 acc["bipop"], acc["nbipop"]))
    print()
    tot = {}
    for m in methods:
        t = float(np.mean([x["conv"] for x in rows if x["solver"] == m]))
        tot[m] = t
        print("TOTAL %-10s = %.4f" % (m, t))
    save_json("sota_compare_summary", {"n_runs": n_runs, "maxfevals": maxfevals,
                                       "per_problem": summ, "total": tot})
    print("saved sota_compare*")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_runs", type=int, default=30)
    ap.add_argument("--maxfevals", type=int, default=MAXFEVALS)
    args = ap.parse_args()
    main(n_runs=args.n_runs, maxfevals=args.maxfevals)
