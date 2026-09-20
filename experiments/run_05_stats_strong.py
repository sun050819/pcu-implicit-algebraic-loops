# -*- coding: utf-8 -*-
"""run_05_stats_strong.py - 50 runs statistical strengthening + de-oracled ablation matrix (v6.8).

Four configurations (same benchmark 36 questions, same A^2EP offset, same perturbations):
 - baseline: use_pcu=False, use_struct_id=False (pure main pipeline); 
  v71  : use_pcu=True,  use_struct_id=False (PCU known-parameter fast path)
  v74a : use_pcu=False, use_struct_id=True  (structural identification, de-oracled standalone)
  v74  : use_pcu=True,  use_struct_id=True  (all enabled, v6.8)
Output: results/stats_strong.csv + summary.json (with 95% CI, paired differences).
"""
import io, os, sys, zlib, time
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import save_json, save_csv, RESULTS
from aloop.data.functions_36 import REGISTRY_36
from aloop.solve.hgca import hgca
from run_00_benchmark import make_shifted, _perturb_x0

CONFIGS = [
    ("v54",  {"use_caci": False, "sar_enabled": False, "use_pcu": False, "use_struct_id": False}),
    ("v71",  {"use_caci": False, "sar_enabled": False, "use_pcu": True,  "use_struct_id": False}),
    ("v74a", {"use_caci": False, "sar_enabled": False, "use_pcu": False, "use_struct_id": True}),
    ("v74",  {"use_caci": False, "sar_enabled": False, "use_pcu": True,  "use_struct_id": True}),
]


def ci95(vals):
    """Simplified Wilson interval: mean +/- 1.96*sqrt(p(1-p)/n)."""
    v = np.asarray(vals, dtype=float)
    n = len(v)
    if n == 0:
        return 0.0, 0.0, 0.0
    m = float(v.mean())
    se = float(np.sqrt(max(m * (1.0 - m), 0.0) / n))
    return m, m - 1.96 * se, m + 1.96 * se


def main(n_runs=50, seed=42, problems=None, resume=False):
    import os
    shifts = {}
    for pname, p in REGISTRY_36.items():
        d = p.dim
        x0 = np.asarray(p.x0, dtype=float)
        mag = max(float(np.max(np.abs(x0))), 1.0)
        rng = np.random.RandomState(20240817 + zlib.crc32(pname.encode("utf-8")) % 1000)
        o = mag * (rng.rand(d) * 2.0 - 1.0) * 0.5
        shifts[pname] = o

    probs = sorted(REGISTRY_36.keys()) if problems is None else problems
    total = len(probs) * n_runs * len(CONFIGS)
    rows = []
    done_pairs = set()
    if resume and os.path.exists(os.path.join(RESULTS, "stats_strong.csv")):
        import csv as _csv
        with open(os.path.join(RESULTS, "stats_strong.csv"), encoding="utf-8-sig") as f:
            for row in _csv.DictReader(f):
                rows.append(row)
                done_pairs.add((row["problem"], int(row["run"]), row["solver"]))
        print(f"resume: loaded {len(rows)} rows", flush=True)
    cnt = len(rows)
    t_all = time.time()
    for pname in probs:
        p = REGISTRY_36[pname]
        sp = make_shifted(p, shifts[pname])
        x0 = np.asarray(sp.x0, dtype=float)
        for r in range(n_runs):
            x0r = _perturb_x0(x0, seed=seed + r * 7 + zlib.crc32(pname.encode("utf-8")) % 1000)
            for ver, cfg_extra in CONFIGS:
                if (pname, r, ver) in done_pairs:
                    continue
                cnt += 1
                t0 = time.time()
                cfg = {"residual_fn": sp.grad, "jacobian_fn": sp.hess}
                cfg.update(cfg_extra)
                res = hgca(sp.func, sp.grad, sp.hess, x0r, tol=1e-10, max_iter=3000,
                           cfg=cfg, f_opt=sp.f_opt)
                fb = float(np.asarray(sp.func(np.asarray(res["x"], dtype=float))).item())
                rows.append({"problem": pname, "solver": ver, "run": r,
                             "conv": int(res["success"] and (fb - sp.f_opt) < 1e-4),
                             "time_ms": round((time.time() - t0) * 1000, 1),
                             "nit": int(res.get("nit", 0)),
                             "stage": res.get("stage", "")})
                if cnt % 200 == 0:
                    save_csv("stats_strong", rows)
                    print(f"  [{cnt}/{total}] {time.time()-t_all:.0f}s (flushed)", flush=True)
    save_csv("stats_strong", rows)

    # Summary: acc + 95% CI per question per configuration; TOTAL; stage distribution
    print("\n%-26s" % "problem", end="")
    for ver, _ in CONFIGS:
        print(" %8s" % ver, end="")
    print(" | v74-v54")
    summ = {}
    for pname in probs:
        acc = {}
        for ver, _ in CONFIGS:
            v = [x["conv"] for x in rows if x["problem"] == pname and x["solver"] == ver]
            m, lo, hi = ci95(v)
            acc[ver] = {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
                        "n": len(v)}
        summ[pname] = acc
        if abs(acc["v74"]["mean"] - acc["v54"]["mean"]) > 1e-9:
            print("%-26s" % pname, end="")
            for ver, _ in CONFIGS:
                print(" %8.2f" % acc[ver]["mean"], end="")
            print(" | %+6.2f" % (acc["v74"]["mean"] - acc["v54"]["mean"]))
    print()
    totals = {}
    for ver, _ in CONFIGS:
        v = [x["conv"] for x in rows if x["solver"] == ver]
        m, lo, hi = ci95(v)
        totals[ver] = {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(v)}
        print("TOTAL %-4s = %.4f  [%.4f, %.4f]" % (ver, m, lo, hi))
    print("\ntotal time %.0fs (%.2f ms/run)" % (time.time() - t_all, (time.time() - t_all) / max(cnt, 1) * 1000))

    from collections import Counter
    stage_dist = {}
    for ver, _ in CONFIGS:
        st = Counter(x["stage"] for x in rows if x["solver"] == ver)
        stage_dist[ver] = dict(st)
    print("stage distribution:", {k: {a: b for a, b in v.items()} for k, v in stage_dist.items()})
    save_json("stats_strong_summary", {"n_runs": n_runs, "seed": seed,
                                       "per_problem": summ, "totals": totals,
                                       "stage_dist": stage_dist})
    print("saved stats_strong*")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_runs", type=int, default=50)
    ap.add_argument("--problems", type=str, default=None,
                    help="comma-separated subset of problem names (smoke test)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    plist = None
    if args.problems:
        plist = [s.strip() for s in args.problems.split(",") if s.strip()]
    main(n_runs=args.n_runs, problems=plist, resume=args.resume)
