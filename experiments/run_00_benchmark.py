# -*- coding: utf-8 -*-
"""run_00_benchmark.py - Shifted fair benchmark: 30 runs statistical significance + NBIPOP-aCMA

Building on run_38 (3 runs):
- n_runs=30, add statistical significance (mean +/- 95% CI)
- Add NBIPOP-aCMA-ES (Loshchilov 2012 dual-population restart enhanced version)
- Core comparison of 4 methods: baseline solver / CMA-ES / BIPOP-aCMA / NBIPOP-aCMA

Outputs: shifted_fair_30runs.csv / shifted_fair_30runs_summary.json
"""
import zlib
import argparse
import time
import numpy as np
from common import save_json, save_csv, RESULTS

_OPT_SHIFT_CACHE = {}


def make_shifted(p, o):
    """f'(x) = f(x - o), optimum at o."""
    d = p.dim

    def func(x):
        return float(np.asarray(p.func(np.asarray(x, dtype=float) - o)).item())

    def grad(x):
        return np.asarray(p.grad(np.asarray(x, dtype=float) - o), dtype=float)

    def hess(x):
        return np.asarray(p.hess(np.asarray(x, dtype=float) - o), dtype=float)

    class P:
        pass
    P.func, P.grad, P.hess = func, grad, hess
    P.dim = d
    P.f_opt = p.f_opt
    P.x0 = np.asarray(p.x0, dtype=float) + o
    P.name = getattr(p, "name", None)
    return P


def _perturb_x0(x0, scale=0.1, seed=0):
    rng = np.random.RandomState(seed)
    return x0 + scale * (np.abs(x0) + 0.1) * rng.randn(len(x0))


def _run_bipop_acma(f, x0, maxfevals=30000):
    """pycma official BIPOP-aCMA-ES (consistent with run_36/38)."""
    import cma
    t0 = time.time()
    opts = {
        'maxfevals': int(maxfevals), 'seed': 0, 'verbose': -9,
        'popsize': 24, 'tolx': 1e-12, 'tolfun': 1e-12, 'CMA_active': 1,
    }
    x_best, es = cma.fmin2(f, np.asarray(x0, dtype=float), 1.0,
                            options=opts, restarts=9, bipop=True, restart_from_best=False)
    dt = (time.time() - t0) * 1000
    n_evals = int(es.result.evaluations) if hasattr(es, 'result') else 0
    return {"x": np.asarray(x_best, dtype=float), "success": True,
            "nit": n_evals, "dt": dt}


def _run_nbipop_acma(f, x0, maxfevals=30000):
    """NBIPOP-aCMA-ES (Loshchilov 2012): dual-population restart + population increase + active covariance.

    Differences from BIPOP: finer small-population budget ratio (B/4 instead of a fixed ratio),
    and after each restart the large-population popsize increases by incpopsize=2 (IPOP feature).
    Manual restart loop implementation to precisely control NBIPOP budget allocation.
    """
    import cma
    t0 = time.time()
    x0 = np.asarray(x0, dtype=float)
    d = len(x0)
    budget_used = 0
    best_x = x0.copy()
    best_f = float('inf')
    pop_large = 24
    incpopsize = 2
    restart_idx = 0

    while budget_used < maxfevals:
        # NBIPOP: alternate large population (1/2 of remaining budget) and small population (1/4 of remaining budget)
        is_large = (restart_idx % 2 == 0)
        remaining = maxfevals - budget_used
        if is_large:
            run_budget = int(remaining * 0.5)
            pop = pop_large
        else:
            run_budget = int(remaining * 0.25)
            pop = max(pop_large // 4, 8)  # small population

        if run_budget < pop * 5:
            break

        try:
            opts = {
                'maxfevals': run_budget,
                'seed': 1000 + restart_idx,
                'verbose': -9,
                'popsize': pop,
                'tolx': 1e-12,
                'tolfun': 1e-12,
                'CMA_active': 1,
            }
            x_best_run, es = cma.fmin2(f, x0, 1.0, options=opts,
                                        restarts=0, bipop=False, restart_from_best=False)
            evals_used = int(es.result.evaluations) if hasattr(es, 'result') else run_budget
            f_run = float(np.asarray(f(np.asarray(x_best_run, dtype=float))).item())
            if f_run < best_f:
                best_f = f_run
                best_x = np.asarray(x_best_run, dtype=float).copy()
            budget_used += evals_used
        except Exception:
            budget_used += run_budget

        pop_large = int(pop_large * incpopsize)
        restart_idx += 1
        if restart_idx > 20:
            break

    dt = (time.time() - t0) * 1000
    return {"x": best_x, "success": True, "nit": budget_used, "dt": dt}


def run_benchmark(n_runs=30, seed=42):
    from aloop.data.functions_36 import REGISTRY_36
    from aloop.solve.hgca import hgca
    from aloop.solve.hybrid import hast_cma

    maxfevals = 30000
    rows = []

    # Shift vector (identical to run_38 to ensure comparability)
    shifts = {}
    for pname, p in REGISTRY_36.items():
        d = p.dim
        x0 = np.asarray(p.x0, dtype=float)
        mag = max(float(np.max(np.abs(x0))), 1.0)
        rng = np.random.RandomState(20240817 + zlib.crc32(pname.encode("utf-8")) % 1000)
        o = mag * (rng.rand(d) * 2.0 - 1.0) * 0.5
        shifts[pname] = o

    total = len(REGISTRY_36) * n_runs
    cnt = 0
    for pname, p in REGISTRY_36.items():
        sp = make_shifted(p, shifts[pname])
        d = p.dim
        x0 = np.asarray(sp.x0, dtype=float)

        for r in range(n_runs):
            cnt += 1
            x0r = _perturb_x0(x0, seed=seed + r * 7 + zlib.crc32(pname.encode("utf-8")) % 1000)

            # 1. baseline solver (no PCU)
            t0 = time.time()
            res = hgca(sp.func, sp.grad, sp.hess, x0r, tol=1e-10,
                       max_iter=3000, cfg={"residual_fn": sp.grad,
                                            "jacobian_fn": sp.hess}, f_opt=sp.f_opt)
            f_best = float(np.asarray(sp.func(np.asarray(res["x"], dtype=float))).item())
            rows.append({"problem": pname, "dim": d, "solver": "hgca", "run": r,
                         "conv": int(res["success"] and (f_best - sp.f_opt) < 1e-4),
                         "f": f_best, "time_ms": round((time.time() - t0) * 1000, 3),
                         "nit": int(res.get("nit", 0))})

    # 2. CMA-ES global stage
            t0 = time.time()
            res = hast_cma(sp.func, sp.grad, sp.hess, x0r, tol=1e-10,
                           max_iter=3000, f_opt=sp.f_opt)
            f_best = float(np.asarray(sp.func(np.asarray(res["x"], dtype=float))).item())
            rows.append({"problem": pname, "dim": d, "solver": "hast_cma", "run": r,
                         "conv": int(res["success"] and (f_best - sp.f_opt) < 1e-4),
                         "f": f_best, "time_ms": round((time.time() - t0) * 1000, 3),
                         "nit": int(res.get("nit", 0))})

            # 3. BIPOP-aCMA
            try:
                rr = _run_bipop_acma(sp.func, x0r, maxfevals=maxfevals)
                f_best = float(np.asarray(sp.func(np.asarray(rr["x"], dtype=float))).item())
                conv = rr["success"] and (f_best - sp.f_opt) < 1e-4
            except Exception:
                f_best, conv, rr = float('inf'), False, {"nit": 0, "dt": 0}
            rows.append({"problem": pname, "dim": d, "solver": "bipop_acma", "run": r,
                         "conv": int(conv), "f": f_best,
                         "time_ms": round(rr["dt"], 3), "nit": int(rr["nit"])})

            # 4. NBIPOP-aCMA
            try:
                rr = _run_nbipop_acma(sp.func, x0r, maxfevals=maxfevals)
                f_best = float(np.asarray(sp.func(np.asarray(rr["x"], dtype=float))).item())
                conv = rr["success"] and (f_best - sp.f_opt) < 1e-4
            except Exception:
                f_best, conv, rr = float('inf'), False, {"nit": 0, "dt": 0}
            rows.append({"problem": pname, "dim": d, "solver": "nbipop_acma", "run": r,
                         "conv": int(conv), "f": f_best,
                         "time_ms": round(rr["dt"], 3), "nit": int(rr["nit"])})

            if cnt % 36 == 0:
                print(f"  [{cnt}/{total}] {pname} run {r} done", flush=True)
                # Incremental saving: save intermediate results after each problem completes, to avoid losing all data if the final save fails
                try:
                    save_csv("shifted_fair_30runs_partial", rows)
                except Exception as e:
                    print(f"  [warn] partial save failed: {e}", flush=True)

    # Summary
    methods = ["hgca", "hast_cma", "bipop_acma", "nbipop_acma"]
    summ = {"note": "shifted fair benchmark 30 runs (statistical significance) + NBIPOP-aCMA", "n_runs": n_runs}
    per_problem = {}
    for m in methods:
        mrows = [r for r in rows if r["solver"] == m]
        convs = [r["conv"] for r in mrows]
        times = [r["time_ms"] for r in mrows]
        nits = [r["nit"] for r in mrows]
        n = len(convs)
        mean_conv = float(np.mean(convs))
        # 95% CI (normal approximation to the binomial distribution)
        se = np.sqrt(mean_conv * (1 - mean_conv) / n) if 0 < mean_conv < 1 else 0.0
        ci95 = 1.96 * se * 100
        summ[m] = {
            "conv_rate": round(mean_conv, 4),
            "conv_rate_pct": round(mean_conv * 100, 2),
            "ci95_pct": round(ci95, 2),
            "avg_time_ms": round(float(np.mean(times)), 1),
            "std_time_ms": round(float(np.std(times)), 1),
            "avg_nit": round(float(np.mean(nits)), 0),
            "n_total": n,
            "n_conv": int(np.sum(convs)),
        }
        # per-problem
        for pname in REGISTRY_36:
            prow = [r for r in mrows if r["problem"] == pname]
            if prow:
                per_problem.setdefault(pname, {})[m] = round(float(np.mean([r["conv"] for r in prow])), 3)

    summ["per_problem"] = per_problem

    # Statistical significance test: solver vs each SOTA (paired sign test / McNemar)
    from collections import defaultdict
    paired = defaultdict(dict)
    for pname in REGISTRY_36:
        for r in range(n_runs):
            key = (pname, r)
            for m in methods:
                match = [x for x in rows if x["problem"] == pname and x["run"] == r and x["solver"] == m]
                if match:
                    paired[key][m] = match[0]["conv"]

    for m in ["hast_cma", "bipop_acma", "nbipop_acma"]:
        hgca_win = 0
        sota_win = 0
        tie = 0
        for key, d in paired.items():
            if "hgca" in d and m in d:
                if d["hgca"] > d[m]:
                    hgca_win += 1
                elif d["hgca"] < d[m]:
                    sota_win += 1
                else:
                    tie += 1
        # Binomial sign test p-value (two-sided): P(X <= min(wins,losses)) * 2, truncated to 1
        # (Fix: the original implementation accumulated from min to n, which is actually the complementary event probability, so p was always 1.0)
        n_pairs = hgca_win + sota_win
        if n_pairs > 0:
            from math import comb
            k_min = min(hgca_win, sota_win)
            p_val = sum(comb(n_pairs, k) * 0.5 ** n_pairs
                        for k in range(0, k_min + 1))
            p_val = min(p_val * 2, 1.0)  # two-sided
        else:
            p_val = 1.0
        summ[f"sign_test_hgca_vs_{m}"] = {
            "hgca_win": hgca_win, "sota_win": sota_win, "tie": tie,
            "p_value": round(p_val, 4),
            "significant_005": p_val < 0.05,
        }

    save_csv("shifted_fair_30runs", rows)
    save_json("shifted_fair_30runs_summary", summ)
    print(f"\n=== shifted fair benchmark 30 runs complete ===")
    for m in methods:
        s = summ[m]
        print(f"  {m:14s}: {s['conv_rate_pct']:5.2f}% +/- {s['ci95_pct']:.2f}% (95% CI) | "
              f"{s['avg_time_ms']:6.1f}ms | {s['avg_nit']:.0f} evals")
    for m in ["hast_cma", "bipop_acma", "nbipop_acma"]:
        st = summ[f"sign_test_hgca_vs_{m}"]
        print(f"  sign test solver vs {m}: solver {st['hgca_win']} / SOTA {st['sota_win']} / "
              f"tie {st['tie']} | p={st['p_value']:.4f} {'*' if st['significant_005'] else ''}")
    return summ


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_benchmark(n_runs=args.n_runs, seed=args.seed)
