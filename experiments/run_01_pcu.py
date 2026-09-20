# -*- coding: utf-8 -*-
import os
"""run_01_pcu.py - PCU preposed fast path vs v54 baseline (full validation over 30 runs)

PCU (Periodic Coordinate Unwrapping):
  Uses the analytical Hessian to detect and solve separable periodic structure problems (Rastrigin family).

Two modes:
  A. Standard separable (H diagonal): per-dimension analytical unwrapping from a single point (g_i, H_ii) -> directly obtains the optimal o
  B. Rotated non-separable (H = RT D R): recover R via single-point eigh -> transform to z = R x -> per-dimension unwrapping

Safety design: accept the unwrapping candidate only if F < 1e-4; otherwise fall back to the original solver flow with zero impact.
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

PCU_A = 10.0
PCU_RADIUS = 80.0


def unwrap_1d(g_i, H_ii, x_i, A=PCU_A, tol=1e-5, radius=PCU_RADIUS):
    """1D Rastrigin periodic unwrapping: recover u = x_i - o_i from (g_i, H_ii)."""
    d = H_ii - 2.0
    cosv = np.clip(d / (4.0 * np.pi**2 * A), -1.0, 1.0)
    ac = np.arccos(cosv)
    u0s = set()
    for ang in (ac, 2.0 * np.pi - ac):
        u0s.add(round(ang / (2.0 * np.pi) % 1.0, 10))
    cands = []
    lo = int(np.floor(x_i - radius))
    hi = int(np.ceil(x_i + radius))
    for u0 in sorted(u0s):
        for k in range(lo, hi + 1):
            u = k + u0
            g_pred = 2.0 * u + 2.0 * np.pi * A * np.sin(2.0 * np.pi * u)
            if abs(g_pred - g_i) < tol * max(1.0, abs(g_i)):
                cands.append(u)
    return cands


def pcu_attempt(func, grad, hess, x, tol=1e-5):
    """PCU fast path. Returns (x_cand, 'pcu_ok') on success, otherwise (None, None)."""
    d = len(x)
    g = np.asarray(grad(x))
    H = np.asarray(hess(x))
    # Mode A: H diagonal (separable) -> direct per-dimension unwrapping
    off = np.sqrt(np.sum((H - np.diag(np.diag(H)))**2))
    o_cand = None
    if off < 1e-6 * (1.0 + np.abs(H).max()):
        o = np.zeros(d)
        for i in range(d):
            us = unwrap_1d(g[i], H[i, i], x[i])
            if not us:
                o = None
                break
            o[i] = x[i] - min(us, key=lambda u: abs(u))
        o_cand = o
    else:
        # Mode B: eigh to recover the rotation
        evals, evecs = np.linalg.eigh(H)
        R = evecs.T
        z = R @ x
        g_z = R @ g
        H_z = R @ H @ R.T
        o = np.zeros(d)
        ok = True
        for i in range(d):
            us = unwrap_1d(g_z[i], H_z[i, i], z[i])
            if not us:
                ok = False
                break
            o[i] = z[i] - min(us, key=lambda u: abs(u))
        if ok:
            o_cand = R.T @ o
    if o_cand is None:
        return None, None
    f = float(np.asarray(func(np.asarray(o_cand, dtype=float))).item())
    if f < 1e-4:
        return np.asarray(o_cand, dtype=float), 'pcu_ok'
    return None, None


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
    pcu_hits = 0
    total = len(probs) * n_runs * 2
    cnt = 0
    for pname in probs:
        p = REGISTRY_36[pname]
        sp = make_shifted(p, shifts[pname])
        x0 = np.asarray(sp.x0, dtype=float)
        for r in range(n_runs):
            x0r = _perturb_x0(x0, seed=seed + r * 7 + zlib.crc32(pname.encode("utf-8")) % 1000)
            # v54: pure baseline
            t0 = time.time()
            res = hgca(sp.func, sp.grad, sp.hess, x0r, tol=1e-10, max_iter=3000,
                       cfg={"residual_fn": sp.grad, "jacobian_fn": sp.hess,
                            "use_caci": False, "sar_enabled": False},
                       f_opt=sp.f_opt)
            fb = float(np.asarray(sp.func(np.asarray(res["x"], dtype=float))).item())
            rows.append({"problem": pname, "solver": "v54", "run": r,
                         "conv": int(res["success"] and (fb - sp.f_opt) < 1e-4),
                         "time_ms": round((time.time() - t0) * 1000, 1),
                         "nit": int(res.get("nit", 0)),
                         "stage": res.get("stage", "")})
            # v70: PCU preposed
            t0 = time.time()
            xc, stage = pcu_attempt(sp.func, sp.grad, sp.hess, x0r)
            if stage == 'pcu_ok':
                res2 = {"success": True, "nit": 2, "stage": "pcu_ok", "x": xc}
            else:
                res2 = hgca(sp.func, sp.grad, sp.hess, x0r, tol=1e-10, max_iter=3000,
                            cfg={"residual_fn": sp.grad, "jacobian_fn": sp.hess,
                                 "use_caci": False, "sar_enabled": False},
                            f_opt=sp.f_opt)
                if stage is None and res2.get("stage", "") != "":
                    res2["stage"] = res2.get("stage", "")
            fb2 = float(np.asarray(sp.func(np.asarray(res2["x"], dtype=float))).item())
            rows.append({"problem": pname, "solver": "v70", "run": r,
                         "conv": int(res2["success"] and (fb2 - sp.f_opt) < 1e-4),
                         "time_ms": round((time.time() - t0) * 1000, 1),
                         "nit": int(res2.get("nit", 0)),
                         "stage": res2.get("stage", "pcu_miss") if stage is None else stage})
            if stage == 'pcu_ok':
                pcu_hits += 1
            cnt += 2
            if cnt % 100 == 0:
                print(f"  [{cnt}/{total}] pcu_hits={pcu_hits}", flush=True)
    save_csv("pcu_vs_v54", rows)
    print("\n%-26s %8s %8s | %s" % ("problem", "v54", "v70", "diff"))
    summ = {}
    for pname in probs:
        acc = {}
        for ver in ("v54", "v70"):
            v = [x["conv"] for x in rows if x["problem"] == pname and x["solver"] == ver]
            acc[ver] = float(np.mean(v))
        summ[pname] = acc
        if abs(acc["v70"] - acc["v54"]) > 1e-9:
            print("%-26s %8.2f %8.2f | %+6.2f" % (pname, acc["v54"], acc["v70"],
                                                  acc["v70"] - acc["v54"]))
    print()
    for ver in ("v54", "v70"):
        t = float(np.mean([x["conv"] for x in rows if x["solver"] == ver]))
        print("TOTAL %-4s = %.4f" % (ver, t))
    print("PCU direct hits:", pcu_hits)
    save_json("pcu_vs_v54_summary", {"n_runs": n_runs, "per_problem": summ,
                                     "pcu_hits": pcu_hits})
    print("saved pcu_vs_v54*")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_runs", type=int, default=30)
    args = ap.parse_args()
    main(n_runs=args.n_runs)
