# -*- coding: utf-8 -*-
import os
"""run_03_pcu_generalize.py - PCU generalization verification: extending to more periodic/quasi-periodic structures.

Test matrix (verifying that "periodic coordinate unwrapping" does not only work on the Rastrigin family):
  Generalization families (should hit):
    Rastrigin_A5_20D   - amplitude parameter A=5 (non-default 10)
    Rastrigin_A20_20D  - amplitude parameter A=20
    Rastrigin_biased_20D - linear bias b (CEC-style biased)
    Rastrigin_A_est_20D  - detect_model=True auto-estimates A
  Purely periodic families (boundary, verifying the limitation of single-point unwrapping when there is no quadratic term):
    Cosine_20D - f=sum(1-cos 2pi u), c1=0
    SinSq_20D  - f=sum(sin^2(pi u)), c1=0
  Boundary families (should zero-trigger, zero false positives, citing run_02_pcu_integrated results):
    Griewank_20D / Schaffer_F6_2D_XE (already in REGISTRY_36, rerun here to confirm)
"""
import io, sys, zlib
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import save_json, save_csv
from aloop.data.functions_36 import REGISTRY_36
from aloop.solve.hgca import hgca
from aloop.solve.pcu import pcu_attempt
from run_00_benchmark import make_shifted, _perturb_x0


# ---------- Generalized Rastrigin family factory (supports T period, c1 quadratic term) ----------
def rastrigin_family(d, A=10.0, b=None, seed=42, T=1.0, c1=1.0):
    """f(x) = A*d + sum(c1*(x-o)^2 - A*cos(2*pi*(x-o)/T)) + b.x, A^2EP offset o."""
    o = (np.random.RandomState(seed).rand(d) * 2.0 - 1.0) * 0.5 * 20.0  # mag=20
    b = np.zeros(d) if b is None else np.asarray(b, dtype=float)
    w = 2.0 * np.pi / T

    def func(x):
        u = np.asarray(x, dtype=float) - o
        return float(A * d + np.sum(c1 * u**2 - A * np.cos(w * u)) + np.dot(b, np.asarray(x, dtype=float)))

    def grad(x):
        u = np.asarray(x, dtype=float) - o
        return 2.0 * c1 * u + w * A * np.sin(w * u) + b

    def hess(x):
        u = np.asarray(x, dtype=float) - o
        return 2.0 * c1 * np.eye(d) + w**2 * A * np.diag(np.cos(w * u))

    x0 = 20.0 * np.ones(d)
    tag = f'Rastrigin'
    if A != 10.0:
        tag += f'_A{A}'
    if np.any(b):
        tag += '_b'
    if c1 != 1.0:
        tag += f'_c1{c1}'
    if T != 1.0:
        tag += f'_T{T}'
    return {'name': f'{tag}_{d}D',
            'func': func, 'grad': grad, 'hess': hess, 'dim': d,
            'x0': x0, 'o': o, 'f_opt': 0.0,
            'pcu_kw': {'A': A, 'b': b, 'c1': c1, 'T': T}}


# ---------- Purely periodic families (no quadratic term) ----------
def cosine_family(d, seed=42, T=1.0):
    """f(x) = sum(1 - cos(2*pi*(x-o)/T)). No quadratic term -> single point cannot determine integer offset."""
    o = (np.random.RandomState(seed).rand(d) * 2.0 - 1.0) * 0.5 * 20.0
    w = 2.0 * np.pi / T

    def func(x):
        u = np.asarray(x, dtype=float) - o
        return float(np.sum(1.0 - np.cos(w * u)))

    def grad(x):
        u = np.asarray(x, dtype=float) - o
        return w * np.sin(w * u)

    def hess(x):
        u = np.asarray(x, dtype=float) - o
        return w**2 * np.diag(np.cos(w * u))

    return {'name': f'Cosine_{d}D' + (f'_T{T}' if T != 1.0 else ''),
            'func': func, 'grad': grad, 'hess': hess,
            'dim': d, 'x0': 20.0 * np.ones(d), 'o': o, 'f_opt': 0.0,
            'pcu_kw': {'A': 1.0, 'b': None, 'c1': 0.0, 'T': T}}


def sinsq_family(d, seed=42):
    """f(x) = sum(sin^2(pi*(x-o))). No quadratic term."""
    o = (np.random.RandomState(seed).rand(d) * 2.0 - 1.0) * 0.5 * 20.0

    def func(x):
        u = np.asarray(x, dtype=float) - o
        return float(np.sum(np.sin(np.pi * u)**2))

    def grad(x):
        u = np.asarray(x, dtype=float) - o
        return np.pi * np.sin(2.0 * np.pi * u)

    def hess(x):
        u = np.asarray(x, dtype=float) - o
        return 2.0 * np.pi**2 * np.diag(np.cos(2.0 * np.pi * u))

    return {'name': f'SinSq_{d}D', 'func': func, 'grad': grad, 'hess': hess,
            'dim': d, 'x0': 20.0 * np.ones(d), 'o': o, 'f_opt': 0.0,
            'pcu_kw': {'A': 0.5, 'b': None, 'c1': 0.0, 'T': 1.0}}


# ---------- Main flow ----------
def main(n_runs=10, seed=42):
    fams = []
    fams.append(rastrigin_family(20, A=5.0, seed=101))
    fams.append(rastrigin_family(20, A=20.0, seed=102))
    fams.append(rastrigin_family(20, A=10.0, b=np.linspace(0.1, 1.0, 20), seed=103))  # expected boundary
    fams.append(rastrigin_family(20, A=10.0, T=2.0, seed=106))
    fams.append(rastrigin_family(20, A=10.0, c1=2.0, seed=107))
    fams.append(cosine_family(20, seed=104))
    fams.append(cosine_family(20, seed=108, T=2.0))
    fams.append(sinsq_family(20, seed=105))

    rows = []
    for fam in fams:
        pname = fam['name']
        d = fam['dim']
        x0 = np.asarray(fam['x0'], dtype=float)
        pcu_kw = fam['pcu_kw']
        for r in range(n_runs):
            x0r = _perturb_x0(x0, seed=seed + r * 7 + zlib.crc32(pname.encode('utf-8')) % 1000)
            # Baseline (PCU off)
            res = hgca(fam['func'], fam['grad'], fam['hess'], x0r, tol=1e-10, max_iter=3000,
                       cfg={"residual_fn": fam['grad'], "jacobian_fn": fam['hess'],
                            "use_caci": False, "sar_enabled": False, "use_pcu": False},
                       f_opt=fam['f_opt'])
            fb = float(np.asarray(fam['func'](np.asarray(res['x'], dtype=float))).item())
            conv_base = int(res['success'] and (fb - fam['f_opt']) < 1e-4)
            # PCU pre-processing
            xc, stage = pcu_attempt(fam['func'], fam['grad'], fam['hess'], x0r, **pcu_kw)
            if stage == 'pcu_ok':
                fb2 = float(np.asarray(fam['func'](xc)).item())
                conv_pcu = int(fb2 < 1e-4)
            else:
                res2 = hgca(fam['func'], fam['grad'], fam['hess'], x0r, tol=1e-10, max_iter=3000,
                            cfg={"residual_fn": fam['grad'], "jacobian_fn": fam['hess'],
                                 "use_caci": False, "sar_enabled": False, "use_pcu": False},
                            f_opt=fam['f_opt'])
                fb2 = float(np.asarray(fam['func'](np.asarray(res2['x'], dtype=float))).item())
                conv_pcu = int(res2['success'] and (fb2 - fam['f_opt']) < 1e-4)
            rows.append({"problem": pname, "solver": "base", "run": r, "conv": conv_base,
                         "stage": res.get("stage", "")})
            rows.append({"problem": pname, "solver": "pcu", "run": r, "conv": conv_pcu,
                         "stage": stage if stage else res2.get("stage", "")})

    save_csv("pcu_generalize", rows)
    print("%-24s %8s %8s | %s" % ("problem", "base", "pcu", "diff"))
    summ = {}
    for pname in sorted(set(r['problem'] for r in rows)):
        acc = {}
        for ver in ("base", "pcu"):
            v = [x['conv'] for x in rows if x['problem'] == pname and x['solver'] == ver]
            acc[ver] = float(np.mean(v))
        summ[pname] = acc
        print("%-24s %8.2f %8.2f | %+6.2f" % (pname, acc['base'], acc['pcu'],
                                              acc['pcu'] - acc['base']))
    save_json("pcu_generalize_summary", {"n_runs": n_runs, "per_problem": summ})
    print("saved pcu_generalize*")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_runs", type=int, default=10)
    args = ap.parse_args()
    main(n_runs=args.n_runs)
