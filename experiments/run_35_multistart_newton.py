# -*- coding: utf-8 -*-
"""run_35_multistart_newton.py

Multi-start damped Newton with finite-difference derivatives under a strict
black-box protocol (every FD evaluation is charged to the shared FE budget).

Motivation (reviewer comment): the SOTA comparison in Table II uses
second-order information on the PCU side vs. zero-order sampling on the
baseline side. The L-BFGS row (run_28) already hedges this with a quasi-Newton
baseline (0/25 hits). This script goes one step further: a TRUE Newton method
with FD gradient + FD Hessian, where the 2D^2 evaluations per iteration are
explicitly charged, so the comparison is information-fair even under the
"derivatives must be obtained by black-box sampling" reading.

Expected outcome: the cost of acquiring second-order information by finite
differences (2*D^2 + 2*D ~ 1,860 FE per iteration at D=30) leaves far too few
iterations to escape the periodic multi-modality; hits stay at 0/25, matching
L-BFGS, PSO, GWO, CMA-ES etc. This isolates the phenomenon as a landscape
property (periodic multi-modality defeats local refinement from arbitrary
starts), not an information-level artifact.

Problem: CEC2017 F5 M_orth (30D), f_opt calibrated, T=19.53.
Budget: 396,993 FE (same as Table II).
Output: results/newton_fd_25runs.json
"""
import io, sys, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'cec2017'))
from cec2017.transforms import rotations, shifts

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
BUDGET = 396993
S5 = 5.12 / 100.0
D = 30
H_FD = 1e-6          # central-difference step (noiseless black-box evaluations)
SEEDS = list(range(20260912, 20260937))  # same 25 seeds as run_26/run_28

Mr = rotations[D][4]
u, _, vt = np.linalg.svd(Mr)
M = u @ vt
o = shifts[4][:D].copy()
oM = o @ M.T


def f5(X):
    Xa = np.atleast_2d(np.asarray(X, float))
    z = S5 * (Xa @ M.T - oM)
    return np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0, axis=1)


def fd_grad(f, x, cnt, h=H_FD):
    """Central-difference gradient; counts 2*D function evaluations."""
    x = np.asarray(x, float)
    g = np.empty(D)
    for i in range(D):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        g[i] = (float(np.asarray(f(np.atleast_2d(xp))).reshape(-1)[0]) -
                float(np.asarray(f(np.atleast_2d(xm))).reshape(-1)[0])) / (2.0 * h)
        cnt[0] += 2
    return g


def fd_hess(f, x, cnt, h=H_FD):
    """Central-difference Hessian (full matrix); counts 2*D^2 evaluations
    (2*D^2 - 2*D unique pairs, charged conservatively as 2*D^2)."""
    x = np.asarray(x, float)
    H = np.empty((D, D))
    for i in range(D):
        for j in range(i, D):
            xpp = x.copy(); xpp[i] += h; xpp[j] += h
            xpm = x.copy(); xpm[i] += h; xpm[j] -= h
            xmp = x.copy(); xmp[i] -= h; xmp[j] += h
            xmm = x.copy(); xmm[i] -= h; xmm[j] -= h
            H[i, j] = H[j, i] = (
                float(np.asarray(f(np.atleast_2d(xpp))).reshape(-1)[0]) -
                float(np.asarray(f(np.atleast_2d(xpm))).reshape(-1)[0]) -
                float(np.asarray(f(np.atleast_2d(xmp))).reshape(-1)[0]) +
                float(np.asarray(f(np.atleast_2d(xmm))).reshape(-1)[0])) / (4.0 * h * h)
            cnt[0] += 4
    return H


def damped_newton_step(f, x, cnt, max_ls=10):
    """One damped Newton step with Tikhonov regularization and backtracking
    line search. Returns (x_new, converged)."""
    g = fd_grad(f, x, cnt)
    H = fd_hess(f, x, cnt)
    lam = 1e-8 * max(1.0, float(np.max(np.abs(np.diag(H)))))
    try:
        delta = np.linalg.solve(H + lam * np.eye(D), g)
    except np.linalg.LinAlgError:
        return x, False
    if not np.all(np.isfinite(delta)):
        return x, False
    delta = np.clip(delta, -1.0, 1.0)
    f0 = float(np.asarray(f(np.atleast_2d(x))).reshape(-1)[0])
    cnt[0] += 1
    alpha = 1.0
    gd = float(np.dot(g, delta))
    for _ in range(max_ls):
        xa = x - alpha * delta
        fa = float(np.asarray(f(np.atleast_2d(xa))).reshape(-1)[0])
        cnt[0] += 1
        if fa <= f0 + 1e-4 * alpha * gd:
            return xa, float(np.max(np.abs(alpha * delta))) < 1e-10
        alpha *= 0.5
    return x, False


def multistart_newton(f, budget=BUDGET, seed=0):
    """Multi-start damped Newton with FD derivatives; every evaluation counted."""
    rng = np.random.RandomState(seed)
    cnt = [0]
    best_f = float('inf')
    n_start = 0
    max_iter_per_start = 60
    while cnt[0] < budget:
        x = rng.uniform(-100.0, 100.0, D)
        n_start += 1
        for _ in range(max_iter_per_start):
            if cnt[0] >= budget:
                break
            fv = float(np.asarray(f(np.atleast_2d(x))).reshape(-1)[0])
            cnt[0] += 1
            if fv < best_f:
                best_f = fv
            if best_f < 1e-4:
                break
            x, conv = damped_newton_step(f, x, cnt)
            if conv:
                break
        if best_f < 1e-4:
            break
    return best_f, cnt[0], n_start


def main():
    t0 = time.time()
    rows = {}
    for seed in SEEDS:
        best_f, fe, n_start = multistart_newton(f5, budget=BUDGET, seed=seed)
        rows[str(seed)] = {'best_F': best_f, 'fe': fe, 'n_starts': n_start}
        print('seed %d: F=%.4g fe=%d starts=%d' % (seed, best_f, fe, n_start), flush=True)
    Fs = [r['best_F'] for r in rows.values()]
    hits = sum(1 for v in Fs if v < 1e-4)
    out = {
        'problem': 'CEC2017 F5 M_orth 30D, f_opt calibrated, T=19.53, budget=396993',
        'method': 'multi-start damped Newton, FD gradient + FD Hessian (central difference h=1e-6), all FE charged',
        'fe_per_iteration': 2 * D * D + 2 * D + 1,
        'budget_fe': BUDGET,
        'n_runs': len(SEEDS),
        'seeds': SEEDS,
        'n_hit': hits,
        'final_f': {
            'median': float(np.median(Fs)),
            'min': float(np.min(Fs)),
            'max': float(np.max(Fs)),
        },
        'per_run': rows,
        'elapsed_s': round(time.time() - t0, 1),
    }
    json.dump(out, io.open(os.path.join(RESULTS, 'newton_fd_25runs.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('SAVED newton_fd_25runs.json | n_hit: %d/%d | median F: %.3f | %.1fs' % (
        hits, len(SEEDS), float(np.median(Fs)), out['elapsed_s']), flush=True)


if __name__ == '__main__':
    main()
