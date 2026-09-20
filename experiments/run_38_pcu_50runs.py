# -*- coding: utf-8 -*-
"""run_38_pcu_50runs.py

Statistical strengthening at the problem level: extend the 50-run panel from
three Rastrigin problems to seven PCU positive instances (responding to
"problem-level evidence n=3 is underpowered"):

  1. CEC2017 f5 (shifted rotated Rastrigin, orth) 30D, analytic derivatives
  2. CEC2017 f8 (non-continuous rotated Rastrigin, orth) 30D, analytic on
     differentiable intervals (discontinuities at odd half-integers have
     measure zero; peaks there are removed by peak-distance filtering)
  3. synthetic consensus-periodic function (T=10, non-Rastrigin)
  4. sin^2-periodic bowl
  5. scaled-period rotated Rastrigin (s=1.4, T=1/s ~= 0.714)
  6. weak-amplitude bowl (A=0.01)
  7. quasi-periodic incommensurate frequencies (robustness positive)

50 independent seeds per problem (20260912..20260961), same estimate_and_pcu
pipeline and F < 1e-4 hit criterion as all other runs. Output
results/pcu_50runs.json
"""
import io, sys, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'cec2017'))
from aloop.solve.struct_id import estimate_and_pcu
from cec2017.transforms import rotations, shifts

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
D = 30
SEEDS = list(range(20260912, 20260962))  # 50 seeds
S5 = 5.12 / 100.0


def fval(f, x):
    return float(np.asarray(f(np.atleast_2d(x))).reshape(-1)[0])


def orth_M(idx):
    Mr = rotations[D][idx].copy()
    u, _, vt = np.linalg.svd(Mr)
    return u @ vt


# ---- 1/2) CEC2017 f5 / f8 rotated Rastrigin (analytic) ----
def make_rastrigin(idx):
    o = shifts[idx][:D].copy()
    M = orth_M(idx)
    def f(X):
        Xa = np.atleast_2d(np.asarray(X, float))
        z = S5 * (Xa @ M.T - o @ M.T)
        return np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0, axis=1)
    def grad(x):
        x = np.asarray(x, float)
        z = S5 * (M @ (x - o))
        gz = S5 * (2.0 * z + 20.0 * np.pi * np.sin(2.0 * np.pi * z))
        return M.T @ gz
    def hess(x):
        x = np.asarray(x, float)
        z = S5 * (M @ (x - o))
        Hzz = S5 * S5 * (2.0 + 40.0 * np.pi * np.pi * np.cos(2.0 * np.pi * z))
        return (M.T * Hzz) @ M
    return f, grad, hess


def make_ncr_rastrigin(idx):
    """CEC2017 f8 non-continuous rotated Rastrigin, analytic on differentiable
    intervals: y_i = z_i if |z_i|<0.5 else round(2 z_i)/2; derivatives w.r.t. z
    vanish on the plateau branches (dy/dz=0) and equal the smooth Rastrigin
    derivative on |z|<0.5."""
    o = shifts[idx][:D].copy()
    M = orth_M(idx)
    def f(X):
        Xa = np.atleast_2d(np.asarray(X, float))
        z = S5 * (Xa @ M.T - o @ M.T)
        y = np.where(np.abs(z) < 0.5, z, np.round(2.0 * z) / 2.0)
        return np.sum(y * y - 10.0 * np.cos(2.0 * np.pi * y) + 10.0, axis=1)
    def grad(x):
        x = np.asarray(x, float)
        z = S5 * (M @ (x - o))
        mask = np.abs(z) < 0.5
        y = np.where(mask, z, np.round(2.0 * z) / 2.0)
        gz = np.zeros_like(z)
        gz[mask] = 2.0 * y[mask] + 20.0 * np.pi * np.sin(2.0 * np.pi * y[mask])
        return M.T @ (S5 * gz)
    def hess(x):
        x = np.asarray(x, float)
        z = S5 * (M @ (x - o))
        mask = np.abs(z) < 0.5
        y = np.where(mask, z, np.round(2.0 * z) / 2.0)
        Hzz = np.zeros_like(z)
        Hzz[mask] = 2.0 + 40.0 * np.pi * np.pi * np.cos(2.0 * np.pi * y[mask])
        return (M.T * (S5 * S5 * Hzz)) @ M
    return f, grad, hess


# ---- 3) synthetic consensus-periodic (T=10, non-Rastrigin) ----
def make_synth():
    T = 10.0
    M = orth_M(4)
    def f(X):
        Xa = np.atleast_2d(np.asarray(X, float))
        Z = Xa @ M.T
        return np.sum(2.0 * (1.0 - np.cos(2.0 * np.pi * Z / T)) + 0.1 * Z * Z, axis=1)
    def grad(x):
        z = M @ np.asarray(x, float)
        return M.T @ (2.0 * np.pi / T * 2.0 * np.sin(2.0 * np.pi * z / T) + 0.2 * z)
    def hess(x):
        z = M @ np.asarray(x, float)
        diag = 0.2 + 2.0 * (2.0 * np.pi / T) ** 2 * np.cos(2.0 * np.pi * z / T)
        return (M.T * diag) @ M
    return f, grad, hess


# ---- 4/5/6/7) run_37 families ----
def make_sin2():
    w = 2.0 * np.pi
    M = orth_M(4)
    def gz_f(Z):
        Z = np.asarray(Z, float)
        return np.sum(np.sin(0.5 * w * Z) ** 2 + 0.2 * Z * Z, axis=1)
    def gz_g(z):
        return 0.5 * w * np.sin(w * z) + 0.4 * z
    def gz_h(z):
        return 0.5 * w * w * np.cos(w * z) + 0.4
    def f(X):
        return gz_f(np.atleast_2d(np.asarray(X, float)) @ M.T)
    def grad(x):
        z = M @ np.asarray(x, float)
        return M.T @ gz_g(z)
    def hess(x):
        z = M @ np.asarray(x, float)
        return (M.T * gz_h(z)) @ M
    return f, grad, hess


def make_scaled():
    s = 1.4
    w = 2.0 * np.pi * s
    M = orth_M(4)
    def gz_f(Z):
        Z = np.asarray(Z, float)
        return np.sum((s * Z) ** 2 - 10.0 * np.cos(w * Z) + 10.0, axis=1)
    def gz_g(z):
        return 2.0 * s * s * z + 10.0 * w * np.sin(w * z)
    def gz_h(z):
        return 2.0 * s * s + 10.0 * w * w * np.cos(w * z)
    def f(X):
        return gz_f(np.atleast_2d(np.asarray(X, float)) @ M.T)
    def grad(x):
        z = M @ np.asarray(x, float)
        return M.T @ gz_g(z)
    def hess(x):
        z = M @ np.asarray(x, float)
        return (M.T * gz_h(z)) @ M
    return f, grad, hess


def make_weak():
    A = 0.01
    w = 2.0 * np.pi
    M = orth_M(4)
    def gz_f(Z):
        Z = np.asarray(Z, float)
        return np.sum(A * (1.0 - np.cos(w * Z)) + 0.5 * Z * Z, axis=1)
    def gz_g(z):
        return A * w * np.sin(w * z) + z
    def gz_h(z):
        return A * w * w * np.cos(w * z) + 1.0
    def f(X):
        return gz_f(np.atleast_2d(np.asarray(X, float)) @ M.T)
    def grad(x):
        z = M @ np.asarray(x, float)
        return M.T @ gz_g(z)
    def hess(x):
        z = M @ np.asarray(x, float)
        return (M.T * gz_h(z)) @ M
    return f, grad, hess


def make_quasi():
    w1 = 2.0 * np.pi
    w2 = 2.0 * np.pi * np.sqrt(2.0)
    a1, a2 = 2.0, 1.0
    M = orth_M(4)
    def gz_f(Z):
        Z = np.asarray(Z, float)
        return np.sum(a1 * (1.0 - np.cos(w1 * Z)) + a2 * (1.0 - np.cos(w2 * Z)), axis=1) + 0.1 * np.sum(Z * Z, axis=1)
    def gz_g(z):
        return a1 * w1 * np.sin(w1 * z) + a2 * w2 * np.sin(w2 * z) + 0.2 * z
    def gz_h(z):
        return a1 * w1 * w1 * np.cos(w1 * z) + a2 * w2 * w2 * np.cos(w2 * z) + 0.2
    def f(X):
        return gz_f(np.atleast_2d(np.asarray(X, float)) @ M.T)
    def grad(x):
        z = M @ np.asarray(x, float)
        return M.T @ gz_g(z)
    def hess(x):
        z = M @ np.asarray(x, float)
        return (M.T * gz_h(z)) @ M
    return f, grad, hess


def run_family(name, make, center=None):
    """center: for shifted CEC functions, x0 = center + U(-5,5) so that the
    scan radius max(10,|x0|+margin) covers >= 2 periods of T~19.5; for
    zero-centered synthetic families (center=None) keep x0 ~ U(-5,5) which
    already covers T<=10."""
    f, grad, hess = make()
    rows = []
    for sd in SEEDS:
        rng = np.random.RandomState(sd)
        if center is None:
            x0 = rng.uniform(-5.0, 5.0, D)
        else:
            x0 = np.asarray(center, float) + rng.uniform(-5.0, 5.0, D)
        t1 = time.time()
        x_cand, status = estimate_and_pcu(f, grad, hess, x0, tol=1e-4)
        hit = status == 'struct_pcu_ok' and fval(f, x_cand) < 1e-4
        rows.append({'seed': sd, 'trigger': status == 'struct_pcu_ok', 'hit': hit,
                     'F': fval(f, x_cand) if x_cand is not None else None,
                     'elapsed_s': round(time.time() - t1, 2)})
    out = {'n_runs': len(rows),
           'n_trigger': sum(1 for r in rows if r['trigger']),
           'n_hit': sum(1 for r in rows if r['hit']),
           'rows': rows}
    print('  %-16s %d/%d trigger, %d/%d hit' % (name, out['n_trigger'], out['n_runs'],
                                                out['n_hit'], out['n_runs']), flush=True)
    return out


def main():
    t0 = time.time()
    out = {'D': D, 'n_runs': len(SEEDS), 'seeds': [SEEDS[0], SEEDS[-1]]}
    out['f5_orth'] = run_family('f5_orth', lambda: make_rastrigin(4), center=shifts[4][:D])
    out['f8_orth'] = run_family('f8_orth', lambda: make_ncr_rastrigin(7), center=shifts[7][:D])
    out['synth_T10'] = run_family('synth_T10', make_synth)
    out['sin2_bowl'] = run_family('sin2_bowl', make_sin2)
    out['scaled_T0.714'] = run_family('scaled_T0.714', make_scaled)
    out['weak_A0.01'] = run_family('weak_A0.01', make_weak)
    out['quasi_2freq'] = run_family('quasi_2freq', make_quasi)
    out['elapsed_s'] = round(time.time() - t0, 1)
    json.dump(out, io.open(os.path.join(RESULTS, 'pcu_50runs.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    n_hit_total = sum(v['n_hit'] for k, v in out.items() if isinstance(v, dict) and 'n_trigger' in v)
    print('TOTAL hits %d/%d problems | %.1fs' % (n_hit_total, 7 * len(SEEDS), out['elapsed_s']), flush=True)
    print('SAVED pcu_50runs.json', flush=True)


if __name__ == '__main__':
    main()
