# -*- coding: utf-8 -*-
"""run_37_periodic_class3.py

Third generalization batch (responding to "positive examples too concentrated
on Rastrigin; add multi-frequency / quasi-periodic / scaled / non-orthogonal
rotation cases"):

  E) Harmonic consensus period: f = sum_k a_k (1-cos(k*2*pi*z/T)) with k=1,2,3
     harmonics sharing the base period T=1 (base frequency dominant in the
     Hessian spectrum). A true multi-frequency consensus-period function.
  F) Scaled period: standard rotated Rastrigin with coordinate scaling s=1.4,
     so the consensus period is T=1/s ~= 0.7143 (non-integer period; stresses
     the FFT period estimator and the Remark-1 bin-quantization bound).
  G) Quasi-periodic two incommensurate frequencies (equal Hessian amplitude):
     no dominant period exists, yet PCU still recovers the true solution
     (H is globally maximal and g=0 at the true offset, so the peak +
     zero-crossing + verification chain is unaffected by the incommensurate
     secondary frequency). A robustness positive example, complementary to
     the Griewank negative (whose period genuinely varies across dimensions).
  H) Non-orthogonal linear transform: L = M_orth @ diag(1+0.1*u), u ~ U(-1,1);
     Hessian is no longer orthogonally diagonalizable; expected boundary
     (either partial trigger or honest rejection).
  I) Weak-amplitude consensus period: A=0.01 periodic term on a 0.5*z^2 bowl
     (Hessian modulation 3.9% of baseline); weak-but-detectable boundary.

10 seeds each, 30D. Output results/periodic_class3.json
"""
import io, sys, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'cec2017'))
from aloop.solve.struct_id import estimate_and_pcu
from cec2017.transforms import rotations

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
D = 30
SEEDS = list(range(20260912, 20260922))  # 10 seeds

Msyn = rotations[D][4]
u, _, vt = np.linalg.svd(Msyn)
Mrot = u @ vt  # fixed orthogonal rotation, consistent with run_29/run_31


def fval(f, x):
    return float(np.asarray(f(np.atleast_2d(x))).reshape(-1)[0])


def orth_wrap(gz_f, gz_g, gz_h, o=None, M=Mrot):
    """z = M (x - o); gradient/hessian pulled back through M.T."""
    oo = np.zeros(D) if o is None else np.asarray(o, float)
    def f(X):
        Xa = np.atleast_2d(np.asarray(X, float))
        Z = (Xa - oo) @ M.T
        return gz_f(Z)
    def grad(x):
        z = M @ (np.asarray(x, float) - oo)
        return M.T @ gz_g(z)
    def hess(x):
        z = M @ (np.asarray(x, float) - oo)
        diag = gz_h(z)
        return M.T @ (diag[:, None] * M)
    return f, grad, hess


# ---- E) harmonic consensus period (T=1, base frequency dominant) ----
def make_harmonic():
    T = 1.0
    w = 2.0 * np.pi / T
    a = np.array([0.5, 0.05, 0.02])      # amplitudes of k=1,2,3 harmonics
    ks = np.array([1.0, 2.0, 3.0])
    def gz_f(Z):
        Z = np.asarray(Z, float)
        acc = np.zeros_like(Z)
        for ak, k in zip(a, ks):
            acc = acc + ak * (1.0 - np.cos(k * w * Z))
        return np.sum(acc, axis=1) + 0.05 * np.sum(Z * Z, axis=1)
    def gz_g(z):
        z = np.asarray(z, float)
        g = 0.1 * z
        for ak, k in zip(a, ks):
            g = g + ak * k * w * np.sin(k * w * z)
        return g
    def gz_h(z):
        z = np.asarray(z, float)
        h = 0.1 * np.ones_like(z)
        for ak, k in zip(a, ks):
            h = h + ak * (k * w) ** 2 * np.cos(k * w * z)
        return h
    return orth_wrap(gz_f, gz_g, gz_h)


# ---- F) scaled-period Rastrigin (s=1.4 -> T=1/1.4) ----
def make_scaled():
    s = 1.4
    w = 2.0 * np.pi * s
    def gz_f(Z):
        Z = np.asarray(Z, float)
        return np.sum((s * Z) ** 2 - 10.0 * np.cos(w * Z) + 10.0, axis=1)
    def gz_g(z):
        z = np.asarray(z, float)
        return 2.0 * s * s * z + 10.0 * w * np.sin(w * z)
    def gz_h(z):
        z = np.asarray(z, float)
        return 2.0 * s * s + 10.0 * w * w * np.cos(w * z)
    return orth_wrap(gz_f, gz_g, gz_h)


# ---- G) quasi-periodic incommensurate frequencies (no consensus) ----
def make_quasi():
    # Equal Hessian amplitude at both frequencies (a1*(2pi)^2 == a2*(2pi*sqrt2)^2):
    # no dominant frequency; robustness case (true solution still recovered).
    w1 = 2.0 * np.pi
    w2 = 2.0 * np.pi * np.sqrt(2.0)
    a1 = 2.0
    a2 = 1.0   # a2*w2^2 = 1*8pi^2 = a1*w1^2 = 2*4pi^2
    def gz_f(Z):
        Z = np.asarray(Z, float)
        return np.sum(a1 * (1.0 - np.cos(w1 * Z)) + a2 * (1.0 - np.cos(w2 * Z)), axis=1) + 0.1 * np.sum(Z * Z, axis=1)
    def gz_g(z):
        z = np.asarray(z, float)
        return a1 * w1 * np.sin(w1 * z) + a2 * w2 * np.sin(w2 * z) + 0.2 * z
    def gz_h(z):
        z = np.asarray(z, float)
        return a1 * w1 * w1 * np.cos(w1 * z) + a2 * w2 * w2 * np.cos(w2 * z) + 0.2
    return orth_wrap(gz_f, gz_g, gz_h)


# ---- H) non-orthogonal linear transform ----
def make_nonorth():
    rng = np.random.RandomState(20260912)
    S = np.diag(1.0 + 0.1 * rng.uniform(-1.0, 1.0, D))
    L = Mrot @ S
    T = 1.0
    w = 2.0 * np.pi / T
    def f(X):
        Xa = np.atleast_2d(np.asarray(X, float))
        Z = Xa @ L.T
        return np.sum(Z * Z - 10.0 * np.cos(w * Z) + 10.0, axis=1)
    def grad(x):
        z = L @ np.asarray(x, float)
        return L.T @ (2.0 * z + 10.0 * w * np.sin(w * z))
    def hess(x):
        z = L @ np.asarray(x, float)
        diag = 2.0 + 10.0 * w * w * np.cos(w * z)
        return L.T @ (diag[:, None] * L)
    return f, grad, hess


# ---- I) weak-amplitude consensus period ----
def make_weak():
    A = 0.01
    w = 2.0 * np.pi
    def gz_f(Z):
        Z = np.asarray(Z, float)
        return np.sum(A * (1.0 - np.cos(w * Z)) + 0.5 * Z * Z, axis=1)
    def gz_g(z):
        z = np.asarray(z, float)
        return A * w * np.sin(w * z) + z
    def gz_h(z):
        z = np.asarray(z, float)
        return A * w * w * np.cos(w * z) + 1.0
    return orth_wrap(gz_f, gz_g, gz_h)


# ---- J) sin^2-periodic bowl (non-Rastrigin, non-cos periodic structure) ----
def make_sin2():
    w = 2.0 * np.pi
    def gz_f(Z):
        Z = np.asarray(Z, float)
        return np.sum(np.sin(0.5 * w * Z) ** 2 + 0.2 * Z * Z, axis=1)
    def gz_g(z):
        z = np.asarray(z, float)
        return 0.5 * w * np.sin(w * z) + 0.4 * z
    def gz_h(z):
        z = np.asarray(z, float)
        return 0.5 * w * w * np.cos(w * z) + 0.4
    return orth_wrap(gz_f, gz_g, gz_h)


def run_family(name, make, x0_rng=None):
    f, grad, hess = make()
    rows = []
    for sd in SEEDS:
        rng = np.random.RandomState(sd)
        x0 = rng.uniform(-5.0, 5.0, D)
        t1 = time.time()
        x_cand, status = estimate_and_pcu(f, grad, hess, x0, tol=1e-4)
        hit = status == 'struct_pcu_ok' and fval(f, x_cand) < 1e-4
        rows.append({'seed': sd, 'trigger': status == 'struct_pcu_ok', 'hit': hit,
                     'F': fval(f, x_cand) if x_cand is not None else None,
                     'elapsed_s': round(time.time() - t1, 1)})
        print('%s seed %d: trigger=%s hit=%s F=%s' % (name, sd, status == 'struct_pcu_ok', hit,
              None if x_cand is None else '%.3g' % fval(f, x_cand)), flush=True)
    return {'n_runs': len(rows),
            'n_trigger': sum(1 for r in rows if r['trigger']),
            'n_hit': sum(1 for r in rows if r['hit']),
            'rows': rows}


def main():
    t0 = time.time()
    out = {'D': D, 'seeds': SEEDS}
    out['harmonic'] = run_family('harmonic', make_harmonic)
    out['sin2_periodic'] = run_family('sin2', make_sin2)
    out['scaled_T0.714'] = run_family('scaled', make_scaled)
    out['quasi_no_consensus'] = run_family('quasi', make_quasi)
    out['nonorth_10pct'] = run_family('nonorth', make_nonorth)
    out['weak_A0.01'] = run_family('weak', make_weak)
    out['elapsed_s'] = round(time.time() - t0, 1)
    json.dump(out, io.open(os.path.join(RESULTS, 'periodic_class3.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    for k in ('harmonic', 'sin2_periodic', 'scaled_T0.714', 'quasi_no_consensus', 'nonorth_10pct', 'weak_A0.01'):
        v = out[k]
        print('  %s: %d/%d trigger, %d/%d hit' % (k, v['n_trigger'], v['n_runs'], v['n_hit'], v['n_runs']), flush=True)
    print('SAVED periodic_class3.json | %.1fs' % out['elapsed_s'], flush=True)


if __name__ == '__main__':
    main()
