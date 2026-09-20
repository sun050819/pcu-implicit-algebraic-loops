# -*- coding: utf-8 -*-
import os
"""run_31_periodic_class2.py
Extended applicability-domain validation (responding to "positive examples too concentrated on Rastrigin"):
  C) Ackley 30D orthogonally rotated (cos(2piz_i) consensus period, T=1) -- second non-Rastrigin positive-example candidate
  D) Schaffer F6 30D (sin^2(sqrt(Sigmaz^2) radial period, non-coordinate consensus) -- extended negative-example boundary
Output results/periodic_class2.json
"""
import io, sys, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'cec2017'))
from aloop.solve.struct_id import estimate_and_pcu, identify_structure
from cec2017.transforms import rotations

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
D = 30
SEEDS = list(range(20260912, 20260922))  # 10 seeds

# Orthogonal rotation matrix (consistent with run_29_periodic_class: CEC2017 official 30D rotation + SVD orthogonalization)
Msyn = rotations[D][4]
u, _, vt = np.linalg.svd(Msyn)
Mrot = u @ vt


def measure_period_spread(hess, x0, T_cand=None, n=128, R=None):
    """Same as run_29_periodic_class: dispersion CV of per-dimension FFT period estimates and number of dominant peaks."""
    from scipy.fft import rfft, rfftfreq
    x = np.atleast_1d(np.asarray(x0, float))
    h = 0.05
    w = max(2.0, 4.0)
    periods = []
    n_main = 0
    for i in range(len(x)):
        xs = np.linspace(x[i] - w, x[i] + w, n)
        diag = np.array([hess(np.where(np.arange(len(x)) == i, v, x))[i, i] for v in xs])
        spec = np.abs(rfft(diag - diag.mean()))
        spec[0] = 0.0
        freqs = rfftfreq(n, d=2 * w / n)
        idx = np.argpartition(spec, -3)[-3:]
        idx = idx[spec[idx] > 0.05 * spec.max()]
        if len(idx) == 0:
            continue
        f_main = freqs[idx[np.argmax(spec[idx])]]
        if f_main > 0:
            periods.append(1.0 / f_main)
        n_main = max(n_main, len(idx))
    if not periods:
        return 0.0, 0
    cv = float(np.std(periods) / np.mean(periods))
    return cv, n_main


# ---------------- Part C: Ackley 30D orthogonal rotation (cos term consensus period T=1) ----------------
def make_ackley():
    M = Mrot
    o = np.zeros(D)
    def f(X):
        Xa = np.atleast_2d(np.asarray(X, float))
        Z = (Xa - o) @ M.T
        s1 = np.sum(Z * Z, axis=1) / D
        s2 = np.sum(np.cos(2.0 * np.pi * Z), axis=1) / D
        return -20.0 * np.exp(-0.2 * np.sqrt(s1)) - np.exp(s2) + 20.0 + np.e
    def grad(x):
        z = M @ (np.asarray(x, float) - o)
        r = np.sqrt(np.sum(z * z) / D)
        gz = (4.0 / D) * np.exp(-0.2 * r) / max(r, 1e-12) * z \
             + (2.0 * np.pi / D) * np.exp(np.sum(np.cos(2.0 * np.pi * z)) / D) * np.sin(2.0 * np.pi * z)
        return M.T @ gz
    def hess(x):
        z = M @ (np.asarray(x, float) - o)
        r = np.sqrt(np.sum(z * z) / D)
        c0 = np.exp(-0.2 * r)
        c1 = np.exp(np.sum(np.cos(2.0 * np.pi * z)) / D)
        H = np.zeros((D, D))
        # Envelope term Hessian (analytically derived: 4/(Dr).e^{-0.2r}.I - 4.8/(D^2r^3).e^{-0.2r}.zzT)
        if r > 1e-12:
            H += (4.0 / (D * r)) * c0 * np.eye(D)
            H -= (4.8 / (D * D * r ** 3)) * c0 * np.outer(z, z)
        # cos term Hessian: diagonal (4pi^2/D)e^{s}cos(2piz_i), cross - (2pi/D)^2e^{s}sin(2piz_i)sin(2piz_j)
        kd = (4.0 * np.pi * np.pi / D) * c1
        kc = (2.0 * np.pi / D) ** 2 * c1
        H += kd * np.diag(np.cos(2.0 * np.pi * z))
        H -= kc * np.outer(np.sin(2.0 * np.pi * z), np.sin(2.0 * np.pi * z))
        return M.T @ H @ M
    return f, grad, hess


# ---------------- Part D: Schaffer F6 30D (sin^2(sqrt(Sigmaz^2), radial period) ----------------
def make_schaffer():
    M = Mrot
    o = np.zeros(D)
    def f(X):
        Xa = np.atleast_2d(np.asarray(X, float))
        Z = (Xa - o) @ M.T
        s = np.sum(Z * Z, axis=1)
        return 0.5 + (np.sin(np.sqrt(s)) ** 2 - 0.5) / (1.0 + 0.001 * s) ** 2
    def grad(x):
        z = M @ (np.asarray(x, float) - o)
        s = np.sum(z * z)
        r = np.sqrt(s) if s > 1e-12 else 1e-12
        d = (1.0 + 0.001 * s) ** 2
        dg = (np.sin(2.0 * r) / r - 0.004 * np.sin(r) ** 2 / d) / d
        gz = dg * z
        return M.T @ gz
    def hess(x):
        z = M @ (np.asarray(x, float) - o)
        s = np.sum(z * z)
        r = np.sqrt(s) if s > 1e-12 else 1e-12
        d = 1.0 + 0.001 * s
        d2 = d * d
        # Radial-direction second-order: f''(r) along the z direction
        fr = (2.0 * np.cos(2.0 * r) * r - np.sin(2.0 * r)) / (r * r * d2) \
             - (0.008 * np.sin(2.0 * r) / (r * d2)) + (0.006 * np.sin(r) ** 2 / (d2 * d))
        f1 = (np.sin(2.0 * r) / r - 0.004 * np.sin(r) ** 2 / d2) / d2  # f'(r)/r
        H = (fr - f1) * np.outer(z, z) / s + f1 * np.eye(D)
        return M.T @ H @ M
    return f, grad, hess


def run_case(name, make_fun, n_seeds=10):
    f, grad, hess = make_fun()
    rows = []
    for seed in SEEDS[:n_seeds]:
        rng = np.random.RandomState(seed)
        x0 = rng.uniform(-25.0, 25.0, D)
        try:
            xc, tag = estimate_and_pcu(f, grad, hess, x0, tol=1e-5, margin=60.0)
        except Exception as e:
            rows.append({'seed': seed, 'trigger': False, 'error': str(e)[:80]})
            print(f'  {name} seed {seed}: error {str(e)[:60]}', flush=True)
            continue
        if tag is None:
            cv, n_main = measure_period_spread(hess, x0)
            rows.append({'seed': seed, 'trigger': False, 'cv_T': cv, 'n_main_peaks': n_main})
            print(f'  {name} seed {seed}: NO trigger (CV_T={cv:.3f}, n_main={n_main})', flush=True)
        else:
            F = float(np.asarray(f(np.atleast_2d(xc))).reshape(-1)[0])
            rows.append({'seed': seed, 'trigger': True, 'hit': bool(F < 1e-4), 'F': F, 'T_est': tag})
            print(f'  {name} seed {seed}: TRIGGER T={tag}, F={F:.2e}', flush=True)
    return {'n_runs': len(rows), 'n_trigger': sum(1 for r in rows if r.get('trigger')),
            'n_hit': sum(1 for r in rows if r.get('hit')), 'rows': rows}


def main():
    t0 = time.time()
    out = {}
    print('Part C: Ackley 30D rotated (consensus cos period T=1)', flush=True)
    out['ackley'] = run_case('ackley', make_ackley)
    print('Part D: Schaffer F6 30D rotated (radial period, non-coordinate)', flush=True)
    out['schaffer_f6'] = run_case('schaffer_f6', make_schaffer)
    out['elapsed_s'] = round(time.time() - t0, 1)
    json.dump(out, io.open(os.path.join(RESULTS, 'periodic_class2.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    for k in ('ackley', 'schaffer_f6'):
        v = out[k]
        print('  %s: %d/%d trigger, %d hit' % (k, v['n_trigger'], v['n_runs'], v['n_hit']), flush=True)
    print('SAVED periodic_class2.json | elapsed %.1fs' % out['elapsed_s'], flush=True)


if __name__ == '__main__':
    main()
