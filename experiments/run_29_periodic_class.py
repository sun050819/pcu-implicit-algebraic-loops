# -*- coding: utf-8 -*-
import os
"""run_29_periodic_class.py
Generalization of the applicability domain (removing the "Rastrigin-only" label):
  A) Synthetic consensus-periodic function (non-Rastrigin structure): f = Sigma[2(1-cos(2piz/T)) + 0.1 z^2], T=10, 30D orthogonal rotation
     verify that PCU holds for the "consensus-periodic multimodal class" (not Rastrigin-specific).
  B) Griewank 30D orthogonal rotation: period diverges with dimension (T_k = 2*pi*sqrt(k)), verifying that the consensus-period assumption is not satisfied -> honest boundary.
Output results/periodic_class.json
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
T_SYN = 10.0
rng0 = np.random.RandomState(7)
Msyn = rotations[D][4]
u, _, vt = np.linalg.svd(Msyn)
Mrot = u @ vt  # orthogonal rotation


# ---------------- Part A: Synthetic consensus-periodic function ----------------
def make_synth():
    T = T_SYN
    M = Mrot
    o = np.zeros(D)
    def f(X):
        Xa = np.atleast_2d(np.asarray(X, float))
        Z = (Xa - o) @ M.T
        return np.sum(2.0 * (1.0 - np.cos(2.0 * np.pi * Z / T)) + 0.1 * Z * Z, axis=1)
    def grad(x):
        z = M @ (np.asarray(x, float) - o)
        gz = 2.0 * (2.0 * np.pi / T) * np.sin(2.0 * np.pi * z / T) + 0.2 * z
        return M.T @ gz
    def hess(x):
        z = M @ (np.asarray(x, float) - o)
        diag = 2.0 * (2.0 * np.pi / T) ** 2 * np.cos(2.0 * np.pi * z / T) + 0.2
        return M.T @ (diag[:, None] * M)
    return f, grad, hess


def synth_fval(f, x):
    return float(np.asarray(f(np.atleast_2d(x))).reshape(-1)[0])


# ---------------- Part B: Griewank 30D orthogonal rotation ----------------
def make_griewank():
    M = Mrot
    o = np.zeros(D)
    sq = np.sqrt(np.arange(1, D + 1)).astype(float)
    def f(X):
        Xa = np.atleast_2d(np.asarray(X, float))
        Z = (Xa - o) @ M.T
        z2 = Z * Z
        prod = np.prod(np.cos(Z / sq), axis=1)
        return 1.0 + np.sum(z2, axis=1) / 4000.0 - prod
    def grad(x):
        z = M @ (np.asarray(x, float) - o)
        P = np.prod(np.cos(z / sq))
        gz = np.empty(D)
        for k in range(D):
            Pk = P / np.cos(z[k] / sq[k]) if abs(np.cos(z[k] / sq[k])) > 1e-12 else 0.0
            gz[k] = z[k] / 2000.0 + (1.0 / sq[k]) * np.sin(z[k] / sq[k]) * Pk
        return M.T @ gz
    def hess(x):
        z = M @ (np.asarray(x, float) - o)
        cosz = np.cos(z / sq)
        sinz = np.sin(z / sq)
        P = np.prod(cosz)
        Hz = np.zeros((D, D))
        for k in range(D):
            Pk = P / cosz[k] if abs(cosz[k]) > 1e-12 else 0.0
            Hz[k, k] = 1.0 / 2000.0 + (1.0 / (sq[k] * sq[k])) * cosz[k] * Pk
            for l in range(k + 1, D):
                Pkl = P / (cosz[k] * cosz[l]) if abs(cosz[k] * cosz[l]) > 1e-12 else 0.0
                v = (1.0 / (sq[k] * sq[l])) * sinz[k] * sinz[l] * Pkl
                Hz[k, l] = Hz[l, k] = v
        return M.T @ Hz @ M
    return f, grad, hess


def run_case(name, make_fun, n_seeds=10):
    f, grad, hess = make_fun()
    rows = []
    for seed in SEEDS[:n_seeds]:
        rng = np.random.RandomState(seed)
        x0 = rng.uniform(-25.0, 25.0, D)
        t0 = time.time()
        try:
            xc, tag = estimate_and_pcu(f, grad, hess, x0, tol=1e-5, margin=60.0)
        except Exception as e:
            rows.append({'seed': seed, 'trigger': False, 'error': str(e)[:80]})
            print(f'  {name} seed {seed}: error {str(e)[:60]}', flush=True)
            continue
        if tag is None:
            # Not triggered: quantify the dispersion of per-dimension period estimates (evidence for consensus)
            cv, n_main = measure_period_spread(hess, x0)
            rows.append({'seed': seed, 'trigger': False, 'cv_T': cv, 'n_main_peaks': n_main})
            print(f'  {name} seed {seed}: NO trigger (CV_T={cv:.3f}, n_main={n_main})', flush=True)
        else:
            F = synth_fval(f, xc) if name.startswith('synth') else float(np.asarray(f(np.atleast_2d(xc))).reshape(-1)[0])
            hit = F < 1e-4
            rows.append({'seed': seed, 'trigger': True, 'hit': bool(hit), 'F': float(F)})
            print(f'  {name} seed {seed}: TRIGGER hit={hit} F={F:.3g}', flush=True)
    trig = sum(1 for r in rows if r.get('trigger'))
    hits = sum(1 for r in rows if r.get('hit'))
    return {'n_runs': len(rows), 'n_trigger': trig, 'n_hit': hits, 'rows': rows}


def measure_period_spread(hess, x0):
    """Along the unrotated path at x0, per-dimension FFT dominant period of the diagonal Hessian: consensus characterized by coefficient of variation CV."""
    from aloop.solve.struct_id import _coarse_period, _scan_dim, _parabolic_peak
    d = len(x0)
    periods = []
    n_main = []
    for i in range(d):
        try:
            ts, hs = _scan_dim(hess, x0, i, radius=25.0, pts_per_period=64)
            T_est = _coarse_period(ts, hs)
            periods.append(T_est if T_est else None)
            n_main.append(1 if T_est else 0)
        except Exception:
            periods.append(None)
            n_main.append(0)
    valid = [p for p in periods if p and p > 0]
    if len(valid) < d * 0.5:
        return float('nan'), 0
    cv = float(np.std(valid) / np.mean(valid))
    return cv, len(valid)


def main():
    t0 = time.time()
    out = {}
    print('Part A: synth consensus-period (non-Rastrigin), 30D rotated', flush=True)
    out['synth_consensus_period'] = run_case('synth', make_synth)
    print('Part B: Griewank 30D rotated (period divergence)', flush=True)
    out['griewank'] = run_case('griewank', make_griewank)
    out['elapsed_s'] = round(time.time() - t0, 1)

    def _sanitize(o):
        # Standard JSON has no NaN/Infinity: missing period estimates are emitted as null.
        if isinstance(o, dict):
            return {k: _sanitize(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_sanitize(v) for v in o]
        if isinstance(o, float) and (o != o or o in (float('inf'), float('-inf'))):
            return None
        return o

    json.dump(_sanitize(out), io.open(os.path.join(RESULTS, 'periodic_class.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    a, b = out['synth_consensus_period'], out['griewank']
    print('SAVED periodic_class.json | synth: %d/%d trigger, %d hit | griewank: %d/%d trigger, %d hit'
          % (a['n_trigger'], a['n_runs'], a['n_hit'], b['n_trigger'], b['n_runs'], b['n_hit']), flush=True)


if __name__ == '__main__':
    main()
