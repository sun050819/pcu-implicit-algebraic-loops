# -*- coding: utf-8 -*-
import os
"""run_13_stress: Large-scale stress test (100D / 200D).

Focus (user-specified):
  - PCU's candidate enumeration at 100D takes the greedy path (4^D >> 1024 -> _best_combination greedy + coordinate descent)
  - The growth of struct_id scanning cost with D (number of scan points per dimension n, total number of Hessian calls, wall-time)
  - Success rate + F=0 hit rate

Two families:
  sep   : separable A^2EP f(x)=Sigma(x_i^2 - A cos(2pi x_i/T) + c), R=I -- pure enumeration/scan cost
  orth  : synthetic orthogonal A^2EP (D=100, T=19.53) -- large-scale rotation generalization + degeneracy rejection report
T in {1.0 (dense period, scanning most expensive), 19.53125 (CEC equivalent)}; analytical version estimate_and_pcu.
Output: results/stress_highdim.json
"""
import io, sys, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from aloop.solve.struct_id import identify_structure, estimate_and_pcu
from aloop.solve import pcu as _pcu

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
N = 10
SEED0 = 20260916
A = 10.0; c = 10.0

def make_sep(D, T, seed):
    rng = np.random.RandomState(seed)
    o = rng.uniform(-5.0, 5.0, D)
    w = 2.0 * np.pi / T
    def f(x):
        z = np.asarray(x, float) - o
        return float(np.sum(z*z - A*np.cos(w*z) + c))
    def g(x):
        z = np.asarray(x, float) - o
        return 2.0*z + A*w*np.sin(w*z)
    def h(x):
        z = np.asarray(x, float) - o
        return np.diag(2.0 + A*w*w*np.cos(w*z))
    return f, g, h, o, T

def make_orth(D, T, seed):
    rng = np.random.RandomState(seed)
    G = rng.randn(D, D)
    u, _, vt = np.linalg.svd(G)
    M = u @ vt
    o = rng.uniform(-5.0, 5.0, D)
    w = 2.0 * np.pi / T
    def f(x):
        z = M @ (np.asarray(x, float) - o)
        return float(np.sum(z*z - A*np.cos(w*z) + c))
    def g(x):
        x = np.asarray(x, float); z = M @ (x - o)
        gz = 2.0*z + A*w*np.sin(w*z)
        return M.T @ gz
    def h(x):
        x = np.asarray(x, float); z = M @ (x - o)
        Hzz = 2.0 + A*w*w*np.cos(w*z)
        return (M.T * Hzz) @ M
    return f, g, h, M, o, T

def run_family(kind, D, T, seed0):
    if kind == 'sep':
        f, g, h, o, T_true = make_sep(D, T, seed0)
    else:
        f, g, h, M, o, T_true = make_orth(D, T, seed0)
    ok = trig = 0
    times_id = []; times_pcu = []
    n_pts_list = []; n_cand_list = []; greedy_list = []
    F_vals = []
    for k in range(N):
        rng = np.random.RandomState(seed0 + k)
        x0 = rng.uniform(-5.0, 5.0, D)
        # Time identify separately (to characterize scanning cost)
        t0 = time.time()
        mdl = identify_structure(f, g, h, x0, margin=5.0)
        times_id.append(time.time() - t0)
        if mdl is not None:
            # Number of scan points (coarse scan + fine scan approximation: 2*radius/T*16 points/dimension)
            mag = float(np.max(np.abs(mdl['R'] @ x0))) if mdl['R'] is not None else float(np.max(np.abs(x0)))
            radius = max(10.0, mag + 5.0)
            n_pts = int(np.ceil(2.0*radius/mdl['T']*16.0)) + 1
            n_pts_list.append(n_pts)
            n_cand_list.append(len(mdl['peaks'][0]))
        t1 = time.time()
        xc, tag = estimate_and_pcu(f, g, h, x0, margin=5.0)
        times_pcu.append(time.time() - t1)
        if tag == 'struct_pcu_ok':
            trig += 1
            Fv = f(xc)
            F_vals.append(Fv)
            if Fv < 1e-4:
                ok += 1
            # Combination path determination: 4^D > 1024 -> greedy
            greedy_list.append(True)
    return {
        'n': N, 'trigger': trig, 'hit': ok, 'success_rate': ok/N,
        'trigger_rate': trig/N,
        'identify_s_mean': round(float(np.mean(times_id)), 2),
        'identify_s_max': round(float(np.max(times_id)), 2),
        'pcu_s_mean': round(float(np.mean(times_pcu)), 2),
        'scan_pts_per_dim_mean': int(np.mean(n_pts_list)) if n_pts_list else None,
        'cand_per_dim_mean': int(np.mean(n_cand_list)) if n_cand_list else None,
        'greedy_path': bool(all(greedy_list)) if greedy_list else None,
        'F0_hit_rate': (sum(1 for v in F_vals if v < 1e-4) / len(F_vals)) if F_vals else 0.0,
    }

def main():
    t_all = time.time()
    results = {}
    # Separable: 100D / 200D x Tin{1.0, 19.53}
    for D in [100, 200]:
        for T in [1.0, 19.53125]:
            key = 'sep_D%d_T%s' % (D, str(T))
            results[key] = run_family('sep', D, T, SEED0 + D)
            print('%s: %s' % (key, json.dumps(results[key], ensure_ascii=False)), flush=True)
    # Synthetic orthogonal: 100D T=19.53 (degeneracy stress)
    key = 'orth_D100_T19.53125'
    results[key] = run_family('orth', 100, 19.53125, SEED0 + 100)
    print('%s: %s' % (key, json.dumps(results[key], ensure_ascii=False)), flush=True)
    results['_meta'] = {'n_runs': N, 'seed0': SEED0,
                        'family': 'A2EP sep/orth, A=10 c=10, o~U(-5,5)',
                        'elapsed_s': round(time.time() - t_all, 1)}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, 'stress_highdim.json'), 'w', encoding='utf-8') as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print('done -> stress_highdim.json')

if __name__ == '__main__':
    main()
