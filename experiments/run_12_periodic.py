# -*- coding: utf-8 -*-
import os
"""run_12_periodic: CEC2017 periodic family analytic verification (F4/F5 Rastrigin, F9 Levy orthogonalization).

Background: In run_11_fulltable full-table black-box, f4 rejection = FD numerical fragility under low-amplitude degradation
(the analytic version of f4 is already 10/10). This script uses analytic g/h to verify that after orthogonalization of the periodic family
the structure is identifiable (unaffected by FD numerical noise).
F9 Levy: sin^2 structure, w=1+0.25(x-1), period 4 (in x space) - verify
"trigger condition = separable periodic peak (both cos and sin^2 work)".
Output: results/cec_periodic_analytic.json
"""
import io, sys, os, json, time
import numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'cec2017'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cec2017.transforms import rotations, shifts
from aloop.solve.struct_id import estimate_and_pcu

D = 30
N = 10
SEED0 = 20260916
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
S5 = 5.12 / 100.0

def make_rastrigin(idx):
    """f4/f5 are Rastrigin-type (orthogonalized M)."""
    o = shifts[idx][:D].copy()
    Mr = rotations[D][idx].copy()
    u, _, vt = np.linalg.svd(Mr)
    M = u @ vt
    def f(x):
        z = S5*(M @ (np.asarray(x, float)-o))
        return float(np.sum(z*z - 10.0*np.cos(2.0*np.pi*z) + 10.0))
    def g(x):
        x = np.asarray(x, float); z = S5*(M@(x-o))
        gz = S5*(2.0*z + 20.0*np.pi*np.sin(2.0*np.pi*z))
        return M.T @ gz
    def h(x):
        x = np.asarray(x, float); z = S5*(M@(x-o))
        Hzz = S5*S5*(2.0 + 40.0*np.pi*np.pi*np.cos(2.0*np.pi*z))
        return (M.T * Hzz) @ M
    return f, g, h

def make_levy(idx=8):
    """CEC2017 F9 Levy: w=1+0.25(x-1), sin^2 periodic term (period 4 in x)."""
    o = shifts[idx][:D].copy()
    Mr = rotations[D][idx].copy()
    u, _, vt = np.linalg.svd(Mr)
    M = u @ vt
    def f(x):
        x = np.asarray(x, float)
        z = M @ (x - o)
        w = 1.0 + 0.25*(z - 1.0)
        t1 = np.sin(np.pi*w[0])**2
        t3 = ((w[-1]-1)**2) * (1 + np.sin(2.0*np.pi*w[-1])**2)
        wi = w[:-1]
        sm = np.sum((wi-1)**2 * (1 + 10*np.sin(np.pi*wi + 1)**2))
        return float(t1 + sm + t3)
    def g(x):
        x = np.asarray(x, float); z = M @ (x - o)
        w = 1.0 + 0.25*(z - 1.0)
        # d/dz_i = 0.25 * d/dw_i
        gz = np.zeros(D)
        # term1: sin^2(piw0) -> pi sin(2piw0)
        gz[0] += 0.25 * np.pi * np.sin(2.0*np.pi*w[0])
        # term3: (w{n-1}-1)^2(1+sin^2(2piw{n-1})) -> 2(w-1)(1+s^2) + (w-1)^2.2pi sin(4piw)
        wl = w[-1]
        s2 = np.sin(2.0*np.pi*wl)**2
        gz[-1] += 0.25 * (2.0*(wl-1)*(1+s2) + (wl-1)**2 * 2.0*np.pi*np.sin(4.0*np.pi*wl))
        # sm: Sigma_{i<n} (w_i-1)^2(1+10 sin^2(piw_i+1))
        for i in range(D-1):
            wi = w[i]
            s2i = np.sin(np.pi*wi + 1.0)**2
            gz[i] += 0.25 * (2.0*(wi-1)*(1+10*s2i) + (wi-1)**2 * 10.0*np.pi*np.sin(2.0*(np.pi*wi+1.0)))
        return M.T @ gz
    def h(x):
        x = np.asarray(x, float); z = M @ (x - o)
        w = 1.0 + 0.25*(z - 1.0)
        Hzz = np.zeros((D, D))
        # term1: d^2/dw^2 sin^2(piw) = 2pi^2 cos(2piw)
        Hzz[0, 0] += 0.25**2 * 2.0*np.pi*np.pi*np.cos(2.0*np.pi*w[0])
        # term3: d^2/dw^2 [(w-1)^2(1+sin^2(2piw))]
        wl = w[-1]
        s2 = np.sin(2.0*np.pi*wl)**2
        s4 = np.sin(4.0*np.pi*wl)
        d1 = 2.0*(wl-1)*(1+s2) + (wl-1)**2 * 2.0*np.pi*s4
        d2 = 2.0*(1+s2) + 2.0*(wl-1)*2.0*np.pi*s4 + 2.0*(wl-1)*2.0*np.pi*s4 + (wl-1)**2*2.0*np.pi*2.0*np.pi*np.cos(4.0*np.pi*wl)
        Hzz[-1, -1] += 0.25**2 * d2
        # sm diagonal
        for i in range(D-1):
            wi = w[i]
            s2i = np.sin(np.pi*wi + 1.0)**2
            s4i = np.sin(2.0*(np.pi*wi + 1.0))
            d2i = 2.0*(1+10*s2i) + 2.0*(wi-1)*10.0*np.pi*s4i + 2.0*(wi-1)*10.0*np.pi*s4i + (wi-1)**2*10.0*2.0*np.pi*np.pi*np.cos(2.0*(np.pi*wi+1.0))
            Hzz[i, i] += 0.25**2 * d2i
        return (M.T * Hzz) @ M
    return f, g, h

def run_case(name, f, g, h, f_opt=0.0):
    trig = hits = 0
    T_ests = []
    for k in range(N):
        rng = np.random.RandomState(SEED0 + k)
        x0 = rng.uniform(-100.0, 100.0, D)
        xc, tag = estimate_and_pcu(f, g, h, x0, margin=100.0)
        if tag == 'struct_pcu_ok':
            trig += 1
            Fv = f(xc)
            if Fv < 1e-4:
                hits += 1
    return {'n': N, 'trigger': trig, 'hit': hits, 'hit_rate': hits/N}

def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    t0 = time.time()
    res = {}
    # F4 / F5 Rastrigin
    for name, idx in [('F4_Rastrigin', 3), ('F5_Rastrigin', 4)]:
        f, g, h = make_rastrigin(idx)
        res[name] = run_case(name, f, g, h)
        print(name, res[name], flush=True)
    # F9 Levy (sin^2 periodic, period 4)
    f, g, h = make_levy(8)
    res['F9_Levy'] = run_case('F9_Levy', f, g, h)
    print('F9_Levy', res['F9_Levy'], flush=True)
    res['_meta'] = {'n_runs': N, 'seed0': SEED0, 'elapsed_s': round(time.time()-t0, 1),
                    'note': 'analytic version; F4/F5 period 19.53 (S5 scaling), F9 period 4 (0.25 scaling, sin^2 structure)'}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, 'cec_periodic_analytic.json'), 'w', encoding='utf-8') as fp:
        json.dump(res, fp, ensure_ascii=False, indent=2)
    print('done')

if __name__ == '__main__':
    main()
