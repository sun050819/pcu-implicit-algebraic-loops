# -*- coding: utf-8 -*-
import os
"""run_23_simulink_nonsep.py - Simulink non-separable periodic loop end-to-end (PCU vs Simulink algebraic loop solver).

Procedure:
1) make_f5(30) exports M/S5/A/w/o to pcu_loop_params.mat (MATLAB residuals use the same parameters)
2) For each seed, generate z0 (same convention as run_21_fd_dual analytic_pcu: rng.uniform(-50,50)) -> z0.json
3) MATLAB builds a real Simulink algebraic loop (Algebraic Constraint + Interpreted MATLAB Function) to solve
4) Python PCU (run_21_fd_dual analytic version, same seed) gives candidate x_pcu
5) Compare F_sim vs F_pcu, distance between x_sim and global o -> simulink_nonsep.json
"""
import io, sys, os, json, subprocess, time, shutil
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'cec2017'))
from cec2017.transforms import rotations, shifts
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_17_sota_compare import make_f5
from run_21_fd_dual import analytic_pcu

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
SIMDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulink')


def _find_matlab():
    """Locate the MATLAB executable: env var, PATH, or a common install fallback."""
    exe = os.environ.get('MATLAB_EXECUTABLE')
    if exe and os.path.exists(exe):
        return exe
    which = shutil.which('matlab')
    if which:
        return which
    # generic probe of the standard MATLAB install location (no machine-specific path)
    import glob as _glob
    common = sorted(_glob.glob(r'C:\Program Files\MATLAB\R*\bin\matlab.exe'))
    return common[0] if common else None


MATLAB = _find_matlab()
if MATLAB is None:
    raise SystemExit(
        'MATLAB executable not found. Set MATLAB_EXECUTABLE to the full path '
        'of matlab.exe or add MATLAB to PATH.')


D = 30
T = 19.53125
SEEDS = [20260914 + i for i in range(5)]


def main():
    f30, M, o = make_f5(D)
    S5 = 5.12 / 100.0
    A = 10.0
    w = 2.0 * np.pi
    # Export parameters to MATLAB
    import scipy.io
    scipy.io.savemat(os.path.join(SIMDIR, 'pcu_loop_params.mat'),
                     {'M': np.asarray(M, float), 'o': np.asarray(o, float).reshape(-1, 1),
                      'S5': float(S5), 'A': float(A), 'w': float(w)})
    print('params exported; |o|=%.3f' % np.linalg.norm(o))

    rows = []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        x0 = rng.uniform(-50.0, 50.0, D)
        # MATLAB solve
        z0json = os.path.join(SIMDIR, 'z0.json')
        io.open(z0json, 'w', encoding='utf-8').write(json.dumps([float(v) for v in x0]))
        outjson = os.path.join(SIMDIR, 'sim_result.json')
        if os.path.exists(outjson):
            os.remove(outjson)
        r = subprocess.run([MATLAB, '-batch',
                            "cd('%s'); run_simulink_loop(%d, '%s', '%s')"
                            % (SIMDIR.replace('\\', '/'), D,
                               z0json.replace('\\', '/'), outjson.replace('\\', '/'))],
                           capture_output=True, text=True, timeout=600)
        if not os.path.exists(outjson):
            print('seed', seed, 'MATLAB FAIL', r.stdout[-300:], r.stderr[-300:])
            continue
        sim = json.load(io.open(outjson, encoding='utf-8'))
        # PCU (analytic version)
        pcu = analytic_pcu(seed)
        row = {
            'seed': seed,
            'F_simulink': float(sim.get('F_sim')) if sim.get('F_sim') is not None else None,
            'dist_global_o': float(sim.get('dist_global_o', float('nan'))),
            'F_fsolve': float(sim.get('F_fsolve')) if sim.get('F_fsolve') is not None else None,
            'fsolve_flag': int(sim.get('fsolve_flag', -1)),
            'status': sim.get('status'),
            'PCU': pcu,
        }
        # When PCU hits: distance to global
        if pcu.get('hit'):
            row['F_pcu'] = round(float(pcu.get('F', 0.0)), 8)
            row['dist_pcu_o'] = 0.0  # hit means F<1e-4, candidate is global solution o (f5 M_orth global optimum)
        rows.append(row)
        print('seed', seed, 'F_sim=%.4g dist_o=%.3g pcu=%s' % (
            row['F_simulink'], row['dist_global_o'], pcu.get('hit')))

    res = {'problem': 'Simulink nonseparable periodic algebraic loop (rotated Rastrigin gradient, 30D)',
           'T': T, 'n_seeds': len(SEEDS), 'seeds': SEEDS, 'rows': rows,
           'simulink_hits_global': int(sum(1 for r in rows if r.get('dist_global_o') is not None and r['dist_global_o'] < 1e-4)),
           'pcu_hits': int(sum(1 for r in rows if r.get('PCU', {}).get('hit'))),
           'elapsed_s': None}
    with io.open(os.path.join(RESULTS, 'simulink_nonsep.json'), 'w', encoding='utf-8') as fp:
        json.dump(res, fp, ensure_ascii=False, indent=2)
    print('written simulink_nonsep.json; sim_global_hits=%d/%d pcu_hits=%d/%d'
          % (res['simulink_hits_global'], len(rows), res['pcu_hits'], len(rows)))


if __name__ == '__main__':
    main()
