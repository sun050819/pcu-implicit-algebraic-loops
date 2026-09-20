# -*- coding: utf-8 -*-
import os
"""run_21_fd_dual.py - PCU FE dual criteria (G2b): FE count when analytical derivatives are available.

Criteria: R recovery for structure identification and Hessian diagonal scan use analytical derivatives (Simulink block derivatives/
    symbolic expression scenarios), candidate zero crossings use analytical gradients (0 f evaluations), only final candidate verification
    counts toward function evaluations. Compared with black-box finite difference full pipeline 396,993 FE.
Problem: CEC2017 F5 M_orth 30D, 10 seeds.
Output: results/pcu_fe_dual.json
"""
import io, sys, os, json
import numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'cec2017'))
from cec2017.transforms import rotations, shifts
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_17_sota_compare import make_f5

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
D = 30
T = 19.53125; A = 10.0; c = 10.0
SEEDS = [20260914 + i for i in range(10)]


def analytic_pcu(seed, margin=100.0):
    """Analytical PCU (30D f5 M_orth):
    - R: eigh analytical Hessian (with rotation recovery K=3 voting, 0 FE)
    - Scan: H_zz diagonal analytical (0 FE)
    - Candidates: analytical gradient zero crossings (0 FE) + cross-dimensional combinatorial greedy
    - Verification: true function evaluation (counts FE)
    Returns {hit, F, fe_verify} or {hit: False, reason}.
    """
    f30, M, o = make_f5(D)
    oM = o @ M.T
    S5 = 5.12 / 100.0
    rng = np.random.RandomState(seed)
    x0 = rng.uniform(-50.0, 50.0, D)

    def H_anal(x):
        z = S5 * (M @ (np.asarray(x, float) - o))
        w_eff = 2.0 * np.pi  # cos(2*pi*z), z=S5*M(x-o) already includes scaling
        diag = 2.0 + A * w_eff ** 2 * np.cos(w_eff * z)
        return M.T @ (diag[:, None] * M)  # row scaling = MT diag M (true Hessian)

    def grad_anal(x):
        z = S5 * (M @ (np.asarray(x, float) - o))
        w_eff = 2.0 * np.pi  # cos(2*pi*z), z=S5*M(x-o) already includes scaling
        gz = 2.0 * z + A * w_eff * np.sin(w_eff * z)
        return M.T @ (S5 * gz)

    # ---- R recovery: K=3 point voting ----
    pts = [x0, x0 + 1.0, x0 - 1.0]
    Hs = [H_anal(p) for p in pts]
    cands = []
    for Hj in Hs:
        _, evecs = np.linalg.eigh(Hj)
        cands.append(evecs.T)
    def off_diag(A):
        A = np.abs(A); np.fill_diagonal(A, 0.0)
        return float(np.sqrt(np.sum(A * A)))
    best_R, best_score = None, float('inf')
    for Rj in cands:
        score = sum(off_diag(Rj @ Hl @ Rj.T) for Hl in Hs)
        if score < best_score:
            best_score, best_R = score, Rj
    # diagonality determination
    H0 = H_anal(x0)
    if off_diag(H0) > 1e-6 * (1.0 + np.abs(H0).max()):
        R = best_R; base = R @ x0
    else:
        R = None; base = x0

    # ---- Scan: analytical diagonal Hessian curve ----
    radius = max(10.0, float(np.max(np.abs(base))) + margin)
    z = base
    # Per dimension: coarse scan to find T, then fine scan to find peak
    from aloop.solve import struct_id as sid
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    T_ests = []
    Rmat = R if R is not None else np.eye(D)
    for i in range(D):
        n_rough = min(max(int(np.ceil(2.0 * radius * 12.0)) + 1, 512), 4000)
        xs = np.linspace(z[i] - radius, z[i] + radius, n_rough)
        Hc = np.empty(n_rough)
        for k in range(n_rough):
            zk = np.asarray(z, dtype=float).copy(); zk[i] = xs[k]
            xk = Rmat.T @ zk
            Hc[k] = float((Rmat @ H_anal(xk) @ Rmat.T)[i, i])
        Tc = sid._coarse_period(xs, Hc)
        if Tc is None:
            return {'hit': False, 'reason': 'no_T_dim_%d' % i}
        T_ests.append(Tc)
    T_c = float(np.median(T_ests))
    if any(abs(Ti - T_c) / T_c > 0.2 for Ti in T_ests):
        return {'hit': False, 'reason': 'T_inconsistent'}

    peaks_all = []
    for i in range(D):
        n = int(np.ceil(2.0 * radius / T_c * 16.0)) + 1
        xs = np.linspace(z[i] - radius, z[i] + radius, n)
        Hc = np.empty(n)
        for k in range(n):
            zk = np.asarray(z, dtype=float).copy(); zk[i] = xs[k]
            xk = Rmat.T @ zk
            Hc[k] = float((Rmat @ H_anal(xk) @ Rmat.T)[i, i])
        peaks = []
        for j in range(1, n - 1):
            if Hc[j] >= Hc[j - 1] and Hc[j] >= Hc[j + 1]:
                if peaks and (xs[j] - peaks[-1]) < 2 * (xs[1] - xs[0]):
                    continue
                peaks.append(sid._parabolic_peak(xs, Hc, j))
        if len(peaks) < 2:
            return {'hit': False, 'reason': 'peaks<2_dim_%d' % i}
        peaks_all.append(np.array(peaks))

    # ---- Candidates: analytical gradient zero crossings (0 FE) ----
    tol_gz = 5e-3
    o_cand_list = []
    for i in range(D):
        entries = []
        for p in peaks_all[i]:
            zk = np.asarray(z, dtype=float).copy(); zk[i] = float(p)
            xk = Rmat.T @ zk
            gv = float((Rmat @ grad_anal(xk))[i])
            if abs(gv) < tol_gz:
                entries.append((abs(gv), round(float(p), 10)))
        if not entries:
            return {'hit': False, 'reason': 'no_cand_dim_%d' % i}
        entries.sort(key=lambda e: e[0])
        o_cand_list.append([p for _, p in entries])

    from aloop.solve import pcu as _pcu
    fe_cnt = [0]
    if R is None:
        def _f_cnt(o):
            fe_cnt[0] += 1
            return float(np.asarray(f30(o)).item())
        o_best, f_best = _pcu._best_combination(_f_cnt, D, o_cand_list)
        if o_best is None:
            return {'hit': False, 'reason': 'combo_fail'}
        return {'hit': f_best < 1e-4, 'F': round(float(f_best), 8), 'fe_verify': fe_cnt[0]}

    # Rotation case: combination + candidate direct verification
    def _f_rot(o_z):
        fe_cnt[0] += 1
        return float(np.asarray(f30(R.T @ np.asarray(o_z, float))).item())
    o_z, f_best = _pcu._best_combination(_f_rot, D, o_cand_list)
    if o_z is None:
        return {'hit': False, 'reason': 'combo_fail_rot'}
    return {'hit': f_best < 1e-4, 'F': round(float(f_best), 8), 'fe_verify': fe_cnt[0]}


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    out = {'problem': 'CEC2017 F5 M_orth 30D analytic-H PCU (Simulink block-derivative scenario)',
           'seeds': SEEDS, 'fe_blackbox_reference': 396993}
    hits = []
    for sd in SEEDS:
        r = analytic_pcu(sd)
        out['s%d' % sd] = r
        hits.append(r.get('hit', False))
        print('seed %d: %s' % (sd, r), flush=True)
    out['hit_rate'] = '%d/10' % sum(hits)
    fvs = [r.get('fe_verify', 0) for r in out.values() if isinstance(r, dict) and 'fe_verify' in r]
    out['fe_verify_max'] = max(fvs) if fvs else None
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, 'pcu_fe_dual.json'), 'w', encoding='utf-8') as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print('done', out['hit_rate'])


if __name__ == '__main__':
    main()
