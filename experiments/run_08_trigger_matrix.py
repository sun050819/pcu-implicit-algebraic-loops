# -*- coding: utf-8 -*-
import os
"""run_08_trigger_matrix: CEC2017 trigger/rejection matrix (30D).

Positive examples (triggerable under method assumptions): F5 Rastrigin (analytic, proven), F9 Levy (analytic, sin^2 periodic)
Negative examples (should reject): F4 Rosenbrock, F6 Schaffer F7, F8 Non-Cont Rastrigin, F10 Schwefel
             -- difference gradient/Hessian (black-box perspective), test identify rejection (no false positives)
Dual-track comparison of boundary behavior between M_raw (official non-orthogonal) and M_orth (orthogonalized).
"""
import io, sys, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'cec2017'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cec2017.transforms import rotations, shifts
from aloop.solve.struct_id import estimate_and_pcu

D = 30
N = 10
SEED0 = 20260913
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
T = 2.048 / 100.0   # Rosenbrock official scaling

def get_o_M(idx):
    o = shifts[idx][:D].copy()
    M = rotations[D][idx].copy()
    return o, M

def make_rosenbrock(o, M):
    def f(x):
        z = T * (M @ (np.asarray(x, float) - o))
        s = 0.0
        for i in range(D - 1):
            s += 100.0 * (z[i]**2 - z[i+1])**2 + (z[i] - 1.0)**2
        return float(s)
    return f

def make_schaffer(o, M):
    def f(x):
        z = M @ (np.asarray(x, float) - o)
        s = 0.0
        for i in range(D - 1):
            si = np.sqrt(z[i]**2 + z[i+1]**2)
            s += si**0.5 * (np.sin(50.0 * si**0.2) + 1.0)
        return float(s * s)
    return f

def make_noncont_rastrigin(o, M):
    S5 = 5.12 / 100.0
    def f(x):
        x = np.asarray(x, float)
        y = x.copy()
        for i in range(D):
            if abs(y[i] - o[i]) > 0.5:
                y[i] = o[i] + np.floor(2.0 * (y[i] - o[i]) + 0.5) / 2.0
        z = S5 * (M @ (y - o))
        return float(np.sum(z*z - 10.0*np.cos(2.0*np.pi*z) + 10.0))
    return f

def make_schwefel(o, M):
    def f(x):
        z = M @ (np.asarray(x, float) - o)
        return float(np.sum(-z * np.sin(np.sqrt(np.abs(z)))) + 4.189828872724338e+02 * D)
    return f

def make_levy(o, M):
    def f(x):
        z = M @ (np.asarray(x, float) - o)
        w = 1.0 + (z - 1.0) / 4.0
        return float(np.sin(np.pi*w[0])**2
                     + np.sum((w[:-1]-1.0)**2 * (1.0 + 10.0*np.sin(np.pi*w[:-1]+1.0)**2))
                     + (w[-1]-1.0)**2 * (1.0 + np.sin(2.0*np.pi*w[-1])**2))
    def g(x):
        x = np.asarray(x, float); z = M@(x-o); w = 1.0 + (z-1.0)/4.0
        gw = np.zeros(D); gw[0] = 2.0*np.pi*np.sin(np.pi*w[0])*np.cos(np.pi*w[0])
        for i in range(D-1):
            a = w[i]-1.0; s = np.sin(np.pi*w[i]+1.0); c = np.cos(np.pi*w[i]+1.0)
            gw[i] += 2.0*a*(1.0+10.0*s**2) + a**2*20.0*np.pi*s*c
        i = D-1; a = w[i]-1.0; s = np.sin(2.0*np.pi*w[i]); c = np.cos(2.0*np.pi*w[i])
        gw[i] += 2.0*a*(1.0+s**2) + a**2*4.0*np.pi*s*c
        return M.T @ (gw/4.0)
    def h(x):
        x = np.asarray(x, float); z = M@(x-o); w = 1.0 + (z-1.0)/4.0
        def dd(i):
            if i == 0:
                s = np.sin(np.pi*w[0]); c = np.cos(np.pi*w[0])
                return 2.0*np.pi**2*(c**2 - s**2)
            if i == D-1:
                a = w[i]-1.0; s = np.sin(2.0*np.pi*w[i]); c = np.cos(2.0*np.pi*w[i])
                return 2.0*(1.0+s**2) + 8.0*a*np.pi*s*c + a**2*8.0*np.pi**2*(c**2 - s**2)
            a = w[i]-1.0; s = np.sin(np.pi*w[i]+1.0); c = np.cos(np.pi*w[i]+1.0)
            return 2.0*(1.0+10.0*s**2) + 40.0*a*np.pi*s*c + a**2*20.0*np.pi**2*(c**2 - s**2)
        Hw = np.array([dd(i) for i in range(D)])
        return (M.T * (Hw/16.0)) @ M
    return f, g, h

def num_grad_h(fn, x, h=1e-5):
    x = np.asarray(x, float); d = len(x)
    g = np.zeros(d); H = np.zeros((d, d))
    f0 = fn(x)
    for i in range(d):
        xp = x.copy(); xm = x.copy()
        xp[i] += h; xm[i] -= h
        g[i] = (fn(xp) - fn(xm)) / (2.0*h)
        H[i, i] = (fn(xp) - 2.0*f0 + fn(xm)) / h**2
    return g, H

def run_matrix(case_name, factory, idx, M_raw, M_orth, n=N):
    """factory(M) -> (f, g, h): rebuild functions with the given rotation matrix (closure rebinds correctly)."""
    res = {}
    for tag, M in [('M_raw', M_raw), ('M_orth', M_orth)]:
        f, g, h = factory(M)
        trig = hits = 0
        for k in range(n):
            rng = np.random.RandomState(SEED0 + idx*100 + k)
            x0 = rng.uniform(-100.0, 100.0, D)
            xc, t2 = estimate_and_pcu(f, g, h, x0, margin=100.0)
            if t2 == 'struct_pcu_ok':
                trig += 1
                if f(xc) < 1e-4:
                    hits += 1
        res[tag] = {'trigger': trig, 'hit': hits, 'n': n, 'trigger_rate': trig/n, 'hit_rate': hits/n}
        print('%s|%s: trigger %d/%d, hit %d/%d' % (case_name, tag, trig, n, hits, n), flush=True)
    return res

results = {}

# --- Positive example: F5 Rastrigin (analytic) ---
o5, M5 = get_o_M(4)
S5 = 5.12/100.0
def factory_f5(M):
    def f(x, Mm=M):
        z = S5*(Mm@(np.asarray(x,float)-o5)); return float(np.sum(z*z-10.0*np.cos(2.0*np.pi*z)+10.0))
    def g(x, Mm=M):
        x=np.asarray(x,float); z=S5*(Mm@(x-o5)); gz=S5*(2.0*z+20.0*np.pi*np.sin(2.0*np.pi*z)); return Mm.T@gz
    def h(x, Mm=M):
        x=np.asarray(x,float); z=S5*(Mm@(x-o5)); Hzz=S5*S5*(2.0+40.0*np.pi*np.pi*np.cos(2.0*np.pi*z)); return (Mm.T*Hzz)@Mm
    return f, g, h
u5, _, vt5 = np.linalg.svd(M5)
results['F5_Rastrigin'] = run_matrix('F5_Rastrigin', factory_f5, 4, M5, u5@vt5)

# --- Positive example: F9 Levy (analytic) ---
o9, M9 = get_o_M(8)
def factory_f9(M):
    def f(x, Mm=M):
        z = Mm@(np.asarray(x,float)-o9); w = 1.0+(z-1.0)/4.0
        return float(np.sin(np.pi*w[0])**2 + np.sum((w[:-1]-1.0)**2*(1.0+10.0*np.sin(np.pi*w[:-1]+1.0)**2)) + (w[-1]-1.0)**2*(1.0+np.sin(2.0*np.pi*w[-1])**2))
    def g(x, Mm=M):
        x=np.asarray(x,float); z=Mm@(x-o9); w=1.0+(z-1.0)/4.0
        gw=np.zeros(D); gw[0]=2.0*np.pi*np.sin(np.pi*w[0])*np.cos(np.pi*w[0])
        for i in range(D-1):
            a=w[i]-1.0; s=np.sin(np.pi*w[i]+1.0); c=np.cos(np.pi*w[i]+1.0)
            gw[i]+=2.0*a*(1.0+10.0*s**2)+a**2*20.0*np.pi*s*c
        i=D-1; a=w[i]-1.0; s=np.sin(2.0*np.pi*w[i]); c=np.cos(2.0*np.pi*w[i])
        gw[i]+=2.0*a*(1.0+s**2)+a**2*4.0*np.pi*s*c
        return Mm.T@(gw/4.0)
    def h(x, Mm=M):
        x=np.asarray(x,float); z=Mm@(x-o9); w=1.0+(z-1.0)/4.0
        def dd(i):
            if i==0:
                s=np.sin(np.pi*w[0]); c=np.cos(np.pi*w[0])
                return 2.0*np.pi**2*(c**2-s**2)
            if i==D-1:
                a=w[i]-1.0; s=np.sin(2.0*np.pi*w[i]); c=np.cos(2.0*np.pi*w[i])
                return 2.0*(1.0+s**2)+8.0*a*np.pi*s*c+a**2*8.0*np.pi**2*(c**2-s**2)
            a=w[i]-1.0; s=np.sin(np.pi*w[i]+1.0); c=np.cos(np.pi*w[i]+1.0)
            return 2.0*(1.0+10.0*s**2)+40.0*a*np.pi*s*c+a**2*20.0*np.pi**2*(c**2-s**2)
        Hw=np.array([dd(i) for i in range(D)])
        return (Mm.T*(Hw/16.0))@Mm
    return f, g, h
u9, _, vt9 = np.linalg.svd(M9)
results['F9_Levy'] = run_matrix('F9_Levy', factory_f9, 8, M9, u9@vt9)

# --- Negative example (pure black-box perspective: full FD-PCU, only call fn, trigger = includes F<1e-4 verification) ---
HS_BB = 1e-3
def _fd_hess_full(fn, x):
    x = np.asarray(x, float)
    H = np.zeros((D, D))
    for i in range(D):
        xp = x.copy(); xm = x.copy()
        xp[i] += HS_BB; xm[i] -= HS_BB
        H[i, i] = (fn(xp) - 2.0*fn(x) + fn(xm)) / HS_BB**2
        for j in range(i + 1, D):
            xpp = x.copy(); xpm = x.copy(); xmp = x.copy(); xmm = x.copy()
            xpp[i] += HS_BB; xpp[j] += HS_BB
            xpm[i] += HS_BB; xpm[j] -= HS_BB
            xmp[i] -= HS_BB; xmp[j] += HS_BB
            xmm[i] -= HS_BB; xmm[j] -= HS_BB
            H[i, j] = H[j, i] = (fn(xpp) - fn(xpm) - fn(xmp) + fn(xmm)) / (4.0 * HS_BB**2)
    return H

def _off_diag(A):
    A = np.abs(A); np.fill_diagonal(A, 0.0)
    return float(np.sqrt(np.sum(A * A)))

def _fd_hess_dir(fn, x, r):
    x = np.asarray(x, float)
    return (fn(x + HS_BB*r) - 2.0*fn(x) + fn(x - HS_BB*r)) / HS_BB**2

def _fd_grad(fn, x):
    x = np.asarray(x, float)
    g = np.zeros(D)
    for i in range(D):
        xp = x.copy(); xm = x.copy()
        xp[i] += HS_BB; xm[i] -= HS_BB
        g[i] = (fn(xp) - fn(xm)) / (2.0 * HS_BB)
    return g

def blackbox_identify(fn, x, margin=100.0):
    x = np.asarray(x, float)
    pts = [x, x + 1.0, x - 1.0]
    Hs = [_fd_hess_full(fn, p) for p in pts]
    cands = [np.linalg.eigh(Hj)[1].T for Hj in Hs]
    best_R, best_score = None, float('inf')
    for Rj in cands:
        s = sum(_off_diag(Rj @ Hl @ Rj.T) for Hl in Hs)
        if s < best_score:
            best_score, best_R = s, Rj
    H0 = Hs[0]
    if _off_diag(H0) > 1e-6 * (1.0 + np.abs(H0).max()):
        base = best_R @ x; R = best_R
    else:
        base = x; R = None
    radius = max(10.0, float(np.max(np.abs(base))) + margin)
    T_ests = []
    for i in range(D):
        n_rough = min(max(int(np.ceil(2.0 * radius * 12.0)) + 1, 512), 4000)
        xs = np.linspace(base[i] - radius, base[i] + radius, n_rough)
        Hc = np.empty(n_rough)
        for k in range(n_rough):
            xk = base.copy(); xk[i] = xs[k]
            rvec = R[i] if R is not None else np.eye(D)[i]
            Hc[k] = _fd_hess_dir(fn, R.T @ xk if R is not None else xk, rvec)
        from aloop.solve import struct_id as _sid
        Tc = _sid._coarse_period(xs, Hc)
        if Tc is None:
            return None
        T_ests.append(Tc)
    T_c = float(np.median(T_ests))
    if any(abs(Ti - T_c) / T_c > 0.2 for Ti in T_ests):
        return None
    peaks_all = []
    for i in range(D):
        n = int(np.ceil(2.0 * radius / T_c * 16.0)) + 1
        xs = np.linspace(base[i] - radius, base[i] + radius, n)
        Hc = np.empty(n)
        for k in range(n):
            xk = base.copy(); xk[i] = xs[k]
            rvec = R[i] if R is not None else np.eye(D)[i]
            Hc[k] = _fd_hess_dir(fn, R.T @ xk if R is not None else xk, rvec)
        peaks = []
        for j in range(1, n - 1):
            if Hc[j] >= Hc[j-1] and Hc[j] >= Hc[j+1]:
                if peaks and (xs[j] - peaks[-1]) < 2 * (xs[1] - xs[0]):
                    continue
                peaks.append(_sid._parabolic_peak(xs, Hc, j))
        if len(peaks) < 2:
            return None
        peaks_all.append(np.array(peaks))
        gaps = np.diff(np.sort(peaks))
        gaps = gaps[(gaps > 0.5*T_c) & (gaps < 1.5*T_c)]
        if len(gaps) == 0:
            return None
    return {'T': T_c, 'peaks': peaks_all, 'R': R}

def blackbox_pcu(fn, x, margin=100.0):
    mdl = blackbox_identify(fn, x, margin=margin)
    if mdl is None:
        return None, None
    peaks_all = mdl['peaks']; R = mdl['R']
    if R is None:
        g0 = _fd_grad(fn, x)
        tol_g = 5e-3 * max(1.0, float(np.max(np.abs(g0))))
        o_cand_list = []
        for i in range(D):
            entries = []
            for p in peaks_all[i]:
                xk = np.asarray(x, dtype=float).copy()
                xk[i] = float(p)
                gv = _fd_grad(fn, xk)[i]
                if abs(gv) < tol_g:
                    entries.append((abs(gv), round(float(p), 10)))
            if not entries:
                return None, None
            entries.sort(key=lambda e: e[0])
            o_cand_list.append([p for _, p in entries])
        o_best, f_best = _pcu._best_combination(fn, D, o_cand_list)
        if o_best is None or f_best >= 1e-4:
            return None, None
        return np.asarray(o_best, float), 'blackbox_pcu_ok'
    else:
        def grad_z(zv): return R @ _fd_grad(fn, R.T @ np.asarray(zv, float))
        z = R @ x
        g_z0 = grad_z(z)
        tol_gz = 5e-3 * max(1.0, float(np.max(np.abs(g_z0))))
        o_cand_list = []
        for i in range(D):
            entries = []
            for p in peaks_all[i]:
                zk = np.asarray(z, dtype=float).copy()
                zk[i] = float(p)
                gv = float(grad_z(zk)[i])
                if abs(gv) < tol_gz:
                    entries.append((abs(gv), round(float(p), 10)))
            if not entries:
                return None, None
            entries.sort(key=lambda e: e[0])
            o_cand_list.append([p for _, p in entries])
        def _f_rot(o_z):
            return float(np.asarray(fn(R.T @ np.asarray(o_z, float))).item())
        o_z, f_best = _pcu._best_combination(_f_rot, D, o_cand_list)
        if o_z is None or f_best >= 1e-4:
            return None, None
        return np.asarray(R.T @ o_z, float), 'blackbox_pcu_ok'

def run_neg(case_name, fn, idx, M):
    trig = hits = 0
    for k in range(N):
        rng = np.random.RandomState(SEED0 + idx*100 + k)
        x0 = rng.uniform(-100.0, 100.0, D)
        xc, t2 = blackbox_pcu(fn, x0, margin=100.0)
        if t2 == 'blackbox_pcu_ok' and fn(xc) < 1e-4:
            trig += 1; hits += 1
    results[case_name] = {'trigger': trig, 'hit': hits, 'n': N, 'trigger_rate': trig/N, 'hit_rate': hits/N, 'mode': 'blackbox_fd_full'}
    print('%s: trigger %d/%d, hit %d/%d' % (case_name, trig, N, hits, N), flush=True)

run_neg('F4_Rosenbrock', make_rosenbrock(*get_o_M(3)), 3, rotations[D][3])
run_neg('F6_SchafferF7', make_schaffer(*get_o_M(5)), 5, rotations[D][5])
run_neg('F8_NonContRastrigin', make_noncont_rastrigin(*get_o_M(7)), 7, rotations[D][7])
run_neg('F10_Schwefel', make_schwefel(*get_o_M(9)), 9, rotations[D][9])

os.makedirs(RESULTS, exist_ok=True)
with open(os.path.join(RESULTS, 'cec_trigger_matrix.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print('done -> cec_trigger_matrix.json')
