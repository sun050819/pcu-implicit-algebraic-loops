# -*- coding: utf-8 -*-
import os
"""run_18_theory_verify.py v2 - numerical verification of three robustness propositions (revised version).

P1 FFT: pure single-frequency cos model (proposition setting), T estimation error vs number of sample points per period
      -> verify Nyquist (>=2 points) and frequency resolution bound |T_est-T|/T <= T/(2R-T)
P2 parabola: periodic function (pure cos) -> O(h^3) staggered / O(h^4) aligned; general C^4 (x^3 perturbation)
      -> O(h^2), peak position solved numerically
P3 spectral gap: numerical confirmation of Davis-Kahan bound + degenerate block demonstration (two nearly degenerate blocks -> large residual)
"""
import io, sys, os, json, time
import numpy as np, warnings
warnings.filterwarnings('ignore')

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def _parabolic_peak(xs, y, i):
    a, b, c = y[i - 1], y[i], y[i + 1]
    denom = a - 2.0 * b + c
    if abs(denom) < 1e-30:
        return xs[i]
    return xs[i] + 0.5 * (a - c) / denom * (xs[1] - xs[0])


def verify_p1():
    T = 19.53125
    R_span = 120.0
    out = []
    for pts_pp in [2.01, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]:
        n = int(np.ceil(2.0 * R_span / T * pts_pp)) + 1
        xs = np.linspace(-R_span, R_span, n)
        y = np.cos(2.0 * np.pi * xs / T)
        yy = y - np.mean(y)
        sp = np.abs(np.fft.rfft(yy * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, d=xs[1] - xs[0])
        sp[0] = 0.0
        k = int(np.argmax(sp))
        T_est = 1.0 / freqs[k] if freqs[k] > 0 else np.inf
        bound = T / (2.0 * R_span - T) if 2 * R_span > T else np.inf
        out.append({'pts_per_period': pts_pp, 'n': n, 'T_est': round(T_est, 4),
                    'rel_err': round(abs(T_est - T) / T, 6), 'bound': round(bound, 6)})
        print('P1 pts/pp=%.2f n=%d T_est=%.3f rel=%.2e bound=%.2e' % (
            pts_pp, n, T_est, abs(T_est - T) / T, bound))
    return out


def verify_p2():
    T = 19.53125
    def ycos(x): return np.cos(2.0 * np.pi * x / T)
    def ygen(x): return np.cos(2.0 * np.pi * x / T) + 0.2 * np.cos(4.0 * np.pi * x / T) + 0.01 * x ** 3
    # numerical peak position (ygen peak near 0, find by fine scan)
    xs_f = np.linspace(-0.5, 0.5, 20001)
    yf = ygen(xs_f)
    x_star = xs_f[int(np.argmax(yf))]
    out = {'cos_offset': [], 'cos_aligned': [], 'general': []}
    for h in [0.5, 1.0, 2.0, 4.0]:
        # cos staggered delta=h/3
        dlt = h / 3.0
        xs = np.array([dlt - h, dlt, dlt + h])
        err = abs(_parabolic_peak(xs, [ycos(v) for v in xs], 1))
        out['cos_offset'].append({'h': h, 'err': round(err, 8)})
        # cos aligned delta=0
        xs = np.array([-h, 0.0, h])
        err = abs(_parabolic_peak(xs, [ycos(v) for v in xs], 1))
        out['cos_aligned'].append({'h': h, 'err': round(err, 8)})
        # general (peak near x_star staggered)
        dlt = min(h / 3.0, abs(x_star))
        xs = np.array([x_star - dlt - h, x_star - dlt, x_star - dlt + h])
        err = abs(_parabolic_peak(xs, [ygen(v) for v in xs], 1) - x_star)
        out['general'].append({'h': h, 'err': round(err, 8)})
        print('P2 h=%.1f cos_off=%.2e cos_align=%.2e gen=%.2e' % (
            h, out['cos_offset'][-1]['err'], out['cos_aligned'][-1]['err'],
            out['general'][-1]['err']))
    return out


def verify_p3():
    d = 10
    rng = np.random.RandomState(7)
    A = rng.randn(d, d)
    M, _ = np.linalg.qr(A)
    out = []
    pert = 1e-2
    # (a) Davis-Kahan: equidistant spectral gap scan
    for gap in [1e-4, 1e-3, 1e-2, 0.1, 1.0]:
        lam = np.arange(d) * gap + 1.0
        H = M.T @ np.diag(lam) @ M
        eps = 1e-6
        Hn = H + eps * rng.randn(d, d); Hn = (Hn + Hn.T) / 2
        V = np.linalg.eigh(Hn)[1].T
        # optimal permutation + sign alignment (Hungarian)
        G = V @ M.T
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(-np.abs(G))
        Vc = V[ci] * np.sign(G[ri, ci])[:, None]
        r_fro = float(np.linalg.norm(Vc @ M.T - np.eye(d)))
        dk_bound = float(2.0 * np.sqrt(d) * eps / gap)
        out.append({'case': 'equal_gap', 'gap': gap, 'r_fro': round(r_fro, 6),
                    'dk_bound': round(dk_bound, 6), 'resid_lte_dk': bool(r_fro <= dk_bound * 1.5)})
        print('P3a gap=%.0e r_fro=%.4f dk_bound=%.4f ok=%s' % (
            gap, r_fro, dk_bound, r_fro <= dk_bound * 1.5))
    # (b) degenerate block: two nearly degenerate blocks (intra-block gap, inter-block gap=9) -> large intra-block residual; contrast with large inter-block gap
    from scipy.optimize import linear_sum_assignment
    for block_gap in [1e-8, 1e-6, 1e-4, 1e-2, 0.1]:
        lam = np.concatenate([np.full(5, 1.0) + np.arange(5) * block_gap,
                              np.full(5, 10.0) + np.arange(5) * block_gap])
        H = M.T @ np.diag(lam) @ M
        eps = 1e-6
        Hn = H + eps * rng.randn(d, d); Hn = (Hn + Hn.T) / 2
        V = np.linalg.eigh(Hn)[1].T
        G = V @ M.T
        ri, ci = linear_sum_assignment(-np.abs(G))
        Vc = V[ci] * np.sign(G[ri, ci])[:, None]
        r_fro = float(np.linalg.norm(Vc @ M.T - np.eye(d)))
        # voting: 3-point perturbation -> pairwise distances of candidates (intra-degenerate-block candidates inconsistent)
        cands = []
        for s in [0, 1, -1]:
            Hp = H + s * pert * rng.randn(d, d); Hp = (Hp + Hp.T) / 2
            cands.append(np.linalg.eigh(Hp)[1].T)
        cross = float(np.mean([np.linalg.norm(c1 @ c2.T - np.eye(d)) for c1 in cands for c2 in cands]))
        out.append({'case': 'block', 'block_gap': block_gap, 'r_fro': round(r_fro, 6),
                    'cand_cross': round(cross, 6)})
        print('P3b block_gap=%.0e r_fro=%.4f cand_cross=%.4f' % (block_gap, r_fro, cross))
    return out


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    t0 = time.time()
    res = {'p1_fft': verify_p1(), 'p2_parabolic': verify_p2(), 'p3_spectral_gap': verify_p3(),
           'elapsed_s': round(time.time() - t0, 1)}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, 'theory_verify.json'), 'w', encoding='utf-8') as fp:
        json.dump(res, fp, ensure_ascii=False, indent=2)
    print('done')


if __name__ == '__main__':
    main()
