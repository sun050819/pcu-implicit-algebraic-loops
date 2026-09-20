# -*- coding: utf-8 -*-
"""pcu.py - PCU (Periodic Coordinate Unwrapping) core implementation, no import side effects.

Use analytical Hessian to detect and solve separable periodic structure problems (Rastrigin family):
  A. Standard separable (H diagonal): single-point (g_i, H_ii) per-dimension analytical unwrapping
  B. Rotated non-separable (H = RT D R): single-point eigh recovers R -> transform z=R x -> per-dimension unwrapping

Unwrapping ambiguity handling: each dimension may produce multiple candidates o_i (different integer basins),
use "candidate combinations + F(o_cand) verification" to select the globally optimal combination (true solution F=0 minimum).
Only accept when F(o_cand) < 1e-4 (zero false-positive safety design).
"""
import numpy as np

PCU_A = 10.0
PCU_RADIUS = 80.0


def unwrap_1d(g_i, H_ii, x_i, A=PCU_A, tol=1e-5, radius=PCU_RADIUS, b=0.0,
              c1=1.0, T=1.0):
    """1D generalized periodic unwrapping: recover candidate u = x_i - o_i list from (g_i, H_ii).

    Supports target family h(u) = c1*u^2 + A*(1 - cos(2*pi*u/T)) + b*u:
      - c1: quadratic term coefficient (Rastrigin is 1; pure periodic family is 0)
      - b:  linear bias (biased Rastrigin)
      - A:  cos periodic term amplitude
      - T:  period (Rastrigin=1; T=2 is half-frequency periodic family)
    Gradient g = 2*c1*u + (2*pi*A/T)*sin(2*pi*u/T) + b
    Hessian H = 2*c1 + (4*pi^2*A/T^2)*cos(2*pi*u/T)
    """
    omega = 2.0 * np.pi / T
    d = H_ii - 2.0 * c1
    cosv = np.clip(d / (4.0 * np.pi**2 * A / T**2), -1.0, 1.0)
    ac = np.arccos(cosv)
    u0s = set()
    for ang in (ac, omega * T - ac):
        u0s.add(round(ang / omega % T, 10))
    cands = []
    lo = int(np.floor((x_i - radius) / T))
    hi = int(np.ceil((x_i + radius) / T))
    for u0 in sorted(u0s):
        for k in range(lo, hi + 1):
            u = k * T + u0
            g_pred = 2.0 * c1 * u + (2.0 * np.pi * A / T) * np.sin(2.0 * np.pi * u / T) + b
            if abs(g_pred - g_i) < tol * max(1.0, abs(g_i)):
                cands.append(u)
    return cands


def _best_combination(func, dim, o_cand_list):
    """Candidate combination search: list of candidate o values per dimension, find the combination with minimum F (true solution F=0).

    Strategy: candidate combinations are usually very few (1 for most dimensions, 2-4 for a few).
    1) If total combinations <= 1024: exhaustive search for minimum F.
    2) Otherwise: greedy (take first per dimension) + two rounds of coordinate descent replacement.
    """
    n_comb = 1
    for cands in o_cand_list:
        n_comb *= max(len(cands), 1)
        if n_comb > 1024:
            break
    best_o = None
    best_f = float('inf')
    if n_comb <= 1024:
        # Exhaustive
        idx = [0] * dim
        while True:
            o = np.array([o_cand_list[i][idx[i]] for i in range(dim)], dtype=float)
            f = float(np.asarray(func(o)).item())
            if f < best_f:
                best_f, best_o = f, o
            # Increment
            p = dim - 1
            while p >= 0:
                idx[p] += 1
                if idx[p] < len(o_cand_list[p]):
                    break
                idx[p] = 0
                p -= 1
            if p < 0:
                break
    else:
        # Greedy + coordinate descent
        o = np.array([o_cand_list[i][0] for i in range(dim)], dtype=float)
        best_f = float(np.asarray(func(o)).item())
        best_o = o
        for _ in range(2):
            improved = False
            for i in range(dim):
                for cand in o_cand_list[i]:
                    o2 = o.copy()
                    o2[i] = cand
                    f2 = float(np.asarray(func(o2)).item())
                    if f2 < best_f - 1e-12:
                        best_f, best_o, o = f2, o2, o2
                        improved = True
            if not improved:
                break
    return best_o, best_f


def pcu_attempt(func, grad, hess, x, tol=1e-5, A=PCU_A, b=None, c1=1.0,
                T=1.0, detect_model=False):
    """PCU fast path. Returns (x_cand, 'pcu_ok') on success, otherwise (None, None).

    Parameters
    ----
    A : cos periodic term amplitude (Rastrigin=10)
    b : linear bias vector (length d or None)
    c1 : quadratic term coefficient (Rastrigin=1; pure periodic family=0)
    T : period (Rastrigin=1)
    detect_model : when True, automatically estimate A from single-point hess diagonal amplitude (generalization)
    """
    d = len(x)
    g = np.asarray(grad(x))
    H = np.asarray(hess(x))
    if b is None:
        b = np.zeros(d)
    b = np.asarray(b, dtype=float)
    if detect_model:
        # Estimate A from H diagonal element amplitude: H_ii = 2*c1 + (4*pi^2*A/T^2)*cos(2*pi*u/T)
        # A single point can only measure one cos value -> A cannot be uniquely estimated; use amplitude upper bound for conservative estimation,
        # rely on F verification as fallback (when A is too large, cosv matching deviation is rejected by F(o_cand)<1e-4).
        amp = float(np.max(np.abs(np.diag(H) - 2.0 * c1)))
        A_est = amp / (4.0 * np.pi**2 / T**2)
        if A_est > 0:
            A = A_est
    off = np.sqrt(np.sum((H - np.diag(np.diag(H)))**2))
    o_cand_list = None
    R = None
    if off < 1e-6 * (1.0 + np.abs(H).max()):
        # Mode A: H diagonal -> separable
        o_cand_list = []
        ok = True
        for i in range(d):
            us = unwrap_1d(g[i] - b[i], H[i, i], x[i], A=A, c1=c1, T=T)
            if not us:
                ok = False
                break
            # Candidate o = x_i - u (deduplicate)
            o_cands = sorted(set(round(x[i] - u, 10) for u in us))
            o_cand_list.append(o_cands)
        if not ok:
            return None, None
    else:
        # Mode B: eigh recovers rotation (bias b also needs rotation: b_z = R @ b)
        evals, evecs = np.linalg.eigh(H)
        R = evecs.T
        z = R @ x
        g_z = R @ g
        b_z = R @ b
        H_z = R @ H @ R.T
        o_cand_list = []
        ok = True
        for i in range(d):
            us = unwrap_1d(g_z[i] - b_z[i], H_z[i, i], z[i], A=A, c1=c1, T=T)
            if not us:
                ok = False
                break
            o_cands = sorted(set(round(z[i] - u, 10) for u in us))
            o_cand_list.append(o_cands)
        if not ok:
            return None, None
    if o_cand_list is None:
        return None, None
    # Combination search + F verification
    if R is None:
        o_best, f_best = _best_combination(func, d, o_cand_list)
    else:
        def _f_rot(o_z):
            return float(np.asarray(func(R.T @ np.asarray(o_z, dtype=float))).item())
        o_z, f_best = _best_combination(_f_rot, d, o_cand_list)
        o_best = R.T @ np.asarray(o_z, dtype=float)
    if f_best < 1e-4:
        return np.asarray(o_best, dtype=float), 'pcu_ok'
    return None, None
