# -*- coding: utf-8 -*-
"""Numba JIT accelerated kernels: compile the most time-consuming objective functions in CMA-ES with @njit.

Enabled only when numba is available; automatically falls back to the pure NumPy version when unavailable.
Covers: func + grad for michalewicz / rastrigin / schaffer_f6 (hess remains numerical differentiation).
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        """Pass-through decorator when numba is unavailable."""
        if args and callable(args[0]):
            return args[0]
        def deco(f):
            return f
        return deco


@njit(cache=True, nogil=True)
def michalewicz_f(x, m=10.0):
    n = len(x)
    total = 0.0
    for i in range(n):
        xi = x[i]
        total += np.sin(xi) * (np.sin((i + 1) * xi * xi / np.pi)) ** (2 * m)
    return -total


@njit(cache=True, nogil=True)
def michalewicz_g(x, m=10.0):
    n = len(x)
    g = np.zeros(n)
    for i in range(n):
        xi = x[i]
        idx = i + 1
        sin_xi = np.sin(xi)
        inner = np.sin(idx * xi * xi / np.pi)
        power = 2 * m
        if abs(inner) < 1e-12:
            g[i] = np.cos(xi) * inner ** power
        else:
            d_inner = (2 * idx * xi / np.pi) * np.cos(idx * xi * xi / np.pi)
            g[i] = np.cos(xi) * inner ** power + sin_xi * power * inner ** (power - 1) * d_inner
    return -g


@njit(cache=True, nogil=True)
def rastrigin_f(x):
    A = 10.0
    n = len(x)
    s = 0.0
    for i in range(n):
        s += x[i] * x[i] - A * np.cos(2 * np.pi * x[i])
    return A * n + s


@njit(cache=True, nogil=True)
def rastrigin_g(x):
    A = 10.0
    n = len(x)
    g = np.empty(n)
    for i in range(n):
        g[i] = 2 * x[i] + 2 * np.pi * A * np.sin(2 * np.pi * x[i])
    return g


@njit(cache=True, nogil=True)
def schaffer_f6_f(x):
    x1 = x[0]
    x2 = x[1]
    r = np.sqrt(x1 * x1 + x2 * x2)
    numerator = np.sin(r) * np.sin(r) - 0.5
    denominator = (1 + 0.001 * (x1 * x1 + x2 * x2)) ** 2
    return 0.5 + numerator / denominator


@njit(cache=True, nogil=True)
def schaffer_f6_g(x):
    x1 = x[0]
    x2 = x[1]
    r = np.sqrt(x1 * x1 + x2 * x2)
    if r < 1e-12:
        return np.zeros(2)
    sin_r = np.sin(r)
    cos_r = np.cos(r)
    s = x1 * x1 + x2 * x2
    denom = (1 + 0.001 * s) ** 2
    dnum_dr = 2 * sin_r * cos_r
    dr_dx1 = x1 / r
    dr_dx2 = x2 / r
    ddenom_dx1 = 2 * (1 + 0.001 * s) * 0.002 * x1
    ddenom_dx2 = 2 * (1 + 0.001 * s) * 0.002 * x2
    num = sin_r * sin_r - 0.5
    g1 = (dnum_dr * dr_dx1 * denom - num * ddenom_dx1) / (denom * denom)
    g2 = (dnum_dr * dr_dx2 * denom - num * ddenom_dx2) / (denom * denom)
    return np.array([g1, g2])


# Single-parameter wrapper (TestProblem.f only accepts x) - directly JIT the fixed m=10 version
@njit(cache=True, nogil=True)
def michalewicz_f10(x):
    n = len(x)
    total = 0.0
    m = 10.0
    for i in range(n):
        xi = x[i]
        total += np.sin(xi) * (np.sin((i + 1) * xi * xi / np.pi)) ** (2 * m)
    return -total


@njit(cache=True, nogil=True)
def michalewicz_g10(x):
    n = len(x)
    g = np.zeros(n)
    m = 10.0
    for i in range(n):
        xi = x[i]
        idx = i + 1
        sin_xi = np.sin(xi)
        inner = np.sin(idx * xi * xi / np.pi)
        power = 2 * m
        if abs(inner) < 1e-12:
            g[i] = np.cos(xi) * inner ** power
        else:
            d_inner = (2 * idx * xi / np.pi) * np.cos(idx * xi * xi / np.pi)
            g[i] = np.cos(xi) * inner ** power + sin_xi * power * inner ** (power - 1) * d_inner
    return -g


NUMBA_AVAILABLE = _NUMBA_AVAILABLE


# ============================================================
# Non-separable rotated functions (accept rotation matrix R as a parameter)
# ============================================================
@njit(cache=True, nogil=True)
def rotated_rastrigin_f(x, R):
    y = R @ x
    n = len(y)
    s = 0.0
    for i in range(n):
        s += y[i] * y[i] - 10.0 * np.cos(2.0 * np.pi * y[i])
    return 10.0 * n + s


@njit(cache=True, nogil=True)
def rotated_rastrigin_g(x, R, Rt):
    y = R @ x
    n = len(y)
    grad_y = np.empty(n)
    for i in range(n):
        grad_y[i] = 2.0 * y[i] + 20.0 * np.pi * np.sin(2.0 * np.pi * y[i])
    return Rt @ grad_y


@njit(cache=True, nogil=True)
def griewank_f(x, sqrt_idx):
    n = len(x)
    sum_sq = 0.0
    for i in range(n):
        sum_sq += x[i] * x[i]
    prod_cos = 1.0
    for i in range(n):
        prod_cos *= np.cos(x[i] / sqrt_idx[i])
    return 1.0 + sum_sq / 4000.0 - prod_cos


@njit(cache=True, nogil=True)
def griewank_g(x, sqrt_idx):
    n = len(x)
    prod_cos = 1.0
    for i in range(n):
        prod_cos *= np.cos(x[i] / sqrt_idx[i])
    g = np.empty(n)
    for i in range(n):
        g[i] = x[i] / 2000.0 + prod_cos * np.tan(x[i] / sqrt_idx[i]) / sqrt_idx[i]
    return g


@njit(cache=True, nogil=True)
def rotated_ackley_f(x, R):
    y = R @ x
    n = len(y)
    sum_sq = 0.0
    sum_cos = 0.0
    for i in range(n):
        sum_sq += y[i] * y[i]
        sum_cos += np.cos(2.0 * np.pi * y[i])
    return -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20.0 + np.e


@njit(cache=True, nogil=True)
def rotated_ackley_g(x, R, Rt):
    y = R @ x
    n = len(y)
    sum_sq = 0.0
    sum_cos = 0.0
    for i in range(n):
        sum_sq += y[i] * y[i]
        sum_cos += np.cos(2.0 * np.pi * y[i])
    term1 = 20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n)) * 0.2 / np.sqrt(n * sum_sq + 1e-30)
    grad_y = np.empty(n)
    for i in range(n):
        grad_y[i] = term1 * y[i] + (2.0 * np.pi / n) * np.exp(sum_cos / n) * np.sin(2.0 * np.pi * y[i])
    return Rt @ grad_y
