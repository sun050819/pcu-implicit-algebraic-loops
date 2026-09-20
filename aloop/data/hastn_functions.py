"""
test_functions.py
Benchmark test functions with f, grad, hess, CI value, initial points.
"""
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass
class TestProblem:
    name: str
    dim: int
    f: Callable[[np.ndarray], float]
    grad: Callable[[np.ndarray], np.ndarray]
    hess: Callable[[np.ndarray], np.ndarray]
    x_opt: np.ndarray
    f_opt: float
    x0: np.ndarray
    ci_value: float
    description: str = ""


# ============================================================
# 1. Rosenbrock function family
# ============================================================
def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))

def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    n = len(x)
    g = np.zeros(n)
    g[0] = -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0])
    for i in range(1, n - 1):
        g[i] = 200.0 * (x[i] - x[i - 1] ** 2) - 400.0 * x[i] * (x[i + 1] - x[i] ** 2) - 2.0 * (1.0 - x[i])
    g[-1] = 200.0 * (x[-1] - x[-2] ** 2)
    return g

def rosenbrock_hess(x: np.ndarray) -> np.ndarray:
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n - 1):
        H[i, i] += 1200.0 * x[i] ** 2 - 400.0 * x[i+1] + 2.0
        H[i, i+1] = -400.0 * x[i]
        H[i+1, i] = -400.0 * x[i]
        H[i+1, i+1] += 200.0
    return H


# ============================================================
# 2. Beale function family
# ============================================================
def beale(x: np.ndarray) -> float:
    x1, x2 = x[0], x[1]
    return float((1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2 ** 2) ** 2 + (2.625 - x1 + x1 * x2 ** 3) ** 2)

def beale_grad(x: np.ndarray) -> np.ndarray:
    x1, x2 = x[0], x[1]
    t1 = 1.5 - x1 + x1 * x2
    t2 = 2.25 - x1 + x1 * x2 ** 2
    t3 = 2.625 - x1 + x1 * x2 ** 3
    g1 = 2 * t1 * (-1 + x2) + 2 * t2 * (-1 + x2 ** 2) + 2 * t3 * (-1 + x2 ** 3)
    g2 = 2 * t1 * x1 + 2 * t2 * 2 * x1 * x2 + 2 * t3 * 3 * x1 * x2 ** 2
    return np.array([g1, g2])

def beale_hess(x: np.ndarray) -> np.ndarray:
    x1, x2 = x[0], x[1]
    t1 = 1.5 - x1 + x1 * x2
    t2 = 2.25 - x1 + x1 * x2 ** 2
    t3 = 2.625 - x1 + x1 * x2 ** 3
    df1_dx1 = -1 + x2; df1_dx2 = x1
    df2_dx1 = -1 + x2**2; df2_dx2 = 2 * x1 * x2
    df3_dx1 = -1 + x2**3; df3_dx2 = 3 * x1 * x2**2
    H = np.zeros((2, 2))
    H[0, 0] = 2 * (df1_dx1**2 + df2_dx1**2 + df3_dx1**2)
    H[0, 1] = 2 * (df1_dx1 * df1_dx2 + df2_dx1 * df2_dx2 + df3_dx1 * df3_dx2)
    H[1, 0] = H[0, 1]
    H[1, 1] = 2 * (df1_dx2**2 + df2_dx2**2 + df3_dx2**2) + 2 * (t2 * 2 * x1 + t3 * 6 * x1 * x2)
    return H


# ============================================================
# 3. Powell singular function (4D)
# ============================================================
def powell(x: np.ndarray) -> float:
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    return float((x1 + 10 * x2) ** 2 + 5 * (x3 - x4) ** 2 + (x2 - 2 * x3) ** 4 + 10 * (x1 - x4) ** 4)

def powell_grad(x: np.ndarray) -> np.ndarray:
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    g1 = 2 * (x1 + 10 * x2) + 40 * (x1 - x4) ** 3
    g2 = 20 * (x1 + 10 * x2) + 4 * (x2 - 2 * x3) ** 3
    g3 = 10 * (x3 - x4) - 8 * (x2 - 2 * x3) ** 3
    g4 = -10 * (x3 - x4) - 40 * (x1 - x4) ** 3
    return np.array([g1, g2, g3, g4])

def powell_hess(x: np.ndarray) -> np.ndarray:
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    H = np.zeros((4, 4))
    H[0, 0] = 2.0 + 120.0 * (x1 - x4)**2
    H[0, 1] = 20.0
    H[0, 3] = -120.0 * (x1 - x4)**2
    H[1, 0] = 20.0
    H[1, 1] = 200.0 + 12.0 * (x2 - 2*x3)**2
    H[1, 2] = -24.0 * (x2 - 2*x3)**2
    H[2, 1] = -24.0 * (x2 - 2*x3)**2
    H[2, 2] = 10.0 + 48.0 * (x2 - 2*x3)**2
    H[2, 3] = -10.0
    H[3, 0] = -120.0 * (x1 - x4)**2
    H[3, 2] = -10.0
    H[3, 3] = 10.0 + 120.0 * (x1 - x4)**2
    return H


# ============================================================
# 4. Wood function (4D)
# ============================================================
def wood(x: np.ndarray) -> float:
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    return float(100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2 + 90 * (x4 - x3 ** 2) ** 2 +
                 (1 - x3) ** 2 + 10.1 * ((x2 - 1) ** 2 + (x4 - 1) ** 2) + 19.8 * (x2 - 1) * (x4 - 1))

def wood_grad(x: np.ndarray) -> np.ndarray:
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    g1 = -400 * x1 * (x2 - x1 ** 2) - 2 * (1 - x1)
    g2 = 200 * (x2 - x1 ** 2) + 20.2 * (x2 - 1) + 19.8 * (x4 - 1)
    g3 = -360 * x3 * (x4 - x3 ** 2) - 2 * (1 - x3)
    g4 = 180 * (x4 - x3 ** 2) + 20.2 * (x4 - 1) + 19.8 * (x2 - 1)
    return np.array([g1, g2, g3, g4])

def wood_hess(x: np.ndarray) -> np.ndarray:
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    H = np.zeros((4, 4))
    H[0, 0] = 1200.0 * x1**2 - 400.0 * x2 + 2.0
    H[0, 1] = -400.0 * x1
    H[1, 0] = -400.0 * x1
    H[1, 1] = 200.0 + 20.2
    H[1, 3] = 19.8
    H[2, 2] = 1080.0 * x3**2 - 360.0 * x4 + 2.0
    H[2, 3] = -360.0 * x3
    H[3, 1] = 19.8
    H[3, 2] = -360.0 * x3
    H[3, 3] = 180.0 + 20.2
    return H

# ============================================================
# 5. Rastrigin multimodal function (massive local minima)
# ============================================================
def rastrigin(x: np.ndarray) -> float:
    A = 10.0
    n = len(x)
    return float(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))

def rastrigin_grad(x: np.ndarray) -> np.ndarray:
    A = 10.0
    return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)

def rastrigin_hess(x: np.ndarray) -> np.ndarray:
    A = 10.0
    n = len(x)
    return 2 * np.eye(n) + 4 * np.pi**2 * A * np.diag(np.cos(2 * np.pi * x))


# ============================================================
# 5b. Schaffer F6 multimodal function (2D, massive local minima)
# ============================================================
def schaffer_f6(x: np.ndarray) -> float:
    x1, x2 = x[0], x[1]
    r = np.sqrt(x1**2 + x2**2)
    numerator = np.sin(r)**2 - 0.5
    denominator = (1 + 0.001 * (x1**2 + x2**2))**2
    return float(0.5 + numerator / denominator)

def schaffer_f6_grad(x: np.ndarray) -> np.ndarray:
    x1, x2 = x[0], x[1]
    r = np.sqrt(x1**2 + x2**2)
    if r < 1e-12:
        return np.zeros(2)
    sin_r = np.sin(r)
    cos_r = np.cos(r)
    s = x1**2 + x2**2
    denom = (1 + 0.001 * s)**2
    dnum_dr = 2 * sin_r * cos_r
    dr_dx1 = x1 / r
    dr_dx2 = x2 / r
    ddenom_dx1 = 2 * (1 + 0.001 * s) * 0.002 * x1
    ddenom_dx2 = 2 * (1 + 0.001 * s) * 0.002 * x2
    num = sin_r**2 - 0.5
    g1 = (dnum_dr * dr_dx1 * denom - num * ddenom_dx1) / denom**2
    g2 = (dnum_dr * dr_dx2 * denom - num * ddenom_dx2) / denom**2
    return np.array([g1, g2])

def schaffer_f6_hess(x: np.ndarray) -> np.ndarray:
    # Numerical Hessian via finite differences for robustness
    n = len(x)
    H = np.zeros((n, n))
    eps = 1e-5
    g0 = schaffer_f6_grad(x)
    for i in range(n):
        xp = x.copy()
        xp[i] += eps
        gp = schaffer_f6_grad(xp)
        H[:, i] = (gp - g0) / eps
    return (H + H.T) / 2


# ============================================================
# 5c. Michalewicz multimodal function (n-D, steep valleys)
# ============================================================
def michalewicz(x: np.ndarray, m: float = 10.0) -> float:
    n = len(x)
    total = 0.0
    for i in range(n):
        xi = x[i]
        total += np.sin(xi) * (np.sin((i + 1) * xi**2 / np.pi))**(2 * m)
    return float(-total)

def michalewicz_grad(x: np.ndarray, m: float = 10.0) -> np.ndarray:
    n = len(x)
    g = np.zeros(n)
    for i in range(n):
        xi = x[i]
        idx = i + 1
        sin_xi = np.sin(xi)
        inner = np.sin(idx * xi**2 / np.pi)
        power = 2 * m
        if abs(inner) < 1e-12:
            g[i] = np.cos(xi) * inner**power
        else:
            d_inner = (2 * idx * xi / np.pi) * np.cos(idx * xi**2 / np.pi)
            g[i] = np.cos(xi) * inner**power + sin_xi * power * inner**(power - 1) * d_inner
    return -g

def michalewicz_hess(x: np.ndarray, m: float = 10.0) -> np.ndarray:
    n = len(x)
    H = np.zeros((n, n))
    eps = 1e-5
    g0 = michalewicz_grad(x, m)
    for i in range(n):
        xp = x.copy()
        xp[i] += eps
        gp = michalewicz_grad(xp, m)
        H[:, i] = (gp - g0) / eps
    return (H + H.T) / 2


# ============================================================
# 6. Simulink equivalent algebraic loop problems
# ============================================================
def simulink_linear_loop(x: np.ndarray) -> float:
    y = x[0]
    residual = y - 0.5 * y - 1.0
    return float(residual ** 2)

def simulink_linear_loop_grad(x: np.ndarray) -> np.ndarray:
    y = x[0]
    residual = y - 0.5 * y - 1.0
    return np.array([2.0 * residual * 0.5])

def simulink_linear_loop_hess(x: np.ndarray) -> np.ndarray:
    return np.array([[0.5]])


def simulink_nonlinear_loop(x: np.ndarray) -> float:
    y1, y2 = x[0], x[1]
    r1 = y1 - 0.3 * y1 - 0.5 * y2 - np.sin(y1 * y2) - 0.1
    r2 = y2 - 0.4 * y2 - 0.2 * y1 ** 2 - 0.3
    return float(r1 ** 2 + r2 ** 2)

def simulink_nonlinear_loop_grad(x: np.ndarray) -> np.ndarray:
    y1, y2 = x[0], x[1]
    r1 = y1 - 0.3 * y1 - 0.5 * y2 - np.sin(y1 * y2) - 0.1
    r2 = y2 - 0.4 * y2 - 0.2 * y1 ** 2 - 0.3
    dr1_dy1 = 0.7 - y2 * np.cos(y1 * y2)
    dr1_dy2 = -0.5 - y1 * np.cos(y1 * y2)
    dr2_dy1 = -0.4 * y1
    dr2_dy2 = 0.6
    g1 = 2 * r1 * dr1_dy1 + 2 * r2 * dr2_dy1
    g2 = 2 * r1 * dr1_dy2 + 2 * r2 * dr2_dy2
    return np.array([g1, g2])

def simulink_nonlinear_loop_hess(x: np.ndarray) -> np.ndarray:
    y1, y2 = x[0], x[1]
    r1 = y1 - 0.3*y1 - 0.5*y2 - np.sin(y1*y2) - 0.1
    r2 = y2 - 0.4*y2 - 0.2*y1**2 - 0.3
    dr1_dy1 = 0.7 - y2 * np.cos(y1*y2)
    dr1_dy2 = -0.5 - y1 * np.cos(y1*y2)
    d2r1_dy12 = y2**2 * np.sin(y1*y2)
    d2r1_dy1dy2 = -np.cos(y1*y2) + y1*y2 * np.sin(y1*y2)
    d2r1_dy22 = y1**2 * np.sin(y1*y2)
    dr2_dy1 = -0.4 * y1
    dr2_dy2 = 0.6
    d2r2_dy12 = -0.4
    H = np.zeros((2, 2))
    H[0,0] = 2*(dr1_dy1**2 + r1*d2r1_dy12 + dr2_dy1**2 + r2*d2r2_dy12)
    H[0,1] = 2*(dr1_dy1*dr1_dy2 + r1*d2r1_dy1dy2)
    H[1,0] = H[0,1]
    H[1,1] = 2*(dr1_dy2**2 + r1*d2r1_dy22 + dr2_dy2**2)
    return H


# ============================================================
# CI computation  (all three components are computable BEFORE solving)
# CI = w1*D + w2*Delta + w3*S
#   D     : dimension score  = min(dim/10, 1)  -> Jacobian size / conditioning
#   Delta : residual score   = min(log10(1+||grad f(x0)||)/L, 1)
#           -> global-search difficulty measured by the initial gradient
#              (residual) norm at the starting point; computable online,
#              unlike a distance-to-known-solution term.
#   S     : stiffness score  = min(cond_est/100, 1) -> nonlinear stiffness / ill-conditioning
# ============================================================
def ci_components(dim: int, x0: np.ndarray, grad,
                  condition_est: float = 1.0, L: float = 8.0):
    """Return the three normalized CI components (D, Delta, S).

    All quantities are available at solve time without knowledge of the
    solution x*: the gradient norm is evaluated at the starting point.
    """
    D = min(dim / 10.0, 1.0)
    g = np.asarray(grad(x0), dtype=float).flatten()
    Delta = min(np.log10(1.0 + float(np.linalg.norm(g))) / L, 1.0)
    S = min(condition_est / 100.0, 1.0)
    return D, Delta, S


def compute_ci(dim: int, x0: np.ndarray, grad,
               condition_est: float = 1.0, w: tuple = (0.4, 0.3, 0.3),
               L: float = 8.0) -> float:
    D, Delta, S = ci_components(dim, x0, grad, condition_est, L)
    ci = w[0] * D + w[1] * Delta + w[2] * S
    return round(ci, 4)


# ============================================================
# Problem registry
# ============================================================
ROSENBROCK_OPT_2D = np.array([1.0, 1.0])
ROSENBROCK_OPT_10D = np.ones(10)
BEALE_OPT = np.array([3.0, 0.5])
POWELL_OPT = np.zeros(4)
WOOD_OPT = np.ones(4)
RASTRIGIN_OPT_2D = np.zeros(2)
SIMULINK_LINEAR_OPT = np.array([2.0])
SIMULINK_NONLINEAR_OPT = np.array([1.77864845, 1.5545301])  # exact solution found numerically (f*~0)

INIT_POINTS = {
    'Rosenbrock_2D': {
        '': np.array([-1.2, 1.0]),
        '_M': np.array([-2.0, 2.0]),
        '_H': np.array([-5.0, 5.0]),
        '_C': np.array([-10.0, 10.0]),
        '_E': np.array([-20.0, 20.0]),
    },
        'Beale_2D': {
        '': np.array([1.0, 1.0]),
        '_M': np.array([2.0, -1.0]),
        '_H': np.array([5.0, -3.0]),
        '_C': np.array([15.0, -10.0]),
        '_E': np.array([35.0, -22.0]),
        '_XE': np.array([80.0, -50.0]),
    },
    'Rosenbrock_10D': {
        '': np.array([-1.2, 1.0, -1.2, 1.0, -1.2, 1.0, -1.2, 1.0, -1.2, 1.0]),
        '_M': np.array([-2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0]),
        '_H': np.array([-5.0, 5.0, -5.0, 5.0, -5.0, 5.0, -5.0, 5.0, -5.0, 5.0]),
        '_C': np.array([-8.0, 8.0, -8.0, 8.0, -8.0, 8.0, -8.0, 8.0, -8.0, 8.0]),
        '_E': np.array([-12.0, 12.0, -12.0, 12.0, -12.0, 12.0, -12.0, 12.0, -12.0, 12.0]),
    },
    'Powell_4D': {
        '': np.array([3.0, -1.0, 0.0, 1.0]),
        '_M': np.array([5.0, -3.0, 2.0, -2.0]),
        '_H': np.array([10.0, -5.0, 5.0, -5.0]),
        '_C': np.array([15.0, -8.0, 8.0, -8.0]),
        '_E': np.array([20.0, -12.0, 12.0, -12.0]),
    },
    'Wood_4D': {
        '': np.array([-3.0, -1.0, -3.0, -1.0]),
        '_M': np.array([-5.0, 2.0, -5.0, 2.0]),
        '_H': np.array([-10.0, 5.0, -10.0, 5.0]),
        '_C': np.array([-15.0, 8.0, -15.0, 8.0]),
        '_E': np.array([-20.0, 12.0, -20.0, 12.0]),
    },
}


CONDITION_EST = {
    'Rosenbrock_2D': 30.0,
    'Beale_2D': 50.0,
    'Rosenbrock_10D': 80.0,
    'Powell_4D': 40.0,
    'Wood_4D': 60.0,
}

PROBLEMS: Dict[str, TestProblem] = {}
_base_configs = [
    ('Rosenbrock_2D', 2, rosenbrock, rosenbrock_grad, rosenbrock_hess, ROSENBROCK_OPT_2D, 0.0),
    ('Beale_2D', 2, beale, beale_grad, beale_hess, BEALE_OPT, 0.0),
    ('Rosenbrock_10D', 10, rosenbrock, rosenbrock_grad, rosenbrock_hess, ROSENBROCK_OPT_10D, 0.0),
    ('Powell_4D', 4, powell, powell_grad, powell_hess, POWELL_OPT, 0.0),
    ('Wood_4D', 4, wood, wood_grad, wood_hess, WOOD_OPT, 0.0),
]
_suffixes = ['', '_M', '_H', '_C', '_E']

for base_name, dim, f_func, g_func, h_func, x_opt, f_opt in _base_configs:
    for suffix in _suffixes:
        full_name = base_name + suffix
        x0 = INIT_POINTS[base_name][suffix]
        cond = CONDITION_EST[base_name]
        ci = compute_ci(dim, x0, g_func, cond)
        PROBLEMS[full_name] = TestProblem(
            name=full_name, dim=dim, f=f_func, grad=g_func, hess=h_func,
            x_opt=x_opt, f_opt=f_opt, x0=x0, ci_value=ci,
            description=f"{base_name} function, dim={dim}, difficulty={suffix or 'baseline'}"
        )
# Register the ultra-extreme Beale test case separately (only the Beale function has this tier, does not affect other functions)
_beale_xe_name = 'Beale_2D_XE'
_beale_xe_x0 = np.array([80.0, -50.0])
_beale_xe_cond = CONDITION_EST['Beale_2D']
_beale_xe_ci = compute_ci(2, _beale_xe_x0, beale_grad, _beale_xe_cond)
PROBLEMS[_beale_xe_name] = TestProblem(
    name=_beale_xe_name, dim=2, f=beale, grad=beale_grad, hess=beale_hess,
    x_opt=BEALE_OPT, f_opt=0.0, x0=_beale_xe_x0, ci_value=_beale_xe_ci,
    description="Beale function, dim=2, difficulty=extreme-far (x0=[80, -50])"
)

# Multimodal Rastrigin extreme test case: pure local methods easily get trapped in local optima, demonstrating the global search value of hybrid algorithms
_rastrigin_xe_name = 'Rastrigin_2D_XE'
_rastrigin_xe_x0 = np.array([15.0, 15.0])
_rastrigin_xe_cond = 100.0
_rastrigin_xe_ci = compute_ci(2, _rastrigin_xe_x0, rastrigin_grad, _rastrigin_xe_cond)
PROBLEMS[_rastrigin_xe_name] = TestProblem(
    name=_rastrigin_xe_name, dim=2, f=rastrigin, grad=rastrigin_grad, hess=rastrigin_hess,
    x_opt=RASTRIGIN_OPT_2D, f_opt=0.0, x0=_rastrigin_xe_x0, ci_value=_rastrigin_xe_ci,
    description="Rastrigin multimodal function, dim=2, extreme initial point [15, 15]"
)

# High-dimensional Rastrigin (20D) - tests scalability of global search
_rastrigin_20d_name = 'Rastrigin_20D_XE'
_rastrigin_20d_x0 = 20.0 * np.ones(20)
_rastrigin_20d_opt = np.zeros(20)
_rastrigin_20d_cond = 150.0
_rastrigin_20d_ci = compute_ci(20, _rastrigin_20d_x0, rastrigin_grad, _rastrigin_20d_cond)
PROBLEMS[_rastrigin_20d_name] = TestProblem(
    name=_rastrigin_20d_name, dim=20, f=rastrigin, grad=rastrigin_grad, hess=rastrigin_hess,
    x_opt=_rastrigin_20d_opt, f_opt=0.0, x0=_rastrigin_20d_x0, ci_value=_rastrigin_20d_ci,
    description="Rastrigin multimodal function, dim=20, extreme initial point 20*ones(20)"
)

# High-dimensional Rastrigin (30D) - tests high-dimensional scalability
_rastrigin_30d_name = 'Rastrigin_30D_XE'
_rastrigin_30d_x0 = 20.0 * np.ones(30)
_rastrigin_30d_opt = np.zeros(30)
_rastrigin_30d_cond = 200.0
_rastrigin_30d_ci = compute_ci(30, _rastrigin_30d_x0, rastrigin_grad, _rastrigin_30d_cond)
PROBLEMS[_rastrigin_30d_name] = TestProblem(
    name=_rastrigin_30d_name, dim=30, f=rastrigin, grad=rastrigin_grad, hess=rastrigin_hess,
    x_opt=_rastrigin_30d_opt, f_opt=0.0, x0=_rastrigin_30d_x0, ci_value=_rastrigin_30d_ci,
    description="Rastrigin multimodal function, dim=30, extreme initial point 20*ones(30)"
)

# Schaffer F6 multimodal (2D) - dense local minima with circular symmetry
_schaffer_name = 'Schaffer_F6_2D_XE'
_schaffer_x0 = np.array([100.0, 100.0])
_schaffer_opt = np.zeros(2)
_schaffer_cond = 120.0
_schaffer_ci = compute_ci(2, _schaffer_x0, schaffer_f6_grad, _schaffer_cond)
PROBLEMS[_schaffer_name] = TestProblem(
    name=_schaffer_name, dim=2, f=schaffer_f6, grad=schaffer_f6_grad, hess=schaffer_f6_hess,
    x_opt=_schaffer_opt, f_opt=0.0, x0=_schaffer_x0, ci_value=_schaffer_ci,
    description="Schaffer F6 multimodal function, dim=2, extreme initial point [100, 100]"
)

# Michalewicz multimodal (10D) - steep valleys, strong nonlinearity
_michalewicz_name = 'Michalewicz_10D_XE'
_michalewicz_x0 = 2.0 * np.ones(10)
_michalewicz_opt = np.array([2.20, 1.57, 1.92, 1.73, 1.64, 1.58, 1.53, 1.50, 1.47, 1.45])
_michalewicz_cond = 100.0
_michalewicz_ci = compute_ci(10, _michalewicz_x0, michalewicz_grad, _michalewicz_cond)
PROBLEMS[_michalewicz_name] = TestProblem(
    name=_michalewicz_name, dim=10, f=michalewicz, grad=michalewicz_grad, hess=michalewicz_hess,
    x_opt=_michalewicz_opt, f_opt=michalewicz(_michalewicz_opt), x0=_michalewicz_x0, ci_value=_michalewicz_ci,
    description="Michalewicz multimodal function (m=10), dim=10, initial point 2*ones(10)"
)
PROBLEMS['Simulink_Linear_Loop'] = TestProblem(
    name='Simulink_Linear_Loop', dim=1,
    f=simulink_linear_loop, grad=simulink_linear_loop_grad,
    hess=simulink_linear_loop_hess,
    x_opt=SIMULINK_LINEAR_OPT, f_opt=0.0, x0=np.array([0.0]),
    ci_value=compute_ci(1, np.array([0.0]), simulink_linear_loop_grad, 5.0),
    description='Simulink first-order linear algebraic loop: y = 0.5*y + 1.0'
)
PROBLEMS['Simulink_Nonlinear_Loop'] = TestProblem(
    name='Simulink_Nonlinear_Loop', dim=2,
    f=simulink_nonlinear_loop, grad=simulink_nonlinear_loop_grad,
    hess=simulink_nonlinear_loop_hess,
    x_opt=SIMULINK_NONLINEAR_OPT, f_opt=0.0, x0=np.array([0.0, 0.0]),
    ci_value=compute_ci(2, np.array([0.0, 0.0]), simulink_nonlinear_loop_grad, 20.0),
    description='Simulink two-stage nested nonlinear algebraic loop'
)


# ============================================================
# Non-separable high-dimensional multimodal functions
# (rotated to break separability, critical for high-dim validation)
# ============================================================
from .nonseparable_functions import (
    rotated_rastrigin_factory, griewank_factory, rotated_ackley_factory
)

# 1. Rotated Rastrigin 20D - non-separable via orthogonal rotation
_rr20_f, _rr20_g, _rr20_h, _rr20_opt = rotated_rastrigin_factory(20, seed=42)
_rr20_x0 = 20.0 * np.ones(20)
_rr20_ci = compute_ci(20, _rr20_x0, _rr20_g, 200.0)
PROBLEMS['Rotated_Rastrigin_20D'] = TestProblem(
    name='Rotated_Rastrigin_20D', dim=20, f=_rr20_f, grad=_rr20_g, hess=_rr20_h,
    x_opt=_rr20_opt, f_opt=0.0, x0=_rr20_x0, ci_value=_rr20_ci,
    description='Rotated Rastrigin (non-separable), dim=20, x0=20*ones, orthogonal rotation seed=42'
)

# 2. Griewank 20D - non-separable via product term
_gw20_f, _gw20_g, _gw20_h, _gw20_opt = griewank_factory(20)
_gw20_x0 = 60.0 * np.ones(20)
_gw20_ci = compute_ci(20, _gw20_x0, _gw20_g, 100.0)
PROBLEMS['Griewank_20D'] = TestProblem(
    name='Griewank_20D', dim=20, f=_gw20_f, grad=_gw20_g, hess=_gw20_h,
    x_opt=_gw20_opt, f_opt=0.0, x0=_gw20_x0, ci_value=_gw20_ci,
    description='Griewank function (non-separable via product term), dim=20, x0=60*ones'
)

# 3. Rotated Ackley 20D - non-separable via orthogonal rotation
_ra20_f, _ra20_g, _ra20_h, _ra20_opt = rotated_ackley_factory(20, seed=42)
_ra20_x0 = 20.0 * np.ones(20)
_ra20_ci = compute_ci(20, _ra20_x0, _ra20_g, 150.0)
PROBLEMS['Rotated_Ackley_20D'] = TestProblem(
    name='Rotated_Ackley_20D', dim=20, f=_ra20_f, grad=_ra20_g, hess=_ra20_h,
    x_opt=_ra20_opt, f_opt=0.0, x0=_ra20_x0, ci_value=_ra20_ci,
    description='Rotated Ackley (non-separable), dim=20, x0=20*ones, orthogonal rotation seed=42'
)


# ============================================================
# Numba JIT acceleration (optional): replace the f/grad of the most time-consuming multimodal function
# ============================================================
try:
    from .numba_kernels import (
        NUMBA_AVAILABLE, rastrigin_f, rastrigin_g,
        schaffer_f6_f, schaffer_f6_g,
        michalewicz_f10, michalewicz_g10,
        rotated_rastrigin_f, rotated_rastrigin_g,
        rotated_ackley_f, rotated_ackley_g,
        griewank_f, griewank_g,
    )
    from .nonseparable_functions import make_orthogonal_matrix
    if NUMBA_AVAILABLE:
        _NB_MAP = {
            'Rastrigin_2D_XE': (rastrigin_f, rastrigin_g),
            'Rastrigin_20D_XE': (rastrigin_f, rastrigin_g),
            'Rastrigin_30D_XE': (rastrigin_f, rastrigin_g),
            'Schaffer_F6_2D_XE': (schaffer_f6_f, schaffer_f6_g),
            'Michalewicz_10D_XE': (michalewicz_f10, michalewicz_g10),
        }
        for _nb_name, (_nb_f, _nb_g) in _NB_MAP.items():
            if _nb_name in PROBLEMS:
                PROBLEMS[_nb_name].f = _nb_f
                PROBLEMS[_nb_name].grad = _nb_g

        # Rotated function: fixed seed=42, dim=20 to regenerate the rotation matrix
        _R20 = make_orthogonal_matrix(20, seed=42)
        _Rt20 = _R20.T

        def _rr20_f_nb(x):
            return rotated_rastrigin_f(x, _R20)

        def _rr20_g_nb(x):
            return rotated_rastrigin_g(x, _R20, _Rt20)

        def _ra20_f_nb(x):
            return rotated_ackley_f(x, _R20)

        def _ra20_g_nb(x):
            return rotated_ackley_g(x, _R20, _Rt20)

        _sqrt_idx20 = np.sqrt(np.arange(1, 21, dtype=float))

        def _gw20_f_nb(x):
            return griewank_f(x, _sqrt_idx20)

        def _gw20_g_nb(x):
            return griewank_g(x, _sqrt_idx20)

        _NB_ROTATED_MAP = {
            'Rotated_Rastrigin_20D': (_rr20_f_nb, _rr20_g_nb),
            'Rotated_Ackley_20D': (_ra20_f_nb, _ra20_g_nb),
            'Griewank_20D': (_gw20_f_nb, _gw20_g_nb),
        }
        for _nb_name, (_nb_f, _nb_g) in _NB_ROTATED_MAP.items():
            if _nb_name in PROBLEMS:
                PROBLEMS[_nb_name].f = _nb_f
                PROBLEMS[_nb_name].grad = _nb_g
except Exception:
    pass  # Silently fall back to pure NumPy when numba is unavailable or compilation fails


# ============================================================
# CI-based legacy configuration
# ============================================================
def get_hastn_config(ci: float) -> Dict:
    """Three-tier parameter configuration based on problem complexity."""
    if ci < 0.3:
        return {'T0': 500, 'alpha': 0.98, 'sigma': 1.0, 'n_restarts': 2,
                'sa_iter_per_restart': 300, 'tier': 'low'}
    elif ci < 0.7:
        # Increase the search intensity of medium-difficulty tiers, adapting to 2D multimodal extreme scenarios
        return {'T0': 1500, 'alpha': 0.98, 'sigma': 2.0, 'n_restarts': 6,
                'sa_iter_per_restart': 267, 'tier': 'medium'}
    else:
        return {'T0': 1500, 'alpha': 0.98, 'sigma': 1.5, 'n_restarts': 6,
                'sa_iter_per_restart': 267, 'tier': 'high'}


def get_problem(name: str) -> TestProblem:
    return PROBLEMS[name]

def get_all_problem_names() -> List[str]:
    return list(PROBLEMS.keys())


if __name__ == '__main__':
    print("=== Test Problem Registry ===")
    for name, p in PROBLEMS.items():
        config = get_hastn_config(p.ci_value)
        print(f"{name:30s} dim={p.dim:2d}  CI={p.ci_value:.4f}  tier={config['tier']:6s}  "
              f"T0={config['T0']:4d}  restarts={config['n_restarts']}")
    print(f"\nTotal problems: {len(PROBLEMS)}")