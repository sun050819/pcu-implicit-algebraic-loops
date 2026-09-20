"""nonseparable_functions.py - Non-separable high-dimensional multimodal functions"""
import numpy as np


def make_orthogonal_matrix(n, seed=42):
    """Generate a fixed orthogonal rotation matrix via QR decomposition."""
    rng = np.random.RandomState(seed)
    A = rng.randn(n, n)
    Q, R = np.linalg.qr(A)
    # Ensure deterministic sign convention
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q


# ============================================================
# 1. Rotated Rastrigin (20D) - non-separable via orthogonal rotation
# ============================================================
def rotated_rastrigin_factory(dim=20, seed=42):
    R = make_orthogonal_matrix(dim, seed)
    Rt = R.T

    def f(x):
        y = R @ x
        return 10.0 * dim + np.sum(y**2 - 10.0 * np.cos(2.0 * np.pi * y))

    def grad(x):
        y = R @ x
        grad_y = 2.0 * y + 20.0 * np.pi * np.sin(2.0 * np.pi * y)
        return Rt @ grad_y

    def hess(x):
        y = R @ x
        diag_h = 2.0 + 40.0 * np.pi**2 * np.cos(2.0 * np.pi * y)
        H_y = np.diag(diag_h)
        return Rt @ H_y @ R

    optimum = np.zeros(dim)  # R^T @ 0 = 0
    return f, grad, hess, optimum


# ============================================================
# 2. Griewank (20D) - non-separable via product term
# ============================================================
def griewank_factory(dim=20):
    idx = np.arange(1, dim + 1, dtype=float)
    sqrt_idx = np.sqrt(idx)

    def f(x):
        sum_sq = np.sum(x**2)
        prod_cos = np.prod(np.cos(x / sqrt_idx))
        return 1.0 + sum_sq / 4000.0 - prod_cos

    def grad(x):
        prod_cos = np.prod(np.cos(x / sqrt_idx))
        # d/dx_i [prod cos(x_j/sqrt(j))] = -prod_cos * tan(x_i/sqrt(i)) / sqrt(i)
        tan_terms = np.tan(x / sqrt_idx) / sqrt_idx
        return x / 2000.0 + prod_cos * tan_terms

    def hess(x):
        # Numerical Hessian via finite differences of gradient
        n = len(x)
        H = np.zeros((n, n))
        eps = 1e-5
        g0 = grad(x)
        for i in range(n):
            xp = x.copy()
            xp[i] += eps
            gp = grad(xp)
            H[:, i] = (gp - g0) / eps
        return (H + H.T) / 2.0

    optimum = np.zeros(dim)
    return f, grad, hess, optimum


# ============================================================
# 3. Rotated Ackley (20D) - non-separable via orthogonal rotation
# ============================================================
def rotated_ackley_factory(dim=20, seed=42):
    R = make_orthogonal_matrix(dim, seed)
    Rt = R.T

    def ackley_f(y):
        n = len(y)
        sum_sq = np.sum(y**2)
        sum_cos = np.sum(np.cos(2.0 * np.pi * y))
        return -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20.0 + np.e

    def ackley_grad(y):
        n = len(y)
        sum_sq = np.sum(y**2)
        sum_cos = np.sum(np.cos(2.0 * np.pi * y))
        term1 = 20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n)) * 0.2 / np.sqrt(n * sum_sq + 1e-30)
        term2 = (2.0 * np.pi / n) * np.exp(sum_cos / n) * np.sin(2.0 * np.pi * y)
        return term1 * y + term2

    def f(x):
        y = R @ x
        return ackley_f(y)

    def grad(x):
        y = R @ x
        return Rt @ ackley_grad(y)

    def hess(x):
        # Numerical Hessian
        n = len(x)
        H = np.zeros((n, n))
        eps = 1e-5
        g0 = grad(x)
        for i in range(n):
            xp = x.copy()
            xp[i] += eps
            gp = grad(xp)
            H[:, i] = (gp - g0) / eps
        return (H + H.T) / 2.0

    optimum = np.zeros(dim)
    return f, grad, hess, optimum


if __name__ == '__main__':
    print("=== Non-separable High-Dimensional Functions ===\n")

    # Rotated Rastrigin 20D
    f, g, h, xopt = rotated_rastrigin_factory(20)
    x0 = 20.0 * np.ones(20)
    print(f"Rotated Rastrigin 20D: f(x0)={f(x0):.2f}, f(opt)={f(xopt):.6f}")
    print(f"  grad norm at opt: {np.linalg.norm(g(xopt)):.2e}")
    print(f"  grad norm at x0: {np.linalg.norm(g(x0)):.2e}")

    # Griewank 20D
    f, g, h, xopt = griewank_factory(20)
    x0 = 20.0 * np.ones(20)
    print(f"\nGriewank 20D: f(x0)={f(x0):.2f}, f(opt)={f(xopt):.6f}")
    print(f"  grad norm at opt: {np.linalg.norm(g(xopt)):.2e}")
    print(f"  grad norm at x0: {np.linalg.norm(g(x0)):.2e}")

    # Rotated Ackley 20D
    f, g, h, xopt = rotated_ackley_factory(20)
    x0 = 20.0 * np.ones(20)
    print(f"\nRotated Ackley 20D: f(x0)={f(x0):.2f}, f(opt)={f(xopt):.6f}")
    print(f"  grad norm at opt: {np.linalg.norm(g(xopt)):.2e}")
    print(f"  grad norm at x0: {np.linalg.norm(g(x0)):.2e}")
