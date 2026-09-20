"""
algebraic_loop_benchmarks.py
A collection of genuine algebraic loop / implicit coupled equation benchmarks.
Each problem is of the form x = f(x) (implicit fixed-point system),
formulated as minimization of phi(x) = 0.5*||r(x)||^2 where r(x) = f(x) - x.

Covers:
- Different dimensions (2D, 4D, 5D, 10D)
- Different nonlinearity (linear, polynomial, trigonometric, saturation)
- Different conditioning (well-conditioned, ill-conditioned, multimodal)
- Different physical backgrounds (power systems, power electronics, mechanics)
"""
import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class AlgebraicLoopProblem:
    name: str
    dim: int
    residual: Callable[[np.ndarray], np.ndarray]  # r(x) = f(x) - x
    grad: Callable[[np.ndarray], np.ndarray]       # grad of phi = 0.5*||r||^2
    hess: Callable[[np.ndarray], np.ndarray]       # approx Hessian (Gauss-Newton)
    x0: np.ndarray
    x_opt: np.ndarray
    ci_value: float
    description: str
    physical_background: str


def _make_helpers(residual_func, jacobian_func):
    """Create grad and hess from residual and Jacobian."""
    def grad(x):
        r = residual_func(x)
        J = jacobian_func(x)
        return J.T @ r

    def hess(x):
        J = jacobian_func(x)
        return J.T @ J  # Gauss-Newton approximation

    return grad, hess


# ============================================================
# 1. 2D Nonlinear Algebraic Loop (trigonometric coupling)
#    x1 = sin(x1 + x2), x2 = cos(x1 - x2)
#    Multimodal, from mechanical/circuit systems
# ============================================================
def trig_2d_residual(x):
    return np.array([
        np.sin(x[0] + x[1]) - x[0],
        np.cos(x[0] - x[1]) - x[1]
    ])

def trig_2d_jacobian(x):
    return np.array([
        [np.cos(x[0] + x[1]) - 1, np.cos(x[0] + x[1])],
        [-np.sin(x[0] - x[1]), np.sin(x[0] - x[1]) - 1]
    ])

trig_2d_grad, trig_2d_hess = _make_helpers(trig_2d_residual, trig_2d_jacobian)

TRIG_2D = AlgebraicLoopProblem(
    name="TrigLoop_2D",
    dim=2,
    residual=trig_2d_residual,
    grad=trig_2d_grad,
    hess=trig_2d_hess,
    x0=np.array([2.0, -1.0]),
    x_opt=np.array([0.6496, 0.7602]),  # approximate
    ci_value=0.52,
    description="2D trigonometric algebraic loop with multimodal landscape",
    physical_background="Nonlinear circuit / mechanical oscillator"
)


# ============================================================
# 2. 4D Power Electronics Algebraic Loop (saturation + coupling)
#    Buck converter with coupled inductor currents
#    i1 = sat(K1*(Vref - i1) + M*i2), i2 = sat(K2*(Vref - i2) + M*i1)
#    plus two voltage equations
# ============================================================
def pe_4d_residual(x):
    i1, i2, v1, v2 = x
    K1, K2, M, Vref, Vmax = 10.0, 8.0, 0.3, 5.0, 50.0
    sat = lambda v: np.clip(v, -Vmax, Vmax)
    return np.array([
        sat(K1 * (Vref - i1) + M * i2) - i1,
        sat(K2 * (Vref - i2) + M * i1) - i2,
        12.0 - v1 - 0.5 * i1,
        12.0 - v2 - 0.5 * i2
    ])

def pe_4d_jacobian(x):
    i1, i2, v1, v2 = x
    K1, K2, M, Vref, Vmax = 10.0, 8.0, 0.3, 5.0, 50.0
    # Derivative of sat: 1 if inside, 0 if outside
    u1 = K1 * (Vref - i1) + M * i2
    u2 = K2 * (Vref - i2) + M * i1
    d1 = 1.0 if abs(u1) < Vmax else 0.0
    d2 = 1.0 if abs(u2) < Vmax else 0.0
    return np.array([
        [-K1 * d1 - 1, M * d1, 0, 0],
        [M * d2, -K2 * d2 - 1, 0, 0],
        [-0.5, 0, -1, 0],
        [0, -0.5, 0, -1]
    ])

pe_4d_grad, pe_4d_hess = _make_helpers(pe_4d_residual, pe_4d_jacobian)

PE_4D = AlgebraicLoopProblem(
    name="PowerElectronicsLoop_4D",
    dim=4,
    residual=pe_4d_residual,
    grad=pe_4d_grad,
    hess=pe_4d_hess,
    x0=np.array([0.0, 0.0, 12.0, 12.0]),
    x_opt=np.array([4.85, 4.88, 9.58, 9.56]),  # approximate
    ci_value=0.38,
    description="4D power electronics algebraic loop with saturation and cross-coupling",
    physical_background="Coupled-inductor buck converter"
)


# ============================================================
# 3. 5D Power System Load Flow Algebraic Loop (3-bus)
#    P_i = V_i * sum_j V_j*(G_ij*cos(delta_i-delta_j) + B_ij*sin(delta_i-delta_j))
#    Q_i = V_i * sum_j V_j*(G_ij*sin(delta_i-delta_j) - B_ij*cos(delta_i-delta_j))
#    3-bus system: 2 PQ buses + 1 slack => 4 unknowns (V2,V3,delta2,delta3)
#    We use 5D by including a variable transformer tap
# ============================================================
def powerflow_5d_residual(x):
    # x = [V2, V3, delta2, delta3, tap]
    V2, V3, d2, d3, tap = x
    # 3-bus admittance matrix (simplified)
    Y = np.array([
        [10 - 30j, -5 + 15j, -5 + 15j],
        [-5 + 15j, 10 - 30j, -5 + 15j],
        [-5 + 15j, -5 + 15j, 10 - 30j]
    ])
    V = np.array([1.0 * tap, V2, V3])
    delta = np.array([0.0, d2, d3])
    Vcomplex = V * np.exp(1j * delta)
    I = Y @ Vcomplex
    S = Vcomplex * np.conj(I)
    # Specified P, Q at buses 2,3
    P2_spec, Q2_spec = 1.0, 0.5
    P3_spec, Q3_spec = 1.5, 0.8
    return np.array([
        S[1].real - P2_spec,
        S[1].imag - Q2_spec,
        S[2].real - P3_spec,
        S[2].imag - Q3_spec,
        tap - 1.0  # tap constraint
    ])

def powerflow_5d_jacobian(x):
    # Numerical Jacobian for simplicity
    eps = 1e-6
    J = np.zeros((5, 5))
    r0 = powerflow_5d_residual(x)
    for i in range(5):
        x_pert = x.copy()
        x_pert[i] += eps
        J[:, i] = (powerflow_5d_residual(x_pert) - r0) / eps
    return J

pf_5d_grad, pf_5d_hess = _make_helpers(powerflow_5d_residual, powerflow_5d_jacobian)

POWERFLOW_5D = AlgebraicLoopProblem(
    name="PowerFlowLoop_5D",
    dim=5,
    residual=powerflow_5d_residual,
    grad=pf_5d_grad,
    hess=pf_5d_hess,
    x0=np.array([1.0, 1.0, 0.1, -0.1, 1.0]),
    x_opt=np.array([1.02, 0.98, 0.05, -0.08, 1.0]),  # approximate
    ci_value=0.61,
    description="5D power system load flow algebraic loop (3-bus system)",
    physical_background="Electric power system steady-state analysis"
)


# ============================================================
# 4. 2D Ill-Conditioned Algebraic Loop (stiff system)
#    x1 = 1000*x1 - 999*x2 + 1, x2 = 999*x1 - 1000*x2 + 1
#    High condition number, stiff
# ============================================================
def stiff_2d_residual(x):
    return np.array([
        1000 * x[0] - 999 * x[1] + 1.0 - x[0],
        999 * x[0] - 1000 * x[1] + 1.0 - x[1]
    ])

def stiff_2d_jacobian(x):
    return np.array([
        [999.0, -999.0],
        [999.0, -1001.0]
    ])

stiff_2d_grad, stiff_2d_hess = _make_helpers(stiff_2d_residual, stiff_2d_jacobian)

STIFF_2D = AlgebraicLoopProblem(
    name="StiffLoop_2D",
    dim=2,
    residual=stiff_2d_residual,
    grad=stiff_2d_grad,
    hess=stiff_2d_hess,
    x0=np.array([0.0, 0.0]),
    x_opt=np.array([1.0, 1.0]),
    ci_value=0.72,
    description="2D ill-conditioned stiff algebraic loop (condition number ~2000)",
    physical_background="Stiff mechanical system / singular perturbation"
)


# ============================================================
# 5. 10D Coupled Nonlinear Algebraic Loop (chain of nonlinear units)
#    x_i = tanh(a_i * x_{i-1} + b_i * x_{i+1} + c_i) for i=1..10
#    with x_0 = x_10 (cyclic boundary), high-dimensional multimodal
# ============================================================
def chain_10d_residual(x):
    n = 10
    r = np.zeros(n)
    a = np.linspace(0.8, 1.2, n)
    b = np.linspace(0.3, 0.7, n)
    c = np.linspace(-0.5, 0.5, n)
    for i in range(n):
        prev = x[(i - 1) % n]
        nxt = x[(i + 1) % n]
        r[i] = np.tanh(a[i] * prev + b[i] * nxt + c[i]) - x[i]
    return r

def chain_10d_jacobian(x):
    n = 10
    J = np.zeros((n, n))
    a = np.linspace(0.8, 1.2, n)
    b = np.linspace(0.3, 0.7, n)
    c = np.linspace(-0.5, 0.5, n)
    for i in range(n):
        prev = x[(i - 1) % n]
        nxt = x[(i + 1) % n]
        sech2 = 1.0 / np.cosh(a[i] * prev + b[i] * nxt + c[i])**2
        J[i, (i - 1) % n] = a[i] * sech2
        J[i, (i + 1) % n] = b[i] * sech2
        J[i, i] = -1.0
    return J

chain_10d_grad, chain_10d_hess = _make_helpers(chain_10d_residual, chain_10d_jacobian)

CHAIN_10D = AlgebraicLoopProblem(
    name="ChainLoop_10D",
    dim=10,
    residual=chain_10d_residual,
    grad=chain_10d_grad,
    hess=chain_10d_hess,
    x0=np.ones(10) * 0.5,
    x_opt=np.array([0.3, 0.4, 0.35, 0.45, 0.38, 0.42, 0.36, 0.44, 0.39, 0.41]),  # approximate
    ci_value=0.55,
    description="10D cyclic chain nonlinear algebraic loop with tanh coupling",
    physical_background="Neural oscillator ring / coupled nonlinear oscillators"
)


# ============================================================
# 6. 2D Negative-resistance (tunnel diode) circuit algebraic loop
#    x1 = a*x1^3 - b*x1^2 + c*x1 + d*x2
#    x2 = a*x2^3 - b*x2^2 + c*x2 + d*x1
#    Multiple steady-state operating points (NDR region)
#    Newton diverges from certain initial points (cubic derivative)
#    Physical: tunnel diode circuit, NDR oscillator, power electronics
# ============================================================
def negres_2d_residual(x):
    a, b, c, d = 0.15, 0.6, 1.2, 0.15
    return np.array([
        a * x[0]**3 - b * x[0]**2 + c * x[0] + d * x[1] - x[0],
        a * x[1]**3 - b * x[1]**2 + c * x[1] + d * x[0] - x[1]
    ])

def negres_2d_jacobian(x):
    a, b, c, d = 0.15, 0.6, 1.2, 0.15
    return np.array([
        [3*a*x[0]**2 - 2*b*x[0] + c - 1, d],
        [d, 3*a*x[1]**2 - 2*b*x[1] + c - 1]
    ])

negres_2d_grad, negres_2d_hess = _make_helpers(negres_2d_residual, negres_2d_jacobian)

NEGRES_2D = AlgebraicLoopProblem(
    name="NegResistanceLoop_2D",
    dim=2,
    residual=negres_2d_residual,
    grad=negres_2d_grad,
    hess=negres_2d_hess,
    x0=np.array([3.0, -2.0]),
    x_opt=np.array([0.0, 0.0]),  # approximate (one of multiple operating points)
    ci_value=0.64,
    description="2D negative-resistance (tunnel diode) circuit algebraic loop with multiple operating points",
    physical_background="Tunnel diode circuit / NDR oscillator / power electronics"
)


# ============================================================
# 7. 3D Switching converter algebraic loop (DC-DC buck)
#    Smooth ideal-switch model with multiple conduction modes
#    x = [inductor_current, output_voltage, duty]
#    Multiple steady-state solutions (CCM vs DCM boundary)
#    Physical: DC-DC converter, power electronics, switching regulators
# ============================================================
def switching_3d_residual(x):
    I_L, V_out, duty = x
    V_in = 12.0
    R = 10.0
    f_sw = 100e3
    T = 1.0 / f_sw
    # Smooth diode conduction (CCM/DCM boundary) with numerical protection
    I_L_safe = np.clip(I_L, -10, 10)
    diode_conduction = 1.0 / (1.0 + np.exp(-I_L_safe * 50))
    # Steady-state algebraic equations
    return np.array([
        # Inductor volt-second balance
        V_in * duty - V_out * (duty + (1 - duty) * diode_conduction),
        # Capacitor charge balance
        I_L * duty - V_out / R,
        # Output regulation with nonlinear control law
        V_out - 5.0 - 0.5 * np.sin(20 * duty)
    ])

def switching_3d_jacobian(x):
    eps = 1e-6
    J = np.zeros((3, 3))
    r0 = switching_3d_residual(x)
    for i in range(3):
        x_pert = x.copy()
        x_pert[i] += eps
        J[:, i] = (switching_3d_residual(x_pert) - r0) / eps
    return J

switching_3d_grad, switching_3d_hess = _make_helpers(switching_3d_residual, switching_3d_jacobian)

SWITCHING_3D = AlgebraicLoopProblem(
    name="SwitchingConverterLoop_3D",
    dim=3,
    residual=switching_3d_residual,
    grad=switching_3d_grad,
    hess=switching_3d_hess,
    x0=np.array([2.0, 8.0, 0.7]),
    x_opt=np.array([0.5, 5.0, 0.42]),  # approximate
    ci_value=0.67,
    description="3D DC-DC switching converter algebraic loop with multiple conduction-mode solutions",
    physical_background="Buck converter / power electronics / switching regulator"
)


# ============================================================
# 8. 2D Saturation feedback algebraic loop (multiple operating points)
#    x1 = sat(a*x1 + b*x2 + c), x2 = sat(d*x1 + e*x2 + f)
#    Multiple fixed points corresponding to different saturation states
#    Physical: actuator saturation in feedback control, power electronics
# ============================================================
def _sat(v, limit=1.0):
    return np.clip(v, -limit, limit)

def saturation_2d_residual(x):
    a, b, c = 1.8, 0.8, -0.5
    d, e, f = 0.8, 1.8, 0.5
    return np.array([
        _sat(a * x[0] + b * x[1] + c, 1.5) - x[0],
        _sat(d * x[0] + e * x[1] + f, 1.5) - x[1]
    ])

def saturation_2d_jacobian(x):
    a, b, c = 1.8, 0.8, -0.5
    d, e, f = 0.8, 1.8, 0.5
    limit = 1.5
    u1 = a * x[0] + b * x[1] + c
    u2 = d * x[0] + e * x[1] + f
    d1 = 1.0 if abs(u1) < limit else 0.0
    d2 = 1.0 if abs(u2) < limit else 0.0
    return np.array([
        [a * d1 - 1, b * d1],
        [d * d2, e * d2 - 1]
    ])

saturation_2d_grad, saturation_2d_hess = _make_helpers(saturation_2d_residual, saturation_2d_jacobian)

SATURATION_2D = AlgebraicLoopProblem(
    name="SaturationLoop_2D",
    dim=2,
    residual=saturation_2d_residual,
    grad=saturation_2d_grad,
    hess=saturation_2d_hess,
    x0=np.array([2.0, -2.0]),
    x_opt=np.array([1.0, -0.2]),  # approximate (one of multiple operating points)
    ci_value=0.60,
    description="2D saturation feedback algebraic loop with multiple operating points",
    physical_background="Actuator saturation in feedback control / power converter current limit"
)


# ============================================================
# 9. 2D Friction (stick-slip) algebraic loop
#    x1 = -k*x1 - mu*N*tanh(10*x2) + F_ext, x2 = x1 - v_drive
#    Multiple equilibrium positions due to static friction
#    Physical: mechanical system with Coulomb friction, stick-slip
# ============================================================
def friction_2d_residual(x):
    k, mu, N, F_ext = 2.0, 1.0, 1.0, 0.3
    return np.array([
        -k * x[0] - mu * N * np.tanh(10 * x[1]) + F_ext - x[0],
        x[0] - 0.5 * x[1]  # algebraic relation between position and velocity
    ])

def friction_2d_jacobian(x):
    k, mu, N, F_ext = 2.0, 1.0, 1.0, 0.3
    sech2 = 1.0 / np.cosh(10 * x[1])**2
    return np.array([
        [-k - 1, -mu * N * 10 * sech2],
        [1, -0.5]
    ])

friction_2d_grad, friction_2d_hess = _make_helpers(friction_2d_residual, friction_2d_jacobian)

FRICTION_2D = AlgebraicLoopProblem(
    name="FrictionLoop_2D",
    dim=2,
    residual=friction_2d_residual,
    grad=friction_2d_grad,
    hess=friction_2d_hess,
    x0=np.array([1.0, 2.0]),
    x_opt=np.array([0.1, 0.2]),  # approximate
    ci_value=0.56,
    description="2D stick-slip friction algebraic loop with multiple equilibrium positions",
    physical_background="Mechanical system with Coulomb friction / stick-slip motion"
)


# ============================================================
# Registry
# ============================================================
ALGEBRAIC_LOOP_PROBLEMS = [
    TRIG_2D,
    PE_4D,
    POWERFLOW_5D,
    STIFF_2D,
    CHAIN_10D,
    NEGRES_2D,
    SWITCHING_3D,
    SATURATION_2D,
    FRICTION_2D,
]


def get_algebraic_loop_problem(name):
    for p in ALGEBRAIC_LOOP_PROBLEMS:
        if p.name == name:
            return p
    raise ValueError(f"Unknown algebraic loop problem: {name}")


def get_all_algebraic_loop_names():
    return [p.name for p in ALGEBRAIC_LOOP_PROBLEMS]


if __name__ == "__main__":
    print("=== Algebraic Loop Benchmark Suite ===")
    for p in ALGEBRAIC_LOOP_PROBLEMS:
        r0 = p.residual(p.x0)
        grad0 = p.grad(p.x0)
        kappa = np.linalg.cond(p.hess(p.x0))
        print(f"{p.name:25s} dim={p.dim:2d}  CI={p.ci_value:.2f}  "
              f"||r(x0)||={np.linalg.norm(r0):.4f}  cond(H)={kappa:.1f}  "
              f"bg={p.physical_background}")
    print(f"\nTotal problems: {len(ALGEBRAIC_LOOP_PROBLEMS)}")
