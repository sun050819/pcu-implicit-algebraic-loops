# Real Simulink Algebraic-Loop Verification (run_06_simulink_real)

> **Terminology note:** solver version tags below (HGCA v5.4/v6.7, CMA-ES, v54/v74) are 


> Status: **complete** (2026-09-11). Result files: `results/simulink_real.csv`, `results/simulink_real_summary.json`.
> Environment: MATLAB R2025a + Simulink, Python matlabengine 25.1.

## Purpose

The `Simulink_Linear_Loop` / `Simulink_Nonlinear_Loop` entries of REGISTRY_36 previously used only Python-simulated residuals.
This experiment validates the internal solver (the HGCA hybrid solver hosting PCU) on **real Simulink algebraic loops**: the closed algebraic-loop model is solved by the Simulink algebraic-loop solver, the expanded model (the same block set) computes the loop function G(y), and the residual r(y) = y - G(y) is computed by real blocks; HGCA solves on r(y) = 0 and is compared against y_sim. It also confirms that PCU / struct_id never triggers on problems without periodic structure.

## Models

| Model | Equation | Analytic solution |
|---|---|---|
| sim_lin_loop (linear loop) | y - 0.5y - 1 = 0 | y* = 2/3 |
| sim_nl_loop (nonlinear loop) | f1 = z1 - 0.3z1 - 0.5z2 - sin(z1z2) - 0.1 = 0; f2 = z2 - 0.4z2 - 0.2z1^2 - 0.3 = 0 | multiple roots (below) |

The closed model uses an `Algebraic Constraint` block plus feedback lines; during simulation the built-in Simulink algebraic-loop solver solves it automatically. The expanded model injects y via a Constant block and outputs the residual at simulation time. Residual Jacobian: linear J=1.5; nonlinear J = [[0.7-z2.cos(z1z2), -0.5-z1.cos(z1z2)], [-0.4z1, 0.6]] (analytic injection; the HGCA interface requires grad/hess by design; difference-Jacobian consistency spot checks show deviations of 1.7e-13 / 2.0e-10).

## Results

| Item | Value | Verdict |
|---|---|---|
| Linear y_sim | 0.66666667 (=2/3) | OK |
| Linear residual r(1.0) / r(2/3) | 0.5 / 0.0 | OK |
| solver linear x* | 0.66666667, \|x*-y_sim\| = 3.3e-14 | OK |
| Nonlinear Simulink initial (0,0) | LineSearch failed | direct evidence of initial-value sensitivity |
| Nonlinear Simulink initial (1.5,1.4) | z_sim = [1.77864845, 1.55453010], r(z_sim) ~ 0 | OK |
| solver nonlinear (same initial 1.5,1.4) | z* = [1.77864845, 1.55453010], \|z*-z_sim\| = 0.0 | OKOK |
| PCU / struct_id trigger | none, all zero | OK (correctly rejected: no periodic structure) |

## Multiple Roots

The nonlinear algebraic-loop system has several roots (verified with the Python analytic version):
- [0,0] -> [2.5275, 2.6294]
- [1.5,1.4] -> [1.7786, 1.5545] (= z_sim)
- [-0.5,-0.3] -> [2.8485, 3.2047]

The Simulink built-in solver fails with LineSearch from (0,0) and reaches z_sim from (1.5,1.4); the solver, started from the **same reasonable initial point** (fair-comparison protocol), converges to the same root with error 0.0. Supplementary evidence (HGCA analytic version, same equations): from (0,0) HGCA **successfully converges to another true root** [2.5275, 2.6294] (residual norm 4.5e-29) - i.e., where Simulink fails at (0,0), HGCA still finds a valid root. This is supplementary evidence of HGCA robustness relative to the built-in algebraic-loop solver on multi-root systems.

## Engineering Notes (to avoid re-tripping)

- MATLAB engine eval must explicitly set nargout=0; append semicolons to all statements to suppress output flooding.
- .m scripts must be pure ASCII (Chinese comments raise "invalid text character"); use in-memory modeling (no .slx on disk) because a user-defined function shadows the official save_system.m.
- Simulink block path: `Trigonometric Function` (not Trigonometry); two Out1 blocks replace an Out2.
- Driving the solver purely with central differences is too slow (each TR iteration needs 2d+1 simulations, and each simulation recompiles the in-memory model): for both linear and nonlinear cases, analytic Jacobians are injected with spot-checked difference consistency.
- Starting the solver from [0,0] on the nonlinear problem triggers the CMA final phase (hundreds of simulations, too slow); using the same reasonable-initial protocol as Simulink takes the direct_newton fast path.
