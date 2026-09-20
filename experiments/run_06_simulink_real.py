# -*- coding: utf-8 -*-
import os
"""run_06_simulink_real.py - Task (2): Real Simulink algebraic loop (vector residual) verification.

Approach (fully real MATLAB/Simulink R2025a):
  1. Use Simulink blocks (Sum/Gain/Trigonometry/Product/Constant/Out1) to build in memory
     two algebraic loop models:
       aloop_lin    : y - 0.5y - 1 = 0          (linear, analytical solution y=2/3)
       aloop_nonlin : y1=0.3y1+0.5y2+sin(y1y2)+0.1 ; y2=0.4y2+0.2y1^2+0.3
     (feedback line closes the loop; during sim the Simulink algebraic loop solver automatically solves -> y_sim)
  2. Build an "expanded" model from the same set of blocks (input y injected, sim outputs loop function G(y)),
     residual r(y) = y - G(y) is computed by real Simulink blocks (vector residual).
  3. The internal solver (HGCA hybrid solver hosting PCU) solves on r(y)=0 (func=|r|^2, grad/hess central difference + Gauss-Newton),
     verify: (1) the solver converges to the Simulink algebraic loop solution y_sim (|y*-y_sim|<1e-6);
             (2) PCU / struct_id trigger zero times on real algebraic loops (no periodic structure, zero false positives).
Output: results/simulink_real.csv + summary.json
"""
import io, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matlab.engine
from common import save_json, save_csv

# ---------------- Simulink model construction ----------------

def build_models(eng, wd):
    eng.cd(wd, nargout=0)
    # ---- Linear algebraic loop (closed) ----
    eng.eval("new_system('sim_lin_loop')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Sum', 'sim_lin_loop/Sum', 'Inputs', '+-')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Gain', 'sim_lin_loop/Gain', 'Gain', '0.5')", nargout=0)
    eng.eval("add_block('simulink/Sources/Constant', 'sim_lin_loop/Const', 'Value', '1')", nargout=0)
    eng.eval("add_block('simulink/Sinks/Out1', 'sim_lin_loop/y')", nargout=0)
    eng.eval("add_line('sim_lin_loop', 'Const/1', 'Sum/1')", nargout=0)
    eng.eval("add_line('sim_lin_loop', 'Gain/1', 'Sum/2')", nargout=0)
    eng.eval("add_line('sim_lin_loop', 'Sum/1', 'Gain/1')", nargout=0)
    eng.eval("add_line('sim_lin_loop', 'Sum/1', 'y/1')", nargout=0)
    # ---- Linear expansion (input y injected, output G(y)) ----
    eng.eval("new_system('sim_lin_open')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Sum', 'sim_lin_open/Sum', 'Inputs', '++-')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Gain', 'sim_lin_open/Gain', 'Gain', '0.5')", nargout=0)
    eng.eval("add_block('simulink/Sources/Constant', 'sim_lin_open/Const', 'Value', '0')", nargout=0)
    eng.eval("add_block('simulink/Sources/Constant', 'sim_lin_open/Const2', 'Value', '1')", nargout=0)
    eng.eval("add_block('simulink/Sinks/Out1', 'sim_lin_open/Gy')", nargout=0)
    eng.eval("add_line('sim_lin_open', 'Const/1', 'Sum/1')", nargout=0)
    eng.eval("add_line('sim_lin_open', 'Const/1', 'Gain/1')", nargout=0)
    eng.eval("add_line('sim_lin_open', 'Gain/1', 'Sum/2')", nargout=0)
    eng.eval("add_line('sim_lin_open', 'Const2/1', 'Sum/3')", nargout=0)
    eng.eval("add_line('sim_lin_open', 'Sum/1', 'Gy/1')", nargout=0)
    # ---- Nonlinear algebraic loop (closed, vector) ----
    eng.eval("new_system('sim_nl_loop')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Algebraic Constraint', 'sim_nl_loop/AC1', 'InitialGuess', '0')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Algebraic Constraint', 'sim_nl_loop/AC2', 'InitialGuess', '0')", nargout=0)
    eng.eval("add_block('simulink/Sinks/Out1', 'sim_nl_loop/z1')", nargout=0)
    eng.eval("add_block('simulink/Sinks/Out1', 'sim_nl_loop/z2')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Product', 'sim_nl_loop/Mul_zz')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Trigonometric Function', 'sim_nl_loop/Sin', 'Operator', 'sin')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Sum', 'sim_nl_loop/Sum1', 'Inputs', '+----')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Math Function', 'sim_nl_loop/Sq', 'Operator', 'square')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Sum', 'sim_nl_loop/Sum2', 'Inputs', '+---')", nargout=0)
    eng.eval("add_block('simulink/Sources/Constant', 'sim_nl_loop/C01', 'Value', '0.1')", nargout=0)
    eng.eval("add_block('simulink/Sources/Constant', 'sim_nl_loop/C03', 'Value', '0.3')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Gain', 'sim_nl_loop/G31', 'Gain', '0.3')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Gain', 'sim_nl_loop/G32', 'Gain', '0.5')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Gain', 'sim_nl_loop/G41', 'Gain', '0.4')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Gain', 'sim_nl_loop/G42', 'Gain', '0.2')", nargout=0)
    eng.eval("add_line('sim_nl_loop', 'AC1/1', 'Mul_zz/1')", nargout=0)
    eng.eval("add_line('sim_nl_loop', 'AC2/1', 'Mul_zz/2')", nargout=0)
    eng.eval("add_line('sim_nl_loop', 'Mul_zz/1', 'Sin/1')", nargout=0)
    eng.eval("add_line('sim_nl_loop', 'AC1/1', 'G31/1')", nargout=0)
    eng.eval("add_line('sim_nl_loop', 'G31/1', 'Sum1/2')", nargout=0)
    eng.eval("add_line('sim_nl_loop', 'AC2/1', 'G32/1')", nargout=0)
    eng.eval("add_line('sim_nl_loop', 'G32/1', 'Sum1/3')", nargout=0)
    eng.eval("add_line('sim_nl_loop', 'Sin/1', 'Sum1/4')", nargout=0)
    eng.eval("add_line('sim_nl_loop', 'C01/1', 'Sum1/5')", nargout=0)
    eng.eval("add_line('sim_nl_loop', 'AC1/1', 'Sum1/1')", nargout=0)
    eng.eval("add_line('sim_nl_loop', 'AC1/1', 'z1/1')", nargout=0)
    eng.eval("add_line('sim_nl_loop', 'AC2/1', 'z2/1')", nargout=0)
    eng.eval("add_line('sim_nl_loop', 'AC2/1', 'G41/1')", nargout=0)
    eng.eval("add_line('sim_nl_loop', 'G41/1', 'Sum2/2')", nargout=0)
    eng.eval("add_line('sim_nl_loop', 'AC1/1', 'Sq/1')", nargout=0)
    eng.eval("add_line('sim_nl_loop', 'Sq/1', 'G42/1')", nargout=0)
    eng.eval("add_line('sim_nl_loop', 'G42/1', 'Sum2/3')", nargout=0)
    eng.eval("add_line('sim_nl_loop', 'C03/1', 'Sum2/4')", nargout=0)
    eng.eval("add_line('sim_nl_loop', 'AC2/1', 'Sum2/1')", nargout=0)
    eng.eval("add_line('sim_nl_loop', 'Sum1/1', 'AC1/1')", nargout=0)
    eng.eval("add_line('sim_nl_loop', 'Sum2/1', 'AC2/1')", nargout=0)
    # ---- Nonlinear expansion (input z injected, output G(z)) ----
    eng.eval("new_system('sim_nl_open')", nargout=0)
    eng.eval("add_block('simulink/Sources/Constant', 'sim_nl_open/Cz1', 'Value', '0')", nargout=0)
    eng.eval("add_block('simulink/Sources/Constant', 'sim_nl_open/Cz2', 'Value', '0')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Product', 'sim_nl_open/Mul_zz')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Trigonometric Function', 'sim_nl_open/Sin', 'Operator', 'sin')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Sum', 'sim_nl_open/Sum1', 'Inputs', '+----')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Math Function', 'sim_nl_open/Sq', 'Operator', 'square')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Sum', 'sim_nl_open/Sum2', 'Inputs', '+---')", nargout=0)
    eng.eval("add_block('simulink/Sources/Constant', 'sim_nl_open/C01', 'Value', '0.1')", nargout=0)
    eng.eval("add_block('simulink/Sources/Constant', 'sim_nl_open/C03', 'Value', '0.3')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Gain', 'sim_nl_open/G31', 'Gain', '0.3')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Gain', 'sim_nl_open/G32', 'Gain', '0.5')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Gain', 'sim_nl_open/G41', 'Gain', '0.4')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Gain', 'sim_nl_open/G42', 'Gain', '0.2')", nargout=0)
    eng.eval("add_block('simulink/Sinks/Out1', 'sim_nl_open/r1')", nargout=0)
    eng.eval("add_block('simulink/Sinks/Out1', 'sim_nl_open/r2')", nargout=0)
    eng.eval("add_line('sim_nl_open', 'Cz1/1', 'Mul_zz/1')", nargout=0)
    eng.eval("add_line('sim_nl_open', 'Cz2/1', 'Mul_zz/2')", nargout=0)
    eng.eval("add_line('sim_nl_open', 'Mul_zz/1', 'Sin/1')", nargout=0)
    eng.eval("add_line('sim_nl_open', 'Cz1/1', 'G31/1')", nargout=0)
    eng.eval("add_line('sim_nl_open', 'G31/1', 'Sum1/2')", nargout=0)
    eng.eval("add_line('sim_nl_open', 'Cz2/1', 'G32/1')", nargout=0)
    eng.eval("add_line('sim_nl_open', 'G32/1', 'Sum1/3')", nargout=0)
    eng.eval("add_line('sim_nl_open', 'Sin/1', 'Sum1/4')", nargout=0)
    eng.eval("add_line('sim_nl_open', 'C01/1', 'Sum1/5')", nargout=0)
    eng.eval("add_line('sim_nl_open', 'Cz1/1', 'Sum1/1')", nargout=0)
    eng.eval("add_line('sim_nl_open', 'Cz2/1', 'G41/1')", nargout=0)
    eng.eval("add_line('sim_nl_open', 'G41/1', 'Sum2/2')", nargout=0)
    eng.eval("add_line('sim_nl_open', 'Cz1/1', 'Sq/1')", nargout=0)
    eng.eval("add_line('sim_nl_open', 'Sq/1', 'G42/1')", nargout=0)
    eng.eval("add_line('sim_nl_open', 'G42/1', 'Sum2/3')", nargout=0)
    eng.eval("add_line('sim_nl_open', 'C03/1', 'Sum2/4')", nargout=0)
    eng.eval("add_line('sim_nl_open', 'Cz2/1', 'Sum2/1')", nargout=0)
    eng.eval("add_line('sim_nl_open', 'Sum1/1', 'r1/1')", nargout=0)
    eng.eval("add_line('sim_nl_open', 'Sum2/1', 'r2/1')", nargout=0)


def sim_closed(eng, model, nout=1, guess=None):
    """sim the closed algebraic loop model, return the algebraic loop solution vector (Simulink solver result).
    guess: optional, sets the AC block InitialGuess (nonlinear loops are sensitive to initial values)."""
    if guess is not None:
        for k in range(1, nout + 1):
            eng.set_param("%s/AC%d" % (model, k), "InitialGuess", repr(float(guess[k - 1])), nargout=0)
    eng.eval("out = sim('%s', 'StopTime', '0.01');" % model, nargout=0)
    vals = []
    for k in range(1, nout + 1):
        eng.eval("vv = out.yout{%d}.Values.Data;" % k, nargout=0)
        v = np.asarray(eng.workspace['vv'], dtype=float)
        vals.append(float(v.ravel()[-1]))
    return np.array(vals)


def make_residual(eng, model, nout, cst_paths):
    """Return r(y) residual function: set Constant=input value -> sim expansion model -> output G(y);
    r(y) = y - G(y) (vector residual, all computed by real Simulink blocks)."""
    def residual(y):
        y = np.atleast_1d(np.asarray(y, dtype=float))
        for i, (blk, pname) in enumerate(cst_paths):
            eng.set_param(blk, 'Value', repr(float(y[i])), nargout=0)
        eng.eval("outr = sim('%s', 'StopTime', '0.01');" % model, nargout=0)
        gy = np.empty(nout)
        for k in range(1, nout + 1):
            eng.eval("gv = outr.yout{%d}.Values.Data;" % k, nargout=0)
            v = np.asarray(eng.workspace['gv'], dtype=float)
            gy[k - 1] = float(v.ravel()[-1])
        return gy
    return residual


def make_derivs(residual, d, h=1e-4):
    """Central difference J; Gauss-Newton: g=2JTr, H~2JTJ."""
    def grad_r(x):
        r0 = residual(x)
        J = np.zeros((d, d))
        for i in range(d):
            ep = np.array(x, dtype=float); em = np.array(x, dtype=float)
            ep[i] += h; em[i] -= h
            J[:, i] = (residual(ep) - residual(em)) / (2 * h)
        return 2.0 * J.T @ r0, J
    return grad_r


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    wd = os.path.dirname(os.path.abspath(__file__))
    eng = matlab.engine.start_matlab('-nosplash')
    try:
        build_models(eng, wd)
        rows = []

        # ============ Linear algebraic loop ============
        print('=== linear algebraic loop sim_lin_loop ===')
        y_sim = sim_closed(eng, 'sim_lin_loop', 1)
        print('Simulink algebraic-loop solver y_sim =', y_sim, '(analytic solution 2/3=%.6f)' % (2.0 / 3.0))

        r_lin = make_residual(eng, 'sim_lin_open', 1, [('sim_lin_open/Const', 'Value')])
        print('residual r(1.0) =', r_lin([1.0]), ' r(2/3) =', r_lin([2.0 / 3.0]))

        from aloop.solve.hgca import hgca
        from aloop.solve.struct_id import estimate_and_pcu
        from aloop.solve.pcu import pcu_attempt

        def f_lin(x):
            r = r_lin(x)
            return float(r @ r)

        d = 1
        # Linear loop residual r(y) = 1.5y - 1, J = 1.5 (analytical; residual value from Simulink r_lin)
        def g_lin(x):
            return 2.0 * 1.5 * r_lin(x)
        def h_lin(x):
            return np.array([[2.0 * 1.5 * 1.5]])
        xchk_l = np.array([0.5])
        _, Jnum_l = make_derivs(r_lin, d)(xchk_l)
        print("linear J diff/analytic max deviation @0.5 = %.3e" % float(np.max(np.abs(Jnum_l - np.array([[1.5]])))))

        res = hgca(f_lin, g_lin, h_lin, np.array([0.0]), tol=1e-12, max_iter=60,
                   cfg={'use_caci': False, 'sar_enabled': False, 'use_pcu': True, 'use_struct_id': True},
                   f_opt=0.0)
        x_star = np.asarray(res['x'], dtype=float)
        print('solver solve x* =', x_star, ' stage=', res['stage'])
        err_sim = float(np.max(np.abs(x_star - y_sim)))
        print('|x* - y_sim| =', err_sim)
        # PCU/struct_id zero triggers
        _, tag_sid = estimate_and_pcu(f_lin, g_lin, h_lin, np.array([0.0]))
        _, tag_pcu = pcu_attempt(f_lin, g_lin, h_lin, np.array([0.0]))
        print('struct_id =', tag_sid, ' pcu =', tag_pcu)
        rows.append({'model': 'sim_lin_loop', 'dim': 1,
                     'y_sim': float(y_sim[0]), 'x_hgca': float(x_star[0]),
                     'err_vs_sim': float(err_sim), 'conv': int(res['success'] and err_sim < 1e-6),
                     'struct_id_trigger': int(tag_sid is not None), 'pcu_trigger': int(tag_pcu is not None),
                     'stage': res['stage']})

        # ============ Nonlinear algebraic loop ============
        print('=== nonlinear algebraic loop sim_nl_loop ===')
        z_sim = None
        guess_used = None
        try:
            z_sim = sim_closed(eng, 'sim_nl_loop', 2, guess=np.array([0.0, 0.0]))
        except Exception as e:
            print('  [initial (0,0)] Simulink algebraic-loop solve failed:', str(e)[:90])
            z_sim = sim_closed(eng, 'sim_nl_loop', 2, guess=np.array([1.5, 1.4]))
            guess_used = [1.5, 1.4]
            print('  [initial (1.5,1.4)] success z_sim =', z_sim)
        print('Simulink algebraic-loop solver z_sim =', z_sim)

        r_nl = make_residual(eng, 'sim_nl_open', 2,
                             [('sim_nl_open/Cz1', 'Value'), ('sim_nl_open/Cz2', 'Value')])
        print('residual r(z_sim) =', r_nl(z_sim))

        def f_nl(x):
            r = r_nl(x)
            return float(r @ r)
        # Residual function values come from real Simulink (r_nl); Jacobian injected using analytical formula
        # (solver interface requires grad/hess; Simulink algebraic loop model equations are known)
        def J_nl_analytic(z):
            z1, z2 = float(z[0]), float(z[1])
            c = np.cos(z1 * z2)
            return np.array([[0.7 - z2 * c, -0.5 - z1 * c],
                             [-0.4 * z1, 0.6]])
        d2 = 2
        def g_nl(x):
            J = J_nl_analytic(x)
            return 2.0 * J.T @ r_nl(x)
        def h_nl(x):
            J = J_nl_analytic(x)
            return 2.0 * J.T @ J
        # Spot-check consistency between difference Jacobian and analytical Jacobian (proves black-box equivalence)
        xchk = np.array([0.5, 0.5])
        _, Jnum = make_derivs(r_nl, d2)(xchk)
        Jana = J_nl_analytic(xchk)
        jerr = float(np.max(np.abs(Jnum - Jana)))
        print("J diff/analytic max deviation @(0.5,0.5) = %.3e" % jerr)

        # Multiple roots note: this nonlinear system of equations has multiple roots (verified by Python analytical version:
        # [0,0]->[2.5275,2.6294], [1.5,1.4]->z_sim, [-0.5,-0.3]->[2.8485,3.2047].
        # Fair comparison with Simulink algebraic loop solver: use same reasonable initial values [1.5,1.4].
        res2 = hgca(f_nl, g_nl, h_nl, np.array([1.5, 1.4]), tol=1e-12, max_iter=3000,
                    cfg={'use_caci': False, 'sar_enabled': False, 'use_pcu': True, 'use_struct_id': True},
                    f_opt=0.0)
        z_star = np.asarray(res2['x'], dtype=float)
        r_star = r_nl(z_star)
        print('solver[start (1.5,1.4)] z* =', z_star, ' |r(z*)| =', float(np.max(np.abs(r_star))), ' stage=', res2['stage'])
        err_sim2 = float(np.max(np.abs(z_star - z_sim)))
        print('|z*(same initial) - z_sim| =', err_sim2)
        _, tag_sid2 = estimate_and_pcu(f_nl, g_nl, h_nl, np.array([0.0, 0.0]))
        _, tag_pcu2 = pcu_attempt(f_nl, g_nl, h_nl, np.array([0.0, 0.0]))
        print('struct_id =', tag_sid2, ' pcu =', tag_pcu2)
        rows.append({'model': 'sim_nl_loop', 'dim': 2,
                     'y_sim': ';'.join('%.10g' % v for v in z_sim),
                     'x_hgca': ';'.join('%.10g' % v for v in z_star),
                     'r_norm': float(np.max(np.abs(r_star))),
                     'err_vs_sim': float(err_sim2),
                     'conv': int((float(np.max(np.abs(r_star))) < 1e-6) and err_sim2 < 1e-6),
                     'struct_id_trigger': int(tag_sid2 is not None), 'pcu_trigger': int(tag_pcu2 is not None),
                     'stage': res2['stage']})

        # Cleanup
        for m in ('sim_lin_loop', 'sim_lin_open', 'sim_nl_loop', 'sim_nl_open'):
            try:
                eng.eval("close_system('%s', 0)" % m, nargout=0)
            except Exception:
                pass

        save_csv('simulink_real', rows)
        save_json('simulink_real_summary', {'rows': rows, 'note': (
            'Real MATLAB R2025a Simulink algebraic loop: the closed model is solved by the Simulink algebraic-loop solver (y_sim);'
            'the expanded model (same block set) computes the loop function G(y) and the residual r(y)=y-G(y) with real blocks;'
            'solver solves r(y)=0 and is compared with y_sim; PCU/struct_id is expected to not trigger (no periodic structure).')})
        print('saved simulink_real*')
    finally:
        eng.quit()


if __name__ == '__main__':
    main()
