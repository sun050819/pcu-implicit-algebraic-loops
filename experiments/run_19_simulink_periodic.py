# -*- coding: utf-8 -*-
import os
"""run_19_simulink_periodic.py - Real Simulink periodic algebraic loop + PCU end-to-end validation.

Model (d=2 separable periodic loop, residual computed by real Simulink blocks):
  r1 = y1 - 0.5*cos(y1) - pi/2  ;  r2 = y2 - 0.5*cos(y2) - pi/2
  -> True solution y* = (pi/2, pi/2) (unique, monotonic);
    The Hessian of the least-squares f=||r||^2/2 contains a cos/sin periodic structure (T=2*pi),
    and the true solution is exactly at the peak of H_ii (zero-crossing detection neighborhood +/-0.125T coverage).

Procedure:
  1) Simulink closes the algebraic loop (Algebraic Constraint block + real block network) -> y_sim (built-in solver)
  2) Simulink expanded model (Constant injects y) -> real blocks compute residual r(y)
  3) PCU end-to-end: func=||r||^2/2 (real Simulink evaluation), grad/hess analytically injected (loop equations known,
     same protocol as run_06_simulink_real (internal solver); finite-difference consistency spot check) -> estimate_and_pcu -> x*
  4) Compare |x* - y_sim| with the true residual r(x*); confirm struct_id trigger (periodic structure)

Output: results/simulink_periodic.csv + simulink_periodic_summary.json
"""
import io, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matlab.engine
from common import save_json, save_csv

PI2 = 0.5 * np.pi  # b1 = b2 = pi/2


def build_models(eng):
    # ---------- Close the periodic algebraic loop (AC + real block network) ----------
    eng.eval("new_system('sim_per_loop')", nargout=0)
    eng.eval("set_param('sim_per_loop', 'AlgebraicLoopMsg', 'none')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Algebraic Constraint', 'sim_per_loop/AC1', 'InitialGuess', '0')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Algebraic Constraint', 'sim_per_loop/AC2', 'InitialGuess', '0')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Trigonometric Function', 'sim_per_loop/Cos1', 'Operator', 'cos')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Trigonometric Function', 'sim_per_loop/Cos2', 'Operator', 'cos')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Gain', 'sim_per_loop/G1', 'Gain', '0.5')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Gain', 'sim_per_loop/G2', 'Gain', '0.5')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Sum', 'sim_per_loop/Sum1', 'Inputs', '+--')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Sum', 'sim_per_loop/Sum2', 'Inputs', '+--')", nargout=0)
    eng.eval("add_block('simulink/Sources/Constant', 'sim_per_loop/C1', 'Value', '%.12g')" % PI2, nargout=0)
    eng.eval("add_block('simulink/Sources/Constant', 'sim_per_loop/C2', 'Value', '%.12g')" % PI2, nargout=0)
    eng.eval("add_block('simulink/Sinks/Out1', 'sim_per_loop/z1')", nargout=0)
    eng.eval("add_block('simulink/Sinks/Out1', 'sim_per_loop/z2')", nargout=0)
    # f1 = +y1 - 0.5cos(y1) - pi/2
    eng.eval("add_line('sim_per_loop', 'AC1/1', 'Cos1/1')", nargout=0)
    eng.eval("add_line('sim_per_loop', 'Cos1/1', 'G1/1')", nargout=0)
    eng.eval("add_line('sim_per_loop', 'G1/1', 'Sum1/2')", nargout=0)
    eng.eval("add_line('sim_per_loop', 'AC1/1', 'Sum1/1')", nargout=0)
    eng.eval("add_line('sim_per_loop', 'C1/1', 'Sum1/3')", nargout=0)
    eng.eval("add_line('sim_per_loop', 'Sum1/1', 'AC1/1')", nargout=0)
    eng.eval("add_line('sim_per_loop', 'AC1/1', 'z1/1')", nargout=0)
    # f2 = +y2 - 0.5cos(y2) - pi/2
    eng.eval("add_line('sim_per_loop', 'AC2/1', 'Cos2/1')", nargout=0)
    eng.eval("add_line('sim_per_loop', 'Cos2/1', 'G2/1')", nargout=0)
    eng.eval("add_line('sim_per_loop', 'G2/1', 'Sum2/2')", nargout=0)
    eng.eval("add_line('sim_per_loop', 'AC2/1', 'Sum2/1')", nargout=0)
    eng.eval("add_line('sim_per_loop', 'C2/1', 'Sum2/3')", nargout=0)
    eng.eval("add_line('sim_per_loop', 'Sum2/1', 'AC2/1')", nargout=0)
    eng.eval("add_line('sim_per_loop', 'AC2/1', 'z2/1')", nargout=0)

    # ---------- Expanded model (Constant injects y, real blocks output residual r) ----------
    eng.eval("new_system('sim_per_open')", nargout=0)
    eng.eval("set_param('sim_per_open', 'AlgebraicLoopMsg', 'none')", nargout=0)
    eng.eval("add_block('simulink/Sources/Constant', 'sim_per_open/Cy1', 'Value', '0')", nargout=0)
    eng.eval("add_block('simulink/Sources/Constant', 'sim_per_open/Cy2', 'Value', '0')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Trigonometric Function', 'sim_per_open/Cos1', 'Operator', 'cos')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Trigonometric Function', 'sim_per_open/Cos2', 'Operator', 'cos')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Gain', 'sim_per_open/G1', 'Gain', '0.5')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Gain', 'sim_per_open/G2', 'Gain', '0.5')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Sum', 'sim_per_open/Sum1', 'Inputs', '+--')", nargout=0)
    eng.eval("add_block('simulink/Math Operations/Sum', 'sim_per_open/Sum2', 'Inputs', '+--')", nargout=0)
    eng.eval("add_block('simulink/Sources/Constant', 'sim_per_open/C1', 'Value', '%.12g')" % PI2, nargout=0)
    eng.eval("add_block('simulink/Sources/Constant', 'sim_per_open/C2', 'Value', '%.12g')" % PI2, nargout=0)
    eng.eval("add_block('simulink/Sinks/Out1', 'sim_per_open/r1')", nargout=0)
    eng.eval("add_block('simulink/Sinks/Out1', 'sim_per_open/r2')", nargout=0)
    eng.eval("add_line('sim_per_open', 'Cy1/1', 'Cos1/1')", nargout=0)
    eng.eval("add_line('sim_per_open', 'Cos1/1', 'G1/1')", nargout=0)
    eng.eval("add_line('sim_per_open', 'G1/1', 'Sum1/2')", nargout=0)
    eng.eval("add_line('sim_per_open', 'Cy1/1', 'Sum1/1')", nargout=0)
    eng.eval("add_line('sim_per_open', 'C1/1', 'Sum1/3')", nargout=0)
    eng.eval("add_line('sim_per_open', 'Sum1/1', 'r1/1')", nargout=0)
    eng.eval("add_line('sim_per_open', 'Cy2/1', 'Cos2/1')", nargout=0)
    eng.eval("add_line('sim_per_open', 'Cos2/1', 'G2/1')", nargout=0)
    eng.eval("add_line('sim_per_open', 'G2/1', 'Sum2/2')", nargout=0)
    eng.eval("add_line('sim_per_open', 'Cy2/1', 'Sum2/1')", nargout=0)
    eng.eval("add_line('sim_per_open', 'C2/1', 'Sum2/3')", nargout=0)
    eng.eval("add_line('sim_per_open', 'Sum2/1', 'r2/1')", nargout=0)


def sim_closed(eng, model, nout=1, guess=None):
    """sim closes the algebraic loop model, returns the solution vector from the Simulink algebraic loop solver."""
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
    """Return the r(y) residual function (real Simulink expanded model: Constant injection -> sim -> output r)."""
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


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    t0 = time.time()
    eng = matlab.engine.start_matlab('-nosplash')
    try:
        build_models(eng)
        rows = []

        # ============ 1. Simulink algebraic loop solution y_sim ============
        print('=== periodic algebraic loop sim_per_loop ===')
        y_sim = sim_closed(eng, 'sim_per_loop', 2, guess=np.array([0.0, 0.0]))
        print('Simulink algebraic loop solver y_sim =', y_sim, '(analytic true solution pi/2 = %.6f)' % (np.pi / 2))

        # ============ 2. Real Simulink residual r(y) ============
        r_per = make_residual(eng, 'sim_per_open', 2,
                              [('sim_per_open/Cy1', 'Value'), ('sim_per_open/Cy2', 'Value')])
        r_sim = r_per(y_sim)
        print('real residual r(y_sim) =', r_sim)

        def f_per(x):
            r = r_per(x)
            return float(r @ r) / 2.0

        # Analytical Jacobian (loop equations known; residual values come from real Simulink - same protocol as run_06_simulink_real)
        def J_analytic(y):
            return np.diag([1.0 + 0.5 * np.sin(float(y[0])),
                            1.0 + 0.5 * np.sin(float(y[1]))])

        def g_per(x):
            J = J_analytic(x)
            return 2.0 * J @ r_per(x)

        def h_per(x):
            J = J_analytic(x)
            return 2.0 * J @ J

        # Finite-difference Jacobian consistency spot check (black-box equivalence evidence)
        xchk = np.array([0.7, 0.8])
        r0 = r_per(xchk)
        hh = 1e-5
        Jnum = np.zeros((2, 2))
        for i in range(2):
            ep = xchk.copy(); em = xchk.copy()
            ep[i] += hh; em[i] -= hh
            Jnum[:, i] = (r_per(ep) - r_per(em)) / (2 * hh)
        jerr = float(np.max(np.abs(Jnum - J_analytic(xchk))))
        print('J finite-difference/analytic max deviation @(0.7,0.8) = %.3e' % jerr)

        # ============ 3. PCU end-to-end (no oracle: initial x0=0 same as Simulink) ============
        from aloop.solve.struct_id import estimate_and_pcu
        x0 = np.array([0.0, 0.0])
        x_star, tag = estimate_and_pcu(f_per, g_per, h_per, x0)
        print('struct_id + PCU:', tag, 'x* =', x_star)
        if x_star is not None:
            x_star = np.asarray(x_star, dtype=float)
            r_star = r_per(x_star)
            err_sim = float(np.max(np.abs(x_star - y_sim)))
            print('|x* - y_sim| = %.3e   F(x*) = %.3e   max|r(x*)| = %.3e' % (
                err_sim, float(r_star @ r_star) / 2.0, float(np.max(np.abs(r_star)))))
            rows.append({'model': 'sim_per_loop', 'dim': 2,
                         'y_sim': ';'.join('%.10g' % v for v in y_sim),
                         'x_pcu': ';'.join('%.10g' % v for v in x_star),
                         'err_vs_sim': float(err_sim),
                         'F_at_x': float(r_star @ r_star) / 2.0,
                         'max_r_at_x': float(np.max(np.abs(r_star))),
                         'struct_id_trigger': 1, 'pcu_ok': int(tag == 'struct_pcu_ok'),
                         'jdiff_consistency': float(jerr)})
        else:
            print('PCU miss (should trigger periodic structure)')
            rows.append({'model': 'sim_per_loop', 'dim': 2,
                         'y_sim': ';'.join('%.10g' % v for v in y_sim),
                         'x_pcu': '', 'err_vs_sim': np.nan, 'F_at_x': np.nan,
                         'max_r_at_x': np.nan, 'struct_id_trigger': 0, 'pcu_ok': 0,
                         'jdiff_consistency': float(jerr)})

        # Cleanup
        for m in ('sim_per_loop', 'sim_per_open'):
            try:
                eng.eval("close_system('%s', 0)" % m, nargout=0)
            except Exception:
                pass

        save_csv('simulink_periodic', rows)
        save_json('simulink_periodic_summary', {
            'rows': rows,
            'elapsed_s': round(time.time() - t0, 1),
            'note': ('Real Simulink R2025a periodic algebraic loop: r1=y1-0.5cos(y1)-pi/2, r2=y2-0.5cos(y2)-pi/2, '
                     'true solution (pi/2,pi/2). The closed model is solved for y_sim by the Simulink algebraic loop solver; the expanded model uses real blocks '
                     'to compute the residual r(y); PCU with x0=0 (same initial value as Simulink) end-to-end: structure identification (periodic trigger) -> '
                     'zero-crossing candidates -> Newton -> F validation (real residual).')})
        print('saved simulink_periodic*')
    finally:
        eng.quit()


if __name__ == '__main__':
    main()
