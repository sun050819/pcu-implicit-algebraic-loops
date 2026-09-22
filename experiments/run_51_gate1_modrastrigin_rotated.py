# -*- coding: utf-8 -*-
"""run_51_gate1_modrastrigin_rotated.py
Gate 1: EA + PCU on rotated modulated Rastrigin (nonseparable).

This addresses the "only separable" criticism by testing the same EA+PCU
initialization protocol on a rotated version of the modulated Rastrigin family.
The rotation makes the problem nonseparable while preserving the periodic
Hessian structure (in the rotated coordinate system).

Output: results/gate1_modrastrigin_rotated.json
"""
import io, sys, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from aloop.solve.struct_id import estimate_and_pcu
import cma

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
BUDGET = 396993
D = 30
S5 = 5.12 / 100.0
N_RUNS = 100

SEEDS = list(range(20260950, 20260950 + N_RUNS))


# ---- Build orthogonal rotation matrix (fixed across all runs) ----
_rng = np.random.RandomState(42)
_A = _rng.randn(D, D)
_Q, _R = np.linalg.qr(_A)
# Ensure proper rotation (det = +1)
if np.linalg.det(_Q) < 0:
    _Q[:, 0] = -_Q[:, 0]
R_orth = _Q  # R @ R.T = I, det = +1


# ---- Rotated Modulated Rastrigin function ----
# f(x) = sum_i envelope(z_i) * (z_i^2 - 10*cos(2*pi*z_i) + 10)
# where z = S5 * R_orth @ x
def f_mod_rastrigin_rot(X):
    Xa = np.atleast_2d(np.asarray(X, float))
    Z = S5 * (R_orth @ Xa.T).T  # z = S5 * R @ x
    envelope = 1.0 + 0.3 * np.sin(0.1 * Z)
    return np.sum(envelope * (Z * Z - 10.0 * np.cos(2.0 * np.pi * Z) + 10.0), axis=1)


def grad_mod_rastrigin_rot(x):
    z = S5 * (R_orth @ x)
    envelope = 1.0 + 0.3 * np.sin(0.1 * z)
    denv_dz = 0.3 * 0.1 * np.cos(0.1 * z)
    f_base = z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0
    gz = envelope * (2.0 * z + 20.0 * np.pi * np.sin(2.0 * np.pi * z)) + denv_dz * f_base
    # grad_x = S5 * R.T @ gz
    return S5 * (R_orth.T @ gz)


def hess_mod_rastrigin_rot(x):
    z = S5 * (R_orth @ x)
    envelope = 1.0 + 0.3 * np.sin(0.1 * z)
    denv_dz = 0.3 * 0.1 * np.cos(0.1 * z)
    d2env_dz2 = -0.3 * 0.01 * np.sin(0.1 * z)
    f_base = z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0
    df_dz = 2.0 * z + 20.0 * np.pi * np.sin(2.0 * np.pi * z)
    d2f_dz2 = 2.0 + 40.0 * np.pi ** 2 * np.cos(2.0 * np.pi * z)
    Hzz_diag = S5 ** 2 * (envelope * d2f_dz2 + 2 * denv_dz * df_dz + d2env_dz2 * f_base)
    # H_x = S5^2 * R.T @ diag(Hzz_diag) @ R
    return S5 ** 2 * (R_orth.T @ np.diag(Hzz_diag) @ R_orth)


# ---- CMA-ES runner ----
def cmaes_from(f_func, x0, budget, seed):
    opts = {'maxfevals': budget, 'seed': seed, 'verbose': -1, 'CMA_diagonal': False}
    es = cma.CMAEvolutionStrategy(np.asarray(x0, float), 2.0, opts)
    first_hit = None
    best = float('inf')
    while es.countevals < budget:
        X = es.ask()
        F = f_func(X)
        es.tell(X, F)
        fb = float(es.result.fbest)
        if fb < best:
            best = fb
        if first_hit is None and best < 1e-4:
            first_hit = es.countevals
    return best, first_hit, es.countevals


# ---- SHADE runner ----
def shade_optimize(f_func, x0, budget, seed):
    rng = np.random.RandomState(seed)
    n = D
    NP = 100
    H = 10
    m_cr = np.ones(H) * 0.5
    m_f = np.ones(H) * 0.5
    k = 0
    p = max(2, int(round(NP * 0.1)))

    pop = np.tile(x0, (NP, 1)) + rng.uniform(-10, 10, (NP, n))
    fit = f_func(pop)
    best_idx = np.argmin(fit)
    best_x = pop[best_idx].copy()
    best_f = fit[best_idx]
    first_hit_fe = None
    fe_used = NP
    archive = []

    while fe_used < budget:
        new_pop = np.zeros_like(pop)
        new_fit = np.zeros(NP)
        s_cr, s_f, good_fit = [], [], []

        for i in range(NP):
            r_i = rng.randint(H)
            cr_i = np.clip(rng.normal(m_cr[r_i], 0.1), 0.0, 1.0)
            f_i = rng.standard_cauchy() * 0.1 + m_f[r_i]
            while f_i <= 0:
                f_i = rng.standard_cauchy() * 0.1 + m_f[r_i]
            f_i = min(f_i, 1.0)

            candidates = list(range(NP))
            candidates.remove(i)
            p_best_idx = pop[np.argpartition(fit, max(1, int(p)))[:max(1, int(p))]]
            p_best = p_best_idx[rng.randint(len(p_best_idx))]

            xr1 = pop[rng.choice(candidates)]
            if len(archive) > 0:
                xr2 = archive[rng.randint(len(archive))]
            else:
                xr2 = pop[rng.choice(candidates)]

            u_i = pop[i] + f_i * (p_best - pop[i]) + f_i * (xr1 - xr2)

            j_rand = rng.randint(n)
            trial = np.where(rng.rand(n) < cr_i, u_i, pop[i])
            trial[j_rand] = u_i[j_rand]

            f_u = f_func(trial.reshape(1, -1))[0]
            fe_used += 1

            if f_u <= fit[i]:
                new_pop[i] = trial
                new_fit[i] = f_u
                s_cr.append(cr_i)
                s_f.append(f_i)
                good_fit.append(fit[i] - f_u)
                if f_u < best_f:
                    best_f = f_u
                    best_x = trial.copy()
                    if first_hit_fe is None and best_f < 1e-4:
                        first_hit_fe = fe_used
            else:
                new_pop[i] = pop[i]
                new_fit[i] = fit[i]

            if fe_used >= budget:
                break

        pop = new_pop
        fit = new_fit
        archive.append(best_x.copy())
        if len(archive) > NP * 2:
            archive = archive[-NP * 2:]

        if s_cr:
            s_cr = np.array(s_cr)
            s_f = np.array(s_f)
            w = np.array(good_fit) / (np.sum(good_fit) + 1e-30)
            m_cr[k] = np.sum(w * s_cr)
            m_f[k] = np.sum(w * s_f * s_f) / (np.sum(w * s_f) + 1e-30)
            k = (k + 1) % H

        if best_f < 1e-4 and first_hit_fe is not None:
            break

    return best_f, first_hit_fe, fe_used


def main():
    print("=" * 60)
    print("GATE 1: Rotated Modulated Rastrigin EA + PCU")
    print("=" * 60)
    print(f"Problem: Rotated Modulated Rastrigin, {D}D")
    print(f"R: fixed orthogonal rotation (det = +1, seed 42)")
    print(f"Runs per condition: {N_RUNS}")
    print(f"Budget: {BUDGET} FE per run")
    print()

    results = {
        'problem': 'Rotated Modulated Rastrigin 30D',
        'budget_fe': BUDGET,
        'n_runs': N_RUNS,
        'seeds': SEEDS,
        'start_range': 'U(-100, 100)^D',
        'rotation': 'fixed orthogonal, seed 42, det=+1',
        'cmaes_pure': [],
        'cmaes_pcu_init': [],
        'shade_pure': [],
        'shade_pcu_init': [],
    }

    # Check for existing results (resume support)
    out_path = os.path.join(RESULTS, 'gate1_modrastrigin_rotated.json')
    if os.path.exists(out_path):
        with open(out_path, 'r') as fp:
            existing = json.load(fp)
        completed_seeds = set(r['seed'] for r in existing['cmaes_pure'])
        print(f"Found existing results: {len(completed_seeds)} runs completed, resuming...")
        results = existing
    else:
        completed_seeds = set()

    for idx, sd in enumerate(SEEDS):
        if sd in completed_seeds:
            continue
        rng = np.random.RandomState(sd)
        x0 = rng.uniform(-100.0, 100.0, D)

        print(f"--- Run {idx+1}/{N_RUNS} (seed {sd}) ---")
        t0 = time.time()

        # 1. CMA-ES pure
        cma_f, cma_hit, cma_fe = cmaes_from(f_mod_rastrigin_rot, x0, BUDGET, sd)
        results['cmaes_pure'].append({
            'seed': sd, 'final_f': cma_f,
            'hit_fe': cma_hit, 'fe_used': cma_fe,
            'hit': cma_f < 1e-4
        })
        print(f"  CMA-ES pure: F={cma_f:.2e}, hit={cma_f<1e-4}")

        # 2. PCU identification
        t_pcu = time.time()
        x_cand, status = estimate_and_pcu(f_mod_rastrigin_rot, grad_mod_rastrigin_rot,
                                          hess_mod_rastrigin_rot, x0, tol=1e-4)
        pcu_triggered = status is not None
        print(f"  PCU: {'triggered' if pcu_triggered else 'no trigger'} ({time.time()-t_pcu:.1f}s)")

        if pcu_triggered:
            # 3. CMA-ES + PCU init
            cma_pcu_f, cma_pcu_hit, cma_pcu_fe = cmaes_from(f_mod_rastrigin_rot, x_cand, BUDGET, sd)
            results['cmaes_pcu_init'].append({
                'seed': sd, 'final_f': cma_pcu_f,
                'hit_fe': cma_pcu_hit, 'fe_used': cma_pcu_fe,
                'hit': cma_pcu_f < 1e-4, 'pcu_triggered': True
            })
            print(f"  CMA-ES+PCU: F={cma_pcu_f:.2e}, hit={cma_pcu_f<1e-4}")

            # 4. SHADE + PCU init
            sh_pcu_f, sh_pcu_hit, sh_pcu_fe = shade_optimize(f_mod_rastrigin_rot, x_cand, BUDGET, sd)
            results['shade_pcu_init'].append({
                'seed': sd, 'final_f': sh_pcu_f,
                'hit_fe': sh_pcu_hit, 'fe_used': sh_pcu_fe,
                'hit': sh_pcu_f < 1e-4, 'pcu_triggered': True
            })
            print(f"  SHADE+PCU: F={sh_pcu_f:.2e}, hit={sh_pcu_f<1e-4}")
        else:
            results['cmaes_pcu_init'].append({
                'seed': sd, 'final_f': cma_f,
                'hit_fe': cma_hit, 'fe_used': cma_fe,
                'hit': cma_f < 1e-4, 'pcu_triggered': False
            })
            results['shade_pcu_init'].append({
                'seed': sd, 'final_f': None, 'hit_fe': None,
                'fe_used': None, 'hit': False, 'pcu_triggered': False
            })

        # 5. SHADE pure
        sh_pure_f, sh_pure_hit, sh_pure_fe = shade_optimize(f_mod_rastrigin_rot, x0, BUDGET, sd)
        results['shade_pure'].append({
            'seed': sd, 'final_f': sh_pure_f,
            'hit_fe': sh_pure_hit, 'fe_used': sh_pure_fe,
            'hit': sh_pure_f < 1e-4
        })
        print(f"  SHADE pure: F={sh_pure_f:.2e}, hit={sh_pure_f<1e-4}")
        print(f"  Time: {time.time()-t0:.1f}s")

        # Save incrementally after each run
        with open(out_path, 'w') as fp:
            json.dump(results, fp, indent=2, default=lambda o: bool(o) if isinstance(o, (np.bool_,)) else float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else o)

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
