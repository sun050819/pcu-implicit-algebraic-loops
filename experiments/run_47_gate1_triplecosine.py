# -*- coding: utf-8 -*-
"""run_47_gate1_triplecosine.py
Gate 1 second function family: EA + PCU initialization on triple-cosine family.

This addresses the "only f5" criticism of Gate 1. Tests whether the EA+PCU
improvement generalizes to a second periodic function family.

Output: results/gate1_triplecosine.json
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
N_RUNS = 30

SEEDS = list(range(20260912, 20260912 + N_RUNS))


# ---- Triple cosine function (rational frequency ratios) ----
def f_triple_cosine(X):
    Xa = np.atleast_2d(np.asarray(X, float))
    z = 0.1 * Xa
    return np.sum(np.cos(2*np.pi*z) + 0.5*np.cos(4*np.pi*z) + 0.25*np.cos(6*np.pi*z) + z*z, axis=1)


def grad_triple_cosine(x):
    z = 0.1 * x
    gz = -0.1 * (2*np.pi*np.sin(2*np.pi*z) + 0.5*4*np.pi*np.sin(4*np.pi*z) + 0.25*6*np.pi*np.sin(6*np.pi*z)) + 2*z*0.1
    return gz


def hess_triple_cosine(x):
    z = 0.1 * x
    Hzz = -0.01 * ((2*np.pi)**2 * np.cos(2*np.pi*z) + 0.5*(4*np.pi)**2 * np.cos(4*np.pi*z) + 0.25*(6*np.pi)**2 * np.cos(6*np.pi*z)) + 2*0.01
    return np.diag(Hzz)


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


# ---- SHADE runner (simple implementation) ----
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
            if fe_used >= budget:
                break
            r_i = rng.randint(H)
            cr_i = np.clip(rng.normal(m_cr[r_i], 0.1), 0, 1)
            r_i = rng.randint(H)
            while True:
                f_i = m_f[r_i] + 0.1 * rng.standard_cauchy()
                if f_i > 0:
                    break
                if f_i > 1:
                    f_i = 1
            f_i = min(f_i, 1.0)

            p_i = rng.choice(p)
            best_indices = np.argsort(fit)[:p]
            x_best_p = pop[rng.choice(best_indices)]
            r1 = rng.randint(NP)
            while r1 == i:
                r1 = rng.randint(NP)
            if len(archive) > 0:
                all_pop = np.vstack([pop, archive])
                r2 = rng.randint(len(all_pop))
            else:
                r2 = rng.randint(NP)
                while r2 == i or r2 == r1:
                    r2 = rng.randint(NP)
                all_pop = pop

            v_i = pop[i] + f_i * (x_best_p - pop[i]) + f_i * (pop[r1] - all_pop[r2])
            j_rand = rng.randint(n)
            u_i = np.where(rng.random(n) < cr_i, v_i, pop[i])
            u_i[j_rand] = v_i[j_rand]

            f_u = float(f_func(u_i.reshape(1, -1))[0])
            fe_used += 1

            if f_u <= fit[i]:
                new_pop[i] = u_i
                new_fit[i] = f_u
                archive.append(pop[i].copy())
                if len(archive) > NP * 2:
                    archive = archive[-NP * 2:]
                s_cr.append(cr_i)
                s_f.append(f_i)
                good_fit.append(fit[i] - f_u)
            else:
                new_pop[i] = pop[i]
                new_fit[i] = fit[i]

            if f_u < best_f:
                best_f = f_u
                best_x = u_i.copy()
                if first_hit_fe is None and best_f < 1e-4:
                    first_hit_fe = fe_used

        pop = new_pop
        fit = new_fit

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
    print("GATE 1 SECOND FAMILY: Triple Cosine EA + PCU")
    print("=" * 60)
    print(f"Problem: Triple cosine, {D}D")
    print(f"Runs per condition: {N_RUNS}")
    print(f"Budget: {BUDGET} FE per run")
    print()

    results = {
        'problem': 'Triple cosine rational frequencies 30D',
        'budget_fe': BUDGET,
        'n_runs': N_RUNS,
        'seeds': SEEDS,
        'start_range': 'U(-100, 100)^D',
        'cmaes_pure': [],
        'cmaes_pcu_init': [],
        'shade_pure': [],
        'shade_pcu_init': [],
    }

    for idx, sd in enumerate(SEEDS):
        rng = np.random.RandomState(sd)
        x0 = rng.uniform(-100.0, 100.0, D)

        print(f"--- Run {idx+1}/{N_RUNS} (seed {sd}) ---")
        t0 = time.time()

        # 1. CMA-ES pure
        cma_f, cma_hit, cma_fe = cmaes_from(f_triple_cosine, x0, BUDGET, sd)
        results['cmaes_pure'].append({
            'seed': sd, 'final_f': cma_f,
            'hit_fe': cma_hit, 'fe_used': cma_fe,
            'hit': cma_f < 1e-4
        })
        print(f"  CMA-ES pure: F={cma_f:.2e}, hit={cma_f<1e-4}")

        # 2. PCU identification
        t_pcu = time.time()
        x_cand, status = estimate_and_pcu(f_triple_cosine, grad_triple_cosine, hess_triple_cosine, x0, tol=1e-4)
        pcu_triggered = status is not None
        print(f"  PCU: {'triggered' if pcu_triggered else 'no trigger'} ({time.time()-t_pcu:.1f}s)")

        if pcu_triggered:
            # 3. CMA-ES + PCU init
            cma_pcu_f, cma_pcu_hit, cma_pcu_fe = cmaes_from(f_triple_cosine, x_cand, BUDGET, sd)
            results['cmaes_pcu_init'].append({
                'seed': sd, 'final_f': cma_pcu_f,
                'hit_fe': cma_pcu_hit, 'fe_used': cma_pcu_fe,
                'hit': cma_pcu_f < 1e-4, 'pcu_triggered': True
            })
            print(f"  CMA-ES+PCU: F={cma_pcu_f:.2e}, hit={cma_pcu_f<1e-4}")

            # 4. SHADE + PCU init
            sh_pcu_f, sh_pcu_hit, sh_pcu_fe = shade_optimize(f_triple_cosine, x_cand, BUDGET, sd)
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

        # 5. SHADE pure (only run once for stats)
        sh_pure_f, sh_pure_hit, sh_pure_fe = shade_optimize(f_triple_cosine, x0, BUDGET, sd)
        results['shade_pure'].append({
            'seed': sd, 'final_f': sh_pure_f,
            'hit_fe': sh_pure_hit, 'fe_used': sh_pure_fe,
            'hit': sh_pure_f < 1e-4
        })
        print(f"  SHADE pure: F={sh_pure_f:.2e}, hit={sh_pure_f<1e-4}")
        print(f"  Time: {time.time()-t0:.1f}s")

    out_path = os.path.join(RESULTS, 'gate1_triplecosine.json')
    with open(out_path, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary
    n_trig = sum(1 for r in results['cmaes_pcu_init'] if r['pcu_triggered'])
    print(f"\n=== Summary ===")
    print(f"PCU triggers: {n_trig}/{N_RUNS} ({n_trig/N_RUNS*100:.1f}%)")


if __name__ == '__main__':
    main()
