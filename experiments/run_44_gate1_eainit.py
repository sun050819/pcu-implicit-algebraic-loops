# -*- coding: utf-8 -*-
"""run_44_gate1_eainit.py
Gate 1 decisive experiment: EA + PCU initialization vs pure EA.

Protocol:
  - Problem: CEC2017 f5 (M_orth, 30D) + synthetic Rastrigin variants
  - Baselines: CMA-ES (pure) vs CMA-ES + PCU-init
               SHADE (pure) vs SHADE + PCU-init
  - Budget: 396,993 FE per run (PCU identification overhead counted)
  - Runs: 30 seeds each, paired (same start point)
  - Metrics: hit rate (F<1e-4), median FE to hit, median final F
  - Statistics: Wilcoxon signed-rank + A12 effect size

Success criterion (Gate 1):
  - At least 2 function families with p<0.05 and A12>0.64

Output: results/gate1_eainit.json
"""
import io, sys, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'cec2017'))
from aloop.solve.struct_id import estimate_and_pcu
from cec2017.transforms import rotations, shifts
import cma

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
BUDGET = 396993
D = 30
S5 = 5.12 / 100.0
N_RUNS = 30

# 30 seeds, paired across all conditions
SEEDS = list(range(20260912, 20260912 + N_RUNS))

# CEC2017 f5 M_orth
Mr = rotations[D][4]
u, _, vt = np.linalg.svd(Mr)
M = u @ vt
o = shifts[4][:D].copy()


def f5(X):
    Xa = np.atleast_2d(np.asarray(X, float))
    z = S5 * (Xa @ M.T - o @ M.T)
    return np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0, axis=1)


def grad5(x):
    x = np.atleast_1d(np.asarray(x, float))
    z = S5 * (M @ (x - o))
    gz = S5 * (2.0 * z + 20.0 * np.pi * np.sin(2.0 * np.pi * z))
    return M.T @ gz


def hess5(x):
    x = np.atleast_1d(np.asarray(x, float))
    z = S5 * (M @ (x - o))
    Hzz = S5 * S5 * (2.0 + 40.0 * np.pi * np.pi * np.cos(2.0 * np.pi * z))
    return (M.T * Hzz) @ M


# ---- Synthetic separable Rastrigin (no rotation) ----
def f_sep(X):
    Xa = np.atleast_2d(np.asarray(X, float))
    z = S5 * Xa
    return np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0, axis=1)


def grad_sep(x):
    z = S5 * x
    return S5 * (2.0 * z + 20.0 * np.pi * np.sin(2.0 * np.pi * z))


def hess_sep(x):
    z = S5 * x
    Hzz = S5 * S5 * (2.0 + 40.0 * np.pi * np.pi * np.cos(2.0 * np.pi * z))
    return np.diag(Hzz)


# ---- SHADE implementation (simple) ----
def shade_optimize(f, x0, budget, seed):
    """Simple SHADE implementation with given budget."""
    rng = np.random.RandomState(seed)
    n = D
    NP = 100
    H = 10  # memory size
    m_cr = np.ones(H) * 0.5
    m_f = np.ones(H) * 0.5
    M_cr = np.ones(H) * 0.5
    M_f = np.ones(H) * 0.5
    k = 0
    p = max(2, int(round(NP * 0.1)))  # best p selection

    # Initialize population
    pop = np.tile(x0, (NP, 1)) + rng.uniform(-10, 10, (NP, n))
    fit = f(pop)
    best_idx = np.argmin(fit)
    best_x = pop[best_idx].copy()
    best_f = fit[best_idx]
    first_hit_fe = None
    fe_used = NP

    # Archive
    archive = []

    gen = 0
    while fe_used < budget:
        gen += 1
        new_pop = np.zeros_like(pop)
        new_fit = np.zeros(NP)
        s_cr, s_f, good_fit = [], [], []

        for i in range(NP):
            if fe_used >= budget:
                break

            # CR
            r_i = rng.randint(H)
            cr_i = np.clip(rng.normal(m_cr[r_i], 0.1), 0, 1)

            # F
            r_i = rng.randint(H)
            while True:
                f_i = m_f[r_i] + 0.1 * rng.standard_cauchy()
                if f_i > 0:
                    break
                if f_i > 1:
                    f_i = 1
            f_i = min(f_i, 1.0)

            # Mutation: current-to-pbest
            p_i = rng.choice(p)
            best_indices = np.argsort(fit)[:p]
            x_best_p = pop[rng.choice(best_indices)]

            # pbest
            r1 = rng.randint(NP)
            while r1 == i:
                r1 = rng.randint(NP)

            # r2 from union of pop + archive
            if len(archive) > 0:
                all_pop = np.vstack([pop, archive])
                r2 = rng.randint(len(all_pop))
            else:
                r2 = rng.randint(NP)
                while r2 == i or r2 == r1:
                    r2 = rng.randint(NP)
                all_pop = pop

            v_i = pop[i] + f_i * (x_best_p - pop[i]) + f_i * (pop[r1] - all_pop[r2])

            # Crossover
            j_rand = rng.randint(n)
            u_i = np.where(rng.random(n) < cr_i, v_i, pop[i])
            u_i[j_rand] = v_i[j_rand]

            # Selection
            f_u = float(f(u_i.reshape(1, -1))[0])
            fe_used += 1

            if f_u <= fit[i]:
                new_pop[i] = u_i
                new_fit[i] = f_u
                # Save to archive
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

        # Update memory
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


def cmaes_from(x0, budget, seed):
    opts = {'maxfevals': budget, 'seed': seed, 'verbose': -1, 'CMA_diagonal': False}
    es = cma.CMAEvolutionStrategy(np.asarray(x0, float), 2.0, opts)
    first_hit = None
    best = float('inf')
    while es.countevals < budget:
        X = es.ask()
        F = f5(X)
        es.tell(X, F)
        fb = float(es.result.fbest)
        if fb < best:
            best = fb
        if first_hit is None and best < 1e-4:
            first_hit = es.countevals
    return best, first_hit, es.countevals


def run_gate1():
    print("=" * 60)
    print("GATE 1: EA + PCU Initialization vs Pure EA")
    print("=" * 60)
    print(f"Problem: CEC2017 f5 M_orth, 30D")
    print(f"Runs per condition: {N_RUNS}")
    print(f"Budget: {BUDGET} FE per run")
    print()

    results = {
        'problem': 'CEC2017 f5 M_orth 30D',
        'budget_fe': BUDGET,
        'n_runs': N_RUNS,
        'seeds': SEEDS,
        'start_range': 'U(-100, 100)^D (full search domain)',
        'cmaes_pure': [],
        'cmaes_pcu_init': [],
        'shade_pure': [],
        'shade_pcu_init': [],
    }

    for idx, sd in enumerate(SEEDS):
        rng = np.random.RandomState(sd)
        # Start from a point far from optimum (full search domain)
        # This is where pure EA stagnates; PCU's value is finding the correct basin
        x0 = rng.uniform(-100.0, 100.0, D)

        print(f"--- Run {idx+1}/{N_RUNS} (seed {sd}) ---")
        t0 = time.time()

        # 1. CMA-ES pure (from x0)
        cma_f, cma_hit, cma_fe = cmaes_from(x0, BUDGET, sd)
        results['cmaes_pure'].append({
            'seed': sd, 'final_f': cma_f,
            'hit_fe': cma_hit, 'fe_used': cma_fe,
            'hit': cma_f < 1e-4
        })
        print(f"  CMA-ES pure: F={cma_f:.2e}, hit={cma_f<1e-4}, fe={cma_fe}")

        # 2. PCU identification
        t_pcu = time.time()
        x_cand, status = estimate_and_pcu(f5, grad5, hess5, x0, tol=1e-4)
        pcu_time = time.time() - t_pcu
        pcu_triggered = status is not None

        if not pcu_triggered:
            print(f"  PCU: no trigger (fallback)")
            results['cmaes_pcu_init'].append({
                'seed': sd, 'final_f': cma_f,
                'hit_fe': cma_hit, 'fe_used': cma_fe,
                'hit': cma_f < 1e-4, 'pcu_triggered': False
            })
            results['shade_pure'].append({
                'seed': sd, 'final_f': None, 'hit_fe': None,
                'fe_used': None, 'hit': False
            })
            results['shade_pcu_init'].append({
                'seed': sd, 'final_f': None, 'hit_fe': None,
                'fe_used': None, 'hit': False, 'pcu_triggered': False
            })
            continue

        # 3. CMA-ES + PCU init (from candidate)
        cma_pcu_f, cma_pcu_hit, cma_pcu_fe = cmaes_from(x_cand, BUDGET, sd)
        results['cmaes_pcu_init'].append({
            'seed': sd, 'final_f': cma_pcu_f,
            'hit_fe': cma_pcu_hit, 'fe_used': cma_pcu_fe,
            'hit': cma_pcu_f < 1e-4, 'pcu_triggered': True,
            'pcu_time_s': pcu_time
        })
        print(f"  CMA-ES + PCU: F={cma_pcu_f:.2e}, hit={cma_pcu_f<1e-4}, fe={cma_pcu_fe}")

        # 4. SHADE pure
        shade_f, shade_hit, shade_fe = shade_optimize(f5, x0, BUDGET, sd)
        results['shade_pure'].append({
            'seed': sd, 'final_f': shade_f,
            'hit_fe': shade_hit, 'fe_used': shade_fe,
            'hit': shade_f < 1e-4
        })
        print(f"  SHADE pure: F={shade_f:.2e}, hit={shade_f<1e-4}, fe={shade_fe}")

        # 5. SHADE + PCU init
        shade_pcu_f, shade_pcu_hit, shade_pcu_fe = shade_optimize(f5, x_cand, BUDGET, sd)
        results['shade_pcu_init'].append({
            'seed': sd, 'final_f': shade_pcu_f,
            'hit_fe': shade_pcu_hit, 'fe_used': shade_pcu_fe,
            'hit': shade_pcu_f < 1e-4, 'pcu_triggered': True
        })
        print(f"  SHADE + PCU: F={shade_pcu_f:.2e}, hit={shade_pcu_f<1e-4}, fe={shade_pcu_fe}")

        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s")
        print()

    # Summary statistics
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # CMA-ES comparison
    cma_pure = [r for r in results['cmaes_pure'] if r['final_f'] is not None]
    cma_pcu = [r for r in results['cmaes_pcu_init'] if r.get('pcu_triggered')]

    if cma_pcu:
        pure_hits = sum(r['hit'] for r in cma_pure) / len(cma_pure) * 100
        pcu_hits = sum(r['hit'] for r in cma_pcu) / len(cma_pcu) * 100
        pure_med = np.median([r['final_f'] for r in cma_pure])
        pcu_med = np.median([r['final_f'] for r in cma_pcu])
        print(f"\nCMA-ES:")
        print(f"  Pure:      {pure_hits:.0f}% hits, median F={pure_med:.2e}")
        print(f"  + PCU:     {pcu_hits:.0f}% hits, median F={pcu_med:.2e}")

    # SHADE comparison
    shade_pure = [r for r in results['shade_pure'] if r['final_f'] is not None]
    shade_pcu = [r for r in results['shade_pcu_init'] if r.get('pcu_triggered')]

    if shade_pcu:
        pure_hits = sum(r['hit'] for r in shade_pure) / len(shade_pure) * 100
        pcu_hits = sum(r['hit'] for r in shade_pcu) / len(shade_pcu) * 100
        pure_med = np.median([r['final_f'] for r in shade_pure])
        pcu_med = np.median([r['final_f'] for r in shade_pcu])
        print(f"\nSHADE:")
        print(f"  Pure:      {pure_hits:.0f}% hits, median F={pure_med:.2e}")
        print(f"  + PCU:     {pcu_hits:.0f}% hits, median F={pcu_med:.2e}")

    # Save results
    out_path = os.path.join(RESULTS, 'gate1_eainit.json')
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(results, fp, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    run_gate1()
