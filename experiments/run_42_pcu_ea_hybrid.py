# -*- coding: utf-8 -*-
"""run_42_pcu_ea_hybrid.py
PCU as a restart operator for evolutionary algorithms.

Experiment A: stagnation-restart comparison for CMA-ES
  - Pure CMA-ES (no restart) vs CMA-ES with PCU-guided restart
  - Measure: hit rate, first-hit FE (cumulative), final best F, restart count

Experiment B: same comparison for SHADE (population-based EA)
  - Pure SHADE vs SHADE with PCU-guided restart
  - On stagnation (no improvement for N generations), PCU generates a new
    candidate from the current best; the run restarts from it if it improves.
  - Random restart is used as the control when PCU fails or does not improve.

Problem: CEC2017 F5 M_orth (30D), T=19.53
Output: results/pcu_ea_hybrid.json
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
BUDGET = 100000
S5 = 5.12 / 100.0
D = 30
SEEDS = [20260912, 20260913, 20260914, 20260915, 20260916,
         20260917, 20260918, 20260919, 20260920, 20260921]  # 10 seeds

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


# ============================================================
# CMA-ES implementations
# ============================================================

def cmaes_pure(x0, budget, seed):
    """Pure CMA-ES, no restart."""
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


def cmaes_pcu_restart(x0, budget, seed, stagnation_gen=50):
    """CMA-ES with PCU restart on stagnation.

    When no improvement for stagnation_gen generations, trigger PCU from
    current best point; if PCU finds a better basin, restart from it.

    Note on accounting: `fe_done` tracks evaluations consumed before the
    current CMA instance, so first-hit FE is cumulative across restarts.
    As in the main paper's PCU experiments, the structure-identification
    cost of PCU itself (sampling-based) is excluded from the FE budget;
    wall-clock time is reported per run.
    """
    rng = np.random.RandomState(seed)
    opts = {'maxfevals': budget, 'seed': seed, 'verbose': -1, 'CMA_diagonal': False}
    es = cma.CMAEvolutionStrategy(np.asarray(x0, float), 2.0, opts)

    first_hit = None
    best = float('inf')
    gens_no_improve = 0
    pcu_restarts = 0
    pcu_tries = 0      # PCU structure-identification attempts
    pcu_ok = 0         # attempts where PCU returned a usable candidate
    fe_done = 0  # FE consumed before the current es instance

    while fe_done + es.countevals < budget:
        X = es.ask()
        F = f5(X)
        es.tell(X, F)
        fb = float(es.result.fbest)

        if fb < best - 1e-12:
            best = fb
            gens_no_improve = 0
        else:
            gens_no_improve += 1

        if first_hit is None and best < 1e-4:
            first_hit = min(fe_done + es.countevals, budget)

        # Stagnation detected: try PCU restart
        if gens_no_improve >= stagnation_gen and best > 1e-4:
            # Close out the current CMA run before restarting
            fe_done += es.countevals

            # Get current best point
            x_best = np.asarray(es.result.xbest, float)

            # Try PCU from current best
            pcu_tries += 1
            try:
                x_cand, status = estimate_and_pcu(f5, grad5, hess5, x_best, tol=1e-4)
            except Exception:
                x_cand, status = None, None

            if status == 'pcu_ok':
                pcu_ok += 1
                # PCU found a candidate - restart from it
                f_cand = float(np.asarray(f5(x_cand)).item())
                if f_cand < best:
                    pcu_restarts += 1
                    best = f_cand
                    # Restart CMA-ES from PCU candidate
                    remaining = budget - fe_done
                    if remaining > 100:
                        opts2 = {'maxfevals': remaining, 'seed': seed + pcu_restarts * 1000,
                                 'verbose': -1, 'CMA_diagonal': False}
                        es = cma.CMAEvolutionStrategy(np.asarray(x_cand, float), 2.0, opts2)
                        gens_no_improve = 0
                        continue

            # PCU failed or no improvement - random restart
            gens_no_improve = 0
            remaining = budget - fe_done
            if remaining > 100:
                x_rand = rng.uniform(-100.0, 100.0, D)
                opts2 = {'maxfevals': remaining, 'seed': seed + pcu_restarts * 1000 + 500,
                         'verbose': -1, 'CMA_diagonal': False}
                es = cma.CMAEvolutionStrategy(np.asarray(x_rand, float), 2.0, opts2)

    # CMA-ES may overshoot the budget by one population; clamp the reported FE
    return best, first_hit, pcu_restarts, min(fe_done + es.countevals, budget), pcu_tries, pcu_ok


# ============================================================
# SHADE implementation (from run_17)
# ============================================================

def shade_pure(x0=None, budget=BUDGET, pop=100, seed=0):
    """Pure SHADE, no restart."""
    rng = np.random.RandomState(seed)
    NP = pop; H = 100
    M_CR = np.full(H, 0.5); M_F = np.full(H, 0.5); k_mem = 0
    
    if x0 is None:
        X = rng.uniform(-100.0, 100.0, (NP, D))
    else:
        # Initialize around x0
        X = np.tile(x0, (NP, 1)) + rng.randn(NP, D) * 5.0
    
    fx = f5(X)
    fe = NP
    archive = np.empty((0, D))
    best_f = float(np.min(fx))
    first_hit = None
    
    while fe < budget:
        rem = budget - fe
        n_new = min(NP, rem)
        r_i = rng.randint(H, size=NP)
        CR_i = np.clip(rng.normal(M_CR[r_i], 0.1), 0.0, 1.0)
        F_i = np.clip(rng.standard_cauchy(size=NP) * 0.1 + M_F[r_i], 0.0, 1.0)
        n_pbest = max(2, int(0.1 * NP))
        order = np.argsort(fx)
        S_CR, S_F, S_dF = [], [], []
        U = X.copy()
        for i in range(NP):
            pb = order[rng.randint(n_pbest)]
            r1 = rng.randint(NP); r2 = rng.randint(NP)
            g = 0
            while (r1 == i or r1 == pb) and g < 10: r1 = rng.randint(NP); g += 1
            g = 0
            while (r2 == i or r2 == pb or r2 == r1) and g < 10: r2 = rng.randint(NP); g += 1
            if len(archive) > 0 and rng.rand() < 0.5:
                xr2 = archive[rng.randint(len(archive))]
            else:
                xr2 = X[r2]
            v = X[i] + F_i[i] * (X[pb] - X[i]) + F_i[i] * (X[r1] - xr2)
            jr = rng.randint(D)
            mask = rng.rand(D) < CR_i[i]; mask[jr] = True
            U[i] = np.clip(np.where(mask, v, X[i]), -100.0, 100.0)
        fu = f5(U[:n_new])
        for i in range(n_new):
            if fu[i] < fx[i]:
                S_CR.append(CR_i[i]); S_F.append(F_i[i]); S_dF.append(abs(fx[i] - fu[i]))
                archive = np.vstack([archive, X[i].copy()]) if len(archive) else X[i:i+1].copy()
                X[i] = U[i]; fx[i] = fu[i]
        if len(archive) > NP:
            archive = archive[rng.choice(len(archive), NP, replace=False)]
        if S_dF:
            w = np.array(S_dF) / np.sum(S_dF)
            M_CR[k_mem] = np.sum(w * np.array(S_CR))
            M_F[k_mem] = np.sum(w * np.array(S_F) ** 2) / np.sum(w * np.array(S_F))
            k_mem = (k_mem + 1) % H
        fe += n_new
        if float(np.min(fx)) < best_f:
            best_f = float(np.min(fx))
            if first_hit is None and best_f < 1e-4:
                first_hit = fe
    
    return best_f, first_hit, fe


def shade_pcu_restart(x0=None, budget=BUDGET, pop=100, seed=0, stagnation_gen=100):
    """SHADE with PCU restart on stagnation."""
    rng = np.random.RandomState(seed)
    NP = pop; H = 100
    M_CR = np.full(H, 0.5); M_F = np.full(H, 0.5); k_mem = 0
    
    if x0 is None:
        X = rng.uniform(-100.0, 100.0, (NP, D))
    else:
        X = np.tile(x0, (NP, 1)) + rng.randn(NP, D) * 5.0
    
    fx = f5(X)
    fe = NP
    archive = np.empty((0, D))
    best_f = float(np.min(fx))
    first_hit = None
    gens_no_improve = 0
    pcu_restarts = 0
    pcu_tries = 0
    pcu_ok = 0
    
    while fe < budget:
        rem = budget - fe
        n_new = min(NP, rem)
        r_i = rng.randint(H, size=NP)
        CR_i = np.clip(rng.normal(M_CR[r_i], 0.1), 0.0, 1.0)
        F_i = np.clip(rng.standard_cauchy(size=NP) * 0.1 + M_F[r_i], 0.0, 1.0)
        n_pbest = max(2, int(0.1 * NP))
        order = np.argsort(fx)
        S_CR, S_F, S_dF = [], [], []
        U = X.copy()
        for i in range(NP):
            pb = order[rng.randint(n_pbest)]
            r1 = rng.randint(NP); r2 = rng.randint(NP)
            g = 0
            while (r1 == i or r1 == pb) and g < 10: r1 = rng.randint(NP); g += 1
            g = 0
            while (r2 == i or r2 == pb or r2 == r1) and g < 10: r2 = rng.randint(NP); g += 1
            if len(archive) > 0 and rng.rand() < 0.5:
                xr2 = archive[rng.randint(len(archive))]
            else:
                xr2 = X[r2]
            v = X[i] + F_i[i] * (X[pb] - X[i]) + F_i[i] * (X[r1] - xr2)
            jr = rng.randint(D)
            mask = rng.rand(D) < CR_i[i]; mask[jr] = True
            U[i] = np.clip(np.where(mask, v, X[i]), -100.0, 100.0)
        fu = f5(U[:n_new])
        for i in range(n_new):
            if fu[i] < fx[i]:
                S_CR.append(CR_i[i]); S_F.append(F_i[i]); S_dF.append(abs(fx[i] - fu[i]))
                archive = np.vstack([archive, X[i].copy()]) if len(archive) else X[i:i+1].copy()
                X[i] = U[i]; fx[i] = fu[i]
        if len(archive) > NP:
            archive = archive[rng.choice(len(archive), NP, replace=False)]
        if S_dF:
            w = np.array(S_dF) / np.sum(S_dF)
            M_CR[k_mem] = np.sum(w * np.array(S_CR))
            M_F[k_mem] = np.sum(w * np.array(S_F) ** 2) / np.sum(w * np.array(S_F))
            k_mem = (k_mem + 1) % H
        fe += n_new
        
        current_best = float(np.min(fx))
        if current_best < best_f - 1e-12:
            best_f = current_best
            gens_no_improve = 0
            if first_hit is None and best_f < 1e-4:
                first_hit = fe
        else:
            gens_no_improve += 1
        
        # Stagnation detected
        if gens_no_improve >= stagnation_gen and best_f > 1e-4:
            gens_no_improve = 0
            x_best = X[np.argmin(fx)]
            
            # Try PCU
            pcu_tries += 1
            try:
                x_cand, status = estimate_and_pcu(f5, grad5, hess5, x_best, tol=1e-4)
            except Exception:
                x_cand, status = None, None
            
            if status == 'pcu_ok':
                pcu_ok += 1
                f_cand = float(np.asarray(f5(x_cand)).item())
                if f_cand < best_f:
                    pcu_restarts += 1
                    best_f = f_cand
                    # Reinitialize population around PCU candidate
                    X = np.tile(x_cand, (NP, 1)) + rng.randn(NP, D) * 2.0
                    fx = f5(X)
                    fe += NP  # count the re-initialization evaluations
                    archive = np.empty((0, D))
                    continue

            # PCU failed - random restart
            X = rng.uniform(-100.0, 100.0, (NP, D))
            fx = f5(X)
            fe += NP  # count the re-initialization evaluations
            archive = np.empty((0, D))
    
    return best_f, first_hit, pcu_restarts, fe, pcu_tries, pcu_ok


# ============================================================
# Main experiment
# ============================================================

if __name__ == '__main__':
    results = {
        'problem': 'CEC2017 f5 M_orth',
        'dim': D,
        'budget': BUDGET,
        'seeds': SEEDS,
        'cmaes': {'pure': [], 'pcu_restart': []},
        'shade': {'pure': [], 'pcu_restart': []},
        'pcu_direct': [],
    }
    
    print("=" * 60)
    print("PCU + EA Hybrid Experiment")
    print("=" * 60)
    
    for si, sd in enumerate(SEEDS):
        rng = np.random.RandomState(sd)
        # Use harder starting points: random in [-100, 100] (like SOTA comparison)
        x0 = rng.uniform(-100.0, 100.0, D)
        print(f"\nSeed {sd} ({si+1}/{len(SEEDS)})...", flush=True)
        
        # 1. PCU direct
        t0 = time.time()
        try:
            x_cand, status = estimate_and_pcu(f5, grad5, hess5, x0, tol=1e-4)
        except Exception as e:
            x_cand, status = None, None
        pcu_time = time.time() - t0
        
        if status == 'pcu_ok':
            f_pcu = float(np.asarray(f5(x_cand)).item())
            results['pcu_direct'].append({'seed': sd, 'f': f_pcu, 'hit': True, 'time_s': pcu_time})
            print(f"  PCU direct: F={f_pcu:.2e} (hit={True})", flush=True)
        else:
            results['pcu_direct'].append({'seed': sd, 'f': None, 'hit': False, 'time_s': pcu_time})
            print(f"  PCU direct: no trigger", flush=True)
        
        # 2. CMA-ES pure
        t0 = time.time()
        best_cma, hit_fe_cma, fe_cma = cmaes_pure(x0, BUDGET, sd)
        time_cma = time.time() - t0
        results['cmaes']['pure'].append({
            'seed': sd, 'f': best_cma, 'hit': best_cma < 1e-4,
            'first_hit_fe': hit_fe_cma, 'fe_used': fe_cma, 'time_s': time_cma
        })
        print(f"  CMA-ES pure: F={best_cma:.2e} (hit={best_cma < 1e-4})", flush=True)
        
        # 3. CMA-ES + PCU restart
        t0 = time.time()
        best_cma_pcu, hit_fe_cma_pcu, restarts, fe_cma_pcu, pcu_tries_c, pcu_ok_c = cmaes_pcu_restart(x0, BUDGET, sd)
        time_cma_pcu = time.time() - t0
        results['cmaes']['pcu_restart'].append({
            'seed': sd, 'f': best_cma_pcu, 'hit': best_cma_pcu < 1e-4,
            'first_hit_fe': hit_fe_cma_pcu, 'pcu_restarts': restarts,
            'fe_used': fe_cma_pcu, 'time_s': time_cma_pcu,
            'pcu_tries': pcu_tries_c, 'pcu_ok': pcu_ok_c
        })
        print(f"  CMA-ES+PCU: F={best_cma_pcu:.2e} (hit={best_cma_pcu < 1e-4}, restarts={restarts}, "
              f"pcu_tries={pcu_tries_c}, pcu_ok={pcu_ok_c})", flush=True)
        
        # 4. SHADE pure
        t0 = time.time()
        best_shade, hit_fe_shade, fe_shade = shade_pure(seed=sd)
        time_shade = time.time() - t0
        results['shade']['pure'].append({
            'seed': sd, 'f': best_shade, 'hit': best_shade < 1e-4,
            'first_hit_fe': hit_fe_shade, 'fe_used': fe_shade, 'time_s': time_shade
        })
        print(f"  SHADE pure: F={best_shade:.2e} (hit={best_shade < 1e-4})", flush=True)
        
        # 5. SHADE + PCU restart
        t0 = time.time()
        best_shade_pcu, hit_fe_shade_pcu, restarts_sh, fe_shade_pcu, pcu_tries_s, pcu_ok_s = shade_pcu_restart(seed=sd)
        time_shade_pcu = time.time() - t0
        results['shade']['pcu_restart'].append({
            'seed': sd, 'f': best_shade_pcu, 'hit': best_shade_pcu < 1e-4,
            'first_hit_fe': hit_fe_shade_pcu, 'pcu_restarts': restarts_sh,
            'fe_used': fe_shade_pcu, 'time_s': time_shade_pcu,
            'pcu_tries': pcu_tries_s, 'pcu_ok': pcu_ok_s
        })
        print(f"  SHADE+PCU: F={best_shade_pcu:.2e} (hit={best_shade_pcu < 1e-4}, restarts={restarts_sh}, "
              f"pcu_tries={pcu_tries_s}, pcu_ok={pcu_ok_s})", flush=True)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for algo in ['cmaes', 'shade']:
        for mode in ['pure', 'pcu_restart']:
            hits = sum(1 for r in results[algo][mode] if r['hit'])
            fs = [r['f'] for r in results[algo][mode] if r['f'] is not None]
            med_f = float(np.median(fs)) if fs else None
            print(f"{algo} {mode}: {hits}/{len(SEEDS)} hits, median F={med_f:.2e}" if med_f else f"{algo} {mode}: {hits}/{len(SEEDS)} hits")
    
    # Save
    out_path = os.path.join(RESULTS, 'pcu_ea_hybrid.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
