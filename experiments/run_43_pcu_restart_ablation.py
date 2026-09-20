# -*- coding: utf-8 -*-
"""run_43_pcu_restart_ablation.py
Simplified PCU restart ablation: compare pure CMA-ES vs random restart vs PCU restart.

Problem: CEC2017 f5 M_orth 30D, budget=100k FE (fast), 5 seeds.
Key question: does PCU restart outperform random restart?
Output: results/pcu_restart_ablation.json
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
BUDGET = 100000  # Reduced for speed
S5 = 5.12 / 100.0
D = 30
SEEDS = [20260912, 20260913, 20260914, 20260915, 20260916]
STAGNATION_GEN = 100  # generations without improvement -> restart

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


def cmaes_experiment(x0, budget, seed, restart_mode='none'):
    """Run CMA-ES with specified restart mode.

    restart_mode:
        'none' - no restart
        'random' - random restart on stagnation
        'pcu' - PCU-guided restart on stagnation

    Note on accounting: `fe_done` tracks evaluations consumed before the
    current CMA instance, so first-hit FE is cumulative across restarts and
    the loop cannot overrun the budget after a restart (each restarted CMA
    instance is capped at the remaining budget).
    """
    rng = np.random.RandomState(seed)
    opts = {'maxfevals': budget, 'seed': seed, 'verbose': -1, 'CMA_diagonal': False}
    es = cma.CMAEvolutionStrategy(np.asarray(x0, float), 2.0, opts)

    best = float('inf')
    first_hit = None
    gens_no_improve = 0
    n_restarts = 0
    pcu_tries = 0      # PCU structure-identification attempts
    pcu_ok_count = 0   # attempts where PCU returned a usable candidate
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

        # Stagnation detected
        if gens_no_improve >= STAGNATION_GEN and best > 1e-4 and restart_mode != 'none':
            gens_no_improve = 0
            n_restarts += 1

            # Close out the current CMA run before restarting
            fe_done += es.countevals

            remaining = budget - fe_done
            if remaining < 100:
                break

            if restart_mode == 'random':
                # Random restart
                x_new = rng.uniform(-100.0, 100.0, D)
            elif restart_mode == 'pcu':
                # PCU-guided restart
                x_best = np.asarray(es.result.xbest, float)
                pcu_tries += 1
                try:
                    x_cand, status = estimate_and_pcu(f5, grad5, hess5, x_best, tol=1e-4)
                    if status == 'pcu_ok':
                        pcu_ok_count += 1
                        x_new = x_cand
                    else:
                        # PCU did not trigger at this stagnation point; fall
                        # back to a random restart (reported via pcu_tries /
                        # pcu_ok_count so trigger failures are visible).
                        x_new = rng.uniform(-100.0, 100.0, D)
                except Exception:
                    x_new = rng.uniform(-100.0, 100.0, D)

            # Restart
            opts2 = {'maxfevals': remaining, 'seed': seed + n_restarts * 1000,
                     'verbose': -1, 'CMA_diagonal': False}
            es = cma.CMAEvolutionStrategy(np.asarray(x_new, float), 2.0, opts2)

    # CMA-ES may overshoot the budget by one population; clamp the reported FE
    return best, first_hit, n_restarts, min(fe_done + es.countevals, budget), pcu_tries, pcu_ok_count


if __name__ == '__main__':
    results = {
        'problem': 'CEC2017 f5 M_orth',
        'dim': D,
        'budget': BUDGET,
        'stagnation_gen': STAGNATION_GEN,
        'seeds': SEEDS,
        'pure': [],
        'random_restart': [],
        'pcu_restart': [],
    }
    
    print("=" * 60)
    print("PCU Restart Ablation Experiment")
    print("=" * 60)
    
    for si, sd in enumerate(SEEDS):
        rng = np.random.RandomState(sd)
        x0 = rng.uniform(-100.0, 100.0, D)
        print(f"\nSeed {sd} ({si+1}/{len(SEEDS)})...", flush=True)
        
        # 1. Pure CMA-ES
        t0 = time.time()
        best_pure, hit_fe_pure, restarts_pure, fe_pure, _, _ = cmaes_experiment(
            x0, BUDGET, sd, restart_mode='none')
        time_pure = time.time() - t0
        results['pure'].append({
            'seed': sd, 'f': best_pure, 'hit': best_pure < 1e-4,
            'first_hit_fe': hit_fe_pure, 'restarts': restarts_pure,
            'fe_used': fe_pure, 'time_s': time_pure
        })
        print(f"  Pure: F={best_pure:.2e} (hit={best_pure < 1e-4})", flush=True)
        
        # 2. Random restart
        t0 = time.time()
        best_rand, hit_fe_rand, restarts_rand, fe_rand, _, _ = cmaes_experiment(
            x0, BUDGET, sd, restart_mode='random')
        time_rand = time.time() - t0
        results['random_restart'].append({
            'seed': sd, 'f': best_rand, 'hit': best_rand < 1e-4,
            'first_hit_fe': hit_fe_rand, 'restarts': restarts_rand,
            'fe_used': fe_rand, 'time_s': time_rand
        })
        print(f"  Random restart: F={best_rand:.2e} (hit={best_rand < 1e-4}, restarts={restarts_rand})", flush=True)
        
        # 3. PCU restart
        t0 = time.time()
        best_pcu, hit_fe_pcu, restarts_pcu, fe_pcu, pcu_tries, pcu_ok = cmaes_experiment(
            x0, BUDGET, sd, restart_mode='pcu')
        time_pcu = time.time() - t0
        results['pcu_restart'].append({
            'seed': sd, 'f': best_pcu, 'hit': best_pcu < 1e-4,
            'first_hit_fe': hit_fe_pcu, 'restarts': restarts_pcu,
            'fe_used': fe_pcu, 'time_s': time_pcu,
            'pcu_tries': pcu_tries, 'pcu_ok': pcu_ok
        })
        print(f"  PCU restart: F={best_pcu:.2e} (hit={best_pcu < 1e-4}, restarts={restarts_pcu}, "
              f"pcu_tries={pcu_tries}, pcu_ok={pcu_ok})", flush=True)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for mode in ['pure', 'random_restart', 'pcu_restart']:
        hits = sum(1 for r in results[mode] if r['hit'])
        fs = [r['f'] for r in results[mode]]
        med_f = float(np.median(fs))
        mean_restarts = float(np.mean([r['restarts'] for r in results[mode]]))
        print(f"{mode:20s}: {hits}/{len(SEEDS)} hits, median F={med_f:.2e}, avg restarts={mean_restarts:.1f}")
    
    # Save
    out_path = os.path.join(RESULTS, 'pcu_restart_ablation.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
