# -*- coding: utf-8 -*-
"""analyze_gate1.py
Analyze Gate 1 experiment results:
  - Wilcoxon signed-rank test (paired)
  - A12 effect size
  - Summary statistics
  - Trigger rate analysis
"""
import json, os, sys
import numpy as np
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

def a12(x, y):
    """Vargha-Delaney A12 effect size.
    A12 > 0.5 means x tends to be larger than y.
    For improvement (x < y), we use 1 - A12 or compute A12(y, x).
    """
    n1, n2 = len(x), len(y)
    count = 0
    for xi in x:
        for yj in y:
            if xi < yj:
                count += 1
            elif xi == yj:
                count += 0.5
    return count / (n1 * n2)


def main():
    path = os.path.join(RESULTS_DIR, 'gate1_eainit.json')
    if not os.path.exists(path):
        print(f"Results file not found: {path}")
        print("Run run_44_gate1_eainit.py first")
        sys.exit(1)

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("=" * 70)
    print("GATE 1: EA + PCU Initialization vs Pure EA")
    print("Statistical Analysis")
    print("=" * 70)
    print(f"Problem: {data['problem']}")
    print(f"Budget: {data['budget_fe']} FE")
    print(f"Runs: {data['n_runs']}")
    print(f"Start range: {data.get('start_range', 'N/A')}")
    print()

    # --- CMA-ES comparison ---
    print("=" * 70)
    print("CMA-ES: Pure vs + PCU-init")
    print("=" * 70)

    cma_pure = [r for r in data['cmaes_pure'] if r.get('final_f') is not None]
    cma_pcu = [r for r in data['cmaes_pcu_init'] if r.get('final_f') is not None and r.get('pcu_triggered')]

    pure_f = [r['final_f'] for r in cma_pure]
    pcu_f = [r['final_f'] for r in cma_pcu]

    print(f"\nPure CMA-ES: {len(pure_f)} runs")
    print(f"  Median F: {np.median(pure_f):.2e}")
    print(f"  Mean F:   {np.mean(pure_f):.2e}")
    print(f"  Hits (F<1e-4): {sum(1 for f in pure_f if f < 1e-4)}/{len(pure_f)} ({sum(1 for f in pure_f if f < 1e-4)/len(pure_f)*100:.0f}%)")

    print(f"\nCMA-ES + PCU-init (triggered only): {len(pcu_f)} runs")
    print(f"  Median F: {np.median(pcu_f):.2e}")
    print(f"  Mean F:   {np.mean(pcu_f):.2e}")
    print(f"  Hits (F<1e-4): {sum(1 for f in pcu_f if f < 1e-4)}/{len(pcu_f)} ({sum(1 for f in pcu_f if f < 1e-4)/len(pcu_f)*100:.0f}%)")

    print(f"\nTrigger rate: {len(cma_pcu)}/{len(cma_pure)} ({len(cma_pcu)/len(cma_pure)*100:.1f}%)")

    # Paired comparison (only runs where PCU triggered)
    # We need to match by seed
    pure_by_seed = {r['seed']: r['final_f'] for r in cma_pure}
    pcu_by_seed = {r['seed']: r['final_f'] for r in cma_pcu}
    paired_seeds = sorted(set(pure_by_seed.keys()) & set(pcu_by_seed.keys()))

    if len(paired_seeds) >= 3:
        paired_pure = [pure_by_seed[s] for s in paired_seeds]
        paired_pcu = [pcu_by_seed[s] for s in paired_seeds]

        print(f"\nPaired comparison (triggered runs only, n={len(paired_seeds)}):")
        print(f"  Pure median:  {np.median(paired_pure):.2e}")
        print(f"  +PCU median:  {np.median(paired_pcu):.2e}")

        # Wilcoxon signed-rank test
        try:
            w_stat, p_val = stats.wilcoxon(paired_pure, paired_pcu, alternative='greater')
            print(f"  Wilcoxon signed-rank: W={w_stat:.1f}, p={p_val:.2e}")
        except Exception as e:
            print(f"  Wilcoxon test failed: {e}")

        # A12 effect size (how often PCU is better)
        a12_val = a12(paired_pcu, paired_pure)  # P(PCU < pure)
        print(f"  A12 effect size (P[PCU < pure]): {a12_val:.3f}")
        if a12_val > 0.64:
            print(f"    → Large effect (>0.64)")
        elif a12_val > 0.56:
            print(f"    → Medium effect (>0.56)")
        else:
            print(f"    → Small/no effect")
    else:
        print(f"\n  Not enough triggered runs for statistical test (n={len(paired_seeds)})")

    # --- SHADE comparison ---
    print()
    print("=" * 70)
    print("SHADE: Pure vs + PCU-init")
    print("=" * 70)

    shade_pure = [r for r in data['shade_pure'] if r.get('final_f') is not None]
    shade_pcu = [r for r in data['shade_pcu_init'] if r.get('final_f') is not None and r.get('pcu_triggered')]

    pure_f = [r['final_f'] for r in shade_pure]
    pcu_f = [r['final_f'] for r in shade_pcu]

    print(f"\nPure SHADE: {len(pure_f)} runs")
    print(f"  Median F: {np.median(pure_f):.2e}")
    print(f"  Hits (F<1e-4): {sum(1 for f in pure_f if f < 1e-4)}/{len(pure_f)} ({sum(1 for f in pure_f if f < 1e-4)/len(pure_f)*100:.0f}%)")

    print(f"\nSHADE + PCU-init (triggered only): {len(pcu_f)} runs")
    print(f"  Median F: {np.median(pcu_f):.2e}")
    print(f"  Hits (F<1e-4): {sum(1 for f in pcu_f if f < 1e-4)}/{len(pcu_f)} ({sum(1 for f in pcu_f if f < 1e-4)/len(pcu_f)*100:.0f}%)")

    # FE to hit analysis
    shade_pcu_hits = [r for r in shade_pcu if r.get('hit_fe') is not None]
    if shade_pcu_hits:
        hit_fes = [r['hit_fe'] for r in shade_pcu_hits]
        print(f"\n  FE to hit (PCU-init):")
        print(f"    Median: {np.median(hit_fes):.0f}")
        print(f"    Mean:   {np.mean(hit_fes):.0f}")
        print(f"    As % of budget: {np.median(hit_fes)/data['budget_fe']*100:.1f}%")

    # Paired comparison
    pure_by_seed = {r['seed']: r['final_f'] for r in shade_pure}
    pcu_by_seed = {r['seed']: r['final_f'] for r in shade_pcu}
    paired_seeds = sorted(set(pure_by_seed.keys()) & set(pcu_by_seed.keys()))

    if len(paired_seeds) >= 3:
        paired_pure = [pure_by_seed[s] for s in paired_seeds]
        paired_pcu = [pcu_by_seed[s] for s in paired_seeds]

        print(f"\nPaired comparison (triggered runs only, n={len(paired_seeds)}):")
        print(f"  Pure median:  {np.median(paired_pure):.2e}")
        print(f"  +PCU median:  {np.median(paired_pcu):.2e}")

        try:
            w_stat, p_val = stats.wilcoxon(paired_pure, paired_pcu, alternative='greater')
            print(f"  Wilcoxon signed-rank: W={w_stat:.1f}, p={p_val:.2e}")
        except Exception as e:
            print(f"  Wilcoxon test failed: {e}")

        a12_val = a12(paired_pcu, paired_pure)
        print(f"  A12 effect size (P[PCU < pure]): {a12_val:.3f}")
    else:
        print(f"\n  Not enough triggered runs for statistical test (n={len(paired_seeds)})")

    # --- Gate 1 verdict ---
    print()
    print("=" * 70)
    print("GATE 1 VERDICT")
    print("=" * 70)

    cma_pass = False
    shade_pass = False

    # Check CMA-ES
    if len(paired_seeds) >= 3:
        # Recompute for CMA
        cma_pure_by_seed = {r['seed']: r['final_f'] for r in data['cmaes_pure'] if r.get('final_f') is not None}
        cma_pcu_by_seed = {r['seed']: r['final_f'] for r in data['cmaes_pcu_init'] if r.get('final_f') is not None and r.get('pcu_triggered')}
        cma_paired = sorted(set(cma_pure_by_seed.keys()) & set(cma_pcu_by_seed.keys()))

        if len(cma_paired) >= 3:
            pp = [cma_pure_by_seed[s] for s in cma_paired]
            pc = [cma_pcu_by_seed[s] for s in cma_paired]
            try:
                _, p_cma = stats.wilcoxon(pp, pc, alternative='greater')
                a_cma = a12(pc, pp)
                cma_pass = (p_cma < 0.05) and (a_cma > 0.64)
                print(f"  CMA-ES: p={p_cma:.2e}, A12={a_cma:.3f} → {'PASS' if cma_pass else 'FAIL'}")
            except:
                print(f"  CMA-ES: statistical test failed")

    # Check SHADE
    if len(paired_seeds) >= 3:
        try:
            p_shade = p_val  # already computed
            a_shade = a12_val
            shade_pass = (p_shade < 0.05) and (a_shade > 0.64)
            print(f"  SHADE:  p={p_shade:.2e}, A12={a_shade:.3f} → {'PASS' if shade_pass else 'FAIL'}")
        except:
            print(f"  SHADE: statistical test failed")

    print()
    if cma_pass or shade_pass:
        print("  ✅ Gate 1: PARTIAL PASS (at least one optimizer shows significant improvement)")
    else:
        print("  ⚠️ Gate 1: NOT YET PASS (need more triggered runs or different problem setup)")

    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
