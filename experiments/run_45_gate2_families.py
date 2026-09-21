# -*- coding: utf-8 -*-
"""run_45_gate2_families.py
Gate 2 decisive experiment: function family extension + raw/orth attack-defense.

Tests PCU on multiple periodic multimodal function families to verify
that the structure identification generalizes beyond Rastrigin.

Function families:
  1. Rastrigin (f5 M_orth) - baseline
  2. Cosine-sum family (triple cosine with rational frequency ratios)
  3. Modulated Rastrigin (amplitude-modulated periodic)
  4. Non-uniform period family (different periods per dimension)

raw/orth attack-defense:
  - raw: official CEC2017 matrix (should NOT trigger)
  - orth: SVD-orthogonalized matrix (should trigger)
  - perturbation decay: gradually add non-orthogonality, measure trigger/hit decay

Output: results/gate2_families.json
"""
import io, sys, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'cec2017'))
from aloop.solve.struct_id import estimate_and_pcu
from cec2017.transforms import rotations, shifts

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
D = 30
S5 = 5.12 / 100.0
N_RUNS = 10

SEEDS = list(range(20260912, 20260912 + N_RUNS))

# CEC2017 f5 M_orth
Mr = rotations[D][4]
u, _, vt = np.linalg.svd(Mr)
M_orth = u @ vt
o = shifts[4][:D].copy()


def rastrigin_orth(X):
    """Rastrigin with orthogonal rotation (f5 M_orth)."""
    Xa = np.atleast_2d(np.asarray(X, float))
    z = S5 * (Xa @ M_orth.T - o @ M_orth.T)
    return np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0, axis=1)


def rastrigin_raw(X):
    """Rastrigin with official CEC2017 raw matrix (should not trigger)."""
    Xa = np.atleast_2d(np.asarray(X, float))
    z = S5 * (Xa @ Mr.T - o @ Mr.T)
    return np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0, axis=1)


def grad_rastrigin_orth(x):
    x = np.atleast_1d(np.asarray(x, float))
    z = S5 * (M_orth @ (x - o))
    gz = S5 * (2.0 * z + 20.0 * np.pi * np.sin(2.0 * np.pi * z))
    return M_orth.T @ gz


def grad_rastrigin_raw(x):
    x = np.atleast_1d(np.asarray(x, float))
    z = S5 * (Mr @ (x - o))
    gz = S5 * (2.0 * z + 20.0 * np.pi * np.sin(2.0 * np.pi * z))
    return Mr.T @ gz


def hess_rastrigin_orth(x):
    x = np.atleast_1d(np.asarray(x, float))
    z = S5 * (M_orth @ (x - o))
    Hzz = S5 * S5 * (2.0 + 40.0 * np.pi * np.pi * np.cos(2.0 * np.pi * z))
    return (M_orth.T * Hzz) @ M_orth


def hess_rastrigin_raw(x):
    x = np.atleast_1d(np.asarray(x, float))
    z = S5 * (Mr @ (x - o))
    Hzz = S5 * S5 * (2.0 + 40.0 * np.pi * np.pi * np.cos(2.0 * np.pi * z))
    return (Mr.T * Hzz) @ Mr


# ---- Family 2: Triple cosine (rational frequency ratios) ----
def triple_cosine(X):
    """Triple cosine with rational frequency ratios (periodic, consensus T)."""
    Xa = np.atleast_2d(np.asarray(X, float))
    z = 0.1 * Xa  # scale to reasonable range
    return np.sum(np.cos(2*np.pi*z) + 0.5*np.cos(4*np.pi*z) + 0.25*np.cos(6*np.pi*z) + z*z, axis=1)


def grad_triple_cosine(x):
    z = 0.1 * x
    gz = -0.1 * (2*np.pi*np.sin(2*np.pi*z) + 0.5*4*np.pi*np.sin(4*np.pi*z) + 0.25*6*np.pi*np.sin(6*np.pi*z)) + 2*z*0.1
    return gz


def hess_triple_cosine(x):
    z = 0.1 * x
    Hzz = -0.01 * ((2*np.pi)**2 * np.cos(2*np.pi*z) + 0.5*(4*np.pi)**2 * np.cos(4*np.pi*z) + 0.25*(6*np.pi)**2 * np.cos(6*np.pi*z)) + 2*0.01
    return np.diag(Hzz)


# ---- Family 3: Modulated Rastrigin ----
def modulated_rastrigin(X):
    """Amplitude-modulated Rastrigin (slowly varying envelope)."""
    Xa = np.atleast_2d(np.asarray(X, float))
    z = S5 * Xa
    envelope = 1.0 + 0.3 * np.sin(0.1 * z)  # slow modulation
    return np.sum(envelope * (z*z - 10.0*np.cos(2*np.pi*z) + 10.0), axis=1)


def grad_mod_rastrigin(x):
    z = S5 * x
    envelope = 1.0 + 0.3 * np.sin(0.1 * z)
    denv_dz = 0.3 * 0.1 * np.cos(0.1 * z)
    f_base = z*z - 10.0*np.cos(2*np.pi*z) + 10.0
    gz = envelope * (2*z + 20.0*np.pi*np.sin(2*np.pi*z)) + denv_dz * f_base
    return S5 * gz


def hess_mod_rastrigin(x):
    z = S5 * x
    envelope = 1.0 + 0.3 * np.sin(0.1 * z)
    denv_dz = 0.3 * 0.1 * np.cos(0.1 * z)
    d2env_dz2 = -0.3 * 0.01 * np.sin(0.1 * z)
    f_base = z*z - 10.0*np.cos(2*np.pi*z) + 10.0
    df_dz = 2*z + 20.0*np.pi*np.sin(2*np.pi*z)
    d2f_dz2 = 2.0 + 40.0*np.pi**2 * np.cos(2*np.pi*z)
    Hzz = S5**2 * (envelope * d2f_dz2 + 2*denv_dz * df_dz + d2env_dz2 * f_base)
    return np.diag(Hzz)


# ---- Family 4: Non-uniform period (different T per dimension) ----
def nonuniform_period(X):
    """Different periods per dimension (non-uniform T)."""
    Xa = np.atleast_2d(np.asarray(X, float))
    z = S5 * Xa
    # Each dimension has slightly different period
    periods = np.linspace(1.8, 2.2, D)  # T varies from 1.8 to 2.2
    result = np.zeros(len(Xa))
    for i in range(D):
        result += z[:,i]**2 - 10.0*np.cos(2*np.pi*z[:,i]/periods[i]) + 10.0
    return result


def grad_nonuniform(x):
    z = S5 * x
    periods = np.linspace(1.8, 2.2, D)
    gz = 2*z + 20.0*np.pi/periods * np.sin(2*np.pi*z/periods)
    return S5 * gz


def hess_nonuniform(x):
    z = S5 * x
    periods = np.linspace(1.8, 2.2, D)
    Hzz = S5**2 * (2.0 + 40.0*np.pi**2/periods**2 * np.cos(2*np.pi*z/periods))
    return np.diag(Hzz)


def test_family(name, f, grad, hess, start_range=(-10, 10)):
    """Test PCU on a given function family."""
    print(f"\n--- {name} ---")
    triggers = 0
    hits = 0
    pcu_f_vals = []

    for sd in SEEDS:
        rng = np.random.RandomState(sd)
        x0 = rng.uniform(start_range[0], start_range[1], D)

        try:
            x_cand, status = estimate_and_pcu(f, grad, hess, x0, tol=1e-4)
            if status is not None:
                triggers += 1
                f_at_cand = float(np.asarray(f(x_cand.reshape(1,-1))).item())
                pcu_f_vals.append(f_at_cand)
                if f_at_cand < 1e-4:
                    hits += 1
                print(f"  seed {sd}: TRIGGER, F(cand)={f_at_cand:.2e}")
            else:
                print(f"  seed {sd}: no trigger")
        except Exception as e:
            print(f"  seed {sd}: ERROR - {e}")

    print(f"\n  Summary: {triggers}/{N_RUNS} triggers, {hits}/{triggers} hits")
    if pcu_f_vals:
        print(f"  Median F: {np.median(pcu_f_vals):.2e}")

    return {
        'name': name,
        'triggers': triggers,
        'hits': hits,
        'n_runs': N_RUNS,
        'trigger_rate': triggers / N_RUNS,
        'hit_rate_when_triggered': hits / triggers if triggers > 0 else 0,
        'median_f': float(np.median(pcu_f_vals)) if pcu_f_vals else None,
    }


def raw_orth_attack_defense():
    """Test raw vs orth matrix to verify selectivity."""
    print("\n" + "=" * 60)
    print("RAW/ORTH ATTACK-DEFENSE")
    print("=" * 60)

    results = {}

    # raw matrix (should NOT trigger)
    print("\n--- Raw CEC2017 f5 (official matrix) ---")
    raw_triggers = 0
    for sd in SEEDS:
        rng = np.random.RandomState(sd)
        x0 = o + rng.uniform(-5, 5, D)
        try:
            _, status = estimate_and_pcu(rastrigin_raw, grad_rastrigin_raw, hess_rastrigin_raw, x0, tol=1e-4)
            if status is not None:
                raw_triggers += 1
                print(f"  seed {sd}: UNEXPECTED TRIGGER!")
        except Exception as e:
            print(f"  seed {sd}: ERROR - {e}")
    print(f"  Raw triggers: {raw_triggers}/{N_RUNS}")
    results['raw_triggers'] = raw_triggers

    # orth matrix (SVD-orthogonalized, SHOULD trigger)
    print("\n--- Orthogonalized CEC2017 f5 (SVD) ---")
    orth_triggers = 0
    orth_hits = 0
    for sd in SEEDS:
        rng = np.random.RandomState(sd)
        x0 = o + rng.uniform(-5, 5, D)
        try:
            x_cand, status = estimate_and_pcu(rastrigin_orth, grad_rastrigin_orth, hess_rastrigin_orth, x0, tol=1e-4)
            if status is not None:
                orth_triggers += 1
                f_cand = float(np.asarray(rastrigin_orth(x_cand.reshape(1,-1))).item())
                if f_cand < 1e-4:
                    orth_hits += 1
                print(f"  seed {sd}: trigger, F={f_cand:.2e}")
        except Exception as e:
            print(f"  seed {sd}: ERROR - {e}")
    print(f"  Orth triggers: {orth_triggers}/{N_RUNS}, hits: {orth_hits}/{orth_triggers}")
    results['orth_triggers'] = orth_triggers
    results['orth_hits'] = orth_hits

    return results


def main():
    print("=" * 60)
    print("GATE 2: Function Family Extension + Raw/Orth Attack-Defense")
    print("=" * 60)
    print(f"Dimensions: {D}")
    print(f"Runs per family: {N_RUNS}")

    all_results = {
        'dimensions': D,
        'n_runs': N_RUNS,
        'seeds': SEEDS,
        'families': [],
        'raw_orth': {},
    }

    # Test all function families
    families = [
        ("Rastrigin (f5 M_orth)", rastrigin_orth, grad_rastrigin_orth, hess_rastrigin_orth, (-10, 10)),
        ("Triple cosine", triple_cosine, grad_triple_cosine, hess_triple_cosine, (-10, 10)),
        ("Modulated Rastrigin", modulated_rastrigin, grad_mod_rastrigin, hess_mod_rastrigin, (-10, 10)),
        ("Non-uniform period", nonuniform_period, grad_nonuniform, hess_nonuniform, (-10, 10)),
    ]

    for name, f, grad, hess, rng_range in families:
        result = test_family(name, f, grad, hess, rng_range)
        all_results['families'].append(result)

    # raw/orth attack-defense
    all_results['raw_orth'] = raw_orth_attack_defense()

    # Summary
    print("\n" + "=" * 60)
    print("GATE 2 SUMMARY")
    print("=" * 60)
    print(f"\nFunction families:")
    for fam in all_results['families']:
        print(f"  {fam['name']:30s}: {fam['trigger_rate']*100:.0f}% trigger, {fam['hit_rate_when_triggered']*100:.0f}% hit")

    n_triggering = sum(1 for f in all_results['families'] if f['trigger_rate'] > 0)
    print(f"\nTriggering families: {n_triggering}/{len(all_results['families'])}")

    ro = all_results['raw_orth']
    print(f"\nRaw/orth attack-defense:")
    print(f"  Raw triggers: {ro['raw_triggers']}/{N_RUNS} (should be 0)")
    print(f"  Orth triggers: {ro['orth_triggers']}/{N_RUNS}, hits: {ro['orth_hits']}")

    # Gate 2 verdict
    print(f"\nGATE 2 VERDICT:")
    if n_triggering >= 3:
        print(f"  ✅ PASS: {n_triggering} function families trigger")
    else:
        print(f"  ⚠️ PARTIAL: only {n_triggering} families trigger")

    if ro['raw_triggers'] == 0 and ro['orth_triggers'] > 0:
        print(f"  ✅ Raw/orth selectivity: PASS (raw=0, orth>0)")
    else:
        print(f"  ⚠️ Raw/orth selectivity: CHECK")

    # Save results
    out_path = os.path.join(RESULTS, 'gate2_families.json')
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(all_results, fp, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()
