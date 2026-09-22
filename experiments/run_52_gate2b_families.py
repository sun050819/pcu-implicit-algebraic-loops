# -*- coding: utf-8 -*-
"""run_52_gate2b_families.py
Gate 2b (re-run): function family generalization under the Section V-B protocol
(initial points near the known solution), plus the rotated modulated-Rastrigin
variant. Replaces the random-start run_45 numbers so that Table IV (tab:families)
has direct data support: Rastrigin (f5 M_orth) near-solution starts, triple
cosine, modulated Rastrigin (diag.), modulated Rastrigin (rot.), non-uniform
period. 10 runs each. Also keeps the raw/orth selectivity check.

Output: results/gate2_families.json
"""
import io, sys, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'cec2017'))
from aloop.solve.struct_id import estimate_and_pcu

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
D = 30
S5 = 5.12 / 100.0
N_RUNS = 10
SEEDS = list(range(20260912, 20260912 + N_RUNS))

# CEC2017 f5 M_orth
Mr = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
             'third_party', 'cec2017', 'transforms', 'rotation_M_D30.npy')) if os.path.exists(
             os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
             'third_party', 'cec2017', 'transforms', 'rotation_M_D30.npy')) else None
from cec2017.transforms import rotations, shifts
Mr = rotations[D][4]
u, _, vt = np.linalg.svd(Mr)
M_orth = u @ vt
o = shifts[4][:D].copy()

def rastrigin_orth(X):
    Xa = np.atleast_2d(np.asarray(X, float))
    z = S5 * (Xa @ M_orth.T - o @ M_orth.T)
    return np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0, axis=1)

def rastrigin_raw(X):
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

def triple_cosine(X):
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

def modulated_rastrigin(X):
    Xa = np.atleast_2d(np.asarray(X, float))
    z = S5 * Xa
    envelope = 1.0 + 0.3 * np.sin(0.1 * z)
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

def nonuniform_period(X):
    Xa = np.atleast_2d(np.asarray(X, float))
    z = S5 * Xa
    periods = np.linspace(1.8, 2.2, D)
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

# ---- Rotated modulated Rastrigin (same construction as run_51) ----
_rng = np.random.RandomState(42)
_A = _rng.randn(D, D)
_Q, _R = np.linalg.qr(_A)
if np.linalg.det(_Q) < 0:
    _Q[:, 0] = -_Q[:, 0]
R_orth = _Q

def f_mod_rastrigin_rot(X):
    Xa = np.atleast_2d(np.asarray(X, float))
    Z = S5 * (R_orth @ Xa.T).T
    envelope = 1.0 + 0.3 * np.sin(0.1 * Z)
    return np.sum(envelope * (Z * Z - 10.0 * np.cos(2.0 * np.pi * Z) + 10.0), axis=1)

def grad_mod_rastrigin_rot(x):
    z = S5 * (R_orth @ x)
    envelope = 1.0 + 0.3 * np.sin(0.1 * z)
    denv_dz = 0.3 * 0.1 * np.cos(0.1 * z)
    f_base = z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0
    gz = envelope * (2.0 * z + 20.0 * np.pi * np.sin(2.0 * np.pi * z)) + denv_dz * f_base
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
    return S5 ** 2 * (R_orth.T @ np.diag(Hzz_diag) @ R_orth)

def test_family(name, f, grad, hess, center):
    """Near-solution starts: x0 = center + uniform(-5,5), Section V-B protocol."""
    print(f"\n--- {name} ---", flush=True)
    triggers, hits, pcu_f_vals, details = 0, 0, [], []
    for sd in SEEDS:
        rng = np.random.RandomState(sd)
        x0 = center + rng.uniform(-5, 5, D)
        t0 = time.time()
        try:
            x_cand, status = estimate_and_pcu(f, grad, hess, x0, tol=1e-4)
            el = time.time() - t0
            if status is not None:
                triggers += 1
                f_at_cand = float(np.asarray(f(x_cand.reshape(1, -1))).item())
                pcu_f_vals.append(f_at_cand)
                if f_at_cand < 1e-4:
                    hits += 1
                print(f"  seed {sd}: TRIGGER, F={f_at_cand:.2e} ({el:.1f}s)", flush=True)
                details.append({'seed': sd, 'trigger': True, 'hit': f_at_cand < 1e-4, 'F': f_at_cand, 'elapsed_s': round(el, 2)})
            else:
                print(f"  seed {sd}: no trigger ({el:.1f}s)", flush=True)
                details.append({'seed': sd, 'trigger': False, 'hit': False, 'F': None, 'elapsed_s': round(el, 2)})
        except Exception as e:
            print(f"  seed {sd}: ERROR - {e}", flush=True)
            details.append({'seed': sd, 'trigger': False, 'hit': False, 'F': None, 'error': str(e)})
    return {
        'name': name, 'triggers': triggers, 'hits': hits, 'n_runs': N_RUNS,
        'trigger_rate': triggers / N_RUNS,
        'hit_rate_when_triggered': hits / triggers if triggers > 0 else 0,
        'median_f': float(np.median(pcu_f_vals)) if pcu_f_vals else None,
        'rows': details,
    }

def raw_orth_attack_defense():
    print("\n" + "=" * 60, "\nRAW/ORTH SELECTIVITY", "=" * 60, sep="\n", flush=True)
    results = {}
    raw_triggers = 0
    for sd in SEEDS:
        rng = np.random.RandomState(sd)
        x0 = o + rng.uniform(-5, 5, D)
        try:
            _, status = estimate_and_pcu(rastrigin_raw, grad_rastrigin_raw, hess_rastrigin_raw, x0, tol=1e-4)
            if status is not None:
                raw_triggers += 1
        except Exception as e:
            print(f"  raw seed {sd}: ERROR {e}", flush=True)
    results['raw_triggers'] = raw_triggers
    orth_triggers, orth_hits = 0, 0
    for sd in SEEDS:
        rng = np.random.RandomState(sd)
        x0 = o + rng.uniform(-5, 5, D)
        try:
            x_cand, status = estimate_and_pcu(rastrigin_orth, grad_rastrigin_orth, hess_rastrigin_orth, x0, tol=1e-4)
            if status is not None:
                orth_triggers += 1
                f_cand = float(np.asarray(rastrigin_orth(x_cand.reshape(1, -1))).item())
                if f_cand < 1e-4:
                    orth_hits += 1
        except Exception as e:
            print(f"  orth seed {sd}: ERROR {e}", flush=True)
    results['orth_triggers'] = orth_triggers
    results['orth_hits'] = orth_hits
    print(f"  Raw triggers: {raw_triggers}/{N_RUNS}; Orth triggers: {orth_triggers}/{N_RUNS}, hits: {orth_hits}", flush=True)
    return results

def main():
    t_start = time.time()
    print("GATE 2b: family generalization under near-solution protocol", flush=True)
    all_results = {'dimensions': D, 'n_runs': N_RUNS, 'seeds': SEEDS,
                   'protocol': 'near-solution starts: x0 = center + U(-5,5)^D (Section V-B); rot uses run_51 QR(det=+1)',
                   'families': [], 'raw_orth': {}, 'elapsed_s': None}
    families = [
        ("Rastrigin (f5 M_orth)", rastrigin_orth, grad_rastrigin_orth, hess_rastrigin_orth, o),
        ("Triple cosine", triple_cosine, grad_triple_cosine, hess_triple_cosine, np.zeros(D)),
        ("Modulated Rastrigin", modulated_rastrigin, grad_mod_rastrigin, hess_mod_rastrigin, np.zeros(D)),
        ("Modulated Rastrigin (rot.)", f_mod_rastrigin_rot, grad_mod_rastrigin_rot, hess_mod_rastrigin_rot, np.zeros(D)),
        ("Non-uniform period", nonuniform_period, grad_nonuniform, hess_nonuniform, np.zeros(D)),
    ]
    for name, f, grad, hess, center in families:
        all_results['families'].append(test_family(name, f, grad, hess, center))
    all_results['raw_orth'] = raw_orth_attack_defense()
    all_results['elapsed_s'] = round(time.time() - t_start, 1)
    print("\nSUMMARY", flush=True)
    for fam in all_results['families']:
        print(f"  {fam['name']:28s}: {fam['triggers']}/{N_RUNS} trig, {fam['hits']}/{fam['triggers']} hit", flush=True)
    with open(os.path.join(RESULTS, 'gate2_families.json'), 'w', encoding='utf-8') as fp:
        json.dump(all_results, fp, indent=1)
    print(f"Saved gate2_families.json ({all_results['elapsed_s']}s)", flush=True)

if __name__ == '__main__':
    main()
