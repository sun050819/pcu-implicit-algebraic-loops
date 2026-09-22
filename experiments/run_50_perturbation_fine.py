# -*- coding: utf-8 -*-
"""run_50_perturbation_fine.py
Perturbation decay experiment with fine resolution near the threshold.

Measures trigger rate at alpha = 0.0 to 0.1 with 0.01 step,
to find exactly where the threshold is.

Output: results/perturbation_fine.json
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

Mr = rotations[D][4]
u, _, vt = np.linalg.svd(Mr)
M_orth = u @ vt
o = shifts[4][:D].copy()


def make_perturbed_matrix(alpha):
    _, s_raw, _ = np.linalg.svd(Mr)
    _, s_orth, _ = np.linalg.svd(M_orth)
    s_interp = s_orth + alpha * (s_raw - s_orth)
    return (u * s_interp) @ vt


def rastrigin_perturbed_factory(M):
    def f(X):
        Xa = np.atleast_2d(np.asarray(X, float))
        z = S5 * (Xa @ M.T - o @ M.T)
        return np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0, axis=1)

    def grad(x):
        x = np.atleast_1d(np.asarray(x, float))
        z = S5 * (M @ (x - o))
        gz = S5 * (2.0 * z + 20.0 * np.pi * np.sin(2.0 * np.pi * z))
        return M.T @ gz

    def hess(x):
        x = np.atleast_1d(np.asarray(x, float))
        z = S5 * (M @ (x - o))
        Hzz = S5 * S5 * (2.0 + 40.0 * np.pi * np.pi * np.cos(2.0 * np.pi * z))
        return (M.T * Hzz) @ M

    return f, grad, hess


def compute_nonorthogonality(M):
    return float(np.linalg.norm(M @ M.T - np.eye(D)))


def main():
    print("=" * 60)
    print("PERTURBATION DECAY - FINE RESOLUTION")
    print("=" * 60)
    print(f"Dimensions: {D}")
    print(f"Runs per perturbation level: {N_RUNS}")
    print()

    # Fine levels between 0 and 0.1
    alphas = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

    results = {
        'dimensions': D,
        'n_runs': N_RUNS,
        'seeds': SEEDS,
        'levels': [],
    }

    for alpha in alphas:
        M = make_perturbed_matrix(alpha)
        nonorth = compute_nonorthogonality(M)
        f, grad, hess = rastrigin_perturbed_factory(M)

        triggers = 0
        hits = 0

        for sd in SEEDS:
            rng = np.random.RandomState(sd)
            x0 = o + rng.uniform(-5, 5, D)
            try:
                x_cand, status = estimate_and_pcu(f, grad, hess, x0, tol=1e-4)
                if status is not None:
                    triggers += 1
                    f_at_cand = float(np.asarray(f(x_cand.reshape(1, -1))).item())
                    if f_at_cand < 1e-4:
                        hits += 1
            except Exception as e:
                print(f"  alpha={alpha:.2f}, seed={sd}: ERROR - {e}")

        trigger_rate = triggers / N_RUNS
        hit_rate = hits / triggers if triggers > 0 else 0

        print(f"alpha={alpha:.2f} | non-orth={nonorth:.3e} | "
              f"triggers={triggers}/{N_RUNS} | hits={hits}/{triggers}")

        results['levels'].append({
            'alpha': alpha,
            'nonorthogonality': nonorth,
            'triggers': triggers,
            'hits': hits,
            'trigger_rate': trigger_rate,
            'hit_rate_when_triggered': hit_rate,
        })

    out_path = os.path.join(RESULTS, 'perturbation_fine.json')
    with open(out_path, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
