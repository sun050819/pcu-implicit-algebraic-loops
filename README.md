# PCU - Periodic Coordinate Unwrapping for Periodic Implicit Algebraic Loops

Companion repository for the manuscript:

> **Periodic Coordinate Unwrapping: Structure-Aware Optimization for a Class of Periodic Implicit Algebraic Loops** (submitted to IEEE Transactions on Evolutionary Computation)

PCU is a structure-aware optimization framework for a class of periodic implicit algebraic loops. Instead of blind black-box sampling, it (i) identifies the periodic structure of the least-squares objective along Hessian-diagonal directions (with rotation recovery), and (ii) generates candidate roots by zero-crossing detection at interpolated peaks. The paper states four propositions that separate strict guarantees from falsifiable conditions and negative diagnostics, and validates end-to-end on CEC2017 (60 configurations), up to 500D, and Simulink periodic algebraic loops.

---

## Repository layout

```
aloop/                        # core implementation (structure identification, PCU, solvers)
experiments/
  run_00.py .. run_43.py      # one script per experiment, run in order
  results/                      # 59 raw result files (49 JSON + 8 CSV + 2 logs, authoritative data)
  simulink/                     # MATLAB/Simulink model scripts (runs 06/19/23)
third_party/cec2017/          # official CEC2017 benchmark package (incl. data.pkl)
tools/
  audit_data.py                 # `--hallucination`: paper numbers vs JSON backtrace
```

The manuscript sources (IEEEtran `paper.tex`, figures, cover letter,
supplementary trigger matrix) are kept private during single-blind review and
will be released upon acceptance; all experimental code and data above are
public and reproduce every number in the paper.

## Dependencies

- Python 3.11
- `numpy`, `scipy`, `cma` - required (minimum versions in `requirements.txt`)
- `torch` with CUDA - only for the GPU (250-500D) experiments
- MATLAB/Simulink - only for the Simulink algebraic-loop validation (runs 06/19/23); the Python experiments run without it

### Optional dependencies

- **GPU experiments:** `pip install torch --index-url https://download.pytorch.org/whl/cu121` (CUDA 12.1).
- **MATLAB engine (Python↔MATLAB bridge):** `pip install matlabengine` requires a local MATLAB installation (R2024b or newer) and its `extern/engines/python` directory on `PATH`. This is only needed if you want to run the Simulink loop validation directly from Python; the numerical experiments do not require it.
- **Testing:** `pip install pytest` (runs `tests/` and the CEC2017 package tests).

## Reproducing the experiments

```powershell
# 1. Numerical verification of the identifiability theory (Prop. 1-3)
python -X utf8 experiments/run_18_theory_verify.py

# 2. CEC2017 60-combination trigger/hit matrix
python -X utf8 experiments/run_07_cec.py
python -X utf8 experiments/run_08_trigger_matrix.py

# 3. SOTA comparison, same budget (396,993 FE), 11 optimizers
python -X utf8 experiments/run_17_sota_compare.py
python -X utf8 experiments/run_24_cmaes_5runs.py
python -X utf8 experiments/run_28_lbfgs_25runs.py
python -X utf8 experiments/run_33_bipop.py        # BIPOP-CMA-ES baseline

# 4. 50-run statistical strengthening
python -X utf8 experiments/run_38_pcu_50runs.py

# 5. High-dimensional stress tests (CPU; GPU variants use run_27/30/34/40 with torch+CUDA)
python -X utf8 experiments/run_13_stress.py

# 6. Noise robustness and ablations
python -X utf8 experiments/run_14_noise.py
python -X utf8 experiments/run_15_ablation.py
python -X utf8 experiments/run_32_vote_ablation.py

# 7. Simulink periodic algebraic loops (requires MATLAB)
#    run_19_simulink_periodic.py, run_23_simulink_nonsep.py

# 8. PCU--EA integration (new; initialization-gain plus stagnation-restart ablation)
python -X utf8 experiments/run_43_pcu_restart_ablation.py   # fast: 100k FE, 5 seeds, restart boundary
python -X utf8 experiments/run_42_pcu_ea_hybrid.py          # 100,000 FE, 10 seeds, trigger accounting, trigger accounting
```

Scripts write into `experiments/results/` and are idempotent: re-running
reproduces the same JSON when the fixed seeds are kept. All 44 `run_*.py`
scripts in `experiments/` (including `run_09` finite-difference, `run_11`
full 60-config CEC table, `run_16` data profile, `run_25` CMA-ES trajectories,
`run_35` multistart Newton, `run_36` CPU high-dim reruns, `run_41` CEC matrix
norms, `run_42` PCU+EA hybrid (PCU candidates feeding CMA-ES/SHADE, with
trigger-rate accounting), `run_43` PCU-vs-random stagnation-restart ablation,
and the GPU variants
`run_27/30/34/40`) feed the 59 committed result files
(generator mapping per file in the provenance paragraph below; running
`run_42`/`run_43` additionally writes `pcu_ea_hybrid.json` and
`pcu_restart_ablation.json`); the pipelines
above cover every figure and table of the paper.

## Data provenance (every number in the paper)

Every numeric claim in the manuscript traces to a named file in
`experiments/results/`. The audit script re-verifies this automatically:

```powershell
python -X utf8 tools/audit_data.py --hallucination
```

It greps `paper.tex` for all locked values and independently recomputes the
statistics (medians, min/max, hit rates) from the raw JSON. As of the current
revision: **0 mismatches** (verified locally with the manuscript present).
In this public repository the manuscript is absent, so `--hallucination`
skips the paper-vs-JSON backtrace and only the structural cross-check runs;
the committed result files remain independently verifiable via the JSON
recomputation section of the audit script.

Of the 59 result files, ten GPU/orth records
(`gpu_*`, `orth500_round3.json`, `pcu_gpu_*`) were captured from GPU
sessions and support the 250-500-D results in the paper; the identical
protocol is re-runnable via the GPU repro scripts in `experiments/repro/`
(`r3_gpu_identify.py`, `r3_gpu_full.py`, `r3_gpu_gap.py`,
`r3_boundary.py`, `r3_supp.py`), which write `_r3_*` snapshots so the
committed data files are never overwritten. `gpu_300400d_10seeds_merged.json`
is the 10-seed assembly (7 new GPU seeds plus the 3 earlier GPU seeds) used
for Table VI; `*_summary.json` and the `*_smoke`-free CSV sidecars are
aggregation views of the raw JSON. `pcu_vs_v54`,
`pcu_integrated`, and `sota_compare_summary.json` are early
integrated-baseline records (column names such as `hgca_v67`/`hgca_v54` refer
to historical solver versions of this project) kept for completeness; all
other result files are produced by the 44 `run_*.py` scripts
(`run_42` writes `pcu_ea_hybrid.json`, `run_43` writes
`pcu_restart_ablation.json` when executed).

## Notes

- Authors: Jichen Sun (first author) and Shiyi Yi (corresponding author),
  College of Art and Information Engineering, Dalian Polytechnic University.
- MATLAB/Simulink is used only as the validation tool for algebraic-loop
  solving; the method itself is tool-agnostic.


## License

All code and data in this repository are released under the MIT License
(see `LICENSE`). The paper text is under the authors' copyright.
The bundled CEC2017 benchmark package under `third_party/cec2017/` is
distributed under its own MIT license (© 2022 Duncan Tilley), included as
`third_party/cec2017/LICENSE.txt`.
