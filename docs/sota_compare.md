# SOTA Timeliness Comparison (2026-09)

> **Terminology note:** solver version tags below (HGCA v5.4/v6.7, CMA-ES, v54/v74) are 


## 1. Fair Comparison on the Same Benchmark (run_04_sota_timeliness, 36 problems x 30 runs x maxfevals=30000)

Protocol identical to run_00_benchmark: A^2EP shifts (mag.U(-0.5,0.5)) + `make_shifted` + `_perturb_x0` perturbation + convergence criterion |f-f_opt|<1e-4.

| Solver | TOTAL convergence rate | Version / implementation |
|---|---|---|
| **solver v6.7 (with PCU)** | **0.9750** | this work, PCU stage 0b |
| solver v5.4 (without PCU) | 0.8917 | this work, baseline |
| BIPOP-aCMA-ES | 0.8611 | pycma 4.4.4 official fmin2(bipop=True) |
| NBIPOP-aCMA-ES | 0.8565 | Loshchilov 2012, manual implementation |

**Conclusion**: solver v6.7 beats the two SOTA CMA-ES restart variants on the same benchmark by +0.114 / +0.119; PCU alone contributes +0.0833 (v5.4 -> v6.7).

## 2. Key Problem-Level Differences (evidence that PCU is irreplaceable)

| Problem | v67 | v54 | bipop | nbipop |
|---|---|---|---|---|
| Rastrigin_20D_XE | **1.00** | 0.00 | 0.00 | 0.00 |
| Rastrigin_30D_XE | **1.00** | 0.00 | 0.00 | 0.00 |
| Rotated_Rastrigin_20D | **1.00** | 0.00 | 0.00 | 0.00 |
| Simulink_Nonlinear_Loop | **1.00** | 1.00 | 0.00 | 0.00 |
| Schaffer_F6_2D_XE | **0.23** | 0.23 | 0.00 | 0.00 |

- Shifted Rastrigin family: BIPOP/NBIPOP all 0.00 - the sample-iterate paradigm **systematically fails** on the "periodic center is the global optimum" structure; PCU is the only way through (analytic closed form, nit=2).
- Simulink_Nonlinear_Loop: CMA-ES family 0.00 (non-smooth / numerical-gradient scenario); solver homotopy + PCU 1.00.
- Slight lag on our side: Rotated_Ackley_20D (v67=0.93 vs bipop=1.00), Rosenbrock_10D_C (0.93 vs 1.00) - non-periodic family, PCU zero trigger; the gap comes from the base HGCA pipeline and is within an acceptable range.

## 3. Latest SOTA Literature Reference (2026; not directly comparable)

### MSC-CMA-ES (Nedanovski, Nenov, Pilev, arXiv:2606.15830, 2026)
- Method: structure-aware restart - Sobol pre-sampling + nearest-better clustering into basins, cycle detection, redundant-basin removal + local refinement.
- Benchmark: CEC 2014/2017/2020/2022, 10 (suite,dim) units, dimensions 5-30, 51 runs/function, official budgets.
- Results: best fixed-budget target coverage across the four suites on composition functions (2.7x BIPOP); best median error on base functions but low deep-target coverage (cost of spending budget on landscape discovery); CMA family lags DE family on hybrid functions.
- **Relation to this work**:
  1. **Orthogonal** to PCU - MSC is still in the sample-iterate paradigm (restart-strategy level), PCU is an analytic fast path (structure-identification level);
  2. different benchmarks (official CEC vs this work's A^2EP-shift fair benchmark) - **direct numbers are not comparable**; convergence rates must not be transferred across benchmarks;
  3. its "low deep-target coverage" cost is exactly the point: on the Rastrigin family the budget is consumed by basin discovery, so it cannot reach the optimum with nit=2 as PCU does.
- Code: the paper states "All results and scripts are publicly available" (see the arXiv page).

### Avoiding Redundant Restarts (De Nobel et al., arXiv:2405.01226, 2024)
- Quantifies redundant restarts on multimodal benchmarks and proposes a CMA-ES repelling-restart mechanism.
- Relation to this work: also a "restart structure" improvement, still sampling-iterative; no analytic fast-path concept.

### HR-CMA-ES (Lou et al., arXiv:1903.09085, 2019)
- Uses cNrGA search history with a BSP tree to aid restart ROI localization.
- Same as above: a restart-layer improvement, orthogonal to PCU.

## 4. Timeliness Conclusions

1. **The solver-architecture comparison (the CMA-ES 86.20% / BIPOP 86.11% / NBIPOP 85.28% figures cited in the docstring) still holds as of 2026-09**: re-running BIPOP/NBIPOP on this benchmark gives 86.11% / 85.65%, consistent with the literature figures (BIPOP fully reproduced at 86.11%).
2. **The 2026 MSC-CMA-ES is a restart-layer improvement and cannot undermine PCU's advantage on periodic structures**; its existence is honestly presented in the paper's SOTA section, with the benchmark-caliber difference noted.
3. To formally cite specific MSC-CMA-ES convergence rates in the paper, its public scripts would need to be downloaded and re-run on this A^2EP benchmark (costly; listed as optional future work).

*Data sources: run_04_sota_timeliness (sota_compare.csv / sota_compare_summary.json); literature: arXiv:2606.15830, arXiv:2405.01226, arXiv:1903.09085.*
