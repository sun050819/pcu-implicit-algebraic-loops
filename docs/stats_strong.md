# Statistical Strengthening: 50-run Four-Config Ablation (run_05_stats_strong)

> **Terminology note:** solver version tags below (HGCA v5.4/v6.7, CMA-ES, v54/v74) are 


> Status: **complete** (2026-09-11). Results `results/stats_strong.csv` (7200 rows) + `stats_strong_summary.json`.
> Protocol: 36 problems (REGISTRY_36) x 50 runs x 4 configs, same A^2EP shifts, same perturbation seeds. Convergence criterion |f-f_opt|<1e-4.

## Configurations

| Config | use_pcu | use_struct_id | Meaning |
|---|---|---|---|
| v54 | NO | NO | pure main pipeline (homotopy + TR + CMA finish) |
| v71 | OK | NO | PCU fast path with known parameters |
| v74a | NO | OK | struct_id black-box structure identification alone |
| v74 | OK | OK | everything on |

## TOTAL (36 problems x 50 runs, 95% Wilson CI)

| Config | Convergence rate | 95% CI | Success / total |
|---|---|---|---|
| v54 | 0.8928 | [0.8785, 0.9071] | 1607/1800 |
| v71 | **0.9761** | [0.9691, 0.9832] | 1757/1800 |
| v74a | **0.9761** | [0.9691, 0.9832] | 1757/1800 |
| v74 | **0.9761** | [0.9691, 0.9832] | 1757/1800 |

v74 - v54 = +0.0833 (150 failures turned into successes), CIs do not overlap, **statistically significant**.
Total wall time 4914s (682.55 ms/run).

## Rastrigin Family (20D/30D/Rotated 20D + 2D, 200 runs per config)

| Config | Rastrigin_20D_XE | Rastrigin_30D_XE | Rotated_Rastrigin_20D | Family total |
|---|---|---|---|---|
| v54 | 0/50 = 0.00 | 0/50 = 0.00 | 0/50 = 0.00 | 0.250 (50/200) |
| v71 | 50/50 = 1.00 | 50/50 = 1.00 | 50/50 = 1.00 | 1.000 (200/200) |
| v74a | 50/50 = 1.00 | 50/50 = 1.00 | 50/50 = 1.00 | 1.000 (200/200) |
| v74 | 50/50 = 1.00 | 50/50 = 1.00 | 50/50 = 1.00 | 1.000 (200/200) |

- The sample-iterate paradigm (v54) fails all 50 runs on 20D/30D/rotated Rastrigin;
- Both PCU (v71) and automatic struct_id application (v74a) succeed 50/50 - **the effect is not accidental, and de-oracling causes no loss**.

## Zero Harm on Other Problem Families (identical across all four configs)

| Family | Convergence rate | Note |
|---|---|---|
| Beale 6 problems | 1.000 (300/300) | - |
| Powell 5 problems | 1.000 (250/250) | - |
| Wood 5 problems | 1.000 (250/250) | - |
| Rosenbrock 10 problems | 0.994 (497/500) | same in all four configs; PCU does not trigger |
| Other 6 problems | 0.867 (260/300) | Griewank/Michalewicz/Rotated_Ackley/Schaffer_F6/Simulink two loops; identical in all four configs |

PCU/struct_id has **zero harm** on non-periodic structures (the F<1e-4 acceptance criterion is the safety valve).

## Stage Distribution

| Config | pcu/struct | hgca_cma_failed | hgca_cma_refined | direct_newton | homotopy_direct |
|---|---|---|---|---|---|
| v54 | 0 | 193 | 701 | 752 | 99 |
| v71 | 200 (pcu_ok) | 43 | 651 | 752 | 99 |
| v74a | 200 (struct_pcu_ok) | 43 | 651 | 752 | 99 |
| v74 | 200 (pcu_ok) | 43 | 651 | 752 | 99 |

PCU/struct_id converts 150 hgca_cma_failed cases (Rastrigin family) into direct hits; the remaining 43 failures come from other families (identical across configs, unrelated to PCU).

## Paper Reporting

1. At 50-run statistical strength, PCU raises the Rastrigin family (20D/30D/rotated) convergence rate from 0 to 1.0 (200/200), TOTAL +0.0833 (CIs non-overlapping).
2. struct_id (automatic black-box structure identification) and PCU with known parameters give **identical results** - de-oracling costs no performance, supporting the "parameter-adaptive PCU" claim.
3. Non-periodic families show identical convergence rates across all four configs - the method has zero harm.
4. Compared with the SOTA benchmarks (run_04_sota_timeliness: BIPOP-aCMA 0.8611 / NBIPOP-aCMA 0.8565, Rastrigin family 0.00), the PCU version at 0.9761 is the highest on the same benchmark.
