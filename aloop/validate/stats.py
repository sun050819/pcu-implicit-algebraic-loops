"""Statistical testing tools (validation protocol 9.1/9.2).

- McNemar test: significant difference in paired binary classification (success/failure), used for comparing point selection/solving success rates.
- Wilcoxon signed-rank test: significant difference in paired continuous quantities (iteration count/residual).
- Paired t-test + bootstrap 95% CI.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def mcnemar_test(y1: np.ndarray, y2: np.ndarray,
                 alternative: str = "two-sided") -> dict:
    """McNemar test: y1,y2 are 0/1 success indicators.

    b = y1=1 and y2=0 (y1 wins); c = y1=0 and y2=1 (y2 wins).
    Test statistic (|b-c|-1)^2/(b+c) ~ chi2(1) (continuity correction).
    """
    y1 = np.asarray(y1).astype(int)
    y2 = np.asarray(y2).astype(int)
    b = int(((y1 == 1) & (y2 == 0)).sum())
    c = int(((y1 == 0) & (y2 == 1)).sum())
    n_disc = b + c
    if n_disc == 0:
        return {"stat": 0.0, "p": 1.0, "b": b, "c": c, "significant": False}
    from scipy.stats import chi2
    stat = (abs(b - c) - 1) ** 2 / n_disc
    p = chi2.sf(stat, 1)
    if alternative == "greater":
        p = 0.5 * p
    elif alternative == "less":
        p = 1.0 - 0.5 * p
    return {"stat": float(stat), "p": float(p), "b": b, "c": c,
            "significant": bool(p < 0.05)}


def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray, alternative: str = "two-sided",
                          idx_x=None, idx_y=None) -> dict:
    """Wilcoxon signed-rank test (paired continuous quantities). Returns {'stat','p','significant','n','note'}.

    Pairing convention (v2.3.14 fix): when the two methods' converged samples are of unequal length, must compute the
    true intersection by case index, not simply truncate x[:n]/y[:n] (which would misalign the pairing).

    Args:
        x, y: arrays of continuous quantities to test (e.g., iteration count, time taken).
        idx_x, idx_y: the case index corresponding to each sample in x/y (e.g., problem_id + run_id).
            When provided, compute the intersection by index and then pair; when not provided, fall back to equal-length truncation (correct only when len(x)==len(y)
            and the samples are naturally aligned).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    from scipy.stats import wilcoxon
    if idx_x is not None and idx_y is not None:
        # Compute the true intersection by case index.
        set_y = set(idx_y)
        common = [(i, j) for i, ix in enumerate(idx_x) if ix in set_y
                  for j, iy in enumerate(idx_y) if iy == ix]
        if len(common) < 2:
            return {"stat": float("nan"), "p": float("nan"), "significant": False,
                    "n": 0, "note": "paired intersection < 2"}
        xx = np.array([x[i] for i, _ in common])
        yy = np.array([y[j] for _, j in common])
        n = len(common)
        note = "paired by case index"
    else:
        n = min(len(x), len(y))
        if n < 2:
            return {"stat": float("nan"), "p": float("nan"), "significant": False,
                    "n": 0, "note": "paired intersection < 2"}
        xx = x[:n]
        yy = y[:n]
        note = "truncated (no case index)" if len(x) != len(y) else ""
    try:
        stat, p = wilcoxon(xx, yy, alternative=alternative, zero_method="wilcox")
    except ValueError as e:
        return {"stat": float("nan"), "p": float("nan"), "significant": False,
                "n": int(n), "note": "wilcoxon error: %s" % str(e)[:80]}
    return {"stat": float(stat), "p": float(p), "significant": bool(p < 0.05),
            "n": int(n), "note": note}


def paired_t_test(x: np.ndarray, y: np.ndarray) -> dict:
    from scipy.stats import ttest_rel
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    stat, p = ttest_rel(x, y)
    return {"stat": float(stat), "p": float(p), "significant": bool(p < 0.05)}


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 42,
                 alpha: float = 0.05) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for the mean. Returns (mean, lo, hi)."""
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return (float("nan"),) * 3
    means = np.array([x[rng.randint(n, size=n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)


def kfold_split(n: int, k: int = 5, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """5-fold cross-validation index pairs (train, test)."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    out = []
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        out.append((train, test))
    return out


def bootstrap_binary_ci(y: np.ndarray, n_boot: int = 5000, seed: int = 42,
                        alpha: float = 0.05) -> dict:
    """Bootstrap confidence interval for a proportion. Returns {rate, ci_lo, ci_hi, n_boot}."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return {"rate": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n_boot": n_boot}
    rates = np.array([y[rng.randint(n, size=n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(rates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"rate": float(y.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "n_boot": n_boot}


def summarize_binary(y: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    return {"n": int(len(y)), "success": int(y.sum()),
            "rate": float(y.mean()) if len(y) else float("nan")}
