"""Legacy 36-problem unified benchmark interface.

Unified solver interface for each problem:
    func(x) / grad(x) / hess(x) / x0 / ci_value / name / dim / x_opt / f_opt
Used by the solver layer 9.1.2 for comparison and stress testing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from .hastn_functions import PROBLEMS


@dataclass
class Problem36:
    name: str
    dim: int
    func: Callable
    grad: Callable
    hess: Callable
    x0: np.ndarray
    x_opt: np.ndarray
    f_opt: float
    ci_value: float
    description: str


def build_registry() -> Dict[str, Problem36]:
    reg = {}
    for name, p in PROBLEMS.items():
        reg[name] = Problem36(
            name=p.name, dim=p.dim, func=p.f, grad=p.grad, hess=p.hess,
            x0=np.asarray(p.x0, dtype=float),
            x_opt=np.asarray(p.x_opt, dtype=float),
            f_opt=float(p.f_opt), ci_value=float(p.ci_value),
            description=p.description)
    return reg


REGISTRY_36: Dict[str, Problem36] = build_registry()

# Complexity classification (used for stress-test statistics)
def tier_of(ci: float) -> str:
    if ci < 0.3:
        return "low"
    if ci < 0.7:
        return "medium"
    return "high"


def list_36() -> List[str]:
    return list(REGISTRY_36.keys())


def summary_36() -> str:
    lines = []
    for name, p in REGISTRY_36.items():
        lines.append(f"{name:28s} dim={p.dim:2d} CI={p.ci_value:.2f} "
                     f"{tier_of(p.ci_value):6s} |r(x0)|={np.linalg.norm(p.func(p.x0)):.3f}")
    return "\n".join(lines)
