"""Robust solving layer: HGCA hybrid solver (with the PCU fast path) + Newton/SA baselines + breakpoint residuals.

Solver aliases are retained for backward compatibility
with earlier algebraic-loop solvers; the PCU structure-identification path is described
in the manuscript."""
from .newton import NEWTON_SOLVERS
from .sa import SA_SOLVERS
from .hybrid import SOLVER_REGISTRY, hast_n, hast_n_ci_adaptive
from .hgca import hgca
from .pipeline import (solve_loop_with_breakpoints, solve_loop_direct,
                       solve_breakpoint_brentq, loop_ci)
