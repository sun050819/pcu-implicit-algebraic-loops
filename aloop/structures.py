"""Core explicit data structures (inter-module decoupling contracts).

Corresponds to Section 7 data flow of the design specification:
    signal_flow_graph -> LoopDB -> BreakpointSet -> SolveResult -> labels -> write back to labels

All cross-layer data uses these dataclass / dictionaries, with accompanying to_dict / from_dict to support JSON schema.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Callable


# ---------------------------------------------------------------------------
# (1) Loop Topology Database LoopDB
# ---------------------------------------------------------------------------
@dataclass
class Loop:
    """Single algebraic loop: outer loop (SCC) or inner loop."""
    nodes: List[int]          # Node order on the loop (along directed graph order)
    ci: float = 0.0           # Composite Complexity Index
    tier: str = "low"         # simple / medium / hard  (low / medium / high)
    is_inner: bool = False    # True=inner loop, False=outer loop
    scc_id: int = -1          # ID of the strongly connected component it belongs to
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LoopDB:
    """Output of the loop structure parsing layer: outer loops + all inner loops + CI per loop."""
    n_nodes: int = 0
    outer_loops: List[Loop] = field(default_factory=list)   # Outer loops (SCC)
    inner_loops: List[Loop] = field(default_factory=list)   # All inner loops
    # Graph information (for use by subsequent layers; optional to carry)
    adj: Optional[Any] = None
    block_types: Optional[Any] = None
    node_fn: Optional[Callable] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_loops(self) -> List[Loop]:
        return self.outer_loops + self.inner_loops

    def to_json(self) -> str:
        d = {
            "n_nodes": self.n_nodes,
            "outer_loops": [l.to_dict() for l in self.outer_loops],
            "inner_loops": [l.to_dict() for l in self.inner_loops],
            "meta": self.meta,
        }
        return json.dumps(d, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# (2) Breakpoint Set BreakpointSet
# ---------------------------------------------------------------------------
@dataclass
class BreakpointSet:
    """Output of the intelligent breakpoint selection layer: top-k breakpoint sequence for each loop (in descending score order)."""
    loop_id: int = -1
    loop_nodes: List[int] = field(default_factory=list)
    candidates: List[int] = field(default_factory=list)     # top-k candidates (ordered)
    scores: List[float] = field(default_factory=list)       # sorting head scores (higher is better)
    reliabilities: List[float] = field(default_factory=list)  # convergence reliability (0~1)
    method: str = "GAT"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# (3) Solve Result SolveResult
# ---------------------------------------------------------------------------
@dataclass
class SolveResult:
    """Output of the robust solving layer: solution, iteration count, residual, convergence flag."""
    converged: bool = False
    x: Any = None                 # np.ndarray
    iterations: int = 0
    residual: float = float("nan")
    time_ms: float = 0.0
    stage: str = ""               # direct_newton / sa_refined / fixed_point / brentq / failed
    breakpoint: Optional[int] = None   # adopted breakpoint (None=directly solve the whole system)
    retries: int = 0              # number of breakpoint retries (top-k in sequence)
    nit_breakdown: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.x is not None:
            try:
                d["x"] = list(self.x)
            except TypeError:
                d["x"] = float(self.x)
        return d
