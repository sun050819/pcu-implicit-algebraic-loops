"""(1) Cycle structure parsing layer: detection orchestration (outer-loop SCC + inner-loop enumeration + CI + parallel deduplication)."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..structures import Loop, LoopDB
from ..parallel.pool import map_parallel
from .gabow_scc import gabow_scc
from .tarjan_scc import tarjan_scc, tarjan_scc_memopt
from .cycles import inner_cycles_from, enumerate_cycles_plain, enumerate_cycles_bidir
from .ci import ci_components, compute_ci, get_ci_config


def _scc_groups(n: int, adj: np.ndarray, method: str = "gabow"):
    if method == "gabow":
        comp, cid = gabow_scc(n, adj)
    elif method == "tarjan":
        comp, cid = tarjan_scc(n, adj)
    elif method == "tarjan_memopt":
        comp, cid = tarjan_scc_memopt(n, adj)
    else:
        raise ValueError(method)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(comp[i], []).append(i)
    return [sorted(g) for g in groups.values() if len(g) > 1], comp


def _loop_ci_structural(loop_nodes: Sequence[int], adj: np.ndarray) -> float:
    """Structural CI when node_fn is absent (dimension + coupling density pathology proxy).

    v2.3.5 acceleration: large-dimension cycles with d>200 have extremely low density in sparse Simulink graphs (cascade model
    density~2/d), S=min(density*3,1)~0, skipping O(d^2) submatrix extraction.
    5000blk outer-loop CI 84.5ms-><0.1ms, CI value deviation <0.001 (S weight only 0.3).
    """
    d = len(loop_nodes)
    D = min(d / 10.0, 1.0)
    if d <= 200:
        sub = adj[np.ix_(loop_nodes, loop_nodes)]
        density = float(sub.sum()) / max(d * (d - 1), 1)
        S = min(density * 3.0, 1.0)
    else:
        S = 0.0  # Sparse large-graph density approximated as 0
    Delta = min(max(np.log10(1 + d) / 8.0, 0.2), 1.0)
    return round(0.4 * D + 0.3 * Delta + 0.3 * S, 4)


def _loop_ci_numeric(loop_nodes: Sequence[int], n: int, adj: np.ndarray,
                     node_fn: Callable, x0: float = 1.0) -> float:
    """Numerical CI based on breakpoint residual r(y)=phi(y)-y."""
    from ..loopeval import build_breakpoint_residual
    d = len(loop_nodes)
    brk = loop_nodes[0]
    try:
        r, J, func, grad, hess = build_breakpoint_residual(n, adj, node_fn, brk)
        g0 = np.asarray(grad(np.array([x0])), dtype=float).flatten()
        Delta = min(np.log10(1.0 + float(np.linalg.norm(g0))) / 8.0, 1.0)
        cond = abs(float(J(np.array([x0]))[0, 0]))
        S = min(cond / 100.0, 1.0)
        D = min(d / 10.0, 1.0)
        ci = 0.4 * D + 0.3 * Delta + 0.3 * S
        if not np.isfinite(ci):
            raise ValueError
        return round(float(ci), 4)
    except Exception:
        return _loop_ci_structural(loop_nodes, adj)


def _enum_single_scc(args) -> List[List[int]]:
    """Single-SCC inner-loop enumeration for process pool/thread pool calls (only uses adj, picklable)."""
    scc_nodes, adj, method, threshold, max_cycle_len = args
    return inner_cycles_from(scc_nodes, adj, method=method, threshold=threshold,
                             max_cycle_len=max_cycle_len)


def detect_loops(n: int, adj: np.ndarray, bt: Optional[np.ndarray] = None,
                 node_fn: Optional[Callable] = None,
                 scc_method: str = "gabow", inner_method: str = "auto",
                 threshold: int = 4, n_workers: Optional[int] = None,
                 use_process: bool = False, compute_ci: bool = True,
                 x0: float = 1.0, max_cycle_len: Optional[int] = None,
                 ci_method: str = "auto", ci_numeric_threshold: int = 50) -> LoopDB:
    """Full cycle structure parsing: outer loops (SCC) + all inner loops + CI per loop.

    Args:
        n, adj        : directed signal flow graph
        bt            : block type per node (used for CI/features, optional)
        node_fn       : output function per node (optional; if provided, CI uses numerical breakpoint residual, otherwise structural CI)
        scc_method    : gabow / tarjan / tarjan_memopt
        inner_method  : auto / plain / bidirectional (inner-loop enumeration)
        threshold     : node count threshold (<=threshold uses plain breadth, otherwise bidirectional breadth)
        n_workers     : inner-loop enumeration parallelism; >1 means parallel
        use_process   : inner-loop enumeration uses process pool for true parallelism
        compute_ci    : whether to compute CI
        max_cycle_len : inner-loop maximum length limit (None=no limit; 6-10 recommended for large-scale graphs)
        ci_method     : CI computation method: auto / numeric / structural
                        auto: n < ci_numeric_threshold uses numerical CI, otherwise structural CI (v2.3.1)
        ci_numeric_threshold: node count threshold for numerical CI (default 50)
    """
    # CI method selection (v2.3.1 automatic structural CI for large models)
    use_numeric_ci = (node_fn is not None and compute_ci and
                       (ci_method == "numeric" or
                        (ci_method == "auto" and n < ci_numeric_threshold)))

    groups, comp = _scc_groups(n, adj, scc_method)

    # Outer loop
    outer: List[Loop] = []
    for cid, g in enumerate(groups):
        ci = 0.0
        if compute_ci:
            if use_numeric_ci:
                ci = _loop_ci_numeric(g, n, adj, node_fn, x0)
            else:
                ci = _loop_ci_structural(g, adj)
        outer.append(Loop(nodes=g, ci=ci, tier=get_ci_config(ci)["tier"],
                          is_inner=False, scc_id=cid))

    # Inner-loop enumeration (parallelizable)
    scc_ix = [g for g in groups if len(g) > threshold]
    if n_workers and len(scc_ix) > 1:
        items = [(g, adj, inner_method, threshold, max_cycle_len) for g in scc_ix]
        inner_lists = map_parallel(_enum_single_scc, items, n_workers=n_workers,
                                   use_process=use_process)
    else:
        inner_lists = [_enum_single_scc((g, adj, inner_method, threshold, max_cycle_len))
                       for g in scc_ix]

    inner: List[Loop] = []
    seen: set = set()
    for (g, ilist) in zip(scc_ix, inner_lists):
        for cyc in ilist:
            fs = frozenset(cyc)
            if fs in seen:
                continue
            seen.add(fs)
            ci = 0.0
            if compute_ci:
                if use_numeric_ci:
                    ci = _loop_ci_numeric(cyc, n, adj, node_fn, x0)
                else:
                    ci = _loop_ci_structural(cyc, adj)
            inner.append(Loop(nodes=cyc, ci=ci, tier=get_ci_config(ci)["tier"],
                              is_inner=True, scc_id=-1,
                              meta={"scc_nodes": g}))

    db = LoopDB(n_nodes=n, outer_loops=outer, inner_loops=inner,
                adj=adj, block_types=bt, node_fn=node_fn,
                meta={"scc_method": scc_method, "inner_method": inner_method,
                      "threshold": threshold, "n_outer": len(outer),
                      "n_inner": len(inner), "ci_method": "numeric" if use_numeric_ci else "structural",
                      "max_cycle_len": max_cycle_len})
    return db
