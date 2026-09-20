"""aloop - integrated analysis system for algebraic loops (the companion code of the PCU manuscript).

PCU (Periodic Coordinate Unwrapping) is a structure-aware optimization framework
for a class of periodic implicit algebraic loops; this package provides the full
pipeline: loop finding -> breakpoint selection -> solving, where the solving layer
contains the PCU fast path (see aloop/solve/hgca.py and aloop/solve/pcu.py).

Module organization (five layers):
    graph   : (1) Loop structure parsing layer (Gabow SCC outer loop + Tarjan-optimized bidirectional BFS inner loop + CI)
    select  : (2) Intelligent breakpoint selection layer (16-dim features + GAT dual-head + top-k)
    solve   : (3) Robust solving layer (three-stage + baseline solver)
    parallel: (4) Engineering acceleration layer (thread pool)
    validate: (5) Continuous learning and validation layer (statistical tests / benchmarks / ablation)
    data    : Benchmark and dataset generator
"""
__version__ = "2.3.17"
