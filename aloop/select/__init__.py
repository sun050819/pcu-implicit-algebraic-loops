"""(2) Intelligent breakpoint selection layer: 16-dimensional features + GAT dual-head + baseline + top-k."""
from .features import build_features, extract_features
from .selector import select_breakpoints, select_all_loops, top_k, BreakpointSet
