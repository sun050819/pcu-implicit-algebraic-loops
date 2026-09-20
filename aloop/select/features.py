"""16-dimensional node features.

Feature composition: 4-dimensional block type one-hot + 12-dimensional topological features
    4  one-hot : [Gain, Sum, Math, Product]
    12 topological: in-degree / out-degree / degree centrality / node count n / out-ratio / in-out ratio /
                 second-order out-neighbor count / second-order in-neighbor count / clustering coefficient / branching complexity /
                 intra-cycle depth / betweenness approximation
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def extract_features(n: int, A: np.ndarray) -> np.ndarray:
    """12-dimensional topological features (excluding type one-hot)."""
    in_deg = A.sum(axis=0)
    out_deg = A.sum(axis=1)
    deg_cent = (in_deg + out_deg) / max(n - 1, 1)
    A2 = A @ A
    second_out = A2.sum(axis=1)
    second_in = A2.sum(axis=0)
    cluster = np.zeros(n)
    for i in range(n):
        neigh = list(np.where((A[i] > 0) | (A[:, i] > 0))[0])
        k = len(neigh)
        if k > 1:
            edges = 0
            for a in range(k):
                for b in range(a + 1, k):
                    if A[neigh[a], neigh[b]] or A[neigh[b], neigh[a]]:
                        edges += 1
            cluster[i] = 2 * edges / (k * (k - 1))
    branch = out_deg * second_out
    loop_depth = np.maximum(1, n) / np.maximum(out_deg, 1)
    between = deg_cent * (np.full(n, n) / max(n, 1))
    out_ratio = out_deg / np.maximum(in_deg + out_deg, 1)
    in_out = in_deg / np.maximum(out_deg, 1)
    topo = np.stack([in_deg, out_deg, deg_cent, np.full(n, n), out_ratio, in_out,
                     second_out, second_in, cluster, branch, loop_depth, between], axis=1)
    return topo


def build_features(n: int, A: np.ndarray, bt: Optional[Sequence[int]] = None) -> np.ndarray:
    """16-dimensional features: 4 one-hot types + 12 topological. When bt is omitted, one-hot is set to zero."""
    f = extract_features(n, A)
    if bt is None:
        onehot = np.zeros((n, 4))
    else:
        bt = np.asarray(bt, dtype=int)
        onehot = np.zeros((n, 4))
        onehot[np.arange(n), bt % 4] = 1.0
    return np.concatenate([onehot, f], axis=1)
