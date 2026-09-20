"""GNN-enhanced minimum feedback vertex set (FVS) -- v2.3.3 P1-7.

Integrate the breakpoint ranking prior of a pretrained Graph Transformer into CI-guided greedy FVS:
  1. GNN predicts rank (breakpoints with fewer iterations have higher rank) -> normalize to [0,1]
  2. Fuse into node CI: ci = alpha * rank_norm + (1-alpha) * ci_structural
  3. FVS greedy node selection score = deg / (ci + eps) -> nodes that the GNN considers "good solutions" are prioritized

Motivation: pure structural CI (out/in-degree) only reflects topological coupling, without the solvability prior of
"whether the residual at this breakpoint is a good solution"; the GNN learns this prior from a large number of synthetic + industrial samples, which can improve breakpoint quality (faster convergence, fewer iterations)
without increasing the number of breakpoints.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .torchgnn_infer import TorchGNNInfer  # noqa: F401  (for type hints)
from ..graph.fvs import ci_guided_fvs, node_ci_structural


def fvs_with_gnn(n: int, adj: np.ndarray, gnn,
                 feats: np.ndarray, alpha: float = 0.5) -> List[int]:
    """GNN-enhanced FVS: greedy node selection fusing GNN rank prior + structural CI.

    Args:
        n: number of nodes
        adj: adjacency matrix (n,n), adj[i,j]>0 means i->j
        gnn: TorchGNNInfer instance (predict(feats, adj_aug) -> (rank, conv_p))
        feats: 16-dimensional node features (output of build_features)
        alpha: GNN prior weight (0=pure structural CI, 1=pure GNN rank)

    Returns:
        list of breakpoint nodes (ascending)
    """
    adj_aug = (np.asarray(adj, dtype=float) > 0).astype(float)
    np.fill_diagonal(adj_aug, 1.0)
    rank, _ = gnn.predict(np.asarray(feats, dtype=float), adj_aug)
    rank = np.asarray(rank, dtype=float).ravel()
    r_norm = (rank - rank.min()) / (rank.max() - rank.min() + 1e-12)
    ci = {}
    for v in range(n):
        cs = node_ci_structural(v, n, np.asarray(adj, dtype=float))
        ci[v] = alpha * float(r_norm[v]) + (1.0 - alpha) * cs
    return ci_guided_fvs(n, np.asarray(adj, dtype=float), node_ci=ci)


def build_feats_from_model(mo, adj: np.ndarray) -> np.ndarray:
    """Build 16-dimensional GNN features from SlxModel (block type mapped to one-hot index).

    one-hot type indices (consistent with the bt convention of features.build_features):
      0=Gain, 1=Sum, 2=Math, 3=Product; other types are treated as 0.
    """
    from ..select.features import build_features
    bt = []
    type_map = {"Gain": 0, "Sum": 1, "Math Function": 2, "Product": 3}
    for b in mo.blocks:
        t = b.block_type
        bt.append(type_map.get(t, 0))
    return build_features(mo.n, np.asarray(adj, dtype=float), np.asarray(bt, dtype=int))
