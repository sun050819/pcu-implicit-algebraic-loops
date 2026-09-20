"""(2) Intelligent breakpoint selection layer: unified entry for point selection strategies + top-k breakpoint sequence.

- For each candidate node on a cycle, each strategy provides a score/preference; sort head scores in descending order and take top-k (k<=2).
- Nested/overlapping cycle scenarios: call point selection separately for each inner cycle, achieving "optimal entry per cycle for nested cycles".
- Output BreakpointSet (candidates/scores/reliabilities).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..structures import BreakpointSet
from .features import build_features
from . import baselines as bl


# ---------------------------------------------------------------------------
# "scoring/point selection" adapters for each strategy
# ---------------------------------------------------------------------------
def _scores_outdeg(feats: np.ndarray) -> np.ndarray:
    return feats[:, 5].astype(float)


def _scores_type(feats: np.ndarray) -> np.ndarray:
    return feats[:, 0] * 4 + feats[:, 1] * 3 + feats[:, 2] * 2 + feats[:, 3] * 1


def _scores_random(n: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.rand(n)


def _scores_dt(clf, feats: np.ndarray) -> np.ndarray:
    return -clf.predict(feats)          # Smaller iteration count is better -> negate the score


def _scores_rl(w, feats: np.ndarray) -> np.ndarray:
    return feats @ w


def _scores_mlp(mlp, feats: np.ndarray) -> np.ndarray:
    return -mlp.predict(feats)


def _scores_gnn(predict, feats: np.ndarray, adj: np.ndarray) -> np.ndarray:
    return predict(feats, adj)


# ---------------------------------------------------------------------------
# top-k selection
# ---------------------------------------------------------------------------
def top_k(scores: np.ndarray, k: int = 2) -> List[int]:
    """Take top-k breakpoints in descending order of score (higher score is better)."""
    k = min(k, len(scores))
    order = np.argsort(-scores, kind="stable")
    return [int(i) for i in order[:k]]


def select_breakpoints(n: int, adj: np.ndarray, bt: Optional[np.ndarray],
                       loop_nodes: Sequence[int], method: str,
                       models: Optional[Dict] = None, topk: int = 2,
                       seed: int = 0, loop_id: int = 0,
                       feats_pre: Optional[np.ndarray] = None) -> BreakpointSet:
    """Select points for a single cycle according to the specified strategy.

    method: outdeg / type / random / optimal / dt / rl / mlp / sage / gcn / gat / tg
    models: learning model container ({'gat':..., 'dt':..., 'rl':..., 'mlp':..., 'sage':..., 'gcn':..., 'tg':...})
    feats_pre: precomputed features (same as training, e.g. LabelDataset.feats[sample]); if None, use
               build_features(n, adj, bt) to reconstruct (only for the generic entry without labels).
    """
    loop_nodes = list(loop_nodes)
    if feats_pre is not None:
        feats = np.asarray(feats_pre)[loop_nodes]
    else:
        feats = build_features(n, adj, bt)[loop_nodes]
    sub_adj = adj[np.ix_(loop_nodes, loop_nodes)]
    k = min(topk, len(loop_nodes))

    if method == "outdeg":
        scores = _scores_outdeg(feats)
        rel = np.zeros(len(loop_nodes))
    elif method == "type":
        scores = _scores_type(feats)
        rel = np.zeros(len(loop_nodes))
    elif method == "random":
        scores = _scores_random(len(loop_nodes), seed)
        rel = np.zeros(len(loop_nodes))
    elif method == "optimal":
        if models is None or "iters" not in models:
            raise ValueError("optimal requires models['iters'] to provide the true iteration count")
        iters = np.asarray(models["iters"])[loop_nodes]
        conv = np.asarray(models.get("conv", np.ones_like(iters)))[loop_nodes]
        j = bl.optimal_select(iters, conv)
        scores = np.zeros(len(loop_nodes))
        scores[j] = 1.0
        rel = np.zeros(len(loop_nodes)); rel[j] = 1.0
    elif method == "dt":
        clf = models["dt"]
        scores = _scores_dt(clf, feats)
        rel = np.zeros(len(loop_nodes))
    elif method == "rl":
        w = models["rl"]
        scores = _scores_rl(w, feats)
        rel = np.zeros(len(loop_nodes))
    elif method == "mlp":
        scores = _scores_mlp(models["mlp"], feats)
        rel = np.zeros(len(loop_nodes))
    elif method in ("sage", "gcn"):
        scores = _scores_gnn(models[method]["predict"], feats, sub_adj)
        rel = np.zeros(len(loop_nodes))
    elif method == "gat":
        model = models["gat"]
        rank_v, conv_p = model.predict(feats, sub_adj, beta=1.0)
        # Both heads fully used: ranking score + beta.convergence reliability (paper score = rank + beta*sigmoid(conv_logit))
        # The reliability head truly participates in point selection, avoiding reliance only on the ranking head
        beta = models.get("gat_beta", 1.0)
        scores = rank_v + beta * conv_p
        rel = conv_p
    elif method == "tg":
        # PyTorch Graph Transformer (numpy inference): same dual-head interface
        model = models["tg"]
        rank_v, conv_p = model.predict(feats, sub_adj, beta=1.0)
        scores = rank_v + conv_p
        rel = conv_p
    else:
        raise ValueError(method)

    candidates = top_k(scores, k)
    return BreakpointSet(
        loop_id=loop_id, loop_nodes=loop_nodes, candidates=candidates,
        scores=[float(scores[c]) for c in candidates],
        reliabilities=[float(rel[c]) for c in candidates], method=method)


# ---------------------------------------------------------------------------
# Select points cycle by cycle for the whole graph (optimal entry per cycle for nested cycles)
# ---------------------------------------------------------------------------
def select_all_loops(n: int, adj: np.ndarray, bt: Optional[np.ndarray],
                     loops: Sequence[Sequence[int]], method: str,
                     models: Optional[Dict] = None, topk: int = 2,
                     seed: int = 0) -> List[BreakpointSet]:
    """Select points independently for each cycle in LoopDB. loops: list of cycle node sequences."""
    out = []
    for i, loop in enumerate(loops):
        out.append(select_breakpoints(n, adj, bt, loop, method, models,
                                      topk=topk, seed=seed + i, loop_id=i))
    return out


def candidate_to_global(cands: BreakpointSet) -> List[int]:
    """Map candidates within a cycle back to global node indices."""
    return cands.candidates
