"""Breakpoint selection baselines.

Classical heuristics: outdeg / type / random; learning baselines: dt / rl / MLP / SAGE / GCN;
Theoretical upper bound: optimal (unreachable).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def outdeg_select(feats: np.ndarray) -> int:
    """Out-degree priority: take the node with the largest feats[:,5] (out-degree)."""
    return int(np.argmax(feats[:, 5]))


def type_select(feats: np.ndarray) -> int:
    """Type priority: Gain>Sum>Math>Product."""
    prio = feats[:, 0] * 4 + feats[:, 1] * 3 + feats[:, 2] * 2 + feats[:, 3] * 1
    return int(np.argmax(prio))


def random_select(n: int, seed: Optional[int] = None) -> int:
    rng = np.random.RandomState(seed)
    return int(rng.randint(n))


def optimal_select(iters: np.ndarray, conv: Optional[np.ndarray] = None) -> int:
    """Exhaustive optimal (unreachable upper bound): among the convergence candidates, the one with the fewest iterations."""
    conv = np.ones(len(iters)) if conv is None else np.asarray(conv).astype(bool)
    cand = np.where(conv)[0]
    if len(cand):
        return int(cand[np.argmin(iters[cand])])
    return int(np.argmin(iters))


def dt_predict(clf, feats: np.ndarray) -> int:
    pred = clf.predict(feats)
    return int(np.argmin(pred))


def rl_select(w: np.ndarray, feats: np.ndarray) -> int:
    return int(np.argmax(feats @ w))


# ---------------------------------------------------------------------------
# Learning baseline training
# ---------------------------------------------------------------------------
def train_decision_tree(X_all: np.ndarray, Y_all: np.ndarray, seed: int = 42, max_depth: int = 6):
    from sklearn.tree import DecisionTreeRegressor
    clf = DecisionTreeRegressor(max_depth=max_depth, random_state=seed)
    clf.fit(X_all, Y_all)
    return clf


def train_rl(X_all: np.ndarray, Y_all: np.ndarray, case_ix: Sequence[int],
             mask: Sequence[np.ndarray], seed: int = 42, epochs: int = 300, lr: float = 0.05):
    """Linear softmax policy REINFORCE (trained aggregated by case).

    X_all/Y_all are organized by case (3D [n,L,F] / 2D [n,L]), and mask is the valid mask for each case.
    """
    rng = np.random.RandomState(seed)
    n_feat = X_all.shape[2]
    w = rng.randn(n_feat) * 0.1
    for _ in range(epochs):
        g = np.zeros(n_feat)
        for j, i in enumerate(case_ix):
            X = X_all[j][mask[j]]
            it = Y_all[j][mask[j]]
            logits = X @ w
            logits -= logits.max()
            e = np.exp(logits - np.max(logits))
            p = e / e.sum()
            a = rng.choice(len(p), p=p)
            R = -it[a] / (it.mean() + 1e-6)
            g += R * (np.eye(len(p))[a] - p) @ X
        w += lr * g / max(len(case_ix), 1)
    return w


def train_mlp_sklearn(X_all: np.ndarray, Y_all: np.ndarray, seed: int = 42,
                      hidden=(64, 64), max_iter: int = 800):
    from sklearn.neural_network import MLPRegressor
    mlp = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", max_iter=max_iter,
                       random_state=seed, early_stopping=True, n_iter_no_change=30)
    mlp.fit(X_all, Y_all)
    return mlp


def mlp_predict(mlp, feats: np.ndarray) -> int:
    return int(np.argmin(mlp.predict(feats)))


# ---------------------------------------------------------------------------
# Custom numpy GNN (SAGE / GCN) -- based on autograd, training is reproducible
# ---------------------------------------------------------------------------
def sage_embed(X: np.ndarray, adj: np.ndarray, W1: np.ndarray, b1: np.ndarray,
               W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """2-layer GraphSAGE (mean-pooling) forward (for inference, computed directly with numpy)."""
    n = X.shape[0]
    A = np.logical_or(adj > 0, np.eye(n) > 0).astype(float)
    deg = A.sum(axis=1, keepdims=True).clip(min=1)
    mean_neigh = A @ X / deg
    h1 = np.maximum(0.0, np.concatenate([X, mean_neigh], axis=1) @ W1 + b1)
    mean2 = A @ h1 / deg
    h2 = np.maximum(0.0, np.concatenate([h1, mean2], axis=1) @ W2 + b2)
    return h2


def gcn_embed(X: np.ndarray, adj: np.ndarray, W1: np.ndarray, b1: np.ndarray,
              W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """2-layer GCN (symmetric normalization) forward."""
    n = X.shape[0]
    A = np.logical_or(adj > 0, np.eye(n) > 0).astype(float)
    deg = A.sum(axis=1).clip(min=1)
    Dinv = np.diag(1.0 / np.sqrt(deg))
    Ahat = Dinv @ A @ Dinv
    h1 = np.maximum(0.0, Ahat @ X @ W1 + b1)
    h2 = np.maximum(0.0, Ahat @ h1 @ W2 + b2)
    return h2


def train_gnn_np(X_all: np.ndarray, Y_all: np.ndarray, case_ix: Sequence[int],
                 mask: np.ndarray, adj_all: np.ndarray, kind: str = "SAGE",
                 hid: int = 64, epochs: int = 300, lr: float = 2e-3, seed: int = 42):
    """Train a 2-layer GNN with autograd (ranking loss, fully differentiable), returning (weight dict, inference function)."""
    from . import autograd as ag
    from .autograd import Parameter, Adam
    rng = np.random.RandomState(seed)
    in_f = X_all.shape[2]           # Feature dim of 3D [n, L, F]
    w1_rows = 2 * in_f if kind == "SAGE" else in_f
    w2_rows = 2 * hid if kind == "SAGE" else hid      # The second GCN layer does not concat
    Ps = [Parameter(rng.randn(w1_rows, hid) * 0.1, "W1"),
          Parameter(np.zeros(hid), "b1"),
          Parameter(rng.randn(w2_rows, hid) * 0.1, "W2"),
          Parameter(np.zeros(hid), "b2"),
          Parameter(rng.randn(hid, 1) * 0.1, "Wh")]
    opt = Adam(Ps, lr=lr, wd=1e-5)

    def forward_p(X, A):
        n = X.shape[0]
        Ab = np.logical_or(A > 0, np.eye(n) > 0).astype(float)
        deg = Ab.sum(axis=1, keepdims=True).clip(min=1)
        if kind == "SAGE":
            mean = ag.div(ag.matmul(ag.Tensor(Ab), ag.Tensor(X)), ag.Tensor(deg))
            # SAGE: concat[x, mean_neigh] @ W1
            cat1 = ag.cat([ag.Tensor(X), mean], axis=1)
            h1 = ag.relu(ag.add(ag.matmul(cat1, Ps[0].t), ag.expand(Ps[1].t, 0)))
            h1d = h1.data
            mean2 = ag.div(ag.matmul(ag.Tensor(Ab), h1), ag.Tensor(deg))
            cat2 = ag.cat([h1, mean2], axis=1)
            h2 = ag.relu(ag.add(ag.matmul(cat2, Ps[2].t), ag.expand(Ps[3].t, 0)))
        else:  # GCN
            d = 1.0 / np.sqrt(Ab.sum(axis=1).clip(min=1))
            Ahat = np.diag(d) @ Ab @ np.diag(d)
            h1 = ag.relu(ag.add(ag.matmul(ag.Tensor(Ahat @ X), Ps[0].t), ag.expand(Ps[1].t, 0)))
            h2 = ag.relu(ag.add(ag.matmul(ag.Tensor(Ahat @ h1.data), Ps[2].t), ag.expand(Ps[3].t, 0)))
        return ag.reshape(ag.matmul(h2, Ps[4].t), (-1,))

    def rank_loss(s: ag.Tensor, iters, margin: float = 0.3):
        N = s.data.shape[0]
        s_i = ag.expand(s, 1)
        s_j = ag.expand(s, 0)
        mask = (iters[:, None] < iters[None, :]).astype(float)
        loss_ij = ag.relu(ag.add(ag.sub(s_j, s_i), ag.Tensor(margin)))
        loss = ag.mul(loss_ij, ag.Tensor(mask))
        return ag.mul(ag.sum_op(loss), ag.Tensor(1.0 / max(mask.sum(), 1.0)))

    for _ in range(epochs):
        total = 0.0
        cnt = 0
        for i in case_ix:
            feats = X_all[i][mask[i]]
            iters = Y_all[i][mask[i]]
            adj = adj_all[i][mask[i]][:, mask[i]]
            N = feats.shape[0]
            if N < 2:
                continue
            s = forward_p(feats, adj)
            loss = rank_loss(s, iters)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.data
            cnt += 1
        if cnt == 0:
            break

    def predict(feats, adj):
        return forward_p(feats, adj).data

    weights = [p.t.data.copy() for p in Ps]
    return {"weights": weights, "predict": predict, "kind": kind}
