"""GAT dual-head network (ranking head + convergence reliability head).

[WARNING] Known issues (discovered in v2.3.14 review):
    GAT training based on the in-house autograd micro-framework, under full configuration (150 epochs + joint data),
    degenerates to an "all-zero output" pathological attractor (rank head == 0, conv head == 0.5, loss constant ~ 0.536),
    causing node selection to degenerate to taking the first two nodes in the cycle. The root cause is suspected to be numerical stability issues in the interaction between autograd gradient backpropagation and the mono_loss
    mask, requiring deep debugging.
    Current status: an experimental comparison method, not the primary node selection scheme. The primary node selection method is tg (Graph Transformer,
    trained with PyTorch, convergence rate 1.000). GAT historical experimental data (run_01/run_01c) is retained but marked as
    "historical version, training stability to be fixed", and cannot be used as a reproducible performance claim for the current code.


Structure (consistent with the paper):
    GATRanking: 3-layer GAT (16 -> 64x2h -> 64x2h -> 64) + residual skip(16->64)
                output concatenation [h, skip] = 128-dim encoding
    GATRankingConv (dual-head):
        rank_head: Linear(128->1)  breakpoint score (higher is better)
        conv_head: Linear(128->1)  convergence reliability logit (sigmoid = probability of convergence)
    Inference node selection: score = rank + beta * sigmoid(conv_logit)

Training loss (exp_conv.py):
    loss = MarginRanking(rank, iters) + 0.5*BCE(conv_logit, conv) + 0.1*monotonicity regularization(rank, outdeg)

All operators are based on pure NumPy micro-automatic differentiation in autograd.py, with no torch dependency.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import autograd as ag
from .autograd import Parameter, Adam, xavier


def add_self_loops(adj: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    return (adj > 0) | np.eye(n, dtype=bool)


# ==========================================================================
# GAT convolution layer
# ==========================================================================
class GATConv:
    def __init__(self, in_f: int, out_f: int, heads: int = 2, concat: bool = True,
                 dropout: float = 0.0, seed: int = 42, leaky: float = 0.1):
        self.rng = np.random.RandomState(seed)
        self.in_f, self.out_f, self.heads, self.concat = in_f, out_f, heads, concat
        self.dropout, self.leaky = dropout, leaky
        self.W = Parameter(xavier((heads, out_f, in_f), self.rng), "W")
        self.a1 = Parameter(xavier((heads, out_f), self.rng), "a1")
        self.a2 = Parameter(xavier((heads, out_f), self.rng), "a2")

    def forward(self, x: ag.Tensor, adj: np.ndarray, training: bool = False) -> ag.Tensor:
        N = x.data.shape[0]
        mask = np.where(add_self_loops(adj), 0.0, -1e9)          # [N,N]
        outs: List[ag.Tensor] = []
        for h in range(self.heads):
            Wh = self._Wh(x, h)                                   # [N,O]
            a1 = ag.Tensor(self.a1.data[h][:, None])              # [O,1]
            a2 = ag.Tensor(self.a2.data[h][:, None])
            e1 = ag.matmul(Wh, a1)                                # [N,1]
            e2 = ag.matmul(Wh, a2)                                # [N,1]
            e = ag.add(e1, ag.transpose(e2))                      # [N,1]+[1,N]=[N,N]
            e = ag.add(e, ag.Tensor(mask))
            e = ag.leaky_relu(e, self.leaky)
            alpha = ag.softmax(e, axis=1)                         # [N,N]
            if training and self.dropout > 0:
                m = (self.rng.rand(N, N) >= self.dropout).astype(float)
                alpha = ag.mul(alpha, ag.Tensor(m / (1.0 - self.dropout)))
            out = ag.matmul(alpha, Wh)                            # [N,O]
            out = ag.leaky_relu(out, self.leaky)
            outs.append(out)
        if self.concat:
            return ag.cat(outs, axis=1)                           # [N, H*O]
        s = outs[0]
        for o in outs[1:]:
            s = ag.add(s, o)
        return ag.mul(s, ag.Tensor(1.0 / self.heads))

    def _Wh(self, x: ag.Tensor, h: int) -> ag.Tensor:
        """x @ W[h].T (gradient of W[h] is correctly backpropagated)."""
        Wt = ag.transpose(ag.take(self.W.t, h, axis=0))           # [F,O]
        return ag.matmul(x, Wt)

    def params(self) -> List[Parameter]:
        return [self.W, self.a1, self.a2]


# ==========================================================================
# GATRanking encoder (3 layers + skip)
# ==========================================================================
class GATRanking:
    def __init__(self, in_f: int = 16, hid: int = 64, heads: int = 2,
                 n_layers: int = 3, dropout: float = 0.15, seed: int = 42):
        self.hid = hid
        self.layers: List[GATConv] = []
        cur = in_f
        for k in range(n_layers):
            concat = k < n_layers - 1
            hh = heads if concat else 1
            layer = GATConv(cur, hid, heads=hh, concat=concat, dropout=dropout, seed=seed + k)
            self.layers.append(layer)
            cur = hh * hid if concat else hid
        self.skip = Parameter(xavier((in_f, hid), np.random.RandomState(seed + 100)), "skip")
        self.params_list = [self.skip]
        for l in self.layers:
            self.params_list.extend(l.params())

    def encode(self, x: ag.Tensor, adj: np.ndarray, training: bool = False) -> ag.Tensor:
        h = x
        for layer in self.layers:
            h = layer.forward(h, adj, training)
        sk = ag.matmul(x, ag.Tensor(self.skip.data))               # x @ W_skip
        return ag.cat([h, sk], axis=1)                             # [N, 2*hid]

    def params(self) -> List[Parameter]:
        return self.params_list


# ==========================================================================
# GAT dual-head network (ranking + convergence reliability)
# ==========================================================================
class GATRankingConv:
    def __init__(self, in_f: int = 16, hid: int = 64, heads: int = 2,
                 n_layers: int = 3, dropout: float = 0.15, seed: int = 42):
        self.gat = GATRanking(in_f, hid, heads, n_layers, dropout, seed)
        rng = np.random.RandomState(seed + 1000)
        self.rank_head = Parameter(xavier((2 * hid, 1), rng), "rank_head")
        self.conv_head = Parameter(xavier((2 * hid, 1), rng), "conv_head")
        self.params_list = self.gat.params() + [self.rank_head, self.conv_head]

    def forward(self, x: ag.Tensor, adj: np.ndarray, training: bool = False):
        e = self.gat.encode(x, adj, training)                      # [N, 2*hid]
        rank = ag.reshape(ag.matmul(e, ag.Tensor(self.rank_head.data)), (-1,))
        clogit = ag.reshape(ag.matmul(e, ag.Tensor(self.conv_head.data)), (-1,))
        return rank, clogit

    def params(self) -> List[Parameter]:
        return self.params_list

    def predict(self, feats: np.ndarray, adj: np.ndarray, beta: float = 1.0):
        """Inference: return (ranking score, convergence probability)."""
        rank, clogit = self.forward(ag.Tensor(feats), adj, training=False)
        conv_p = ag.sigmoid_np(clogit.data)
        return rank.data, conv_p

    def state_dict(self) -> dict:
        # Use indices as keys in parameter order (multi-layer parameters with the same name 'W'/'a1'/'a2' cannot be used as keys)
        return {str(i): p.t.data.copy() for i, p in enumerate(self.params())}

    def load_state_dict(self, sd: dict):
        for i, p in enumerate(self.params()):
            if str(i) in sd:
                p.t.data = sd[str(i)]


# ==========================================================================
# Differentiable loss (all built on the autograd computation graph)
# ==========================================================================
def ranking_loss(rank: ag.Tensor, iters: np.ndarray, margin: float = 0.3) -> ag.Tensor:
    """Pairwise MarginRanking: nodes with higher rank corresponding to fewer iterations are better."""
    N = rank.data.shape[0]
    s_i = ag.expand(rank, 1)                       # [N,1]
    s_j = ag.expand(rank, 0)                       # [1,N]
    it_i = iters[:, None]
    it_j = iters[None, :]
    mask = (it_i < it_j).astype(float)             # [N,N]
    loss_ij = ag.relu(ag.add(ag.sub(s_j, s_i), ag.Tensor(margin)))
    loss = ag.mul(loss_ij, ag.Tensor(mask))
    denom = max(mask.sum(), 1.0)
    return ag.mul(ag.sum_op(loss), ag.Tensor(1.0 / denom))


def mono_loss(rank: ag.Tensor, outdeg: np.ndarray) -> ag.Tensor:
    """Monotonicity regularization: larger out-degree -> higher cost -> lower score. Penalizes outdeg_i<outdeg_j and s_i<s_j."""
    N = rank.data.shape[0]
    s_i = ag.expand(rank, 1)
    s_j = ag.expand(rank, 0)
    o_i = outdeg[:, None]
    o_j = outdeg[None, :]
    mask = ((o_i < o_j) * (s_i.data < s_j.data)).astype(float)
    loss = ag.mul(ag.sub(s_j, s_i), ag.Tensor(mask))
    denom = max(mask.sum(), 1.0)
    return ag.mul(ag.sum_op(loss), ag.Tensor(1.0 / denom))


def total_loss(model: GATRankingConv, feats: np.ndarray, iters: np.ndarray,
               conv: np.ndarray, margin: float = 0.3, conv_w: float = 0.5,
               lam: float = 0.1) -> ag.Tensor:
    x = ag.Tensor(feats)
    rank, clogit = model.forward(x, feats_adj(feats), training=True)
    rl = ranking_loss(rank, iters, margin)
    cl = ag.bce_with_logits(clogit, conv.astype(float))
    loss = ag.add(rl, ag.mul(cl, ag.Tensor(conv_w)))
    if lam > 0:
        ml = mono_loss(rank, feats[:, 5])
        loss = ag.add(loss, ag.mul(ml, ag.Tensor(lam)))
    return loss


def feats_adj(feats: np.ndarray) -> np.ndarray:
    """Recover cycle adjacency from features (degrades to a fully connected cycle when features are unavailable)."""
    n = feats.shape[0]
    ring = np.zeros((n, n))
    if n > 1:
        for k in range(n):
            ring[k, (k + 1) % n] = 1
            ring[k, (k - 1) % n] = 1
    return ring


def train_gat_dual(model: GATRankingConv, dataset, train_idx: Sequence[int],
                   epochs: int = 400, lr: float = 2e-3, wd: float = 1e-5,
                   margin: float = 0.3, conv_w: float = 0.5, lam: float = 0.1,
                   patience: int = 6, seed: int = 42,
                   extra: Optional[Sequence[Tuple[np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray]]] = None,
                   extra_w: float = 1.0) -> GATRankingConv:
    """Train the dual-head GAT on the labeled set. dataset[i] -> (feats, iters, conv, adj).

    extra: list of additional training samples, each item (feats[N,F], iters[N], conv[N], adj[N,N]),
           participating in training together with train_idx in each epoch (for proxy/large joint data).
    extra_w: loss weight for additional samples.
    """
    rng = np.random.RandomState(seed)
    opt = Adam(model.params(), lr=lr, wd=wd)
    best_loss = float("inf")
    best_state = None
    bad = 0
    for ep in range(epochs):
        total = 0.0
        cnt = 0
        for i in train_idx:
            feats, iters, conv, adj = dataset[i]
            N = feats.shape[0]
            if N < 2:
                continue
            x = ag.Tensor(feats)
            rank, clogit = model.forward(x, adj, training=True)
            rl = ranking_loss(rank, iters, margin)
            cl = ag.bce_with_logits(clogit, conv.astype(float))
            loss = ag.add(rl, ag.mul(cl, ag.Tensor(conv_w)))
            if lam > 0:
                ml = mono_loss(rank, feats[:, 5])
                loss = ag.add(loss, ag.mul(ml, ag.Tensor(lam)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.data
            cnt += 1
        if extra:
            for feats, iters, conv, adj in extra:
                if feats.shape[0] < 2:
                    continue
                x = ag.Tensor(feats)
                rank, clogit = model.forward(x, adj, training=True)
                rl = ranking_loss(rank, iters, margin)
                cl = ag.bce_with_logits(clogit, conv.astype(float))
                loss = ag.add(rl, ag.mul(cl, ag.Tensor(conv_w * extra_w)))
                if lam > 0:
                    ml = mono_loss(rank, feats[:, 5])
                    loss = ag.add(loss, ag.mul(ml, ag.Tensor(lam)))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.data
                cnt += 1
        if cnt == 0:
            continue
        avg = total / cnt
        if avg < best_loss - 1e-4:
            best_loss = avg
            best_state = model.state_dict()
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model
