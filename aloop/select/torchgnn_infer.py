"""Pure NumPy inference module for the PyTorch Graph Transformer node-selection model.

Training is done in system Python 3.11 (torch), and the weights are exported as .npz (with keys matching the torch state_dict
consistent); this module replicates the same forward pass with NumPy, so the main pipeline (sandbox Python, without torch dependency)
can call it directly, with an interface compatible with GATRankingConv.predict:
    model = TorchGNNInfer(path)
    rank_v, conv_p = model.predict(feats, adj, beta=1.0)

The structure is strictly consistent with the GraphTransformer in experiments/train_torchgnn.py:
    enc: Linear(16->64)+LN -> 4xTransformerBlock(adjacency-constrained multi-head attention+FFN+LN+residual)
       -> LN -> dual-head rank/conv
Export format: npz contains 'enc.0.weight/bias', 'enc.1.weight/bias'(LN),
    'blocks.{i}.q/k/v/o.weight/bias', 'blocks.{i}.ln1/ln2.*',
    'blocks.{i}.ffn.0.*', 'blocks.{i}.ffn.2.*', 'ln.*', 'rank_head.*', 'conv_head.*'
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def _gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU (erf), consistent with PyTorch's default F.gelu(exact)."""
    from math import erf
    f = np.vectorize(lambda t: 0.5 * t * (1.0 + erf(t / np.sqrt(2.0))))
    return f(x)


class TorchGNNInfer:
    """Pure NumPy Graph Transformer inference (structure mirrors experiments/train_torchgnn.py)."""

    def __init__(self, npz_path: str, hid: int = 64, heads: int = 4, layers: int = 4,
                 in_f: int = 16):
        d = np.load(npz_path, allow_pickle=True)
        self.P = {k: d[k] for k in d.files}
        self.hid, self.heads, self.layers, self.in_f = hid, heads, layers, in_f
        self.dh = hid // heads

    # ---- Basic layers ----
    def _linear(self, x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        return x @ w.T + b

    def _block(self, x: np.ndarray, adj: np.ndarray, i: int) -> np.ndarray:
        """x [L,d], adj [L,L] bool (including self-loops) -> h [L,d] (residual + LN + FFN)."""
        L = x.shape[0]
        P = self.P
        # ---- Multi-head attention (adjacency-constrained) ----
        q = self._linear(x, P[f"blocks.{i}.q.weight"], P[f"blocks.{i}.q.bias"])   # [L,d]
        k = self._linear(x, P[f"blocks.{i}.k.weight"], P[f"blocks.{i}.k.bias"])
        v = self._linear(x, P[f"blocks.{i}.v.weight"], P[f"blocks.{i}.v.bias"])
        q = q.reshape(L, self.heads, self.dh).transpose(1, 0, 2)   # [H,L,dh]
        k = k.reshape(L, self.heads, self.dh).transpose(1, 0, 2)
        v = v.reshape(L, self.heads, self.dh).transpose(1, 0, 2)
        scores = q @ k.transpose(0, 2, 1) / np.sqrt(self.dh)        # [H,L,L]
        mask = np.where(adj > 0, 0.0, -1e9)                          # mask non-edges
        scores = scores + mask[None, :, :]
        attn = _softmax(scores, axis=-1)                             # [H,L,L]
        out = attn @ v                                               # [H,L,dh]
        out = out.transpose(1, 0, 2).reshape(L, self.hid)
        out = self._linear(out, P[f"blocks.{i}.o.weight"], P[f"blocks.{i}.o.bias"])
        h = _layer_norm(x + out, P[f"blocks.{i}.ln1.weight"], P[f"blocks.{i}.ln1.bias"])
        # ---- FFN ----
        ff = self._linear(h, P[f"blocks.{i}.ffn.0.weight"], P[f"blocks.{i}.ffn.0.bias"])
        ff = _gelu(ff)
        ff = self._linear(ff, P[f"blocks.{i}.ffn.2.weight"], P[f"blocks.{i}.ffn.2.bias"])
        h = _layer_norm(h + ff, P[f"blocks.{i}.ln2.weight"], P[f"blocks.{i}.ln2.bias"])
        return h

    def encode(self, feats: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """[L,16] -> [L,hid] (excluding the final LN/heads)."""
        P = self.P
        x = self._linear(feats, P["enc.0.weight"], P["enc.0.bias"])
        x = _layer_norm(x, P["enc.1.weight"], P["enc.1.bias"])
        a = (adj > 0) | np.eye(adj.shape[0], dtype=bool)
        for i in range(self.layers):
            x = self._block(x, a, i)
        x = _layer_norm(x, P["ln.weight"], P["ln.bias"])
        return x

    def forward_scores(self, feats: np.ndarray, adj: np.ndarray):
        h = self.encode(feats, adj)                                   # [L,hid]
        rank = self._linear(h, self.P["rank_head.weight"], self.P["rank_head.bias"]).ravel()
        clogit = self._linear(h, self.P["conv_head.weight"], self.P["conv_head.bias"]).ravel()
        return rank, clogit

    def predict(self, feats: np.ndarray, adj: np.ndarray, beta: float = 1.0):
        """Inference: returns (ranking scores, convergence probabilities). Interface compatible with GATRankingConv.predict."""
        feats = np.asarray(feats, dtype=np.float64)
        adj = np.asarray(adj, dtype=np.float64)
        rank, clogit = self.forward_scores(feats, adj)
        conv_p = 1.0 / (1.0 + np.exp(-np.clip(clogit, -50, 50)))
        return rank, conv_p
