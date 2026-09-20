"""Micro automatic differentiation framework (pure NumPy, for GAT / GCN / SAGE / MLP training).

The delivered baseline requires "no deep learning toolbox dependency", so we implement minimal automatic differentiation ourselves:
    Tensor(value, requires_grad) + operators (matmul/add/mul/leaky_relu/softmax/...) + Adam.

Operators support 2D tensors, broadcasting (add/sub/mul), reshape/transpose/take; verified by unit tests
and numerical gradients to be consistent.

Implementation key points: each operator first constructs an output Tensor (_backward=None); the backward closure references
this output Tensor (rather than intermediate values), so that out.grad is correctly available during backpropagation.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def _sum_to(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """reduce the gradient back to the target shape (handle broadcasting)."""
    grad = np.asarray(grad, dtype=np.float64)   # 0-dim operations may return np.float64 scalars
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    __slots__ = ("data", "grad", "_backward", "_prev", "requires_grad")

    def __init__(self, data, requires_grad: bool = False,
                 _backward=None, _prev: Tuple["Tensor", ...] = ()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad: Optional[np.ndarray] = np.zeros_like(self.data) if requires_grad else None
        self._backward = _backward
        self._prev = _prev
        self.requires_grad = requires_grad

    def backward(self):
        # Start from the root (loss); its gradient is always 1; intermediate/leaf gradients are accumulated by _backward
        self.grad = np.ones_like(self.data)
        topo = []
        visited = set()

        def build(t):
            if id(t) in visited:
                return
            visited.add(id(t))
            for p in t._prev:
                build(p)
            topo.append(t)

        build(self)
        for t in reversed(topo):
            if t._backward is not None:
                t._backward()
        return self

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
def matmul(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data @ b.data, a.requires_grad or b.requires_grad, None, (a, b))

    def backward():
        if a.requires_grad and a.grad is not None:
            a.grad += out.grad @ b.data.T
        if b.requires_grad and b.grad is not None:
            b.grad += a.data.T @ out.grad

    out._backward = backward
    return out


def add(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data + b.data, a.requires_grad or b.requires_grad, None, (a, b))

    def backward():
        if a.requires_grad and a.grad is not None:
            a.grad += _sum_to(out.grad, a.data.shape)
        if b.requires_grad and b.grad is not None:
            b.grad += _sum_to(out.grad, b.data.shape)

    out._backward = backward
    return out


def sub(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data - b.data, a.requires_grad or b.requires_grad, None, (a, b))

    def backward():
        if a.requires_grad and a.grad is not None:
            a.grad += _sum_to(out.grad, a.data.shape)
        if b.requires_grad and b.grad is not None:
            b.grad += -_sum_to(out.grad, b.data.shape)

    out._backward = backward
    return out


def mul(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data * b.data, a.requires_grad or b.requires_grad, None, (a, b))

    def backward():
        if a.requires_grad and a.grad is not None:
            a.grad += _sum_to(out.grad * b.data, a.data.shape)
        if b.requires_grad and b.grad is not None:
            b.grad += _sum_to(out.grad * a.data, b.data.shape)

    out._backward = backward
    return out


def div(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data / b.data, a.requires_grad or b.requires_grad, None, (a, b))

    def backward():
        if a.requires_grad and a.grad is not None:
            a.grad += _sum_to(out.grad / b.data, a.data.shape)
        if b.requires_grad and b.grad is not None:
            b.grad += _sum_to(-out.grad * a.data / (b.data ** 2), b.data.shape)

    out._backward = backward
    return out


def neg(a: Tensor) -> Tensor:
    return mul(a, Tensor(-1.0))


def transpose(a: Tensor, axes: Optional[Sequence[int]] = None) -> Tensor:
    axes_t = tuple(axes) if axes is not None else tuple(range(a.data.ndim - 1, -1, -1))
    out = Tensor(a.data.transpose(axes_t), a.requires_grad, None, (a,))

    def backward():
        if a.requires_grad and a.grad is not None:
            a.grad += out.grad.transpose(np.argsort(axes_t))

    out._backward = backward
    return out


def take(a: Tensor, index: int, axis: int = 0) -> Tensor:
    out = Tensor(np.take(a.data, index, axis=axis), a.requires_grad, None, (a,))

    def backward():
        if a.requires_grad and a.grad is not None:
            g = np.zeros_like(a.data)
            sl = [slice(None)] * a.data.ndim
            sl[axis] = index
            g[tuple(sl)] = out.grad
            a.grad += g

    out._backward = backward
    return out


def reshape(a: Tensor, shape: Tuple[int, ...]) -> Tensor:
    out = Tensor(a.data.reshape(shape), a.requires_grad, None, (a,))

    def backward():
        if a.requires_grad and a.grad is not None:
            a.grad += out.grad.reshape(a.data.shape)

    out._backward = backward
    return out


def expand(a: Tensor, axis: int) -> Tensor:
    """Expand along axis to [N,1] (axis=1) or [1,N] (axis=0), with broadcasting backward correct."""
    out = Tensor(np.expand_dims(a.data, axis), a.requires_grad, None, (a,))

    def backward():
        if a.requires_grad and a.grad is not None:
            a.grad += out.grad.squeeze(axis)

    out._backward = backward
    return out


def leaky_relu(a: Tensor, slope: float = 0.1) -> Tensor:
    out = Tensor(np.where(a.data > 0, a.data, slope * a.data),
                 a.requires_grad, None, (a,))

    def backward():
        if a.requires_grad and a.grad is not None:
            a.grad += out.grad * np.where(a.data > 0, 1.0, slope)

    out._backward = backward
    return out


def relu(a: Tensor) -> Tensor:
    return leaky_relu(a, 0.0)


def sigmoid(a: Tensor) -> Tensor:
    s = 1.0 / (1.0 + np.exp(-np.clip(a.data, -50, 50)))
    out = Tensor(s, a.requires_grad, None, (a,))

    def backward():
        if a.requires_grad and a.grad is not None:
            a.grad += out.grad * s * (1.0 - s)

    out._backward = backward
    return out


def softmax(a: Tensor, axis: int = 1) -> Tensor:
    m = a.data - a.data.max(axis=axis, keepdims=True)
    e = np.exp(m)
    s = e / e.sum(axis=axis, keepdims=True)
    out = Tensor(s, a.requires_grad, None, (a,))

    def backward():
        g = out.grad
        ss = s * (g - (g * s).sum(axis=axis, keepdims=True))
        a.grad += ss

    out._backward = backward
    return out


def cat(tensors: List[Tensor], axis: int = 1) -> Tensor:
    data = np.concatenate([t.data for t in tensors], axis=axis)
    requires = any(t.requires_grad for t in tensors)
    sizes = [t.data.shape[axis] for t in tensors]
    out = Tensor(data, requires, None, tuple(tensors))

    def backward():
        acc = 0
        for t, sz in zip(tensors, sizes):
            sl = [slice(None)] * data.ndim
            sl[axis] = slice(acc, acc + sz)
            if t.requires_grad and t.grad is not None:
                t.grad += out.grad[tuple(sl)]
            acc += sz

    out._backward = backward
    return out


def sum_op(a: Tensor) -> Tensor:
    out = Tensor(a.data.sum(), a.requires_grad, None, (a,))

    def backward():
        a.grad += out.grad * np.ones_like(a.data)

    out._backward = backward
    return out


def mean_op(a: Tensor) -> Tensor:
    n = a.data.size
    out = Tensor(a.data.mean(), a.requires_grad, None, (a,))

    def backward():
        a.grad += out.grad * np.ones_like(a.data) / n

    out._backward = backward
    return out


def bce_with_logits(logits: Tensor, targets: np.ndarray) -> Tensor:
    """binary cross entropy (logits unnormalized), takes mean over elementwise losses."""
    x = logits.data
    loss = np.clip(x, 0, None) - x * targets + np.log1p(np.exp(-np.abs(x)))
    out = Tensor(loss.mean(), logits.requires_grad, None, (logits,))

    def backward():
        d = (sigmoid_np(x) - targets)
        logits.grad += _sum_to(out.grad * d / max(x.size, 1), x.shape)

    out._backward = backward
    return out


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


# ---------------------------------------------------------------------------
# Parameter container & Adam optimizer
# ---------------------------------------------------------------------------
class Parameter:
    def __init__(self, data, name: str = ""):
        self.t = Tensor(data, requires_grad=True)
        self.name = name

    @property
    def data(self):
        return self.t.data


class Adam:
    def __init__(self, params: List[Parameter], lr: float = 2e-3, wd: float = 1e-5,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.params = params
        self.lr, self.wd = lr, wd
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.m = [np.zeros_like(p.t.data) for p in params]
        self.v = [np.zeros_like(p.t.data) for p in params]
        self.t_ = 0

    def zero_grad(self):
        for p in self.params:
            p.t.zero_grad()

    def step(self):
        self.t_ += 1
        for i, p in enumerate(self.params):
            g = p.t.grad if p.t.grad is not None else np.zeros_like(p.t.data)
            if self.wd:
                g = g + self.wd * p.t.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mh = self.m[i] / (1 - self.b1 ** self.t_)
            vh = self.v[i] / (1 - self.b2 ** self.t_)
            p.t.data -= self.lr * mh / (np.sqrt(vh) + self.eps)


def xavier(shape, rng: np.random.RandomState) -> np.ndarray:
    fan_in = shape[-1] if len(shape) > 1 else shape[0]
    fan_out = shape[-2] if len(shape) > 1 else shape[0]
    bound = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-bound, bound, size=shape).astype(np.float64)
