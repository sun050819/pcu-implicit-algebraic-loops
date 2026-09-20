"""Benchmark/dataset generator: synthetic algebraic loops + industrial scenarios."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..loopeval import make_block_node_fn
from ..graph.gabow_scc import gabow_scc

T_GAIN, T_SUM, T_MATH, T_PRODUCT = 0, 1, 2, 3


def gen_block_params(bt: int) -> np.ndarray:
    k = np.random.uniform(0.35, 0.92)
    b = np.random.uniform(-0.3, 0.3)
    c = np.random.uniform(0.4, 1.6)
    return np.array([k, b, c])


def make_loop_case(n_node: int, scene: str, seed: Optional[int] = None):
    """Generate synthetic algebraic loop test cases: returns (A, bt, params, node_fn)."""
    if seed is not None:
        np.random.seed(seed)
    A = np.zeros((n_node, n_node), dtype=np.int64)
    if scene == "single":
        n_loops = 1
    else:  # overlap / nested
        n_loops = 2
    for k in range(n_loops):
        loop_size = np.random.randint(3, min(10, n_node) + 1)
        ln = np.random.choice(n_node, loop_size, replace=False)
        for i in range(loop_size - 1):
            A[ln[i], ln[i + 1]] = 1
        A[ln[-1], ln[0]] = 1
    n_extra = np.random.randint(int(np.ceil(n_node * 1.5)),
                                int(np.ceil(n_node * 3.5)) + 1)
    for _ in range(n_extra):
        s, d = np.random.randint(n_node), np.random.randint(n_node)
        if s != d:
            A[s, d] = 1
    bt = np.random.randint(0, 4, size=n_node)
    params = np.array([gen_block_params(b) for b in bt])
    fns = {v: make_block_node_fn(int(bt[v]), params[v]) for v in range(n_node)}

    def node_fn(v, pv):
        return fns[v](v, pv)

    return A, bt, params, node_fn


def find_loop_nodes(n: int, adj: np.ndarray) -> List[int]:
    """Take the largest SCC (>1) as the target loop (find_loops protocol)."""
    comp, cid = gabow_scc(n, adj)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(comp[i], []).append(i)
    loops = [g for g in groups.values() if len(g) > 1]
    if not loops:
        return []
    return sorted(max(loops, key=len))


def make_case(n_node: int, scene: str, seed: int) -> Dict:
    """Generate a case dictionary with a picklable spec (for label generation/process pool)."""
    A, bt, params, node_fn = make_loop_case(n_node, scene, seed)
    loop_nodes = find_loop_nodes(n_node, A)
    return {"n": n_node, "adj": A, "bt": bt, "params": params,
            "node_fn": node_fn, "loop_nodes": loop_nodes,
            "scene": scene, "seed": seed}


# ==========================================================================
# Industrial scenarios (source: aloop_scenarios.py)
# ==========================================================================
def build_power_flow(nbus: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    Y = np.zeros((nbus, nbus), dtype=complex)
    for i in range(nbus):
        for j in range(i + 1, nbus):
            y = 1.0 / (rng.uniform(0.05, 0.18) + 1j * rng.uniform(0.2, 0.5))
            Y[i, j] = Y[j, i] = -y
    for i in range(nbus):
        Y[i, i] = -np.sum([Y[i, j] for j in range(nbus) if j != i]) \
                  + rng.uniform(3.0, 6.0) + 1j * rng.uniform(6.0, 12.0)
    P = np.zeros(nbus); Q = np.zeros(nbus)
    for i in range(1, nbus):
        P[i] = rng.uniform(0.06, 0.18)
        Q[i] = rng.uniform(0.02, 0.08)
    Vslack = 1.0 + 0j
    nn = nbus

    def node_fn(v, pred_vals):
        if v == 0:
            return Vslack
        V = np.zeros(nbus, dtype=complex)
        V[0] = Vslack
        for b in range(1, nbus):
            V[b] = pred_vals.get(b, 1.0 + 0j)
        Vi = V[v]
        if abs(Vi) < 1e-9:
            return complex(float("nan"), 0.0)
        s = P[v] - 1j * Q[v]
        num = s / np.conj(Vi) - np.sum([Y[v, j] * V[j] for j in range(nbus) if j != v])
        Vcalc = num / Y[v, v]
        return Vcalc if np.isfinite(Vcalc) else complex(float("nan"), 0.0)

    A = np.zeros((nn, nn), dtype=np.int64)
    for b in range(1, nbus):
        for j in range(nbus):
            A[b, j] = 1
    loop_nodes = list(range(1, nn))
    meta = {"domain": "power_flow", "nbus": nbus,
            "desc": "%d-bus distribution-network AC power flow" % nbus}
    return nn, A, node_fn, loop_nodes, meta


def build_ik2dof(L1=1.0, L2=0.8, target=(1.3, 0.5), k1=0.8, k2=0.8,
                 cross=0.4, seed=0):
    xt, yt = target
    nn = 4

    def node_fn(v, pred_vals):
        t1 = pred_vals.get(0, 0.5)
        t2 = pred_vals.get(1, 0.5)
        x = pred_vals.get(2, 0.0)
        y = pred_vals.get(3, 0.0)
        if v == 0:
            return t1 + k1 * (xt - x) + cross * np.tanh(yt - y)
        if v == 1:
            return t2 + k2 * (yt - y) + cross * np.tanh(xt - x)
        if v == 2:
            return L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
        return L1 * np.sin(t1) + L2 * np.sin(t1 + t2)

    A = np.ones((nn, nn), dtype=np.int64)
    np.fill_diagonal(A, 0)
    loop_nodes = list(range(nn))
    meta = {"domain": "ik2dof", "L1": L1, "L2": L2, "target": list(target),
            "desc": "2-DOF manipulator inverse kinematics"}
    return nn, A, node_fn, loop_nodes, meta


def build_idle_speed(wref=800.0, TL=18.0, Kp=0.02, a=40.0, b=0.05, c=0.06,
                     w0=750.0, kw=0.25, seed=0):
    nn = 4

    def node_fn(v, pred_vals):
        w = pred_vals.get(0, wref)
        u = pred_vals.get(1, 0.0)
        Te = pred_vals.get(2, 0.0)
        if v == 1:
            return Kp * (wref - w) + 0.05 * np.tanh(u)
        if v == 2:
            return a * np.tanh(b * u) - c * (w - w0) + 8.0
        if v == 3:
            return w + 0.01 * u
        return w + kw * (Te - TL) + 0.02 * u

    A = np.zeros((nn, nn), dtype=np.int64)
    A[0, 1] = 1
    A[1, 0] = 1
    A[1, 2] = 1
    A[2, 0] = 1
    A[0, 2] = 1
    A[1, 3] = 1
    loop_nodes = list(range(nn))
    meta = {"domain": "idle_speed", "wref": wref, "TL": TL,
            "desc": "engine idle-speed PID feedback algebraic loop"}
    return nn, A, node_fn, loop_nodes, meta


def build_industrial_cases() -> List[Dict]:
    """40-case industrial suite (power grid power flow / robotic arm IK / engine idle)."""
    cases = []
    for nb in [3, 4, 5, 6]:
        for s in range(5):
            cases.append(("power_flow", build_power_flow(nb, seed=s)))
    targets = [(1.3, 0.5), (1.1, 0.8), (0.9, 0.9), (1.4, 0.3), (1.0, 1.0),
               (1.5, 0.2), (0.8, 0.6), (1.2, 0.6), (1.35, 0.45), (0.95, 0.75)]
    for ti, t in enumerate(targets):
        cases.append(("ik2dof", build_ik2dof(target=t, seed=ti)))
    for (wr, tl) in [(750, 18), (800, 18), (900, 18), (850, 18),
                     (800, 12), (800, 24), (900, 12), (900, 24),
                     (700, 15), (950, 20)]:
        cases.append(("idle_speed", build_idle_speed(wref=wr, TL=tl)))
    return [{"domain": dom, "n": nn, "adj": A, "node_fn": fn,
             "loop_nodes": loop, "meta": meta}
            for dom, (nn, A, fn, loop, meta) in cases]
