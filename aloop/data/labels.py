"""Label generation and dataset.

- Synthetic cycle labels: 3 scenarios (single/overlap/nested) x 50, 15 nodes; cycle = maximum SCC;
  for each node on the cycle, use the real solver (fixed-point + Aitken + Newton) to compute cost; divergent breakpoints recorded as 3000 penalty;
  only cases with at least one convergent breakpoint are kept (about 149 cases).
- Surrogate protocol: 90 cases (larger/more complex graphs) used for "surrogate label accelerated training".
- Large-graph transfer: 30-node cases (15->30 zero-shot transfer test).
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..loopeval import solve_breakpoint
from ..select.features import build_features
from .synth import make_loop_case, find_loop_nodes

PENALTY = 3000.0


class LabelDataset:
    """Cycle-level label dataset (isomorphic to synth_labels.npz, indexed by case)."""

    def __init__(self, feats: np.ndarray, iters: np.ndarray, resids: np.ndarray,
                 times: np.ndarray, adj: np.ndarray, conv: np.ndarray,
                 mask: np.ndarray, scenes: np.ndarray, seeds: np.ndarray,
                 opt_idx: np.ndarray, outdeg_idx: np.ndarray,
                 type_idx: np.ndarray, rand_idx: np.ndarray,
                 case_ids: Optional[np.ndarray] = None,
                 n_node_global: Optional[np.ndarray] = None):
        self.feats = feats              # [C, L, 16]
        self.iters = iters              # [C, L]
        self.resids = resids            # [C, L]
        self.times = times              # [C, L]
        self.adj = adj                  # [C, L, L]
        self.conv = conv                # [C, L] int
        self.mask = mask                # [C, L] bool
        self.scenes = scenes
        self.seeds = seeds
        self.opt_idx = opt_idx
        self.outdeg_idx = outdeg_idx
        self.type_idx = type_idx
        self.rand_idx = rand_idx
        self.case_ids = case_ids if case_ids is not None else np.arange(len(self))
        self.n_node_global = n_node_global

    def __len__(self):
        return self.feats.shape[0]

    def __getitem__(self, i):
        m = self.mask[i].sum()
        return (self.feats[i, :m], self.iters[i, :m].astype(float),
                self.conv[i, :m].astype(float), self.adj[i, :m, :m])

    def padded(self):
        return (self.feats, self.iters, self.resids, self.times, self.adj,
                self.conv, self.mask)

    def save(self, path: str):
        np.savez_compressed(
            path, feats=self.feats, iters=self.iters, resids=self.resids,
            times=self.times, adj=self.adj, conv=self.conv, mask=self.mask,
            scenes=self.scenes, seeds=self.seeds, opt_idx=self.opt_idx,
            outdeg_idx=self.outdeg_idx, type_idx=self.type_idx,
            rand_idx=self.rand_idx, case_ids=self.case_ids)

    @classmethod
    def load(cls, path: str) -> "LabelDataset":
        d = np.load(path, allow_pickle=True)
        return cls(d["feats"], d["iters"], d["resids"], d["times"], d["adj"],
                   d["conv"], d["mask"], d["scenes"], d["seeds"], d["opt_idx"],
                   d["outdeg_idx"], d["type_idx"], d["rand_idx"],
                   d.get("case_ids", None))


def _record_case(case_id, scene, k, seed, n_node, A, bt, node_fn,
                 tol=1e-6) -> Optional[Dict]:
    loop_nodes = find_loop_nodes(n_node, A)
    if not loop_nodes:
        return None
    costs = {v: solve_breakpoint(n_node, A, node_fn, v, x0=1.0, tol=tol)
             for v in loop_nodes}
    feats = build_features(n_node, A, bt)
    feat_nodes = feats[loop_nodes]
    conv = np.array([costs[v]["converged"] for v in loop_nodes])
    if conv.sum() == 0:
        return None
    iters = np.array([costs[v]["iter"] if costs[v]["converged"] else PENALTY
                      for v in loop_nodes], dtype=np.float32)
    resids = np.array([costs[v]["residual"] for v in loop_nodes], dtype=np.float32)
    times = np.array([costs[v]["time_ms"] for v in loop_nodes], dtype=np.float32)
    outdeg = feat_nodes[:, 5]
    type_prio = feat_nodes[:, 0] * 4 + feat_nodes[:, 1] * 3 \
        + feat_nodes[:, 2] * 2 + feat_nodes[:, 3] * 1
    conv_iter = np.where(conv, iters, np.inf)
    opt_idx = int(np.argmin(conv_iter))
    outdeg_idx = int(np.argmax(outdeg))
    type_idx = int(np.argmax(type_prio))
    rng = np.random.RandomState(seed)
    rand_idx = int(rng.randint(len(loop_nodes)))
    return {"case_id": case_id, "scene": scene, "k": k, "seed": seed,
            "n_node": n_node, "nodes": loop_nodes, "n_loop": len(loop_nodes),
            "feats": feat_nodes.astype(np.float32), "iters": iters,
            "resids": resids, "times": times, "conv": conv.astype(np.int64),
            "opt_idx": opt_idx, "outdeg_idx": outdeg_idx,
            "type_idx": type_idx, "rand_idx": rand_idx,
            "adj": A[np.ix_(loop_nodes, loop_nodes)].astype(np.float32)}


def _records_to_dataset(records: List[Dict]) -> LabelDataset:
    L = max((r["n_loop"] for r in records), default=15)
    n_rec = len(records)
    feats_p = np.zeros((n_rec, L, 16), dtype=np.float32)
    iters_p = np.zeros((n_rec, L), dtype=np.float32)
    resids_p = np.zeros((n_rec, L), dtype=np.float32)
    times_p = np.zeros((n_rec, L), dtype=np.float32)
    adj_p = np.zeros((n_rec, L, L), dtype=np.float32)
    conv_p = np.zeros((n_rec, L), dtype=np.int64)
    mask = np.zeros((n_rec, L), dtype=bool)
    for i, r in enumerate(records):
        m = r["n_loop"]
        feats_p[i, :m] = r["feats"]
        iters_p[i, :m] = r["iters"]
        resids_p[i, :m] = r["resids"]
        times_p[i, :m] = r["times"]
        adj_p[i, :m, :m] = r["adj"]
        conv_p[i, :m] = r["conv"]
        mask[i, :m] = True
    return LabelDataset(
        feats_p, iters_p, resids_p, times_p, adj_p, conv_p, mask,
        scenes=np.array([r["scene"] for r in records]),
        seeds=np.array([r["seed"] for r in records]),
        opt_idx=np.array([r["opt_idx"] for r in records]),
        outdeg_idx=np.array([r["outdeg_idx"] for r in records]),
        type_idx=np.array([r["type_idx"] for r in records]),
        rand_idx=np.array([r["rand_idx"] for r in records]),
        case_ids=np.array([r["case_id"] for r in records]),
        n_node_global=np.array([r["n_node"] for r in records]))


def _record_case_from_spec(spec):
    """Process pool worker: reconstruct node_fn from a picklable case spec and generate records.

    spec = (case_id, scene, k, seed, n_node, A, bt, params, tol)
    """
    import warnings
    warnings.filterwarnings("ignore")   # divergent breakpoints trigger overflow warnings, which is expected noise
    case_id, scene, k, seed, n_node, A, bt, params, tol = spec
    from ..parallel.pool import build_node_fn_from_spec
    node_fn = build_node_fn_from_spec(list(zip(bt, params)))
    return _record_case(case_id, scene, k, seed, n_node, A, bt, node_fn, tol)


def _generate_parallel(specs, n_workers, verbose, tag):
    """Perform process-pool parallel label generation on a list of case specs (returns records in order)."""
    t0 = time.time()
    if n_workers and n_workers > 1:
        from ..parallel.pool import map_parallel
        results = map_parallel(_record_case_from_spec, specs, n_workers=n_workers,
                               use_process=True, ordered=True)
    else:
        results = [_record_case_from_spec(s) for s in specs]
    records = [r for r in results if r is not None]
    if verbose:
        print("[labels] %s cases=%d (%.1fs)" % (tag, len(records), time.time() - t0))
    return records


def generate_synth_labels(n_node: int = 15, per: int = 50,
                          scenes: Sequence[str] = ("single", "overlap", "nested"),
                          tol: float = 1e-6, verbose: bool = True,
                          n_workers: Optional[int] = None) -> LabelDataset:
    """Reproduce the 149-case synthetic cycle label protocol (supports process-pool parallelism)."""
    specs = []
    case_id = 0
    for scene in scenes:
        si = list(scenes).index(scene)
        for k in range(per):
            seed = si * 10000 + k * 7 + 3
            A, bt, params, _ = make_loop_case(n_node, scene, seed=seed)
            specs.append((case_id, scene, k, seed, n_node, A, bt, params, tol))
            case_id += 1
    records = _generate_parallel(specs, n_workers, verbose, "synth")
    ds = _records_to_dataset(records)
    if verbose:
        for scene in scenes:
            rs = [r for r in records if r["scene"] == scene]
            if rs:
                conv_mean = np.mean([r["conv"].mean() for r in rs])
                print("  %-8s n=%d conv=%.2f" % (scene, len(rs), conv_mean))
    return ds


def generate_proxy_labels(n_node: int = 15, n_cases: int = 90,
                          scenes: Sequence[str] = ("single", "overlap", "nested"),
                          tol: float = 1e-6, verbose: bool = True,
                          n_workers: Optional[int] = None) -> LabelDataset:
    """90-case surrogate protocol labels (more varied edge densities and seeds), used for accelerated training/surrogate label comparison."""
    specs = []
    si_map = {s: i for i, s in enumerate(scenes)}
    for k in range(n_cases):
        scene = scenes[k % len(scenes)]
        seed = 90000 + k * 11 + si_map[scene] * 1000
        A, bt, params, _ = make_loop_case(n_node, scene, seed=seed)
        specs.append((k, scene, k, seed, n_node, A, bt, params, tol))
    records = _generate_parallel(specs, n_workers, verbose, "proxy")
    return _records_to_dataset(records)


def generate_large_labels(n_node: int = 30, per: int = 40,
                          scenes: Sequence[str] = ("single", "overlap", "nested"),
                          tol: float = 1e-6, verbose: bool = True,
                          n_workers: Optional[int] = None) -> LabelDataset:
    """30-node large-cycle labels (ground truth of the "test domain" for 15->30 zero-shot transfer)."""
    specs = []
    case_id = 0
    for scene in scenes:
        si = list(scenes).index(scene)
        for k in range(per):
            seed = 30000 + si * 10000 + k * 7 + 3
            A, bt, params, _ = make_loop_case(n_node, scene, seed=seed)
            specs.append((case_id, scene, k, seed, n_node, A, bt, params, tol))
            case_id += 1
    records = _generate_parallel(specs, n_workers, verbose, "large(30)")
    return _records_to_dataset(records)
