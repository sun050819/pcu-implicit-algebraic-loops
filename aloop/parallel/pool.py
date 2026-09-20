"""(4) Engineering acceleration layer: thread pool / process pool parallelism.

- `map_parallel`: general-purpose parallel map, returns results in order.
  - use_process=True: process pool (true parallelism, requires fn/items to be picklable; suitable for inner-loop enumeration,
    multi-candidate breakpoint solving, and other CPU-intensive tasks).
  - use_process=False: thread pool (suitable for mixed/IO workloads; under the GIL, limited improvement for pure Python,
    but effective for scipy/BLAS inner layers).
- `PriorityPool`: thread pool wrapper with priority control (corresponds to the paper's "thread priority control").
- `make_node_fn_spec / build_node_fn_from_spec`: picklable representation of standard block-type loops,
  enabling process pool workers to reconstruct node_fn for true parallel solving.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from ..loopeval import make_block_node_fn

__all__ = ["map_parallel", "PriorityPool", "n_cpu", "make_node_fn_spec", "build_node_fn_from_spec"]


def n_cpu() -> int:
    return os.cpu_count() or 1


def map_parallel(fn: Callable, items: Iterable, n_workers: Optional[int] = None,
                 use_process: bool = False, ordered: bool = True) -> List:
    """Parallel map, returns results in order or out of order.

    Args:
        fn         : Single-argument function (accepts a single element from items)
        items      : Input sequence
        n_workers  : Degree of parallelism, None=min(len, cpu)
        use_process: Whether to use a process pool (true parallelism, requires picklable)
        ordered    : True to return results in input order
    """
    items = list(items)
    if len(items) == 0:
        return []
    workers = n_workers or max(1, min(len(items), n_cpu()))
    if workers <= 1:
        return [fn(it) for it in items]
    Executor = ProcessPoolExecutor if use_process else ThreadPoolExecutor
    with Executor(max_workers=workers) as ex:
        if ordered:
            return list(ex.map(fn, items))
        futures = {ex.submit(fn, it): i for i, it in enumerate(items)}
        out = [None] * len(items)
        for fut in as_completed(futures):
            out[futures[fut]] = fut.result()
        return out


class PriorityPool:
    """Thread pool wrapper with priority control (corresponds to the paper's 'thread priority control').

    Usage:
        pool = PriorityPool(max_workers=4)
        f1 = pool.submit(fn, arg, priority=2)   # Smaller value means higher priority, executed first
        ...
        pool.shutdown()
    """

    def __init__(self, max_workers: Optional[int] = None):
        import itertools
        from queue import PriorityQueue
        import threading
        self._pq: PriorityQueue = PriorityQueue()
        self._seq_counter = itertools.count()
        self._workers = [threading.Thread(target=self._run, daemon=True)
                         for _ in range(max_workers or max(1, n_cpu()))]
        self._results = {}
        for w in self._workers:
            w.start()

    def _run(self):
        while True:
            item = self._pq.get()
            if item is None:
                break
            prio, seq, fn, args, kwargs = item
            try:
                res = fn(*args, **kwargs)
                self._results[seq] = (True, res)
            except Exception as e:  # noqa
                self._results[seq] = (False, e)

    def submit(self, fn, *args, priority: int = 0, **kwargs):
        seq = next(self._seq_counter)
        self._pq.put((priority, seq, fn, args, kwargs))
        return seq

    def result(self, seq: int, timeout: Optional[float] = None):
        import time
        t0 = time.time()
        while seq not in self._results:
            if timeout is not None and time.time() - t0 > timeout:
                raise TimeoutError(seq)
            time.sleep(0.001)
        ok, val = self._results.pop(seq)
        if not ok:
            raise val
        return val

    def shutdown(self):
        for _ in self._workers:
            self._pq.put(None)


# ---------------------------------------------------------------------------
# Picklable representation of standard block-type loops (for process pool to reconstruct node_fn)
# ---------------------------------------------------------------------------
def make_node_fn_spec(bt, params) -> Tuple:
    """Pack (block type, parameters) into a picklable spec."""
    return (int(bt), tuple(float(p) for p in params))


def build_node_fn_from_spec(specs) -> Callable:
    """Reconstruct {v: node_fn} from specs (standard block types)."""
    fns = {v: make_block_node_fn(int(bt), params) for v, (bt, params) in enumerate(specs)}

    def node_fn(v, pred_vals):
        return fns[v](v, pred_vals)

    return node_fn
