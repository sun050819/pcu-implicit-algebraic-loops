"""Tarjan strongly connected components algorithm + space-optimized version.

Space optimization ideas:
  - Standard Tarjan uses two int arrays dfn[] and low[] (2x4n bytes).
  - Optimization 1: The back-edge update low[u] = min(low[u], dfn[v]) can be replaced by
    low[u] = min(low[u], low[v]) (since v is on the stack, low[v] <= dfn[v], and low[v]
    already aggregates the minimum dfn reachable in v's subtree, the two are equivalent).
  - Optimization 2: The root test low[u]==dfn[u] is equivalent to "whether low[u] has been modified",
    which can be recorded with a flag status bit, without storing dfn[u].
  - Optimization 3: Use int16 for low (covers <=32767 nodes) instead of int32; use uint8 for flag.
    Memory drops from 8n bytes to 3n bytes.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _adj_lists(n: int, adj: np.ndarray) -> List[List[int]]:
    return [list(np.nonzero(adj[i])[0]) for i in range(n)]


# ==========================================================================
# Standard Tarjan (dual int arrays dfn + low)
# ==========================================================================
def tarjan_scc(n: int, adj: np.ndarray) -> Tuple[List[int], int]:
    adjl = _adj_lists(n, adj)
    dfn = [0] * n
    low = [0] * n
    in_stack = [False] * n
    stack: List[int] = []
    counter = 0
    cid = 0
    comp = [-1] * n

    def strongconnect(v: int) -> None:
        nonlocal counter, cid
        counter += 1
        dfn[v] = low[v] = counter
        stack.append(v)
        in_stack[v] = True
        for w in adjl[v]:
            if dfn[w] == 0:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif in_stack[w]:
                low[v] = min(low[v], dfn[w])
        if low[v] == dfn[v]:
            while True:
                x = stack.pop()
                in_stack[x] = False
                comp[x] = cid
                if x == v:
                    break
            cid += 1

    import sys
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, 10 * n + 100))
    try:
        for v in range(n):
            if dfn[v] == 0:
                strongconnect(v)
    finally:
        sys.setrecursionlimit(old)
    return comp, cid


# ==========================================================================
# Space-optimized Tarjan (int16 low + uint8 flag, eliminating the dfn array)
# ==========================================================================
def tarjan_scc_memopt(n: int, adj: np.ndarray) -> Tuple[List[int], int]:
    """Space-optimized Tarjan.

    State flag[u] (uint8):
        0 = unvisited
        1 = on stack and low not modified (can be an SCC root candidate)
        2 = on stack and low modified (non-root)
        3 = SCC already assigned (completed)
    low[u] (int16, n<=32767 otherwise automatically promoted to int32): minimum reachable dfn.
    """
    adjl = _adj_lists(n, adj)
    dtype = np.int16 if n <= 32767 else np.int32
    low = np.zeros(n, dtype=dtype)          # write dfn on discovery, then take min
    flag = np.zeros(n, dtype=np.uint8)      # 0 unvisited
    stack: List[int] = []
    counter = 0
    cid = 0
    comp = [-1] * n

    for start in range(n):
        if flag[start] != 0:
            continue
        # Iterative DFS: (node, edge_index)
        dfs = [(start, 0)]
        while dfs:
            v, ei = dfs[-1]
            if ei == 0 and flag[v] == 0:
                counter += 1
                low[v] = counter
                flag[v] = 1                # on stack, low unmodified
                stack.append(v)
            advl = adjl[v]
            advanced = False
            for j in range(ei, len(advl)):
                w = advl[j]
                if flag[w] == 0:           # tree edge
                    dfs[-1] = (v, j + 1)
                    dfs.append((w, 0))
                    advanced = True
                    break
                elif flag[w] == 1 or flag[w] == 2:   # back/cross edge to node on stack
                    if low[w] < low[v]:
                        low[v] = low[w]
                        flag[v] = 2        # mark low as modified
            if advanced:
                continue
            # ---- finish v ----
            if flag[v] == 1:               # low unmodified => v is an SCC root
                flag[v] = 3
                while True:
                    x = stack.pop()
                    comp[x] = cid
                    flag[x] = 3
                    if x == v:
                        break
                cid += 1
            else:
                # Non-root: pass low[v] back to parent (parent is one level up in the dfs stack)
                if len(dfs) >= 2:
                    p = dfs[-2][0]
                    if low[v] < low[p]:
                        low[p] = low[v]
                        flag[p] = 2        # parent's low modified
                # v remains on the SCC stack until the root node pops it
            dfs.pop()
    return comp, cid
