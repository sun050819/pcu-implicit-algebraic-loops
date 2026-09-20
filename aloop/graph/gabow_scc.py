"""Gabow's strongly connected components algorithm (single-pass linear, ported from the earlier project's detection layer).

The Gabow algorithm uses two stacks (S: DFS path stack, P: SCC root candidate stack) to replace Tarjan's low-link,
allowing all strongly connected components to be found in linear time without requiring a low array.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def gabow_scc(n: int, adj: np.ndarray) -> Tuple[List[int], int]:
    """Gabow SCC.

    Args:
        n   : number of nodes
        adj : nxn adjacency matrix (nonzero entries are treated as directed edges)

    Returns:
        (comp, n_comp): comp[i] = ID of the SCC that node i belongs to (0..n_comp-1)
    """
    adj_lists = [list(np.nonzero(adj[i])[0]) for i in range(n)]
    pre = [0] * n          # preorder number (discovery timestamp), 0 = unvisited
    comp = [-1] * n
    S: List[int] = []      # DFS path stack
    P: List[int] = []      # SCC root candidate stack
    counter = 0
    cid = 0

    for start in range(n):
        if comp[start] != -1:
            continue
        stack = [(start, 0)]          # (node, next neighbor index)
        while stack:
            v, ei = stack[-1]
            if ei == 0 and pre[v] == 0:
                counter += 1
                pre[v] = counter
                S.append(v)
                P.append(v)
            adjv = adj_lists[v]
            advanced = False
            for j in range(ei, len(adjv)):
                w = adjv[j]
                if pre[w] == 0:
                    stack[-1] = (v, j + 1)
                    stack.append((w, 0))
                    advanced = True
                    break
                elif comp[w] == -1:   # back edge to an unassigned SCC: maintain P
                    while pre[P[-1]] > pre[w]:
                        P.pop()
            if advanced:
                continue
            # ---- finish v ----
            stack.pop()
            if P and P[-1] == v:      # v is the root of the current SCC
                P.pop()
                while True:
                    x = S.pop()
                    comp[x] = cid
                    if x == v:
                        break
                cid += 1
    return comp, cid
