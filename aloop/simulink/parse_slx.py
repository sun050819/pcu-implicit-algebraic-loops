# -*- coding: utf-8 -*-
"""Simulink .slx model parsing: extract block dependency graph (adjacency matrix) and node output functions.

.slx is essentially a ZIP package; blocks and lines are defined in simulink/systems/system_root.xml:
  - Block: <Block BlockType="Gain" Name="Gain" SID="10"><P Name="Gain">0.5</P></Block>
  - Line:  <Line><P Name="Src">13#out:1</P><P Name="Dst">9#in:1</P></Line>
  - Branch: a line may contain <Branch> child elements (fan-out), parsed recursively.

Direction convention (consistent with aloop graph): adj[i, j]=1 indicates signal flows from block i's output to block j's input
(i is a predecessor of j, and pred_vals of node_fn(v, pred_vals) contains the input source values of j).
"""
from __future__ import annotations

import os
import re
import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np


class SlxBlock:
    """A single Simulink block."""

    def __init__(self, sid: int, name: str, block_type: str, params: Dict[str, str]):
        self.sid = sid
        self.name = name
        self.block_type = block_type
        self.params = params

    def __repr__(self):
        return f"SlxBlock(sid={self.sid}, name={self.name}, type={self.block_type})"


class SlxModel:
    """Parsed Simulink model structure."""

    def __init__(self, name: str, blocks: List[SlxBlock], edges: List[Tuple[int, int]],
                 port_edges: Optional[List[Tuple[int, int, int]]] = None):
        self.name = name
        self.blocks = blocks
        self.edges = edges  # (src_sid, dst_sid)
        # (src_sid, dst_sid, dst_port): includes target input port number, used to restore port order for blocks like Sum
        self.port_edges = port_edges if port_edges is not None else \
            [(s, d, 1) for s, d in edges]
        self.sid_to_idx = {b.sid: k for k, b in enumerate(blocks)}
        self.n = len(blocks)
        self._adj = None

    def adjacency(self) -> np.ndarray:
        """Block dependency adjacency matrix adj[i,j]=1 indicates i->j (i's output flows to j's input)."""
        if self._adj is None:
            self._adj = np.zeros((self.n, self.n), dtype=int)
            for s, d in self.edges:
                if s in self.sid_to_idx and d in self.sid_to_idx:
                    self._adj[self.sid_to_idx[s], self.sid_to_idx[d]] = 1
        return self._adj.copy()

    def input_ports(self, sid: int) -> List[int]:
        """Sequence of input port numbers for block sid (sorted ascending by port)."""
        return sorted(p for s, d, p in self.port_edges if d == sid)

    def describe(self) -> str:
        lines = [f"model={self.name} n={self.n} edges={len(self.edges)}"]
        for b in self.blocks:
            lines.append(f"  [{b.sid}] {b.name} ({b.block_type}) {b.params}")
        for s, d in self.edges:
            sn = self.sid_to_idx.get(s, -1)
            dn = self.sid_to_idx.get(d, -1)
            lines.append(f"  {s}->{d}: {self.blocks[sn].name if sn >= 0 else '?'} -> "
                         f"{self.blocks[dn].name if dn >= 0 else '?'}")
        return "\n".join(lines)


def _extract_block_params(elem: ET.Element) -> Dict[str, str]:
    """Extract block <P Name=...> parameters (excluding graphical parameters like Position/ZOrder)."""
    params: Dict[str, str] = {}
    for p in elem.findall("P"):
        nm = p.get("Name", "")
        if nm in ("Position", "ZOrder", "GraphicalSettings", "WindowPosition",
                  "MultipleDisplayCache", "ActiveDisplayYMinimum",
                  "ActiveDisplayYMaximum", "WasSavedAsWebScope",
                  "ScopeFrameLocation", "Floating", "DataLoggingVariableName",
                  "DataLoggingSaveFormat"):
            continue
        params[nm] = (p.text or "").strip()
    return params


def _walk_lines(elem: ET.Element, out: List[Tuple[str, Optional[str]]],
                inherited_src: Optional[str] = None):
    """Recursively extract (Src, Dst) pairs from Line and its Branch.

    Branch elements do not contain their own Src; they inherit the parent Line's Src;
    therefore inherited_src must be passed down. v2.2.1 fix: previously Branch edges were lost.
    """
    src_p = elem.find("P[@Name='Src']")
    dst_p = elem.find("P[@Name='Dst']")
    src = src_p.text.strip() if src_p is not None and src_p.text else inherited_src
    dst = dst_p.text.strip() if dst_p is not None and dst_p.text else None
    if src and dst:
        out.append((src, dst))
    for branch in elem.findall("Branch"):
        _walk_lines(branch, out, inherited_src=src)


def _parse_port(ref: str) -> Tuple[int, str, int]:
    """'13#out:1' -> (13, 'out', 1)."""
    m = re.match(r"(\d+)#(in|out):(\d+)", ref.strip())
    if not m:
        return (-1, "", 0)
    return int(m.group(1)), m.group(2), int(m.group(3))


def parse_slx(path: str) -> SlxModel:
    """Parse .slx file into SlxModel."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as z:
        root_xml = None
        for name in z.namelist():
            if name.endswith("systems/system_root.xml"):
                root_xml = z.read(name)
                break
        if root_xml is None:
            # Compatible with older format simulink/systems/subsystem.xml
            for name in z.namelist():
                if name.endswith("systems/subsystem.xml"):
                    root_xml = z.read(name)
                    break
        if root_xml is None:
            raise ValueError(f"no system xml in {path}")

    root = ET.fromstring(root_xml)
    model_name = os.path.splitext(os.path.basename(path))[0]

    blocks: List[SlxBlock] = []
    for b in root.findall("Block"):
        sid = int(b.get("SID", "0"))
        btype = b.get("BlockType", "")
        name = b.get("Name", f"B{sid}")
        params = _extract_block_params(b)
        # Port count (some blocks omit PortCounts)
        pc = b.find("PortCounts")
        if pc is not None:
            for k in ("in", "out"):
                v = pc.get(k)
                if v:
                    params[f"_n_in" if k == "in" else "_n_out"] = v
        blocks.append(SlxBlock(sid, name, btype, params))

    edges: List[Tuple[int, int]] = []
    port_edges: List[Tuple[int, int, int]] = []
    for line in root.findall("Line"):
        pairs: List[Tuple[str, Optional[str]]] = []
        _walk_lines(line, pairs)
        for src_ref, dst_ref in pairs:
            s_sid, _, _ = _parse_port(src_ref)
            d_sid, _, d_port = _parse_port(dst_ref)
            if s_sid >= 0 and d_sid >= 0:
                edges.append((s_sid, d_sid))
                port_edges.append((s_sid, d_sid, d_port))
    return SlxModel(model_name, blocks, edges, port_edges)


# ==========================================================================
# Node output function generation (node_fn contract: fn(v_idx, pred_vals) -> output value)
# ==========================================================================
def _parse_sum_signs(inputs_str: str, n_in: int) -> List[float]:
    """Sum block Inputs string (e.g. '+-', '++--', '1 2 -3') -> sign for each input port."""
    s = inputs_str.strip()
    if not s:
        return [1.0] * n_in
    # Numeric form (e.g. '1 2')
    if all(ch in "0123456789 .-" for ch in s) and " " in s:
        parts = [p for p in re.split(r"\s+", s) if p]
        signs = []
        for p in parts:
            if "-" in p:
                signs.extend([-1.0] * max(1, int(p.strip("-"))))
            else:
                signs.extend([1.0] * max(1, int(p)))
        return signs[:n_in] if n_in else signs
    # Symbolic form (e.g. '+-')
    signs = [1.0 if ch == "+" else -1.0 for ch in s if ch in "+-"]
    return signs[:n_in] if n_in else signs


def build_node_fn(model: SlxModel, external_inputs: Optional[Dict[int, float]] = None):
    """Construct node_fn(v_idx, pred_vals) -> float for the model.

    Supported block types:
      Step/Constant/SignalGenerator/Inport : constant/external input
      Sum : sum(sign_i * x_i) (signs from Inputs parameter)
      Gain : k * sum(x_i)
      Fcn/Interpreted MATLAB Function : parse expression (sin/cos/tanh/exp/^)
      Trigonometric Function : sin/cos/tan/... etc.
      Math Function : exp/log/sqrt/pow/...
      Product/Divide : multiply/divide
      Abs/Unary Minus : |x| / -x
      Memory/Unit Delay : previous step value (treated as identity pass-through when breaking algebraic loops, engineering approximation)
      Outport/Terminator/Scope/ToWorkspace/Display : identity pass-through
    """
    n = model.n
    idx = model.sid_to_idx

    # Predecessor mapping by port order (according to Simulink input port numbers), used for blocks sensitive to signs such as Sum
    # port_pred[v_idx] = {port number: predecessor node index}
    port_pred: Dict[int, Dict[int, int]] = {}
    for src_sid, dst_sid, dst_port in model.port_edges:
        if dst_sid not in idx:
            continue
        dv = idx[dst_sid]
        if dv not in port_pred:
            port_pred[dv] = {}
        # A port may have multiple predecessors (rare); take the last one
        for s2, d2, p2 in model.port_edges:
            if d2 == dst_sid and p2 == dst_port:
                if s2 in idx:
                    port_pred[dv][dst_port] = idx[s2]
    # Sign sequence for each block (Sum's Inputs correspond in order of ports 1..k)
    sum_signs: Dict[int, List[float]] = {}
    for b in model.blocks:
        if b.block_type == "Sum":
            ports = sorted(port_pred.get(idx[b.sid], {}).keys())
            n_in = len(ports) or int(b.params.get("_n_in", "2") or 2)
            sum_signs[idx[b.sid]] = _parse_sum_signs(b.params.get("Inputs", "+"), n_in)

    def get_block_value(b: SlxBlock) -> float:
        """Constant output of the block (source blocks)."""
        if b.block_type in ("Step",):
            # Step: when t>=StepTime output FinalValue (default 1); take 1 for algebraic steady state
            return 1.0
        if b.block_type in ("Constant", "SignalGenerator"):
            try:
                return float(b.params.get("Value", "1"))
            except Exception:
                return 1.0
        return None

    source_vals = {}
    for b in model.blocks:
        if external_inputs and b.sid in external_inputs:
            source_vals[idx[b.sid]] = float(external_inputs[b.sid])
        else:
            v = get_block_value(b)
            if v is not None:
                source_vals[idx[b.sid]] = v

    def fn(v_idx: int, pred_vals: Dict[int, float]) -> float:
        if v_idx in source_vals:
            return source_vals[v_idx]
        b = model.blocks[v_idx]
        bt = b.block_type
        vals = list(pred_vals.values())
        s = sum(vals) if vals else 0.0
        try:
            if bt == "Sum":
                signs = sum_signs.get(v_idx)
                if signs is not None:
                    # Match predecessors by port number (sign order = Simulink input port order)
                    pp = port_pred.get(v_idx, {})
                    out = 0.0
                    for port in sorted(pp.keys()):
                        pv = pred_vals.get(pp[port])
                        if pv is None:
                            continue
                        sg = signs[port - 1] if port - 1 < len(signs) else 1.0
                        out += sg * pv
                    return out
                # Fallback when no port information: in node index order
                signs2 = _parse_sum_signs(b.params.get("Inputs", "+"), len(vals))
                if len(signs2) < len(vals):
                    signs2 = signs2 + [1.0] * (len(vals) - len(signs2))
                return sum(sg * v for sg, v in zip(signs2, vals))
            if bt == "Gain":
                k = float(b.params.get("Gain", "1"))
                return k * s
            if bt in ("Saturate", "Saturation"):
                try:
                    upper = float(b.params.get("UpperLimit", b.params.get("UpperSat", "inf")))
                    lower = float(b.params.get("LowerLimit", b.params.get("LowerSat", "-inf")))
                except Exception:
                    upper, lower = float("inf"), float("-inf")
                return min(max(s, lower), upper)
            if bt == "DeadZone":
                try:
                    dz_start = float(b.params.get("StartOfDeadZone", "-0.5"))
                    dz_end = float(b.params.get("EndOfDeadZone", "0.5"))
                except Exception:
                    dz_start, dz_end = -0.5, 0.5
                if s < dz_start:
                    return s - dz_start
                if s > dz_end:
                    return s - dz_end
                return 0.0
            if bt == "Relay":
                try:
                    on_pt = float(b.params.get("SwitchOnPoint", "0"))
                    off_pt = float(b.params.get("SwitchOffPoint", "0"))
                    on_out = float(b.params.get("OutputWhenOn", "1"))
                    off_out = float(b.params.get("OutputWhenOff", "0"))
                    init_out = float(b.params.get("InitialOutput", "0"))
                except Exception:
                    on_pt, off_pt, on_out, off_out, init_out = 0, 0, 1, 0, 0
                # Algebraic steady-state approximation: input > turn-on point -> turn-on output; < turn-off point -> turn-off output;
                # within hysteresis band use initial output (conservative approximation when no history)
                if s > on_pt:
                    return on_out
                if s < off_pt:
                    return off_out
                return init_out
            if bt == "Product":
                out = 1.0
                for v in vals:
                    out *= v
                return out
            if bt == "Divide":
                out = vals[0] if vals else 0.0
                for v in vals[1:]:
                    if abs(v) < 1e-12:
                        return 0.0
                    out /= v
                return out
            if bt in ("Fcn", "Interpreted MATLAB Function"):
                expr = b.params.get("Expr") or b.params.get("MATLABFunction") or "u"
                return _eval_fcn_expr(expr, vals[0] if vals else 0.0)
            if bt == "Trigonometric Function":
                op = b.params.get("Operator", "sin")
                u = vals[0] if vals else 0.0
                return _eval_trig(op, u)
            if bt == "Math Function":
                op = b.params.get("Operator", "exp")
                u = vals[0] if vals else 0.0
                return _eval_math(op, u)
            if bt == "Abs":
                return abs(s)
            if bt == "Unary Minus":
                return -s
            if bt in ("Memory", "Unit Delay", "Delay"):
                return s  # after breaking algebraic loop, engineering approximation as identity pass-through
            if bt in ("Outport", "Terminator", "Scope", "ToWorkspace", "Display",
                      "Ground", "Demux", "Mux"):
                return s if vals else 0.0
            if bt in ("Inport",):
                return source_vals.get(v_idx, 0.0)
        except Exception:
            pass
        # Unknown block: fallback to identity pass-through
        return s if vals else 0.0

    return fn


def _eval_fcn_expr(expr: str, u: float) -> float:
    """Evaluate Fcn block expression (u is input). Only common math functions supported."""
    e = expr.replace("^", "**")
    e = e.replace("abs", "np.abs").replace("sqrt", "np.sqrt")
    e = e.replace("sin", "np.sin").replace("cos", "np.cos").replace("tan", "np.tan")
    e = e.replace("tanh", "np.tanh").replace("sinh", "np.sinh").replace("cosh", "np.cosh")
    e = e.replace("exp", "np.exp").replace("log10", "np.log10").replace("log", "np.log")
    e = e.replace("sign", "np.sign")
    try:
        return float(eval(e, {"np": np, "u": float(u)}))
    except Exception:
        return u


def _eval_trig(op: str, u: float) -> float:
    u = float(u)
    m = {
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
        "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
        "asinh": np.arcsinh, "acosh": np.arccosh, "atanh": np.arctanh,
        "sinc": (lambda x: np.sinc(x / np.pi) if np.isfinite(x) else 0.0),
    }
    try:
        return float(m.get(op.lower(), np.sin)(u))
    except Exception:
        return np.sin(u)


def _eval_math(op: str, u: float) -> float:
    u = float(u)
    m = {
        "exp": np.exp, "log": np.log, "log10": np.log10, "log2": np.log2,
        "sqrt": np.sqrt, "magnitude^2": (lambda x: x * x), "square": (lambda x: x * x),
        "pow": (lambda x: x * x), "mod": (lambda x: abs(x)), "rem": (lambda x: abs(x)),
        "hypot": np.abs, "sign": np.sign, "reciprocal": (lambda x: 1.0 / x if abs(x) > 1e-12 else 0.0),
    }
    try:
        return float(m.get(op.lower(), np.exp)(u))
    except Exception:
        return np.exp(u)


# ==========================================================================
# Analytic automatic differentiation (AD) version: single DAG propagation computes values and gradients simultaneously
# v2.3.2 - replaces numerical differentiation (d+1 DAG evaluations), O(n*d) instead of O(d*n*d)
# ==========================================================================
def build_multibreak_residual_ad(model, breaks, eps: float = 1e-6,
                                 use_sparse: Optional[bool] = None):
    """Multi-breakpoint residual construction for the analytic forward automatic differentiation version.

    Fully interface-compatible with fvs.build_multibreak_residual (returns r, J, func, grad, hess),
    but replaces numerical differentiation with analytic AD: a single DAG topological-order propagation computes values of all nodes and gradients w.r.t. breakpoints simultaneously,
    reducing a single J(x) from O(d * n) to O(n * d) (with a small constant factor d times smaller).

    Supported blocks and their analytic gradients:
      Sum (by port signs), Gain, Product, Divide, Trigonometric (sin/cos/tan/...),
      Math Function (exp/log/sqrt/...), Abs, Unary Minus, Saturate, DeadZone,
      Relay (gradient=0), Memory/Delay (identity), Outport/Terminator/... (identity).
      Fcn / unknown blocks: degraded to numerical differentiation w.r.t. that block's inputs (only that block, not the whole graph).

    Args:
        model: SlxModel (already parsed)
        breaks: list of breakpoint SIDs (should be FVS)
        eps: step size for degraded numerical differentiation of Fcn/unknown blocks
        use_sparse: retained for parameter compatibility (v2.3.3 measured that sparse hess is actually slower for cascaded models,
            so always dense by default; this parameter is only for explicit callers).

    Returns:
        (r, J, func, grad, hess) - same as fvs.build_multibreak_residual
    """
    from collections import deque
    n = model.n
    adj = model.adjacency()
    idx = model.sid_to_idx
    breaks = list(breaks)
    d = len(breaks)
    bk_set = set(breaks)

    # ---- Port predecessor mapping + Sum signs (same as build_node_fn) ----
    port_pred: Dict[int, Dict[int, int]] = {}
    for src_sid, dst_sid, dst_port in model.port_edges:
        if dst_sid not in idx:
            continue
        dv = idx[dst_sid]
        if dv not in port_pred:
            port_pred[dv] = {}
        for s2, d2, p2 in model.port_edges:
            if d2 == dst_sid and p2 == dst_port and s2 in idx:
                port_pred[dv][dst_port] = idx[s2]
    sum_signs: Dict[int, List[float]] = {}
    for b in model.blocks:
        if b.block_type == "Sum":
            ports = sorted(port_pred.get(idx[b.sid], {}).keys())
            n_in = len(ports) or int(b.params.get("_n_in", "2") or 2)
            sum_signs[idx[b.sid]] = _parse_sum_signs(b.params.get("Inputs", "+"), n_in)

    # ---- Source block constants ----
    def _get_source_val(b: SlxBlock):
        if b.block_type == "Step":
            return 1.0
        if b.block_type in ("Constant", "SignalGenerator"):
            try:
                return float(b.params.get("Value", "1"))
            except Exception:
                return 1.0
        return None

    source_vals = {}
    for b in model.blocks:
        v = _get_source_val(b)
        if v is not None:
            source_vals[idx[b.sid]] = v

    # ---- Precompute adjacency list (v2.3.4 optimization: eliminate O(n^2) full-graph scan) ----
    # Build the table once using numpy row/column nonzero indices; all subsequent traversals are O(degree)
    adj_list = [np.nonzero(adj[i, :])[0].tolist() for i in range(n)]
    pred_list = [np.nonzero(adj[:, i])[0].tolist() for i in range(n)]

    # ---- Topological order (DAG after cutting breakpoints) ----
    pred = [[] for _ in range(n)]
    for i in range(n):
        if i in bk_set:
            continue
        pred[i] = [j for j in pred_list[i] if j not in bk_set]
    indeg = [len(pred[v]) for v in range(n)]
    q = deque([v for v in range(n) if v not in bk_set and indeg[v] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for w in adj_list[u]:
            if w in bk_set:
                continue
            if indeg[w] > 0:
                indeg[w] -= 1
                if indeg[w] == 0:
                    q.append(w)

    # Input predecessors of breakpoints (including other breakpoints, whose values are already fixed to x)
    break_preds = {v: pred_list[v] for v in breaks}

    def _block_vg(b: SlxBlock, v_idx: int, pv: Dict[int, float],
                   pg: Dict[int, np.ndarray]) -> Tuple[float, np.ndarray]:
        """Compute the value and gradient vector (length d) of a single block."""
        if v_idx in source_vals:
            return source_vals[v_idx], np.zeros(d)
        bt = b.block_type
        vals = list(pv.values())
        s = float(sum(vals)) if vals else 0.0
        # Gradient "sum": the sum of all input gradient vectors
        g_sum = np.zeros(d)
        for gv in pg.values():
            g_sum += gv

        try:
            # ---- Sum (by port sign) ----
            if bt == "Sum":
                signs = sum_signs.get(v_idx)
                if signs is not None:
                    pp = port_pred.get(v_idx, {})
                    out = 0.0
                    gout = np.zeros(d)
                    for port in sorted(pp.keys()):
                        pi = pp[port]
                        if pi not in pv:
                            continue
                        sg = signs[port - 1] if port - 1 < len(signs) else 1.0
                        out += sg * pv[pi]
                        gout += sg * pg.get(pi, np.zeros(d))
                    return out, gout
                signs2 = _parse_sum_signs(b.params.get("Inputs", "+"), len(vals))
                if len(signs2) < len(vals):
                    signs2 = signs2 + [1.0] * (len(vals) - len(signs2))
                out = sum(sg * v for sg, v in zip(signs2, vals))
                gout = np.zeros(d)
                for sg, gv in zip(signs2, pg.values()):
                    gout += sg * gv
                return out, gout

            # ---- Gain ----
            if bt == "Gain":
                k = float(b.params.get("Gain", "1"))
                return k * s, k * g_sum

            # ---- Product ----
            if bt == "Product":
                if not vals:
                    return 1.0, np.zeros(d)
                out = 1.0
                for v in vals:
                    out *= v
                gout = np.zeros(d)
                keys = list(pv.keys())
                for i, ki in enumerate(keys):
                    # Product of all inputs except the i-th
                    prod_except = 1.0
                    for j, kj in enumerate(keys):
                        if j != i:
                            prod_except *= pv[kj]
                    gout += prod_except * pg.get(ki, np.zeros(d))
                return out, gout

            # ---- Divide ----
            if bt == "Divide":
                if not vals:
                    return 0.0, np.zeros(d)
                out = vals[0]
                for v in vals[1:]:
                    if abs(v) < 1e-12:
                        return 0.0, np.zeros(d)
                    out /= v
                # Gradient: d(u0 / (u1*...*uk)) = du0/denom - u0 * sum(dui/(ui*denom))
                keys = list(pv.keys())
                denom = 1.0
                for kj in keys[1:]:
                    denom *= pv[kj]
                gout = pg.get(keys[0], np.zeros(d)) / denom if abs(denom) > 1e-12 else np.zeros(d)
                for ki in keys[1:]:
                    if abs(pv[ki]) > 1e-12:
                        gout -= out * pg.get(ki, np.zeros(d)) / pv[ki]
                return out, gout

            # ---- Trigonometric Function ----
            if bt == "Trigonometric Function":
                op = b.params.get("Operator", "sin").lower()
                u = vals[0] if vals else 0.0
                gu = pg.get(list(pg.keys())[0], np.zeros(d)) if pg else np.zeros(d)
                val = _eval_trig(op, u)
                # Derivative
                if op == "sin":
                    dval = np.cos(u)
                elif op == "cos":
                    dval = -np.sin(u)
                elif op == "tan":
                    dval = 1.0 / (np.cos(u) ** 2 + 1e-12)
                elif op == "tanh":
                    dval = 1.0 - np.tanh(u) ** 2
                elif op == "sinh":
                    dval = np.cosh(u)
                elif op == "cosh":
                    dval = np.sinh(u)
                elif op == "asin":
                    dval = 1.0 / np.sqrt(max(1 - u * u, 1e-12))
                elif op == "acos":
                    dval = -1.0 / np.sqrt(max(1 - u * u, 1e-12))
                elif op == "atan":
                    dval = 1.0 / (1 + u * u)
                else:
                    # Unknown trigonometric function: numerical differentiation
                    dval = (_eval_trig(op, u + eps) - _eval_trig(op, u - eps)) / (2 * eps)
                return val, dval * gu

            # ---- Math Function ----
            if bt == "Math Function":
                op = b.params.get("Operator", "exp").lower()
                u = vals[0] if vals else 0.0
                gu = pg.get(list(pg.keys())[0], np.zeros(d)) if pg else np.zeros(d)
                val = _eval_math(op, u)
                if op == "exp":
                    dval = np.exp(u)
                elif op in ("log", "log10", "log2"):
                    base = {"log": 1.0, "log10": np.log(10), "log2": np.log(2)}[op]
                    dval = 1.0 / (u * base) if abs(u) > 1e-12 else 0.0
                elif op == "sqrt":
                    dval = 1.0 / (2 * np.sqrt(max(u, 1e-12))) if u > 0 else 0.0
                elif op in ("magnitude^2", "square", "pow"):
                    dval = 2 * u
                elif op in ("mod", "rem", "hypot"):
                    dval = np.sign(u)
                elif op == "sign":
                    dval = 0.0
                elif op == "reciprocal":
                    dval = -1.0 / (u * u) if abs(u) > 1e-12 else 0.0
                else:
                    dval = (_eval_math(op, u + eps) - _eval_math(op, u - eps)) / (2 * eps)
                return val, dval * gu

            # ---- Abs ----
            if bt == "Abs":
                return abs(s), np.sign(s) * g_sum

            # ---- Unary Minus ----
            if bt == "Unary Minus":
                return -s, -g_sum

            # ---- Saturate / Saturation ----
            if bt in ("Saturate", "Saturation"):
                try:
                    upper = float(b.params.get("UpperLimit", b.params.get("UpperSat", "inf")))
                    lower = float(b.params.get("LowerLimit", b.params.get("LowerSat", "-inf")))
                except Exception:
                    upper, lower = float("inf"), float("-inf")
                if s > upper:
                    return upper, np.zeros(d)
                if s < lower:
                    return lower, np.zeros(d)
                return s, g_sum

            # ---- DeadZone ----
            if bt == "DeadZone":
                try:
                    dz_start = float(b.params.get("StartOfDeadZone", "-0.5"))
                    dz_end = float(b.params.get("EndOfDeadZone", "0.5"))
                except Exception:
                    dz_start, dz_end = -0.5, 0.5
                if s < dz_start:
                    return s - dz_start, g_sum
                if s > dz_end:
                    return s - dz_end, g_sum
                return 0.0, np.zeros(d)

            # ---- Relay (piecewise constant, gradient=0) ----
            if bt == "Relay":
                try:
                    on_pt = float(b.params.get("SwitchOnPoint", "0"))
                    off_pt = float(b.params.get("SwitchOffPoint", "0"))
                    on_out = float(b.params.get("OutputWhenOn", "1"))
                    off_out = float(b.params.get("OutputWhenOff", "0"))
                    init_out = float(b.params.get("InitialOutput", "0"))
                except Exception:
                    on_pt, off_pt, on_out, off_out, init_out = 0, 0, 1, 0, 0
                if s > on_pt:
                    return on_out, np.zeros(d)
                if s < off_pt:
                    return off_out, np.zeros(d)
                return init_out, np.zeros(d)

            # ---- Memory / Unit Delay / Delay (identity) ----
            if bt in ("Memory", "Unit Delay", "Delay"):
                return s, g_sum

            # ---- Transport Delay (instantaneous algebraic pass-through in algebraic loop, identity) ----
            if bt == "Transport Delay":
                # Instantaneous algebraic delay has no physical meaning in an algebraic loop; pass through as identity (consistent with Delay)
                return s, g_sum

            # ---- LookupTable 1D (linear interpolation) ----
            if bt == "Lookup Table":
                # Parameters: BreakpointsForDimension1 / Table (1D or 2D)
                x_bp = b.params.get("BreakpointsForDimension1")
                table = b.params.get("Table")
                if x_bp and table and vals:
                    import ast
                    try:
                        xbp = np.asarray(ast.literal_eval(x_bp), dtype=float)
                        tbl = np.asarray(ast.literal_eval(table), dtype=float)
                    except Exception:
                        return s, g_sum
                    u = vals[0]
                    gu = pg.get(list(pg.keys())[0], np.zeros(d)) if pg else np.zeros(d)
                    if tbl.ndim == 1 and xbp.ndim == 1:
                        # 1D linear interpolation
                        if len(xbp) >= 2:
                            i = int(np.clip(np.searchsorted(xbp, u) - 1, 0, len(xbp) - 2))
                            x0_, x1_ = xbp[i], xbp[i + 1]
                            t = (u - x0_) / max(x1_ - x0_, 1e-12)
                            t = float(np.clip(t, 0.0, 1.0))
                            val = tbl[i] + t * (tbl[i + 1] - tbl[i])
                            slope = (tbl[i + 1] - tbl[i]) / max(x1_ - x0_, 1e-12)
                            return val, slope * gu
                        return float(tbl[0]) if len(tbl) else 0.0, np.zeros(d)
                    elif tbl.ndim == 2 and len(vals) >= 2:
                        # 2D bilinear interpolation (u1 along dim1, u2 along dim2)
                        y_bp = b.params.get("BreakpointsForDimension2")
                        if y_bp:
                            try:
                                ybp = np.asarray(ast.literal_eval(y_bp), dtype=float)
                            except Exception:
                                ybp = np.arange(tbl.shape[0], dtype=float)
                        else:
                            ybp = np.arange(tbl.shape[0], dtype=float)
                        u1, u2 = vals[0], vals[1]
                        g1 = pg.get(list(pg.keys())[0], np.zeros(d)) if pg else np.zeros(d)
                        g2 = pg.get(list(pg.keys())[1], np.zeros(d)) if len(pg) >= 2 else np.zeros(d)
                        i = int(np.clip(np.searchsorted(xbp, u1) - 1, 0, len(xbp) - 2))
                        j = int(np.clip(np.searchsorted(ybp, u2) - 1, 0, len(ybp) - 2))
                        tx = float(np.clip((u1 - xbp[i]) / max(xbp[i + 1] - xbp[i], 1e-12), 0, 1))
                        ty = float(np.clip((u2 - ybp[j]) / max(ybp[j + 1] - ybp[j], 1e-12), 0, 1))
                        v00, v10 = tbl[j, i], tbl[j, i + 1]
                        v01, v11 = tbl[j + 1, i], tbl[j + 1, i + 1]
                        val = (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 \
                              + (1 - tx) * ty * v01 + tx * ty * v11
                        dval_d1 = (1 - ty) * (v10 - v00) + ty * (v11 - v01)
                        dval_d2 = (1 - tx) * (v01 - v00) + tx * (v11 - v10)
                        return val, dval_d1 * g1 + dval_d2 * g2
                return s, g_sum

            # ---- Quantizer (quantization: y = round(u/Q)*Q, subgradient=1) ----
            if bt == "Quantizer":
                try:
                    q = float(b.params.get("QuantizationInterval", "0.5"))
                except Exception:
                    q = 0.5
                if abs(q) < 1e-12:
                    q = 0.5
                val = np.round(s / q) * q
                # Slope is 1 everywhere except at jump points; jump points (u = (k+0.5)Q) are discontinuous, use subgradient 1
                return val, g_sum

            # ---- Outport / Terminator / ... (identity) ----
            if bt in ("Outport", "Terminator", "Scope", "ToWorkspace", "Display",
                      "Ground", "Demux", "Mux"):
                return (s if vals else 0.0), g_sum

            # ---- Inport ----
            if bt == "Inport":
                return source_vals.get(v_idx, 0.0), np.zeros(d)

            # ---- Fcn / Interpreted MATLAB Function (fallback: numerical differentiation of the block inputs) ----
            if bt in ("Fcn", "Interpreted MATLAB Function"):
                expr = b.params.get("Expr") or b.params.get("MATLABFunction") or "u"
                u = vals[0] if vals else 0.0
                gu = pg.get(list(pg.keys())[0], np.zeros(d)) if pg else np.zeros(d)
                val = _eval_fcn_expr(expr, u)
                dval = (_eval_fcn_expr(expr, u + eps) - _eval_fcn_expr(expr, u - eps)) / (2 * eps)
                return val, dval * gu

        except Exception:
            pass

        # Unknown block: pass through as identity
        return (s if vals else 0.0), g_sum

    # ---- One-time DAG evaluation (values + gradients), with caching ----
    _eval_cache = {"x": None, "values": None, "grads": None}

    def _evaluate(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xa = np.asarray(x, dtype=float).ravel()
        if _eval_cache["x"] is not None and np.array_equal(_eval_cache["x"], xa):
            return _eval_cache["values"], _eval_cache["grads"]
        values = np.zeros(n)
        grads = np.zeros((n, d))
        # Breakpoint initialization: values fixed to x, gradient w.r.t. itself=1
        for k, v in enumerate(breaks):
            values[v] = xa[k]
            grads[v, k] = 1.0
        # Topological-order propagation
        for u in topo:
            preds = pred_list[u]
            pv = {j: values[j] for j in preds}
            pg = {j: grads[j] for j in preds}
            values[u], grads[u] = _block_vg(model.blocks[u], u, pv, pg)
        _eval_cache["x"] = xa.copy()
        _eval_cache["values"] = values
        _eval_cache["grads"] = grads
        return values, grads

    def _phi_from_values(values, grads, with_grad: bool):
        """Compute phi (computed input values) and gradients for all breakpoints from the already evaluated values/grads."""
        rr = np.zeros(d)
        Jm = np.eye(d) if with_grad else None
        for k, v in enumerate(breaks):
            preds = break_preds[v]
            pv = {j: values[j] for j in preds}
            if with_grad:
                pg = {j: grads[j] for j in preds}
                phi, phi_grad = _block_vg(model.blocks[v], v, pv, pg)
                Jm[k, :] -= phi_grad
            else:
                pg_zero = {j: np.zeros(d) for j in preds}
                phi, _ = _block_vg(model.blocks[v], v, pv, pg_zero)
            rr[k] = phi
        return rr, Jm

    # ---- Residual r(x) = x - phi(x) ----
    def r(x):
        xa = np.asarray(x, dtype=float).ravel()
        if xa.shape[0] != d:
            raise ValueError("dim mismatch")
        values, _ = _evaluate(xa)
        phi, _ = _phi_from_values(values, None, with_grad=False)
        return xa - phi

    # ---- Analytical Jacobian J(x) ----
    def J(x):
        xa = np.asarray(x, dtype=float).ravel()
        if xa.shape[0] != d:
            raise ValueError("dim mismatch")
        values, grads = _evaluate(xa)
        _, Jm = _phi_from_values(values, grads, with_grad=True)
        return Jm

    # ---- Compute r and J simultaneously in one evaluation (avoid duplicate _evaluate) ----
    def _rJ(x):
        xa = np.asarray(x, dtype=float).ravel()
        values, grads = _evaluate(xa)
        phi, Jm = _phi_from_values(values, grads, with_grad=True)
        return xa - phi, Jm

    # ---- func / grad / hess ----
    def func(x):
        rr = r(x)
        return float(0.5 * float(np.dot(rr, rr)))

    def grad(x):
        rr, Jv = _rJ(x)
        return (Jv.T @ rr).ravel()

    def hess(x):
        Jv = J(x)
        return Jv.T @ Jv

    return r, J, func, grad, hess
