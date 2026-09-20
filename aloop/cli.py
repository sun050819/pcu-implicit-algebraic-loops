# -*- coding: utf-8 -*-
"""aloop-system command line interface (P3-12).

Usage example:
    py -3.11 -m aloop.cli solve aloop_boost_converter.slx
    py -3.11 -m aloop.cli solve aloop_nonlinear_loop.slx --solver newton --scoring invar
    py -3.11 -m aloop.cli detect aloop_large_200blk.slx
    py -3.11 -m aloop.cli solve aloop_large_5000blk.slx --output-json out.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from aloop.simulink.parse_slx import parse_slx, build_node_fn
from aloop.graph.detector import detect_loops
from aloop.solve.pipeline import solve_loop_with_fvs


def _resolve_model_path(path):
    """Parse the model path: if the direct path does not exist, automatically try the project's built-in models/ directory."""
    if os.path.exists(path):
        return path
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # aloop_system root
    cand = os.path.join(here, "models", path)
    return cand if os.path.exists(cand) else path


def _load_model(path):
    path = _resolve_model_path(path)
    if not os.path.exists(path):
        sys.exit(f"[error] model not found: {path} (searched project models/ directory)")
    mo = parse_slx(path)
    return mo


def cmd_detect(args):
    mo = _load_model(args.model)
    adj = mo.adjacency()
    node_fn = build_node_fn(mo)
    t0 = time.time()
    db = detect_loops(mo.n, adj, node_fn=node_fn)
    dt = (time.time() - t0) * 1000
    names = {mo.sid_to_idx[b.sid]: b.name for b in mo.blocks}
    print(f"model: {args.model}")
    print(f"blocks: {mo.n}  inner loops: {len(db.inner_loops)}  time: {dt:.1f} ms")
    for i, loop in enumerate(db.inner_loops):
        nodes = loop.nodes if hasattr(loop, "nodes") else loop
        path = " -> ".join(names.get(v, str(v)) for v in nodes)
        print(f"  loop {i+1} (len={len(nodes)}): {path}")
    return 0


def cmd_solve(args):
    mo = _load_model(args.model)
    adj = mo.adjacency()
    node_fn = build_node_fn(mo)
    names = {mo.sid_to_idx[b.sid]: b.name for b in mo.blocks}
    ci_map = {v: -float(adj[v, :].sum()) for v in range(mo.n)}

    t0 = time.time()
    db = detect_loops(mo.n, adj, node_fn=node_fn)
    res = solve_loop_with_fvs(
        mo.n, adj, node_fn,
        solver=args.solver,
        scoring=args.scoring,
        ci_map=ci_map,
        max_iter=args.max_iter,
        ad_model=mo,
    )
    dt = (time.time() - t0) * 1000

    result = {
        "model": args.model,
        "n_blocks": mo.n,
        "n_inner_loops": len(db.inner_loops),
        "n_breaks": (res.nit_breakdown or {}).get("breaks",
                     0 if res.x is None else len(np.atleast_1d(res.x))),
        "converged": bool(res.converged),
        "residual": float(res.residual),
        "stage": res.stage,
        "time_ms": round(dt, 1),
        "x": None if res.x is None else np.round(res.x, 6).tolist(),
    }
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[ok] results written to {args.output_json}")

    print(f"model: {args.model}")
    print(f"blocks: {mo.n}  inner loops: {len(db.inner_loops)}  "
          f"break points: {result['n_breaks']}")
    print(f"solver: {args.solver}  scoring: {args.scoring}")
    print(f"converged: {res.converged}  residual: {res.residual:.3e}  "
          f"stage: {res.stage}  time: {dt:.1f} ms")
    if res.x is not None:
        for i, v in enumerate(np.atleast_1d(res.x)):
            print(f"  break[{i}] = {v:.8g}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="aloop",
        description="Implicit algebraic-loop detect-select-solve pipeline (v2.3.17)")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("solve", help="solve algebraic loops of a .slx model end-to-end")
    ps.add_argument("model")
    ps.add_argument("--solver",
                    choices=["hgca", "hast_cma", "hast_n", "adaptive_sa_newton",
                             "trust_region", "vanilla_newton", "lm"],
                    default="hgca",
                    help="solver (default hgca: homotopy-guided CMA-ES, 100% convergence on 36 problems, 4.8x faster; hast_cma: CMA-ES baseline; hast_n: SA fast-path control; others: comparison methods)")
    ps.add_argument("--scoring", choices=["outdeg", "invar"], default="outdeg")
    ps.add_argument("--max-iter", type=int, default=3000)
    ps.add_argument("--output-json")
    ps.set_defaults(fn=cmd_solve)

    pd = sub.add_parser("detect", help="detect algebraic-loop structure only")
    pd.add_argument("model")
    pd.set_defaults(fn=cmd_detect)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
