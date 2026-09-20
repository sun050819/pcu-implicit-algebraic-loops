# -*- coding: utf-8 -*-
"""Common utilities for experiment scripts: load datasets, save CSV/JSON, 5-fold split."""
import os
import json
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_THIS)                        # aloop_system root
sys.path.insert(0, ROOT)
RESULTS = os.path.join(_THIS, "results")
os.makedirs(RESULTS, exist_ok=True)
MODEL_DIR = os.path.join(ROOT, "models")             # Simulink model directory


def model_path(name: str):
    """Return the absolute path of the model file (prefer the models/ directory)."""
    p = os.path.join(MODEL_DIR, name)
    return p if os.path.exists(p) else name


def load_labels(name: str):
    from aloop.data.labels import LabelDataset
    return LabelDataset.load(os.path.join(RESULTS, name + ".npz"))


def save_csv(name: str, rows):
    import csv
    if not rows:
        return
    # Collect all column names (keep order of first appearance)
    cols = []
    seen = set()
    for r in rows:
        if not isinstance(r, dict):
            continue
        for k in r:
            if k not in seen:
                cols.append(k)
                seen.add(k)
    out_path = os.path.join(RESULTS, name + ".csv")
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            if isinstance(r, dict):
                # Convert numpy scalars to native Python types
                clean = {}
                for k, v in r.items():
                    if isinstance(v, (np.floating,)):
                        clean[k] = float(v)
                    elif isinstance(v, (np.integer,)):
                        clean[k] = int(v)
                    elif isinstance(v, np.ndarray):
                        clean[k] = v.tolist()
                    else:
                        clean[k] = v
                w.writerow(clean)


def save_json(name: str, data):
    def _conv(o):
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(os.path.join(RESULTS, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_conv)


def five_folds(n: int, k: int = 5, seed: int = 42):
    from aloop.validate.stats import kfold_split
    return kfold_split(n, k, seed=seed)
