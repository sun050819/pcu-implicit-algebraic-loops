# -*- coding: utf-8 -*-
"""run_41_cec_matrix_norm: record the CEC2017 f5 (30D) raw/orthogonalized
matrix norms used in the paper (Sec. V-B, paragraph on the raw/orth asymmetry).

Computes, for the official CEC2017 f5 rotation matrix M_raw (30D):
  ||M_orth - M_raw||_2  and  ||M_orth M_orth^T - I||_2
with M_orth = U V^T from the SVD M_raw = U S V^T (the paper's
orthogonalization), and writes the values to cec_matrix_norm.json.
"""
import io, sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'cec2017'))
from cec2017.transforms import rotations
import numpy as np

D = 30
M_raw = rotations[D][4].copy()
u, s, vt = np.linalg.svd(M_raw)
M_orth = u @ vt

n2_diff = float(np.linalg.norm(M_orth - M_raw, 2))
n2_orth = float(np.linalg.norm(M_orth @ M_orth.T - np.eye(D), 2))
sv_min, sv_max = float(s.min()), float(s.max())

out = {
    "problem": "CEC2017 f5 (shifted rotated Rastrigin), D=30, official M_raw",
    "D": D,
    "svd": "M_orth = U V^T from M_raw = U S V^T",
    "||M_orth - M_raw||_2": n2_diff,
    "||M_orth M_orth^T - I||_2": n2_orth,
    "singular_value_range_M_raw": [sv_min, sv_max],
}
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'cec_matrix_norm.json')
with open(path, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
print("written:", path)
