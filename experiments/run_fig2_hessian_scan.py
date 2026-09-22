# -*- coding: utf-8 -*-
"""run_fig2_hessian_scan: produce the REAL Hessian-diagonal scan data for Fig. 2.

Scans H_ii along one coordinate direction on the actual CEC2017 f5 (30D):
  - raw : f5 with the official rotation M_raw (non-orthogonal, Lambda scaling);
          PCU scans in x space (R=I) -> multi-frequency diagonal, identification rejects.
  - orth: f5 with M_orth = polar/SVD orthogonalization of M_raw;
          PCU recovers R and scans in the z = R x space -> single-frequency diagonal,
          identification triggers.

Output: experiments/results/fig2_hessian_scan.json (deterministic, no randomness).
"""
import io, sys, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'cec2017'))
from cec2017.transforms import rotations, shifts

D = 30
S = 0.0512
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS, exist_ok=True)

o = shifts[4][:D].copy()
M_raw = rotations[D][4].copy()
u, _, vt = np.linalg.svd(M_raw)
M_orth = u @ vt          # orthogonalized rotation (polar factor)


def make_funcs(Mm):
    """Return (f, g, h) for f5 with rotation Mm (same convention as run_07_cec)."""
    def f5(x):
        z = S * (Mm @ (np.asarray(x, float) - o))
        return float(np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0))
    def g5(x):
        x = np.asarray(x, float)
        z = S * (Mm @ (x - o))
        gz = S * (2.0 * z + 20.0 * np.pi * np.sin(2.0 * np.pi * z))
        return Mm.T @ gz
    def h5(x):
        x = np.asarray(x, float)
        z = S * (Mm @ (x - o))
        Hzz = S * S * (2.0 + 40.0 * np.pi * np.pi * np.cos(2.0 * np.pi * z))
        return (Mm.T * Hzz) @ Mm
    return f5, g5, h5


f_raw, g_raw, h_raw = make_funcs(M_raw)
f_ort, g_ort, h_ort = make_funcs(M_orth)


def scan_curve(hess, base, dim, radius, n):
    """Scan H[dim,dim] along coordinate dim (other coords fixed at base)."""
    xs = np.linspace(base[dim] - radius, base[dim] + radius, n)
    Hc = np.empty(n)
    for k in range(n):
        xk = np.asarray(base, dtype=float).copy()
        xk[dim] = xs[k]
        Hc[k] = np.asarray(hess(xk), dtype=float)[dim, dim]
    return xs, Hc


def fft_spectrum(y, dx, n_out=50):
    yy = np.asarray(y, dtype=float) - np.mean(y)
    sp = np.abs(np.fft.rfft(yy))
    freqs = np.fft.rfftfreq(len(y), d=dx)
    sp[0] = 0.0
    return freqs[1:n_out + 1].tolist(), sp[1:n_out + 1].tolist()


# ---- raw: scan in x space along dim 0 (PCU with R=I; multi-frequency, rejects) ----
x0 = np.zeros(D)
radius = 3.0 / S                       # 3 periods of the base frequency 1/S ~= 58.6
n = 600
xs_raw, H_raw = scan_curve(h_raw, x0, 0, radius, n)
fq_raw, sp_raw = fft_spectrum(H_raw, xs_raw[1] - xs_raw[0])
T_x = 1.0 / S                          # period of z_i along x_i (base frequency)

# ---- orth: scan in z = M_orth x space along z_0 (PCU recovers R; single-frequency, triggers) ----
base_z = np.zeros(D)                   # z-space base (x = M_orth^T z + o)
# hess in z space: Hz = R H R^T with R = M_orth
def hz(zv):
    xv = M_orth.T @ np.asarray(zv, float) + o
    return M_orth @ np.asarray(h_ort(xv), dtype=float) @ M_orth.T

zs_ort, H_ort = scan_curve(hz, base_z, 0, radius, n)
fq_ort, sp_ort = fft_spectrum(H_ort, zs_ort[1] - zs_ort[0])

# dominant peak location (normalized f*T)
def dominant_peak(freqs, spec):
    k = int(np.argmax(spec))
    return float(freqs[k] * T_x), float(freqs[k])

pk_raw = dominant_peak(fq_raw, sp_raw)
pk_ort = dominant_peak(fq_ort, sp_ort)

out = {
    "description": "Real Hessian-diagonal scans on CEC2017 f5 (30D), S=0.0512, T_x=1/S",
    "raw": {"xs": xs_raw.tolist(), "H_ii": H_raw.tolist(),
            "freq": fq_raw, "spec": sp_raw,
            "dom_peak_fT": pk_raw[0], "dom_peak_f": pk_raw[1]},
    "orth": {"xs": zs_ort.tolist(), "H_ii": H_ort.tolist(),
             "freq": fq_ort, "spec": sp_ort,
             "dom_peak_fT": pk_ort[0], "dom_peak_f": pk_ort[1]},
}
with open(os.path.join(RESULTS, 'fig2_hessian_scan.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=1)

# quick statistics for the record
def n_peaks(y):
    yy = np.asarray(y)
    s = np.sign(np.diff(yy))
    return int(np.sum((s[:-1] > 0) & (s[1:] < 0)))
print('raw  : n_peaks=%d dom_fT=%.3f dom_f=%.4f' % (n_peaks(H_raw), pk_raw[0], pk_raw[1]))
print('orth : n_peaks=%d dom_fT=%.3f dom_f=%.4f' % (n_peaks(H_ort), pk_ort[0], pk_ort[1]))
print('spectral top3 raw :', [round(v, 1) for v in np.sort(sp_raw)[-3:][::-1]])
print('spectral top3 orth:', [round(v, 1) for v in np.sort(sp_ort)[-3:][::-1]])
print('saved', os.path.join(RESULTS, 'fig2_hessian_scan.json'))
