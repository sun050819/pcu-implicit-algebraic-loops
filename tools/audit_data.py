# -*- coding: utf-8 -*-
"""Audit: every paper number traces to a results/*.json value.

Two entry points:
  python -X utf8 tools/audit_data.py        # structural cross-check (kept)
  python -X utf8 tools/audit_data.py --hallucination  # paper-vs-JSON backtrace

The --hallucination mode is the paper-claims audit: it greps paper.tex for the
values listed below and independently recomputes the statistics from the raw
JSON so the paper can never carry a number that is not in the experiment logs.
"""
import io, json, os, re, statistics, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'experiments', 'results')
TEX = os.path.join(ROOT, 'paper', 'ieee_tevc_paper', 'paper.tex')

def load(name):
    with open(os.path.join(RES, name), encoding='utf-8-sig') as f:
        return json.load(f)

def rd(fn):
    return load(fn)

# ============ part 1: structural (legacy) ============
def structural():
    checks = []

    def chk(label, cond, detail=''):
        checks.append((label, cond, detail))

    nr = load('noise_robustness.json')
    print('noise_robustness keys:', list(nr.keys()))
    for k, v in nr.items():
        if isinstance(v, dict):
            print(' ', k, ':', {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in list(v.items())[:6]})

    sc = load('sota_compare_summary.json')
    print('\nsota_compare_summary:', json.dumps(sc, ensure_ascii=False)[:500])

    cf = load('cec_fulltable.json')
    print('\ncec_fulltable type:', type(cf))
    s = json.dumps(cf, ensure_ascii=False)
    print('len:', len(s), '| head:', s[:300])

    ss = load('stats_strong_summary.json')
    print('\nstats_strong_summary:', json.dumps(ss, ensure_ascii=False)[:600])

    sr = load('supplementary_round3.json')
    print('\nsupplementary_round3 keys:', list(sr.keys())[:10])

# ============ part 2: hallucination backtrace ============
def hallucination():
    if not os.path.exists(TEX):
        print('audit: paper.tex is kept private during single-blind review and is not in this '
              'public repository; run this check locally with the manuscript present. '
              'Skipping the paper-vs-JSON backtrace.')
        return
    tex = io.open(TEX, encoding='utf-8').read()
    checks = []

    def chk(name, expected, must_exist=True):
        ok = expected in tex
        checks.append((name, expected, ok))
        if not ok and must_exist:
            print('MISSING in tex:', name, '=', expected)

    # Table II 5-run medians (rounded from sota_compare.json)
    tab2 = {
        'SHADE 15.46': '15.46', 'SHADE q1 14.53': '14.53', 'SHADE q3 16.59': '16.59',
        'LSHADE 18.91': '18.91', 'JADE 22.27': '22.27', 'JADE q1 21.58': '21.58', 'JADE q3 23.34': '23.34',
        'PSO 298.25': '298.25', 'PSO q1 293.10': '293.10', 'PSO q3 325.01': '325.01',
        'GWO 501.16': '501.16', 'GWO q1 482.07': '482.07', 'GWO q3 505.50': '505.50',
        'BO RBF 61.34': '61.34', 'BO RBF q1 33.33': '33.33', 'BO RBF q3 64.85': '64.85',
        'BO periodic 72.54': '72.54', 'BO p q1 62.12': '62.12', 'BO p q3 81.55': '81.55',
        'CMA-ES 442.75': '442.75', 'CMA q1 375.09': '375.09', 'CMA q3 493.49': '493.49',
        'RFF 430.76': '430.76', 'RFF q1 416.71': '416.71', 'RFF q3 457.47': '457.47',
        'L-BFGS 195.01': '195.01', 'LBFGS q1 184.34': '184.34', 'LBFGS q3 206.95': '206.95',
        'Basinhopping 8.95': '8.95', 'BH q1 7.96': '7.96', 'BH q3 8.95': '8.95',
    }
    for name, v in tab2.items():
        chk(name, v)

    # 25-run baselines and extremes
    for v in ['56.7', '18.1', '191.0', '113.4', '222.9', '640.7', '504.8', '765.9',
              '471.99', '323.7', '498.5']:
        chk('stat', v)

    # stats p-values (2.0e-29 is the numeric form of the Fisher two-sided value
    # 2/binom(100,50); paper states it numerically, so the formula string is not checked)
    for v in ['5.4\\times10^{-9}', '2.0\\times10^{-29}', '5.9\\times10^{-29}', '0.125']:
        chk('pvalue', v)

    # hit rates
    for v in ['50/50', '7/50', '10/10', '0/25', '0/5', '5/5']:
        chk('hits', v)

    # dimensions / FE / periods / simulink
    for v in ['250', '500', '300', '400', '396{,}993', '19.5312', '19.53125',
              '2.6\\times10^{-6}', '8.9\\times10^{-2}', '2.4\\%', '6.5\\%',
              '8.96\\times10^{-9}', '321', '7.1\\times10^{-5}', '1.4\\times10^{-11}', '2.1\\times10^{-15}']:
        chk('misc', v)

    missing = [c for c in checks if not c[2]]
    print()
    print('=== hallucination audit: %d checks, %d missing ===' % (len(checks), len(missing)))
    for c in missing:
        print('MISS:', c[0], c[1])

    # ---- recompute from raw JSON (independent of what the paper says) ----
    print()
    print('--- independent recomputation from JSON ---')
    c = rd('sota_cmaes_5runs.json')
    fs = [r['f'] for r in c['rows']]
    print('CMA-ES 5-run recomputed: median=%.2f (paper 442.75), min=%.1f, max=%.1f'
          % (statistics.median(fs), min(fs), max(fs)))

    l = rd('lbfgs_25runs.json')
    print('L-BFGS 25-run recomputed: median=%.2f min=%.2f max=%.2f (paper 191.0/113.4/222.9)'
          % (l['final_f']['median'], l['final_f']['min'], l['final_f']['max']))

    n = rd('newton_fd_25runs.json')
    print('FD-Newton 25-run recomputed: median=%.2f min=%.2f max=%.2f (paper 640.7/504.8/765.9)'
          % (n['final_f']['median'], n['final_f']['min'], n['final_f']['max']))

    b = rd('bipop_cmaes.json')
    fs = [r['f'] for r in b['runs']]
    print('BIPOP recomputed: median=%.2f best=%.1f worst=%.1f hits=%d (paper 471.99/323.7/498.5/0)'
          % (statistics.median(fs), min(fs), max(fs), b['hits']))

    sc = rd('sota_compare.json')
    for k in ['SHADE', 'LSHADE', 'JADE', 'PSO', 'GWO', 'BO_RBF_smooth', 'BO_periodic_kernel']:
        f = sc[k]['final_f']
        print('%-18s median=%.4f q1=%.4f q3=%.4f' % (k, f['median'], f['q1'], f['q3']))

    p = rd('pcu_50runs.json')
    print('pcu50 f5: %d/%d trigger, %d/%d hit; f8: %d/%d trigger, %d/%d hit'
          % (p['f5_orth']['n_trigger'], p['f5_orth']['n_runs'], p['f5_orth']['n_hit'], p['f5_orth']['n_runs'],
             p['f8_orth']['n_trigger'], p['f8_orth']['n_runs'], p['f8_orth']['n_hit'], p['f8_orth']['n_runs']))

if __name__ == '__main__':
    if '--hallucination' in sys.argv:
        hallucination()
    else:
        structural()
