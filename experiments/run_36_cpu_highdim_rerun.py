# -*- coding: utf-8 -*-
"""run_36_cpu_highdim_rerun.py

CPU re-validation of the high-dimensional rotated pipeline (250/300/400/500D).

Motivation (reviewer comment): the GPU batch pipeline is verified
equivalent to the CPU pipeline at 100D only; CPU re-validation at 250-500D is
listed as future work in Table IV. This script closes that gap: the SAME code
(r3_gpu_full.full_pcu, torch FP64) is executed with PCU_DEV=cpu, i.e. only the
hardware changes, the protocol (64-point-per-dim batch scan, voting, candidate
solve, F verification) is identical.

Seeds (aligned with Table IV): 250D x5, 300D x3, 400D x3, 500D x2.
Output: results/cpu_highdim_rerun.json
"""
import io, sys, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
os.environ['PCU_DEV'] = 'cpu'          # force CPU (same torch FP64 code path)
import numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'repro'))
from r3_gpu_full import full_pcu

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
PLAN = {
    250: [20260912, 20260913, 20260914, 20260915, 20260916],
    300: [20260912, 20260913, 20260914],
    400: [20260912, 20260913, 20260914],
    500: [20260912, 20260913],
}


def main():
    t0 = time.time()
    out = {'device': 'cpu', 'protocol': 'r3_gpu_full.full_pcu (torch FP64 batch, 64-point-per-dim scan)'}
    for Dd in sorted(PLAN):
        rows = []
        for sd in PLAN[Dd]:
            t1 = time.time()
            r = full_pcu(Dd, sd, margin=5.0, voting=True)
            r['seed'] = sd
            r['elapsed_s'] = round(time.time() - t1, 1)
            rows.append(r)
            print('D=%d seed %d: trigger=%s hit=%s F=%s (%.1fs)' % (
                Dd, sd, r.get('trigger'), r.get('hit'), r.get('F'), r['elapsed_s']), flush=True)
        out[str(Dd)] = {
            'n_runs': len(rows),
            'n_trigger': sum(1 for r in rows if r.get('trigger')),
            'n_hit': sum(1 for r in rows if r.get('hit')),
            'rows': rows,
        }
    out['elapsed_s'] = round(time.time() - t0, 1)
    json.dump(out, io.open(os.path.join(RESULTS, 'cpu_highdim_rerun.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    for Dd in sorted(PLAN):
        v = out[str(Dd)]
        print('  D=%s: %d/%d trigger, %d/%d hit' % (Dd, v['n_trigger'], v['n_runs'], v['n_hit'], v['n_runs']), flush=True)
    print('SAVED cpu_highdim_rerun.json | %.1fs' % out['elapsed_s'], flush=True)


if __name__ == '__main__':
    main()
