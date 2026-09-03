"""Combine saved per-solver SPE9 results (see run_spe9_solver_case.py) and
print a timing / accuracy comparison table."""
import pickle
import sys

import numpy as np


def load(name):
    path = f'scripts/spe9_solver_result_{name}.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    names = sys.argv[1:] or ['amgcl_cpr', 'mumps']
    results = []
    for name in names:
        try:
            results.append(load(name))
        except FileNotFoundError:
            print(f'(missing scripts/spe9_solver_result_{name}.pkl -- run run_spe9_solver_case.py {name} first)')

    print('\n=== SPE9 solver comparison ===')
    header = f'{"Solver":<12}{"OK":<6}{"sim_time[s]":<14}{"nsteps":<9}{"p_min":<14}{"p_max":<14}{"p_last":<14}'
    print(header)
    print('-' * len(header))
    for r in results:
        if r.get('ok'):
            print(f'{r["name"]:<12}{"yes":<6}{r["sim_time"]:<14.2f}{r["nsteps"]:<9}'
                  f'{r["p_min"]:<14.4e}{r["p_max"]:<14.4e}{r["p_last"]:<14.4e}')
        else:
            print(f'{r["name"]:<12}{"NO":<6}{r["sim_time"]:<14.2f}{"-":<9}{"-":<14}{"-":<14}{"-":<14}')
        print(f'  python: {r.get("python", "?").splitlines()[0]}')
        if not r.get('ok'):
            print(f'  error: {r.get("error")}')

    ok_results = [r for r in results if r.get('ok')]
    if len(ok_results) >= 2:
        ref = ok_results[0]
        for r in ok_results[1:]:
            n = min(len(ref['pressures']), len(r['pressures']))
            max_abs_diff = 0.0
            max_rel_diff = 0.0
            for i in range(n):
                p_ref = ref['pressures'][i]
                p_cur = r['pressures'][i]
                if p_ref.shape != p_cur.shape:
                    continue
                diff = np.abs(p_ref - p_cur)
                max_abs_diff = max(max_abs_diff, float(diff.max()))
                denom = max(float(np.abs(p_ref).max()), 1e-30)
                max_rel_diff = max(max_rel_diff, float(diff.max() / denom))
            speedup = r['sim_time'] / ref['sim_time'] if ref['sim_time'] > 0 else float('nan')
            print(f'\n[{ref["name"]} vs {r["name"]}] over {n} common states:')
            print(f'  max abs pressure diff = {max_abs_diff:.4e}')
            print(f'  max rel pressure diff = {max_rel_diff:.4e}')
            print(f'  sim_time ratio ({r["name"]}/{ref["name"]}) = {speedup:.3f}x')


if __name__ == '__main__':
    main()
