"""Compare PETSc CPR with gamg vs hypre pressure preconditioner on T142.

Runs the same N report steps from the same initial state with each
pressure preconditioner and reports wall time, linear-solve stats and
Krylov iterations.

Usage:
    python scripts/compare_t142_gamg_vs_hypre.py [n_steps]
"""
import copy
import sys
import time

import numpy as np

sys.path.insert(0, '.')

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import _get_non_linear_solver
from PRSTCore.ad_core.solvers.petsc_solver_ad import PETScSolverAD
from PRSTCore.ad_core.simulators.simulate_schedule_ad import simulate_schedule_ad


def _linear_summary(records):
    if not records:
        return {}
    st = np.asarray([float(r.get('SolverTime', np.nan)) for r in records], dtype=float)
    it = np.asarray([float(r.get('Iterations', np.nan)) for r in records], dtype=float)
    res = np.asarray([float(r.get('Residual', np.nan)) for r in records], dtype=float)
    return {
        'count': len(records),
        'lin_total': float(np.nansum(st)),
        'lin_avg': float(np.nanmean(st)),
        'kry_avg': float(np.nanmean(it)),
        'kry_total': float(np.nansum(it)),
        'residual_max': float(np.nanmax(res)),
    }


def run_case(pc, state0, model, schedule, n_steps):
    print(f'\n========== PETSc CPR + pressure_precond={pc} ==========', flush=True)
    linear = PETScSolverAD(
        strategy='cpr', pressure_precond=pc, second_stage='ilu',
        tolerance=1e-4, maxIterations=200, verbose=False,
        decoupling='trueIMPES', decoupling_strategy='mrst', reuse_setup=True,
    )
    records = []
    orig = linear.solveLinearProblem

    def traced(problem, model_arg=None):
        dx, res, rep = orig(problem, model_arg)
        if isinstance(rep, dict):
            records.append(dict(rep))
        return dx, res, rep

    linear.solveLinearProblem = traced

    nls = _get_non_linear_solver(model, {
        'Verbose': False, 'errorOnFailure': False, 'usePETSc': True,
        'linearSolverTolerance': 1e-4,
    })
    nls.LinearSolver = linear
    nls.verbose = False
    nls.errorOnFailure = False

    t0 = time.time()
    well_sols, states, report = simulate_schedule_ad(
        copy.deepcopy(state0), model, schedule,
        NonLinearSolver=nls, verbose=False, return_report=True,
    )
    wall = time.time() - t0
    s = _linear_summary(records)
    print(f'{pc}: wall={wall:.1f}s conv={report.get("Converged")} '
          f'newton={report.get("Iterations")} lin_count={s.get("count", 0)} '
          f'lin_total={s.get("lin_total", np.nan):.1f}s '
          f'lin_avg={s.get("lin_avg", np.nan):.3f}s '
          f'krylov_avg={s.get("kry_avg", np.nan):.1f} '
          f'krylov_total={s.get("kry_total", np.nan):.0f}', flush=True)
    if states:
        p = np.asarray(states[-1]['pressure'], dtype=float)
        sw = np.asarray(states[-1]['sW'], dtype=float)
        print(f'  final p={float(p.min()):.3e}..{float(p.max()):.3e} '
              f'sW={float(sw.min()):.4f}..{float(sw.max()):.4f}', flush=True)
    return {'wall': wall, 'linear': s}


def main():
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    t0 = time.time()
    s0, model, schedule, _ = init_eclipse_problem_ad(
        'examples/T142/T142_E100.DATA', RemoveZeroPoreVolume=True)
    schedule = copy.deepcopy(schedule)
    schedule['step']['val'] = schedule['step']['val'][:n_steps]
    schedule['step']['control'] = schedule['step']['control'][:n_steps]
    print(f'init: cells={len(s0["pressure"])} steps={n_steps} ({time.time()-t0:.1f}s)',
          flush=True)

    r_gamg = run_case('gamg', s0, model, schedule, n_steps)
    r_hypre = run_case('hypre', s0, model, schedule, n_steps)

    print('\n========== SUMMARY ==========', flush=True)
    for name, r in (('gamg', r_gamg), ('hypre', r_hypre)):
        s = r['linear'] or {}
        print(f'{name:6s}: wall={r["wall"]:7.1f}s lin_total={s.get("lin_total", np.nan):6.1f}s '
              f'lin_avg={s.get("lin_avg", np.nan):.3f}s '
              f'krylov_avg={s.get("kry_avg", np.nan):5.1f} '
              f'count={s.get("count", 0)}', flush=True)


if __name__ == '__main__':
    main()
