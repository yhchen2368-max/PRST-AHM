"""Compare PETSc CPR vs PETSc fieldsplit (block) on T142.

Both linear solvers are PETScSolverAD with hypre BoomerAMG on the pressure
block and ILU second stage; only the preconditioner *shape* differs:
  * strategy='cpr'       -> two-stage CPR (composite: pressure fieldsplit + ILU)
  * strategy='fieldsplit'-> block-fieldsplit (multiplicative pressure/rest)

Usage:
    python scripts/compare_t142_petsc_cpr_block.py [n_steps]
"""
import copy
import sys
import time

import numpy as np

sys.path.insert(0, '.')

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import _get_non_linear_solver
from PRSTCore.ad_core.solvers import NonLinearSolver
from PRSTCore.ad_core.solvers.petsc_solver_ad import PETScSolverAD
from PRSTCore.ad_core.simulators.simulate_schedule_ad import simulate_schedule_ad


def _trim_schedule(schedule, max_steps):
    if max_steps <= 0:
        return schedule
    out = copy.deepcopy(schedule)
    out['step']['val'] = schedule['step']['val'][:max_steps]
    out['step']['control'] = schedule['step']['control'][:max_steps]
    return out


def _linear_summary(records):
    if not records:
        return {}
    solver_time = np.asarray([float(r.get('SolverTime', np.nan)) for r in records], dtype=float)
    lin_time = np.asarray([float(r.get('LinearSolutionTime', np.nan)) for r in records], dtype=float)
    iters = np.asarray([float(r.get('Iterations', np.nan)) for r in records], dtype=float)
    residual = np.asarray([float(r.get('Residual', np.nan)) for r in records], dtype=float)
    return {
        'count': len(records),
        'solver_time_avg': float(np.nanmean(solver_time)),
        'solver_time_total': float(np.nansum(solver_time)),
        'linear_time_avg': float(np.nanmean(lin_time)),
        'iters_avg': float(np.nanmean(iters)),
        'iters_total': float(np.nansum(iters)),
        'residual_max': float(np.nanmax(residual)),
    }


def run_case(name, strategy, state0, model, schedule, max_steps,
             verbose=False):
    print(f'\n========== {name} (strategy={strategy}) ==========', flush=True)
    lin_tol = 1e-4
    linear_solver = PETScSolverAD(
        strategy=strategy,
        pressure_precond='hypre',
        second_stage='ilu',
        tolerance=lin_tol,
        maxIterations=200,
        verbose=verbose,
        decoupling='trueIMPES',
        decoupling_strategy='mrst',
        reuse_setup=True,
    )
    linear_records = []
    original_solve = linear_solver.solveLinearProblem

    def traced_solve(problem, model_arg=None):
        dx, res, rep = original_solve(problem, model_arg)
        if isinstance(rep, dict):
            linear_records.append(dict(rep))
        return dx, res, rep

    linear_solver.solveLinearProblem = traced_solve

    solver_opts = {
        'Verbose': False,
        'errorOnFailure': False,
        'usePETSc': True,
        'linearSolverTolerance': lin_tol,
    }
    nonlinear_solver = _get_non_linear_solver(model, solver_opts)
    nonlinear_solver.LinearSolver = linear_solver
    nonlinear_solver.verbose = False
    nonlinear_solver.errorOnFailure = False

    t0 = time.time()
    well_sols, states, report = simulate_schedule_ad(
        state0, model, schedule, NonLinearSolver=nonlinear_solver,
        verbose=verbose, return_report=True,
    )
    wall = time.time() - t0

    controls = report.get('ControlstepReports', [])
    total_iter = int(report.get('Iterations', sum(int(c.get('Iterations', 0)) for c in controls)))
    total_mini = sum(int(c.get('MinistepCount', 0)) for c in controls)
    s = _linear_summary(linear_records)
    print(f'{name}: wall={wall:.1f}s conv={report.get("Converged")} '
          f'steps={len(controls)} nonlin_iters={total_iter} ministeps={total_mini}', flush=True)
    if s:
        print(f'  linear: count={s["count"]} avg={s["solver_time_avg"]:.3f}s '
              f'total={s["solver_time_total"]:.1f}s iters_avg={s["iters_avg"]:.1f} '
              f'iters_total={s["iters_total"]:.0f} residual_max={s["residual_max"]:.3e}', flush=True)
    if states:
        p = np.asarray(states[-1]['pressure'], dtype=float)
        sw = np.asarray(states[-1]['sW'], dtype=float)
        print(f'  final p={float(p.min()):.3e}..{float(p.max()):.3e} '
              f'sW={float(sw.min()):.4f}..{float(sw.max()):.4f}', flush=True)
    return {'wall': wall, 'report': report, 'linear': s, 'states': states}


def main():
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    t0 = time.time()
    s0, model, schedule, _ = init_eclipse_problem_ad(
        'examples/T142/T142_E100.DATA', RemoveZeroPoreVolume=True)
    schedule = _trim_schedule(schedule, n_steps)
    print(f'init: cells={len(s0["pressure"])} steps={len(schedule["step"]["val"])} '
          f'({time.time()-t0:.1f}s)', flush=True)

    # independent fresh state for each case
    import copy as _c
    results = {}
    results['CPR'] = run_case('CPR', 'cpr', _c.deepcopy(s0), model, schedule, n_steps)
    results['BLOCK'] = run_case('BLOCK-fieldsplit', 'fieldsplit', _c.deepcopy(s0), model, schedule, n_steps)

    print('\n========== SUMMARY ==========', flush=True)
    for name, r in results.items():
        lin = r['linear'] or {}
        print(f'{name:20s}: wall={r["wall"]:7.1f}s '
              f'lin_total={lin.get("solver_time_total", np.nan):6.1f}s '
              f'lin_avg={lin.get("solver_time_avg", np.nan):.3f}s '
              f'lin_iters_avg={lin.get("iters_avg", np.nan):5.1f} '
              f'lin_count={lin.get("count", 0)}', flush=True)


if __name__ == '__main__':
    main()
