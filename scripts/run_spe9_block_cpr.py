"""Run SPE9 with the MRST-style AMGCL block CPR solver.

Environment knobs:
  SPE9_MAX_STEPS       Limit schedule steps for quick profiling. Default: all.
  SPE9_VERBOSE         Print per-control-step progress. Default: 1.
  SPE9_NEWTON_VERBOSE  Print Newton/block-CPR trace. Default: 0.
  SPE9_MRST_TIMESTEP   Use initEclipseProblemAD's MRST-style selector. Default: 1.
  SPE9_NL_MAXIT        Override nonlinear max iterations. Default: MRST/deck default.
  SPE9_LIN_TOL         AMGCL relative tolerance. Default: 1e-4.
  SPE9_LIN_MAXIT       AMGCL max iterations. Default: 100.
  SPE9_REUSE           Reuse AMGCL CPR hierarchy. Default: 0.
  SPE9_UPDATE_S        Update CPR second-stage preconditioner on reuse. Default: 1.
  SPE9_UPDATE_P        Update CPR pressure transfer on reuse. Default: 0.
  SPE9_SCHUR           Schur approximation: diagonal or full. Default: diagonal.
  SPE9_WELL_TOL_FACTOR Multiply model.toleranceWellRate. Default: 2.0 for saved MRST SPE9 trace.
"""
import copy
import os
import sys
import time

import numpy as np

sys.path.insert(0, '.')

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import _get_non_linear_solver
from PRSTCore.ad_core.solvers import NonLinearSolver, AMGCL_CPRSolverBlockAD
from PRSTCore.ad_core.simulators.simulate_schedule_ad import simulate_schedule_ad


def _env_int(name, default):
    val = os.environ.get(name, '')
    return default if val == '' else int(val)


def _env_float(name, default):
    val = os.environ.get(name, '')
    return default if val == '' else float(val)


def _env_bool(name, default):
    val = os.environ.get(name, '')
    if val == '':
        return bool(default)
    return val.lower() not in ('0', 'false', 'no', 'off')


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
    prep_time = np.asarray([float(r.get('PreparationTime', np.nan)) for r in records], dtype=float)
    iters = np.asarray([float(r.get('Iterations', np.nan)) for r in records], dtype=float)
    residual = np.asarray([float(r.get('Residual', np.nan)) for r in records], dtype=float)
    full_res = []
    kernel_time = []
    types = set()
    for r in records:
        pre = r.get('PreconditionerReport', {})
        if isinstance(pre, dict):
            types.add(str(pre.get('Type', '')))
            full_res.append(float(pre.get('FullSystemResidual', np.nan)))
            kernel_time.append(float(pre.get('KernelTime', np.nan)))
    full_res = np.asarray(full_res, dtype=float)
    kernel_time = np.asarray(kernel_time, dtype=float)
    return {
        'count': len(records),
        'solver_time_avg': float(np.nanmean(solver_time)),
        'solver_time_min': float(np.nanmin(solver_time)),
        'solver_time_max': float(np.nanmax(solver_time)),
        'linear_time_avg': float(np.nanmean(lin_time)),
        'prep_time_avg': float(np.nanmean(prep_time)),
        'kernel_time_avg': float(np.nanmean(kernel_time)) if kernel_time.size else np.nan,
        'iterations_avg': float(np.nanmean(iters)),
        'residual_max': float(np.nanmax(residual)),
        'full_residual_max': float(np.nanmax(full_res)) if full_res.size else np.nan,
        'types': sorted(t for t in types if t),
    }


def _print_summary(report, linear_records, wall):
    controls = report.get('ControlstepReports', [])
    print('--- SPE9 block CPR summary ---', flush=True)
    print(
        f'OK control_steps={report.get("NumControlSteps", len(controls))} '
        f'converged={report.get("Converged", True)} '
        f'wall={wall:.2f}s simtime={report.get("SimulationTime", np.nan):.2f}s',
        flush=True,
    )
    total_mini = sum(int(c.get('MinistepCount', 0)) for c in controls)
    total_iter = sum(int(c.get('Iterations', 0)) for c in controls)
    cuts = sum(
        int(sr.get('Converged', True) is False)
        for c in controls for sr in c.get('StepReports', [])
    )
    print(f'nonlinear total_iterations={total_iter} ministeps={total_mini} failed_attempts={cuts}', flush=True)
    s = _linear_summary(linear_records)
    if s:
        print(
            'linear '
            f'count={s["count"]} '
            f'solver_avg={s["solver_time_avg"]:.3f}s '
            f'solver_min={s["solver_time_min"]:.3f}s '
            f'solver_max={s["solver_time_max"]:.3f}s '
            f'linear_avg={s["linear_time_avg"]:.3f}s '
            f'prep_avg={s["prep_time_avg"]:.3f}s '
            f'kernel_avg={s["kernel_time_avg"]:.3f}s '
            f'iters_avg={s["iterations_avg"]:.2f} '
            f'residual_max={s["residual_max"]:.3e} '
            f'full_residual_max={s["full_residual_max"]:.3e} '
            f'types={s["types"]}',
            flush=True,
        )
    for c in controls:
        print(
            f'step {int(c.get("ControlStep", 0)):03d}: '
            f'dt={float(c.get("Timestep", 0.0)):.6g} '
            f'conv={bool(c.get("Converged", False))} '
            f'iters={int(c.get("Iterations", 0))} '
            f'ministeps={int(c.get("MinistepCount", 0))} '
            f'time={float(c.get("SimulationTime", 0.0)):.2f}s',
            flush=True,
        )


def main():
    step_verbose = _env_bool('SPE9_VERBOSE', True)
    newton_verbose = _env_bool('SPE9_NEWTON_VERBOSE', False)
    mrst_timestep = _env_bool('SPE9_MRST_TIMESTEP', True)
    max_steps = _env_int('SPE9_MAX_STEPS', 0)
    nl_maxit_env = os.environ.get('SPE9_NL_MAXIT', '')
    nl_maxit = None if nl_maxit_env == '' else int(nl_maxit_env)
    lin_tol = _env_float('SPE9_LIN_TOL', 1e-4)
    lin_maxit = _env_int('SPE9_LIN_MAXIT', 100)
    reuse = _env_bool('SPE9_REUSE', False)
    update_s = _env_bool('SPE9_UPDATE_S', True)
    update_p = _env_bool('SPE9_UPDATE_P', False)
    schur = os.environ.get('SPE9_SCHUR', 'diagonal')
    well_tol_factor = _env_float('SPE9_WELL_TOL_FACTOR', 2.0)

    print('--- SPE9 init ---', flush=True)
    t0 = time.time()
    state0, model, schedule, _ = init_eclipse_problem_ad('examples/SPE9/SPE9_CP.DATA')
    if well_tol_factor != 1.0 and hasattr(model, 'toleranceWellRate'):
        model.toleranceWellRate = float(model.toleranceWellRate) * float(well_tol_factor)
    schedule = _trim_schedule(schedule, max_steps)
    print(
        f'cells={len(state0["pressure"])} steps={len(schedule["step"]["val"])} '
        f'gas={model.gas} init={time.time() - t0:.2f}s',
        flush=True,
    )
    print(
        f'linear=AMGCL_CPRSolverBlockAD blockSize=3 tol={lin_tol:g} '
        f'maxit={lin_maxit} nl_maxit={nl_maxit if nl_maxit is not None else "MRST-default"} '
        f'mrst_timestep={mrst_timestep} '
        f'reuse={reuse} update_s={update_s} update_p={update_p} '
        f'schur={schur} well_tol_factor={well_tol_factor:g} '
        f'step_verbose={step_verbose} newton_verbose={newton_verbose}',
        flush=True,
    )

    linear_solver = AMGCL_CPRSolverBlockAD(
        blockSize=3,
        tolerance=lin_tol,
        maxIterations=lin_maxit,
        verbose=newton_verbose,
        reuseMode=reuse,
        update_sprecond=update_s,
        update_ptransfer=update_p,
        schurApproxType=schur,
    )
    linear_records = []
    original_solve = linear_solver.solveLinearProblem

    def traced_solve(problem, model_arg=None):
        dx, res, rep = original_solve(problem, model_arg)
        if isinstance(rep, dict):
            linear_records.append(dict(rep))
        return dx, res, rep

    linear_solver.solveLinearProblem = traced_solve
    if mrst_timestep:
        solver_opts = {
            'Verbose': newton_verbose,
            'errorOnFailure': False,
            'useAMGCL': True,
            'useAMGCLCPR': True,
            'linearSolverTolerance': lin_tol,
        }
        if nl_maxit is not None:
            solver_opts['maxIterations'] = nl_maxit
        nonlinear_solver = _get_non_linear_solver(model, solver_opts)
        nonlinear_solver.LinearSolver = linear_solver
        nonlinear_solver.verbose = newton_verbose
        nonlinear_solver.errorOnFailure = False
    else:
        nonlinear_solver = NonLinearSolver(
            linearSolver=linear_solver,
            verbose=newton_verbose,
            maxIterations=15 if nl_maxit is None else nl_maxit,
            errorOnFailure=False,
        )

    print('--- SPE9 simulate block CPR ---', flush=True)
    t0 = time.time()
    well_sols, states, report = simulate_schedule_ad(
        state0,
        model,
        schedule,
        NonLinearSolver=nonlinear_solver,
        verbose=step_verbose,
        return_report=True,
    )
    wall = time.time() - t0
    _print_summary(report, linear_records, wall)
    if states:
        p0 = np.asarray(states[-1]['pressure'], dtype=float)
        print(f'final pressure min={float(np.min(p0)):.6e} max={float(np.max(p0)):.6e}', flush=True)
    print(f'well_solutions={len(well_sols)} states={len(states)}', flush=True)


if __name__ == '__main__':
    main()
