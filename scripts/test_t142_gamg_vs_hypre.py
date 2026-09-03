"""Test gamg vs hypre pressure preconditioner on late T142 steps.

Warms up with a fixed solver to steps 150 and 250, then on each target
report step runs the *same* initial state with PETSc CPR using
pressure_precond='gamg' vs 'hypre', recording Krylov iterations and
linear-solve time.

Usage:
    python scripts/test_t142_gamg_vs_hypre.py [target_steps ...]
"""
import copy
import sys
import time

import numpy as np

sys.path.insert(0, '.')

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import _get_non_linear_solver
from PRSTCore.ad_core.solvers.petsc_solver_ad import PETScSolverAD

DECK = 'examples/T142/T142_E100.DATA'


def make_linear(pressure_precond):
    return PETScSolverAD(
        strategy='cpr', pressure_precond=pressure_precond,
        second_stage='ilu', tolerance=1e-4, maxIterations=200,
        decoupling='trueIMPES', decoupling_strategy='mrst',
        reuse_setup=True, verbose=False,
    )


def make_nonlinear(model, linear):
    nls = _get_non_linear_solver(model, {
        'Verbose': False, 'errorOnFailure': False, 'usePETSc': True,
        'linearSolverTolerance': 1e-4,
    })
    nls.LinearSolver = linear
    nls.verbose = False
    nls.errorOnFailure = False
    return nls


def collect_linear(linear):
    records = []
    orig = linear.solveLinearProblem

    def traced(problem, model_arg=None):
        dx, res, rep = orig(problem, model_arg)
        if isinstance(rep, dict):
            records.append(dict(rep))
        return dx, res, rep

    linear.solveLinearProblem = traced
    return records


def step_once(state, model, schedule, step_idx, nonlinear):
    ctrl = int(schedule['step']['control'][step_idx])
    forces = model.getDrivingForces(schedule['control'][ctrl])
    model, state = model.updateForChangedControls(state, forces)
    dt = float(schedule['step']['val'][step_idx])
    state, report, _ = nonlinear.solveTimestep(
        copy.deepcopy(state), dt, model, drivingForces=forces,
        initialGuess=copy.deepcopy(state), controlId=ctrl,
    )
    return state, report


def test_target(target, state, model, schedule):
    print(f'\n========== target step {target} ==========', flush=True)
    out = {}
    for name, pc in (('gamg', 'gamg'), ('hypre', 'hypre')):
        linear = make_linear(pc)
        records = collect_linear(linear)
        nls = make_nonlinear(model, linear)
        t0 = time.time()
        _, report = step_once(copy.deepcopy(state), model, schedule, target - 1, nls)
        wall = time.time() - t0
        conv = report.get('Converged')
        iters = int(report.get('Iterations', 0))
        # aggregate linear records
        lt = np.asarray([float(r.get('SolverTime', np.nan)) for r in records], dtype=float)
        li = np.asarray([float(r.get('Iterations', np.nan)) for r in records], dtype=float)
        lin_total = float(np.nansum(lt))
        lin_avg = float(np.nanmean(lt))
        kry_avg = float(np.nanmean(li))
        kry_total = float(np.nansum(li))
        print(f'  {pc:5s}: wall={wall:7.1f}s conv={conv} newton={iters} '
              f'lin_count={len(records)} lin_total={lin_total:7.1f}s '
              f'lin_avg={lin_avg:.3f}s krylov_avg={kry_avg:6.1f} '
              f'krylov_total={kry_total:7.0f}', flush=True)
        out[name] = dict(wall=wall, conv=conv, newton=iters, lin_total=lin_total,
                         lin_avg=lin_avg, kry_avg=kry_avg, kry_total=kry_total,
                         count=len(records))
    return out


def main():
    targets = [int(a) for a in sys.argv[1:]] or [150, 250]
    t0 = time.time()
    s0, model, schedule, solver0 = init_eclipse_problem_ad(DECK, RemoveZeroPoreVolume=True)
    print(f'init: cells={len(s0["pressure"])} ({time.time()-t0:.1f}s)', flush=True)

    # warm-up solver: PETSc CPR + hypre (same as the long run)
    warm_linear = make_linear('hypre')
    warm_nls = make_nonlinear(model, warm_linear)

    state = s0
    for target in targets:
        # warm up from current state up to target-1
        for i in range(len(targets) and 0, 0):
            pass
        # determine warmup window
        # find previous target boundary
        prev = 1
        for t in targets:
            if t < target:
                prev = t
        start = prev  # 0-based: warm from step prev-1's end (i.e. step index prev-1)
        for step_idx in range(prev - 1, target - 1):
            ctrl = int(schedule['step']['control'][step_idx])
            forces = model.getDrivingForces(schedule['control'][ctrl])
            model, state = model.updateForChangedControls(state, forces)
            dt = float(schedule['step']['val'][step_idx])
            tw = time.time()
            state, report = warm_nls.solveTimestep(
                copy.deepcopy(state), dt, model, drivingForces=forces,
                initialGuess=copy.deepcopy(state), controlId=ctrl,
            )
            print(f'  warm step {step_idx+1}: conv={report.get("Converged")} '
                  f'wall={time.time()-tw:.1f}s', flush=True)
        test_target(target, state, model, schedule)

    print(f'\ntotal wall: {time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
