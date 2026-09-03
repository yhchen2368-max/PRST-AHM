"""Run the SPE9 first report step for every MRST AMGCL CPR strategy.

Validates the 1:1 strategy port: mrst / mrst_drs / amgcl / amgcl_drs crossed
with the decoupling methods trueIMPES / quasiIMPES / none, on the MRST-style
AMGCL block CPR solver.  Reports convergence, Newton iterations, wall time
and the first-step final pressure range for each configuration.
"""
import sys
import time
from copy import deepcopy

sys.path.insert(0, '.')

import numpy as np

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.solvers import NonLinearSolver, AMGCL_CPRSolverBlockAD

STRATEGIES = ['mrst', 'mrst_drs', 'amgcl', 'amgcl_drs']
DECOUPLINGS = ['trueIMPES', 'quasiIMPES', 'none']


def run_case(strategy, decoupling, verbose=False):
    s0, model, schedule, _ = init_eclipse_problem_ad('examples/SPE9/SPE9_CP.DATA')
    control_id = int(schedule['step']['control'][0])
    forces = model.getDrivingForces(schedule['control'][control_id])
    model, s0 = model.updateForChangedControls(s0, forces)
    dt = float(schedule['step']['val'][0])

    linear_solver = AMGCL_CPRSolverBlockAD(
        blockSize=3,
        tolerance=1e-4,
        maxIterations=100,
        verbose=verbose,
        strategy=strategy,
        decoupling=decoupling,
        schurApproxType='full',
    )
    linear_records = []
    original_solve = linear_solver.solveLinearProblem

    def traced_solve(problem, model_arg=None):
        dx, res, rep = original_solve(problem, model_arg)
        if isinstance(rep, dict):
            linear_records.append(dict(rep))
        return dx, res, rep

    linear_solver.solveLinearProblem = traced_solve
    solver = NonLinearSolver(
        linearSolver=linear_solver,
        maxIterations=15,
        errorOnFailure=False,
        verbose=verbose,
    )
    t0 = time.time()
    state, report, _ = solver.solveTimestep(
        deepcopy(s0), dt, model,
        drivingForces=forces,
        initialGuess=deepcopy(s0),
        controlId=control_id,
    )
    wall = time.time() - t0

    iters = []
    res_max = []
    for nr in report.get('NonlinearReport', []):
        iters.append(int(nr.get('Iterations', 0)))
        r = nr.get('Residuals')
        if r is not None:
            res_max.append(float(np.abs(np.asarray(r, dtype=float)).max()))
    lin = {}
    if linear_records:
        lin = {
            'lin_count': len(linear_records),
            'lin_iters_avg': float(np.mean([float(r.get('Iterations', 0)) for r in linear_records])),
            'lin_solver_avg': float(np.mean([float(r.get('SolverTime', 0)) for r in linear_records])),
            'kernel_avg': float(np.mean([float(r.get('PreconditionerReport', {}).get('KernelTime', 0))
                                         for r in linear_records])),
            'use_drs': linear_records[-1].get('PreconditionerReport', {}).get('AMGCLUseDRS'),
            'mrst_weighted': linear_records[-1].get('PreconditionerReport', {}).get('MRSTRowWeighted'),
            'trueimpes': linear_records[-1].get('PreconditionerReport', {}).get('TrueIMPESWeighted'),
        }
    return {
        'converged': bool(report.get('Converged', False)),
        'wall': wall,
        'newton_iters': iters,
        'final_residual': max(res_max) if res_max else float('nan'),
        'p_range': (float(state['pressure'].min()), float(state['pressure'].max())),
        **lin,
    }


def main():
    print('cells=9000 first-step AMGCL block CPR strategy sweep', flush=True)
    rows = []
    for strategy in STRATEGIES:
        for decoupling in DECOUPLINGS:
            try:
                r = run_case(strategy, decoupling)
            except Exception as exc:
                r = {'converged': False, 'error': str(exc)}
            rows.append((strategy, decoupling, r))
            print(
                f'{strategy:9s} {decoupling:10s} conv={r.get("converged")} '
                f'wall={r.get("wall", float("nan")):.2f}s '
                f'newton={r.get("newton_iters")} '
                f'lin=(n={r.get("lin_count", "-")}, '
                f'avg_iters={r.get("lin_iters_avg", float("nan")):.1f}, '
                f'avg_solver={r.get("lin_solver_avg", float("nan")):.2f}s, '
                f'kernel={r.get("kernel_avg", float("nan")):.2f}s) '
                f'drs={r.get("use_drs")} mrst_w={r.get("mrst_weighted")} '
                f'trueimp={r.get("trueimpes")} '
                f'p=({r.get("p_range", (0, 0))[0]:.6g},{r.get("p_range", (0, 0))[1]:.6g})',
                flush=True,
            )
    print('\n--- summary ---', flush=True)
    for strategy, decoupling, r in rows:
        if r.get('converged'):
            print(f'{strategy:9s} {decoupling:10s} OK  wall={r["wall"]:.2f}s newton={r["newton_iters"]}', flush=True)
        else:
            print(f'{strategy:9s} {decoupling:10s} FAIL {r.get("error", "")}', flush=True)


if __name__ == '__main__':
    main()
