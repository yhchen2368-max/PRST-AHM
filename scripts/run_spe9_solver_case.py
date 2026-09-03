"""Run the SPE9 simulation with a single named linear solver and save timing
and pressure-state results to disk, so results from solvers that live in
different Python environments (e.g. MUMPS in a conda env with python-mumps,
AMGCL-CPR in the main env with the compiled pyamgcl extension) can be
compared afterwards without needing both in one interpreter.

Usage:
    python scripts/run_spe9_solver_case.py amgcl_cpr
    python scripts/run_spe9_solver_case.py mumps
"""
import os
import pickle
import sys
import time

sys.path.insert(0, '.')

import numpy as np

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.simulators.simulate_schedule_ad import simulate_schedule_ad
from PRSTCore.ad_core.solvers import NonLinearSolver


def build_solver(name):
    if name == 'amgcl_cpr':
        from PRSTCore.ad_core.solvers import AMGCL_CPRSolverAD
        return AMGCL_CPRSolverAD(tolerance=1e-3, maxIterations=100, verbose=True)
    if name == 'mumps':
        from PRSTCore.ad_core.solvers import MUMPSSolverAD, check_mumps
        if not check_mumps():
            raise RuntimeError(
                'python-mumps is not importable in this interpreter. Run this '
                'script with the conda env that has python-mumps + mumps-seq installed.'
            )
        return MUMPSSolverAD(tolerance=1e-3, verbose=True)
    raise ValueError(f'Unknown solver name: {name!r} (expected amgcl_cpr or mumps)')


def main():
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    name = sys.argv[1]
    deck_path = sys.argv[2] if len(sys.argv) == 3 else 'examples/SPE9/SPE9_CP.DATA'
    linear_solver = build_solver(name)

    print(f'--- SPE9 init ({name}, {deck_path}) ---')
    t0 = time.time()
    s0, model, schedule, _ = init_eclipse_problem_ad(deck_path)
    init_time = time.time() - t0
    print(f'cells={len(s0["pressure"])} steps={len(schedule["step"]["val"])} '
          f'gas={model.gas} init_time={init_time:.1f}s')

    nonlinear_solver = NonLinearSolver(linearSolver=linear_solver, verbose=True, maxIterations=10)

    print(f'--- SPE9 simulate ({name}) ---')
    result = {'name': name, 'init_time': init_time, 'python': sys.version}
    t0 = time.time()
    try:
        wsol, states, report = simulate_schedule_ad(
            s0, model, schedule,
            NonLinearSolver=nonlinear_solver,
            return_report=True,
        )
        sim_time = time.time() - t0
        pressures = [np.asarray(s['pressure'], dtype=float) for s in states if s is not None]
        p_vals = [float(p[0]) for p in pressures]
        result.update(
            ok=True,
            sim_time=sim_time,
            nstates=len(states),
            nsteps=report.get('NumControlSteps', len(states)),
            converged=bool(report.get('Converged', True)),
            p_min=min(p_vals) if p_vals else float('nan'),
            p_max=max(p_vals) if p_vals else float('nan'),
            p_last=p_vals[-1] if p_vals else float('nan'),
            pressures=pressures,
        )
        print(f'OK nstates={len(states)} nsteps={result["nsteps"]} '
              f'converged={result["converged"]} sim_time={sim_time:.1f}s')
        print(f'  p_min={result["p_min"]:.4e} p_max={result["p_max"]:.4e}')
    except Exception as e:
        sim_time = time.time() - t0
        result.update(ok=False, sim_time=sim_time, error=f'{type(e).__name__}: {str(e)[:300]}')
        print(f'FAIL {result["error"]}')
        import traceback
        traceback.print_exc()

    suffix = '' if deck_path == 'examples/SPE9/SPE9_CP.DATA' else '_' + os.path.splitext(os.path.basename(deck_path))[0].lower()
    out_path = f'scripts/spe9_solver_result_{name}{suffix}.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(result, f)
    print(f'Saved results to {out_path}')


if __name__ == '__main__':
    main()
