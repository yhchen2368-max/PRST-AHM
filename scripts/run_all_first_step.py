"""Run the first report step of every bundled Eclipse deck.

The solver is the one returned by ``init_eclipse_problem_ad``.  This is
intentional: it is the Python port of MRST ``initEclipseProblemAD``'s
deck/TUNING defaults, not a diagnostic AMGCL/CPR override.
"""
import sys
import time
from copy import deepcopy

sys.path.insert(0, '.')
sys.setrecursionlimit(10000)

import numpy as np

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad


CASES = {
    'SPE1': 'examples/SpE1/BENCH_SPE1.DATA',
    'SPE9': 'examples/SPE9/SPE9_CP.DATA',
    'EGG': 'examples/EGG/Egg_Model_ECL.DATA',
    'NORNE': 'examples/Norne/Norne_simplified/NORNE_ATW2013.DATA',
}


for name, path in CASES.items():
    print(f'\n=== {name} ===')
    try:
        start = time.perf_counter()
        state0, model, schedule, solver = init_eclipse_problem_ad(path)
        control_id = int(schedule['step']['control'][0])
        forces = model.getDrivingForces(schedule['control'][control_id])
        dt = float(schedule['step']['val'][0])
        print(
            f'cells={len(state0["pressure"])} dt={dt:g} gas={model.gas} '
            f'disgas={model.disgas} vapoil={model.vapoil} '
            f'wells={len(forces.get("W", []))} '
            f'nls=({solver.maxIterations}, cuts={solver.maxTimestepCuts})'
        )
        state, report, _ = solver.solveTimestep(
            deepcopy(state0), dt, model,
            drivingForces=forces,
            initialGuess=deepcopy(state0),
            controlId=control_id,
        )
        print(
            f'converged={report["Converged"]} iterations={report["Iterations"]} '
            f'ministeps={report["AcceptedMinisteps"]} '
            f'elapsed={time.perf_counter() - start:.1f}s'
        )
        print(
            f'p_range=[{state["pressure"].min():.7g}, '
            f'{state["pressure"].max():.7g}]'
        )
    except Exception as exc:
        print(f'FAIL {type(exc).__name__}: {exc}')
