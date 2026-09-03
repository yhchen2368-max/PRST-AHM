"""Trace SPE9 first-step Newton residuals without changing solver logic."""
import sys
from copy import deepcopy
import numpy as np
sys.path.insert(0, '.')

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.solvers import NonLinearSolver

s0, model, schedule, _ = init_eclipse_problem_ad('examples/SPE9/SPE9_CP.DATA')
control_id = int(schedule['step']['control'][0])
forces = model.getDrivingForces(schedule['control'][control_id])
dt = float(schedule['step']['val'][0])
solver = NonLinearSolver(verbose=False, errorOnFailure=False, maxTimestepCuts=0, maxIterations=1)
state = deepcopy(s0)
state = model.validateState(state)

for it in range(1, 31):
    problem, state = model.get_equations(s0, state, dt, forces, iteration=it)
    res = np.asarray(problem['Residuals'], dtype=float)
    nc = len(s0['pressure'])
    norms = [float(np.max(np.abs(res[i*nc:(i+1)*nc]))) for i in range(3)]
    print(f'{it:02d} max={max(norms):.8e} water={norms[0]:.8e} oil={norms[1]:.8e} gas={norms[2]:.8e} sGmax={np.max(state["sG"]):.8e}')
    if max(norms) < model.nonlinearTolerance:
        break
    dx, _, report = solver.LinearSolver.solveLinearProblem(problem, model)
    if not np.all(np.isfinite(dx)):
        print('nonfinite dx')
        break
    state = model.updateState(state, problem, dx, forces)
