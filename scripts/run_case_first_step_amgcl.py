"""Run one report step of an Eclipse case with AMGCL CPR."""
import sys
from copy import deepcopy
sys.path.insert(0, '.')
from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.solvers import NonLinearSolver, AMGCL_CPRSolverAD

cases = {
    'EGG': 'examples/EGG/Egg_Model_ECL.DATA',
    'NORNE': 'examples/Norne/Norne_simplified/NORNE_ATW2013.DATA',
}
name = sys.argv[1].upper()
path = cases[name]
s0, model, schedule, _ = init_eclipse_problem_ad(path)
ctrl = int(schedule['step']['control'][0])
forces = model.getDrivingForces(schedule['control'][ctrl])
dt = float(schedule['step']['val'][0])
print(f'{name} cells={len(s0["pressure"])} steps={len(schedule["step"]["val"])} dt={dt} gas={model.gas}')
ls = AMGCL_CPRSolverAD(tolerance=1e-3, maxIterations=100, verbose=False)
nls = NonLinearSolver(linearSolver=ls, maxIterations=5, maxTimestepCuts=0, errorOnFailure=False)
state, report, minis = nls.solveTimestep(deepcopy(s0), dt, model, drivingForces=forces, initialGuess=deepcopy(s0), controlId=ctrl)
print('converged=', report.get('Converged'), 'failure=', report.get('Failure'), 'iterations=', report.get('Iterations'))
print('pressure_range=', float(state['pressure'].min()), float(state['pressure'].max()))
if 'sW' in state: print('sW_range=', float(state['sW'].min()), float(state['sW'].max()))
if 'sG' in state: print('sG_range=', float(state['sG'].min()), float(state['sG'].max()))
