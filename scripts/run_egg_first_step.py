"""Run and diagnose only the first EGG (oil-water) report step.

Ad-hoc regression driver for the GenericBlackOilModel._update_state_mrst_generic_ow
/_check_convergence_mrst_generic_ow branch (2-phase MRST-generic path), which
has no automated pytest coverage. Mirrors run_spe9_first_step.py's structure.
"""
import sys
from copy import deepcopy
sys.path.insert(0, '.')

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad


s0, model, schedule, solver = init_eclipse_problem_ad('examples/EGG/Egg_Model_ECL.DATA')
control_id = int(schedule['step']['control'][0])
forces = model.getDrivingForces(schedule['control'][control_id])
model, s0 = model.updateForChangedControls(s0, forces)
dt = float(schedule['step']['val'][0])

print(f'cells={len(s0["pressure"])} dt={dt} gas={model.gas} mrst_generic={model._use_mrst_generic_assembly}')
solver.verbose = True
solver.errorOnFailure = False
state, report, ministates = solver.solveTimestep(
    deepcopy(s0), dt, model,
    drivingForces=forces,
    initialGuess=deepcopy(s0),
    controlId=control_id,
)
print('Converged:', report.get('Converged'))
print('Failure:', report.get('Failure'), report.get('FailureMsg'))
print('Reports:', len(report.get('NonlinearReport', [])))
print('p range:', float(state['pressure'].min()), float(state['pressure'].max()))
print('sW range:', float(state['sW'].min()), float(state['sW'].max()))
problem, _ = model.get_equations(s0, state, dt, forces)
res = problem['Residuals']
nc = len(s0['pressure'])
print('final residual max:', float(abs(res).max()))
print('water residual max:', float(abs(res[:nc]).max()))
print('oil residual max:', float(abs(res[nc:2*nc]).max()))
