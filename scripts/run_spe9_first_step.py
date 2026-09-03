"""Run and diagnose only the first SPE9 report step."""
import sys
from copy import deepcopy
sys.path.insert(0, '.')

import numpy as np
from pathlib import Path
from scipy.io import loadmat

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad


def _field(obj, name, default=None):
    return getattr(obj, name, default)


def _load_mrst_trace(path):
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    report = data['report']
    reference = []
    for step_report in np.ravel(report.StepReports):
        nonlinear_reports = list(np.ravel(step_report.NonlinearReport))
        residuals = [
            np.asarray(_field(nr, 'Residuals', []), dtype=float).ravel()
            for nr in nonlinear_reports
        ]
        linear_time = []
        for nr in nonlinear_reports:
            if not bool(_field(nr, 'Solved', False)):
                continue
            linear = _field(nr, 'LinearSolver', None)
            linear_time.append(float(_field(linear, 'SolverTime', np.nan)))
        reference.append({
            'dt': float(step_report.Timestep),
            'iterations': int(step_report.Iterations),
            'converged': bool(step_report.Converged),
            'residuals': residuals,
            'linear_time': linear_time,
        })
    return reference


s0, model, schedule, solver = init_eclipse_problem_ad('examples/SPE9/SPE9_CP.DATA')
control_id = int(schedule['step']['control'][0])
forces = model.getDrivingForces(schedule['control'][control_id])
model, s0 = model.updateForChangedControls(s0, forces)
dt = float(schedule['step']['val'][0])

print(f'cells={len(s0["pressure"])} dt={dt} gas={model.gas}')
solver.verbose = True
solver.errorOnFailure = False
reference_path = Path('mrst_spe9_first_step.mat')
if reference_path.is_file():
    mrst_trace = _load_mrst_trace(reference_path)
    solver.setIterationTraceReference(mrst_trace)
    print(f'Loaded MRST trace: {reference_path} ministeps={len(mrst_trace)}')
else:
    print(f'MRST trace not found: {reference_path}')
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
print('sG range:', float(state['sG'].min()), float(state['sG'].max()))
problem, _ = model.get_equations(s0, state, dt, forces)
res = problem['Residuals']
nc = len(s0['pressure'])
print('final residual max:', float(abs(res).max()))
print('water residual max:', float(abs(res[:nc]).max()))
print('oil residual max:', float(abs(res[nc:2*nc]).max()))
print('gas residual max:', float(abs(res[2*nc:]).max()))
for i, r in enumerate(report.get('NonlinearReport', [])):
    print(i, r.get('Converged'), r.get('Iterations'), r.get('Residuals'))
