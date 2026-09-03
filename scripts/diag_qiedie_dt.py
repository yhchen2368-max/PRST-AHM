"""Compare QIEDIE first residual at the report dt vs the tiny mini-step dt."""
import sys

sys.path.insert(0, '.')

import numpy as np

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad

DECK = 'examples/HM/QIEDIE.DATA'

state0, model, schedule, _ = init_eclipse_problem_ad(DECK, RemoveZeroPoreVolume=True)
nc = len(state0['pressure'])
control_id = int(schedule['step']['control'][0])
forces = model.getDrivingForces(schedule['control'][control_id])
model, state0 = model.updateForChangedControls(state0, forces)

dt_report = float(schedule['step']['val'][0])
print('report dt = %.6g s (%.4g d)' % (dt_report, dt_report / 86400))

# The solver's first mini-step dt (what the selector produced in the run):
# replicate the IterationCountTimeStepSelector ramp-up.
from PRSTCore.ad_core.solvers.nonlinear_solver import IterationCountTimeStepSelector
sel = IterationCountTimeStepSelector(targetIterationCount=8,
                                     firstRampupStepRelative=0.1,
                                     firstRampupStep=86400.0)
dt_mini = sel.firstRampupStep  # attribute? fall back to computing
print('selector firstRampupStep attr: %r' % getattr(sel, 'firstRampupStep', None))

for dt in (dt_report, 1181.25, 60480.0, 86400.0):
    model2, st = model.prepareReportstep(state0, state0, dt, forces)
    problem, _ = model2.get_equations(state0, st, dt, forces)
    r = np.asarray(problem['Residuals'], dtype=float).ravel()
    print('\ndt=%.6g (%.4g d):' % (dt, dt / 86400))
    print('  raw |r| max = %.6g  (water %.6g / oil %.6g / gas %.6g)'
          % (np.max(np.abs(r)), np.max(np.abs(r[:nc])),
             np.max(np.abs(r[nc:2 * nc])), np.max(np.abs(r[2 * nc:3 * nc]))))
    try:
        vals, tol, names = model2.getConvergenceValues(problem)
        vals = np.asarray(vals, dtype=float).ravel()
        print('  convergence values: ' +
              ', '.join('%s=%.6g' % (names[i], vals[i]) for i in range(vals.size)))
    except Exception as exc:
        print('  getConvergenceValues error:', exc)
