"""QIEDIE: locate the CNV blow-up cell and the normalization that explodes."""
import sys
import time

sys.path.insert(0, '.')

import numpy as np

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad

DECK = 'examples/HM/QIEDIE.DATA'

state0, model, schedule, _ = init_eclipse_problem_ad(DECK, RemoveZeroPoreVolume=True)
nc = len(state0['pressure'])
control_id = int(schedule['step']['control'][0])
forces = model.getDrivingForces(schedule['control'][control_id])
model, state0 = model.updateForChangedControls(state0, forces)
dt = float(schedule['step']['val'][0])
print('schedule first dt = %r (val[0]=%r)  nsteps=%d'
      % (dt, schedule['step']['val'][0], len(schedule['step']['val'])))
model, state0 = model.prepareReportstep(state0, state0, dt, forces)
problem, _ = model.get_equations(state0, state0, dt, forces)

try:
    vals, tol, names = model.getConvergenceValues(problem)
    vals = np.asarray(vals, dtype=float).ravel()
    tol = np.asarray(tol, dtype=float).ravel()
    print('\ngetConvergenceValues:')
    for i in range(vals.size):
        print('  %-10s val=%.6g tol=%.6g norm=%.6g' % (names[i] if i < len(names) else '?', vals[i], tol[i], vals[i] / tol[i]))
except Exception as exc:
    print('getConvergenceValues error:', exc)
    # fall back: manually inspect the CNV normalizer via the model's state
    print('falling back to manual inspection')

r = np.asarray(problem['Residuals'], dtype=float).ravel()
# mass-balance residuals per phase (raw)
for i, name in enumerate(('water', 'oil', 'gas')):
    seg = r[i * nc:(i + 1) * nc]
    j = int(np.argmax(np.abs(seg)))
    print('raw %-5s max|r| = %.6g at cell %d' % (name, np.abs(seg[j]), j))

# what normalization does CNV use? Look at the model's convergence function source
import inspect
try:
    src = inspect.getsource(type(model).checkConvergence)
    print('\n--- checkConvergence source (first 1200 chars) ---')
    print(src[:1200])
except Exception as exc:
    print('no source:', exc)

# PV / dt denominators
pv = np.asarray(getattr(model, 'porevolume', np.ones(nc)), dtype=float).ravel()
print('\nporevolume min=%.6g max=%.6g' % (pv.min(), pv.max()))
# check a few cells near the argmax of each phase residual
for i, name in enumerate(('water', 'oil', 'gas')):
    seg = np.abs(r[i * nc:(i + 1) * nc])
    js = np.argsort(seg)[-3:][::-1]
    print('  %-5s top cells: %s |r|=%.3g,%.3g,%.3g  pv=%s'
          % (name, js, seg[js[0]], seg[js[1]], seg[js[2]],
             np.round(pv[js], 3)))
