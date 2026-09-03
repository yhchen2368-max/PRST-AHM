"""Check what PRSTCore builds for QIEDIE wells: limits, control types, and how
the well equation enforces rate vs BHP limit."""
import sys

sys.path.insert(0, '.')

import numpy as np

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad

DECK = 'examples/HM/QIEDIE.DATA'
state0, model, schedule, _ = init_eclipse_problem_ad(DECK, RemoveZeroPoreVolume=True)
control_id = int(schedule['step']['control'][0])
forces = model.getDrivingForces(schedule['control'][control_id])
wells = forces.get('W', []) if isinstance(forces, dict) else []
print('wells in first control:', len(wells))
for w in wells:
    name = w.get('name')
    print('\nwell %s: type=%s val=%r sign=%s' % (name, w.get('type'), w.get('val'), w.get('sign')))
    for key in ('lims', 'bhp', 'bhpLimit', 'status', 'compi', 'WI', 'cells', 'val'):
        if key in w:
            v = w[key]
            if isinstance(v, np.ndarray):
                print('  %s: array len=%d sum=%.6g' % (key, v.size, v.sum() if v.size else 0))
            else:
                print('  %s: %r' % (key, v))
# does the well structure carry limits?
import inspect
from PRSTCore.ad_core.models import well_model as wm
src = inspect.getsource(wm)
print('\nwell_model.py mentions lims:', src.count('lims'), ' bhpLimit:', src.count('bhpLimit'),
      ' maxRate:', src.count('maxRate'), ' minBHP:', src.count('minBHP'))
