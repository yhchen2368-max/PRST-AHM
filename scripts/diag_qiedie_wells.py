"""QIEDIE well-model probe: PI interpretation and implied BHP at the target rates."""
import sys

sys.path.insert(0, '.')

import numpy as np

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad

DECK = 'examples/HM/QIEDIE.DATA'
state0, model, schedule, _ = init_eclipse_problem_ad(DECK, RemoveZeroPoreVolume=True)
control_id = int(schedule['step']['control'][0])
forces = model.getDrivingForces(schedule['control'][control_id])
model, state0 = model.updateForChangedControls(state0, forces)
model, state0 = model.prepareReportstep(state0, state0, 604800.0, forces)

wells = forces.get('W', []) if isinstance(forces, dict) else []
print('wells in first control:', len(wells))
p0 = np.asarray(state0['pressure'], dtype=float)
for w in wells:
    name = w.get('name')
    typ = w.get('type')          # control type (rate/bhp/...)
    val = w.get('val')
    sign = w.get('sign', 1)
    comp = w.get('compi', [])
    wc = w.get('wc', [])  # connections? maybe 'cells'/'WI'
    cells = np.asarray(w.get('cells', []), dtype=int)
    WI = np.asarray(w.get('WI', []), dtype=float)
    print('\nwell %s type=%s val=%r sign=%d compi=%s' % (name, typ, val, sign, comp))
    if cells.size:
        print('  nconn=%d, p at conn: %.6g .. %.6g bar' % (
            cells.size, p0[cells].min()/1e5, p0[cells].max()/1e5))
    if WI.size:
        print('  WI (well index) range: %.6g .. %.6g  (sum=%.6g)'
              % (WI.min(), WI.max(), WI.sum()))
        print('  PI total (SI) = %.6g' % WI.sum())
        if cells.size:
            avgp = float(p0[cells].mean())
            # q = PI * (p_res - BHP) => BHP needed for target val (m3/s)
            # target rate for water injector: q_surface; reservoir q = q_surface (water)
            bhp_needed = avgp - float(val) / max(WI.sum(), 1e-30)
            print('  implied BHP for val=%.6g: %.6g Pa (%.6g bar)'
                  % (val, bhp_needed, bhp_needed/1e5))
