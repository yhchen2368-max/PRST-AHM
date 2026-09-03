"""QIEDIE initial-state diagnostic: why is the first residual ~1e23 (CNV_O)?"""
import sys
import time

sys.path.insert(0, '.')

import numpy as np

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad

DECK = 'examples/HM/QIEDIE.DATA'

t0 = time.time()
state0, model, schedule, _ = init_eclipse_problem_ad(DECK, RemoveZeroPoreVolume=True)
print('init: %.2f s, cells=%d, steps=%d, phases(w/o/g)=%s/%s/%s disgas=%s'
      % (time.time() - t0, len(state0['pressure']), len(schedule['step']['val']),
         bool(model.water), bool(model.oil), bool(model.gas),
         bool(getattr(model, 'disgas', False))))

nc = len(state0['pressure'])
p = np.asarray(state0['pressure'], dtype=float)
sW = np.asarray(state0['sW'], dtype=float)
sG = np.asarray(state0['sG'], dtype=float)
print('state:')
print('  p    range: %.6g .. %.6g Pa (%.4g .. %.4g bar)'
      % (p.min(), p.max(), p.min()/1e5, p.max()/1e5))
print('  sW   range: %.6g .. %.6g  (cells out of [0,1]: %d)'
      % (sW.min(), sW.max(), int(np.sum((sW < 0) | (sW > 1)))))
print('  sG   range: %.6g .. %.6g  (cells out of [0,1]: %d)'
      % (sG.min(), sG.max(), int(np.sum((sG < 0) | (sG > 1)))))
print('  sW+sG range: %.6g .. %.6g  (cells > 1+1e-9: %d)'
      % ((sW+sG).min(), (sW+sG).max(), int(np.sum(sW+sG > 1 + 1e-9))))
if 'rs' in state0:
    rs = np.asarray(state0['rs'], dtype=float)
    print('  rs   range: %.6g .. %.6g' % (rs.min(), rs.max()))
if 'rv' in state0:
    rv = np.asarray(state0['rv'], dtype=float)
    print('  rv   range: %.6g .. %.6g' % (rv.min(), rv.max()))

# PVT sanity at the initial state
try:
    pvt = model._phase_pvt(p)
    for k in ('bo', 'bw', 'bg', 'muo', 'muw', 'mug'):
        if k in pvt:
            a = np.asarray(pvt[k], dtype=float).ravel()
            print('pvt[%s] range: %.6g .. %.6g' % (k, a.min(), a.max()))
except Exception as exc:
    print('pvt eval error:', exc)

# pore volume sanity
pv = getattr(model, 'porevolume', None)
if pv is not None:
    pva = np.asarray(pv, dtype=float).ravel()
    print('porevolume range: %.6g .. %.6g  (<=0: %d)'
          % (pva.min(), pva.max(), int(np.sum(pva <= 0))))

# assemble the first system and look at the residual per phase
control_id = int(schedule['step']['control'][0])
forces = model.getDrivingForces(schedule['control'][control_id])
model, state0 = model.updateForChangedControls(state0, forces)
model, state0 = model.prepareReportstep(state0, state0, schedule['step']['val'][0], forces)
dt = float(schedule['step']['val'][0])
problem, _ = model.get_equations(state0, state0, dt, forces)
r = np.asarray(problem['Residuals'], dtype=float).ravel()
print('\nfirst system residual (n=%d):' % r.size)
print('  max |r| overall   : %.6g' % np.max(np.abs(r)))
for i, name in enumerate(('water', 'oil', 'gas')):
    if i * nc < r.size:
        seg = r[i * nc:min((i + 1) * nc, r.size)]
        print('  max |r| %-5s    : %.6g' % (name, np.max(np.abs(seg))))
    if (i + 1) * nc >= r.size:
        break
print('  well block max|r| : %.6g' % np.max(np.abs(r[3 * nc:])) if r.size > 3 * nc else '')
# where is the huge residual? top cells?
idx = int(np.argmax(np.abs(r[:nc])))
print('  argmax|r| (oil) at cell %d, depth-ish z from grid' % idx)
G = getattr(model, 'G', {})
if isinstance(G, dict) and G.get('cells', {}).get('centroids') is not None:
    cz = np.asarray(G['cells']['centroids'], dtype=float)[:, 2]
    print('  z at that cell: %.3f m' % cz[idx])
