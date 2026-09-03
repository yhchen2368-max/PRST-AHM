"""Break down SPE9 residuals by equation block after one Newton step."""
import sys
from copy import deepcopy
import numpy as np
sys.path.insert(0, '.')
sys.setrecursionlimit(10000)

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad

s0, model, schedule, _ = init_eclipse_problem_ad('examples/SPE9/SPE9_CP.DATA')
ctrl = int(schedule['step']['control'][0])
forces = model.getDrivingForces(schedule['control'][ctrl])
dt = float(schedule['step']['val'][0])

s = model.validateState(deepcopy(s0))
prob, s = model.get_equations(s0, s, dt, forces)

res = np.asarray(prob['Residuals'], dtype=float)
names = prob['equationNames']
types = prob['types']

nc = model._num_cells()
ngas = 3 if model.gas else 2

print(f'total residuals: {res.size}  (reservoir={nc*ngas}  facility={res.size - nc*ngas})')
print(f'res_w   max={np.max(np.abs(res[:nc])):.3e}')
print(f'res_o   max={np.max(np.abs(res[nc:2*nc])):.3e}')
if model.gas:
    print(f'res_g   max={np.max(np.abs(res[2*nc:3*nc])):.3e}')
well_res = res[nc*ngas:]
print(f'well closure+control: size={well_res.size}  max={np.max(np.abs(well_res)):.3e}  mean={np.mean(np.abs(well_res)):.3e}')
# Print per-well residual
wells_names = names[nc*ngas:]
wells_res = res[nc*ngas:]
for i, (n, r) in enumerate(zip(wells_names, wells_res)):
    if abs(r) > 1e-2:
        print(f'  [{i}] {n}: {r:.4e}')
