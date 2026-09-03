"""Inspect SPE9 first Newton residual and Jacobian structure."""
import sys
sys.path.insert(0, '.')
from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
import numpy as np

s0, model, schedule, _ = init_eclipse_problem_ad('examples/SPE9/SPE9_CP.DATA')
ctrl = int(schedule['step']['control'][0])
forces = model.getDrivingForces(schedule['control'][ctrl])
dt = float(schedule['step']['val'][0])
problem, state = model.get_equations(s0, s0, dt, forces)

print('System shape:', problem['Jacobian'].shape, 'res size:', problem['Residuals'].size)
print('facility_qs:', state.get('facility_qs', [])[:6])
print('facility_bhp:', state.get('facility_bhp', [])[:3])
print('wellSol qWs sample:', [w['qWs'] for w in problem['wellSol'][:3]])
print('Reservoir residuals [0,9000,18000]:', [float(problem['Residuals'][i]) for i in [0,9000,18000]])
print('Facility residuals [27000,27001,27002,27103]:', [float(problem['Residuals'][i]) for i in [27000,27001,27002,27103]])
print('Residual max:', float(np.abs(problem['Residuals']).max()))
