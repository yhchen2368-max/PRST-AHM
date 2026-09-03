"""Diagnose why reservoir residuals are large after facility coupling."""
import sys
import numpy as np
sys.path.insert(0, '.')

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad

s0, model, schedule, _ = init_eclipse_problem_ad('examples/SPE9/SPE9_CP.DATA')
ctrl = int(schedule['step']['control'][0])
forces = model.getDrivingForces(schedule['control'][ctrl])
dt = float(schedule['step']['val'][0])

# Get problem before and after facility augmentation
model_nofac = model
model_nofac.enable_facility_unknowns = False
problem_nofac, _ = model_nofac.get_equations(s0, s0, dt, forces)

model.enable_facility_unknowns = True
problem_fac, state_fac = model.get_equations(s0, s0, dt, forces)

nc = len(s0['pressure'])
print(f'Reservoir residuals WITHOUT facility coupling:')
print(f'  water: max={np.max(np.abs(problem_nofac["Residuals"][:nc])):.3e}')
print(f'  oil:   max={np.max(np.abs(problem_nofac["Residuals"][nc:2*nc])):.3e}')
print(f'  gas:   max={np.max(np.abs(problem_nofac["Residuals"][2*nc:3*nc])):.3e}')

print(f'\nReservoir residuals WITH facility coupling:')
print(f'  water: max={np.max(np.abs(problem_fac["Residuals"][:nc])):.3e}')
print(f'  oil:   max={np.max(np.abs(problem_fac["Residuals"][nc:2*nc])):.3e}')
print(f'  gas:   max={np.max(np.abs(problem_fac["Residuals"][2*nc:3*nc])):.3e}')

print(f'\nFacility unknowns:')
print(f'  facility_qs: {state_fac.get("facility_qs", [])[:6]}')
print(f'  facility_bhp: {state_fac.get("facility_bhp", [])[:3]}')

print(f'\nWell struct sample:')
wells = [w for w in forces.get('W', []) if w.get('status')]
for w in wells[:2]:
    print(f'  {w.get("name")}: sign={w.get("sign")} val={w.get("val")} cells={len(w.get("cells", []))}')

print(f'\nWellSol sample:')
for ws in problem_fac['wellSol'][:2]:
    print(f'  {ws.get("name")}: qWs={ws.get("qWs"):.1f} qOs={ws.get("qOs"):.1f}')
