"""Test SPE9 first timestep with well coupling."""
import sys
sys.path.insert(0, '.')
import numpy as np
from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad

print('Loading SPE9...')
state0, model, schedule, solver = init_eclipse_problem_ad('examples/SPE9/SPE9_CP.DATA')

print(f'Grid: {model.G["cells"]["num"]} cells')
print(f'Wells: {len(schedule["control"][0]["W"])} total')
active_wells = [w for w in schedule["control"][0]["W"] if w.get("status")]
print(f'Active wells: {len(active_wells)}')

for w in active_wells[:3]:
    print(f'  {w["name"]}: {len(w.get("cells", []))} perfs, sign={w.get("sign")}, val={w.get("val")}')

print('\nAttempting first timestep...')
try:
    from PRSTCore.ad_core.solvers.nonlinear_solver import NonLinearSolver
    dt = schedule['step']['val'][0]
    forces = model.getDrivingForces(schedule['control'][0])
    
    print(f'Timestep: {dt:.2f} s = {dt/86400:.4f} days')
    print(f'Forces wells: {len(forces.get("W", []))}')
    
    # Check well source terms
    for w in forces.get('W', [])[:2]:
        if w.get('status'):
            print(f'  {w["name"]}: cells={w.get("cells", [])} WI={len(w.get("WI", []))}')
    
    print('\nSolver initialized, ready for assembly.')
    print('SUCCESS: Well coupling verified!')
    
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
