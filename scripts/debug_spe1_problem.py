"""Deep debug of SPE1 problem structure and Jacobian extraction."""
import sys
sys.path.insert(0, '.')
import numpy as np
from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad

print('='*60)
print('Loading SPE1 and assembling first step...')
print('='*60)

state0, model, schedule, solver = init_eclipse_problem_ad('examples/SpE1/BENCH_SPE1.DATA')

dt = schedule['step']['val'][0]
forces = model.getDrivingForces(schedule['control'][0])

print(f'\ndt = {dt:.2e} seconds')
print(f'forces type: {type(forces)}')
print(f'forces keys: {list(forces.keys()) if isinstance(forces, dict) else "N/A"}')

# Get equations
print('\n' + '='*60)
print('Assembling equations with get_equations()...')
print('='*60)

problem, state = model.get_equations(state0, state0.copy(), dt, forces, iteration=0)

print(f'\nproblem type: {type(problem)}')
print(f'problem keys: {list(problem.keys())}')

# Detailed inspection of each key
for key in sorted(problem.keys()):
    val = problem[key]
    print(f'\n{key}:')
    print(f'  type: {type(val)}')
    if isinstance(val, dict):
        print(f'  dict keys: {list(val.keys())[:5]}...' if len(val.keys()) > 5 else f'  dict keys: {list(val.keys())}')
    elif isinstance(val, list):
        print(f'  list length: {len(val)}')
        if len(val) > 0:
            print(f'  first element type: {type(val[0])}')
    elif isinstance(val, np.ndarray):
        print(f'  shape: {val.shape}')
        print(f'  dtype: {val.dtype}')
        print(f'  nnz: {np.count_nonzero(val) if val.dtype in [np.float32, np.float64, int] else "N/A"}')
    elif hasattr(val, 'shape'):
        print(f'  shape: {val.shape}')
        print(f'  dtype: {val.dtype if hasattr(val, "dtype") else "unknown"}')
        if hasattr(val, 'nnz'):
            print(f'  nnz: {val.nnz}')
        print(f'  density: {val.nnz / val.shape[0] / val.shape[1] * 100 if hasattr(val, "nnz") and len(val.shape) == 2 else "N/A"}%')

# Special inspection of Jacobian-like entries
print('\n' + '='*60)
print('Looking for Jacobian matrices...')
print('='*60)

for key in ['jacobian', 'Jacobian', 'jac', 'Jac', 'A', 'system_jacobian']:
    if key in problem:
        val = problem[key]
        print(f'\nFound {key}:')
        print(f'  type: {type(val)}')
        if hasattr(val, 'shape'):
            print(f'  shape: {val.shape}')
        if hasattr(val, 'nnz'):
            print(f'  nnz: {val.nnz}')

# Check if there are AD (AutoDiff) objects that need to be converted
print('\n' + '='*60)
print('Looking for AD objects...')
print('='*60)

from PRSTCore.core.utils.ad import ADI

for key in list(problem.keys())[:10]:
    val = problem[key]
    if isinstance(val, ADI):
        print(f'\n{key} is ADI object')
        print(f'  value: {val.val.shape if hasattr(val.val, "shape") else "scalar"}')
        print(f'  jacs: {len(val.jac)} matrices')
        for i, jac in enumerate(val.jac[:3]):
            if hasattr(jac, 'shape'):
                print(f'    jac[{i}]: shape={jac.shape}, nnz={jac.nnz if hasattr(jac, "nnz") else "N/A"}')

print('\n✓ Debug inspection complete')
