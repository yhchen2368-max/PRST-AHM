"""Detailed inspection of sparse Jacobian and residuals."""
import sys
sys.path.insert(0, '.')
import numpy as np
import scipy.sparse as sp
from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad

print('Loading SPE1...')
state0, model, schedule, solver = init_eclipse_problem_ad('examples/SpE1/BENCH_SPE1.DATA')

dt = schedule['step']['val'][0]
forces = model.getDrivingForces(schedule['control'][0])

problem, state = model.get_equations(state0, state0.copy(), dt, forces, iteration=0)

jac = problem['Jacobian']
residuals = problem['Residuals']

print(f'\n=== Jacobian Analysis ===')
print(f'Shape: {jac.shape}')
print(f'NNZ: {jac.nnz}')

# Get dense form to inspect
jac_dense = jac.toarray()
print(f'\nNon-zero elements:')
rows, cols = jac_dense.nonzero()
for i, (r, c) in enumerate(zip(rows, cols)):
    print(f'  [{r:3d}, {c:3d}] = {jac_dense[r, c]:12.6e}')

print(f'\n=== Residuals Analysis ===')
print(f'Shape: {residuals.shape}')
print(f'NNZ: {np.count_nonzero(residuals)}')
print(f'Min: {residuals.min():.6e}')
print(f'Max: {residuals.max():.6e}')
print(f'Norm: {np.linalg.norm(residuals):.6e}')

# Find non-zero residuals
nz_idx = np.nonzero(residuals)[0]
print(f'\nNon-zero residuals at indices: {nz_idx}')
for idx in nz_idx:
    print(f'  [{idx}] = {residuals[idx]:.6e}')

print(f'\n=== Equation names ===')
eq_names = problem['equationNames']
print(f'Total equations: {len(eq_names)}')
print(f'Equation types: {set(problem["types"])}')

# Map residuals to equation names
print(f'\nNon-zero residuals with equation names:')
for idx in nz_idx:
    print(f'  Eq {idx}: {eq_names[idx]} (type={problem["types"][idx]}) = {residuals[idx]:.6e}')

# Check well solution
print(f'\n=== Well Solutions ===')
for i, wsol in enumerate(problem['wellSol']):
    print(f'Well {i}:')
    for key in ['name', 'qWs', 'qOs', 'qGs', 'bhp', 'status']:
        if key in wsol:
            val = wsol[key]
            if isinstance(val, (int, float)):
                print(f'  {key}: {val}')
            else:
                print(f'  {key}: {type(val)}')

# Check facility primary variables
print(f'\n=== Facility Primary Variables ===')
print(f'Variables: {problem["facilityPrimaryVariables"]}')
print(f'Number of vars: {len(problem["facilityPrimaryVariables"])}')

# Try to understand the block structure
print(f'\n=== Block Structure Analysis ===')
print(f'300 cells → 900 reservoir equations (3 phases)')
print(f'2 wells × 4 vars each → 8 facility equations')
print(f'Total: 908 equations')

# Check which rows/cols have non-zeros
print(f'\nRows with non-zeros: {set(rows)}')
print(f'Cols with non-zeros: {set(cols)}')

print(f'\nResidual structure: first 20 values')
print(residuals[:20])

# Check if problem is in correct units (seconds vs days)
print(f'\n=== Unit Check ===')
print(f'dt in problem: {problem["dt"]}')
print(f'Expected in seconds: 1.0')
print(f'If in days, would be: {problem["dt"]/86400:.6e}')

# Print first pressure to check if state makes sense
print(f'\n=== State Check ===')
print(f'First pressure: {state0["pressure"][0]:.6e} Pa')
print(f'All pressures equal? {np.all(state0["pressure"] == state0["pressure"][0])}')
print(f'State keys: {list(state0.keys())}')
