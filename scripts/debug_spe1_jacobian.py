#!/usr/bin/env python
"""
Debug SPE1 Jacobian extraction.
Check AD object types and proper matrix construction.
"""
import sys
sys.path.insert(0, 'c:\\Users\\junji\\Desktop\\github\\Cgnet')

import numpy as np
from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad

# Load SPE1
print('Loading SPE1...')
state0, model, schedule, _ = init_eclipse_problem_ad(
    'examples/SpE1/BENCH_SPE1.DATA'
)

G = model.G

# Get first timestep
dt = schedule['step']['val'][0]
forces = model.getDrivingForces(schedule['control'][0])

print(f'dt = {dt:.6e} s')
print(f'Model type: {type(model).__name__}')

# Get equations
print('\nAssembling equations...')
problem, state = model.get_equations(state0, state0.copy(), dt, forces, iteration=0)

print(f'\nProblem dict keys: {list(problem.keys())}')
print(f'Problem Jacobian type: {type(problem["Jacobian"]).__name__}')

# Deep inspection of Jacobian
jac = problem['Jacobian']
print(f'Jacobian shape: {jac.shape}')

# Check if it's an AD object
if hasattr(jac, 'jac'):
    print(f'AD jacobian attribute exists: {type(jac.jac).__name__}')
    if hasattr(jac.jac, '__len__'):
        print(f'AD jacobian length: {len(jac.jac)}')
        if len(jac.jac) > 0:
            print(f'AD jacobian[0] type: {type(jac.jac[0]).__name__}')

# Try different extraction methods
print('\n========== Jacobian Extraction Methods ==========')

# Method 1: Direct conversion
try:
    jac_dense1 = jac.toarray() if hasattr(jac, 'toarray') else np.array(jac.todense())
    print(f'Method 1 (toarray/todense): shape={jac_dense1.shape}, nnz={np.count_nonzero(jac_dense1)}')
except Exception as e:
    print(f'Method 1 failed: {e}')

# Method 2: AD evaluation via val
try:
    if hasattr(jac, 'val'):
        jac_val = jac.val
        print(f'Method 2 (jac.val): type={type(jac_val).__name__}, shape={jac_val.shape if hasattr(jac_val, "shape") else "scalar"}')
        if hasattr(jac_val, 'toarray'):
            jac_dense2 = jac_val.toarray()
            print(f'  -> toarray: shape={jac_dense2.shape}, nnz={np.count_nonzero(jac_dense2)}')
except Exception as e:
    print(f'Method 2 failed: {e}')

# Method 3: Check residuals
print('\n========== Residuals ==========')
residuals = problem.get('Residuals')
if residuals is not None:
    if isinstance(residuals, dict):
        print(f'Residuals (dict): {list(residuals.keys())}')
        for key, val in residuals.items():
            if hasattr(val, 'val'):
                print(f'  {key}: shape={val.val.shape}')
    else:
        print(f'Residuals type: {type(residuals).__name__}, shape={residuals.shape if hasattr(residuals, "shape") else "N/A"}')

# Method 4: Check jacobian structure
print('\n========== Jacobian Structure ==========')
print(f'Jacobian dtype: {jac.dtype}')
print(f'Jacobian format: {jac.format if hasattr(jac, "format") else "N/A"}')

# Check if jacobian is a block structure
if hasattr(jac, 'blocks'):
    print(f'Has blocks attribute: {len(jac.blocks)} blocks')

# Try manual block assembly if needed
print('\n========== Checking Model Equations ==========')
print(f'Number of grid equations: 3 * {G["cells"]["num"]}')
print(f'Expected total equations: 3*{G["cells"]["num"]} + well equations')

# Verify state variables
print(f'\nState keys: {list(state.keys())}')
for key in ['pressure', 'sW', 'sG']:
    if key in state:
        val = state[key]
        print(f'  {key}: shape={val.shape}, type={type(val).__name__}')
