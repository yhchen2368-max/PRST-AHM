"""Test SPE1 first timestep and export Jacobian for comparison with MRST."""
import sys
sys.path.insert(0, '.')
import numpy as np
import scipy.io as sio
from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad

print('='*60)
print('Loading SPE1 model...')
print('='*60)
state0, model, schedule, solver = init_eclipse_problem_ad('examples/SpE1/BENCH_SPE1.DATA')

print(f'\nGrid: {model.G["cells"]["num"]} cells')
print(f'Active cells: {model.G["cells"]["num"]}')
# Check which phases are active
phases = []
if hasattr(model, 'water') and model.water:
    phases.append('water')
if hasattr(model, 'oil') and model.oil:
    phases.append('oil')
if hasattr(model, 'gas') and model.gas:
    phases.append('gas')
print(f'Phases: {", ".join(phases) if phases else "unknown"}')
print(f'Wells: {len(schedule["control"][0]["W"])} total')

active_wells = [w for w in schedule["control"][0]["W"] if w.get("status")]
print(f'Active wells: {len(active_wells)}')
for w in active_wells:
    cells = w.get('cells', [])
    wi = w.get('WI', [])
    print(f'  {w["name"]}: {len(cells)} perfs, sign={w.get("sign")}, type={w.get("type")}, val={w.get("val")}')
    if len(cells) > 0:
        print(f'    cells={cells[:5]}{"..." if len(cells) > 5 else ""}')

print('\n' + '='*60)
print('First timestep assembly...')
print('='*60)

try:
    dt = schedule['step']['val'][0]
    print(f'Timestep dt = {dt:.2f} s = {dt/86400:.4f} days')
    
    forces = model.getDrivingForces(schedule['control'][0])
    print(f'Driving forces: {len(forces.get("W", []))} wells')
    
    # Assemble equations for first step
    print('\nAssembling equations...')
    state = state0.copy()
    
    # Get equations and Jacobian (returns tuple: problem, state_updated)
    problem, state = model.get_equations(state0, state, dt, forces, iteration=0)
    
    print(f'\nEquations assembled:')
    # problem is a dict with uppercase keys
    if 'equationNames' in problem:
        eq_names = problem['equationNames']
        print(f'  Number of equations: {len(eq_names)}')
        print(f'  Equation names: {eq_names}')
    
    # Extract Jacobian
    print('\nExtracting Jacobian...')
    jac = problem.get('Jacobian')
    print(f'Jacobian type: {type(jac)}')
    
    if jac is not None:
        if hasattr(jac, 'toarray'):
            jac_dense = jac.toarray()
        elif hasattr(jac, 'todense'):
            jac_dense = np.array(jac.todense())
        else:
            jac_dense = np.array(jac)
    else:
        print('  WARNING: Jacobian not found in problem dict')
        jac_dense = None
    
    if jac_dense is not None:
        print(f'Jacobian shape: {jac_dense.shape}')
        print(f'Jacobian nnz: {np.count_nonzero(jac_dense)}')
        print(f'Jacobian density: {np.count_nonzero(jac_dense) / jac_dense.size * 100:.2f}%')
        print(f'Jacobian norm: {np.linalg.norm(jac_dense):.6e}')
    
    # Extract residual
    residual = problem.get('Residuals')
    if residual is not None:
        if isinstance(residual, dict):
            # Residual is a dict of residuals for each equation
            print(f'\nResiduals (dict format):')
            for res_name, res_val in residual.items():
                res_array = res_val.val if hasattr(res_val, 'val') else res_val
                print(f'  {res_name}: shape={res_array.shape if hasattr(res_array, "shape") else "scalar"}, norm={np.linalg.norm(res_array):.6e}')
            residual_array = np.concatenate([
                (res_val.val if hasattr(res_val, 'val') else res_val).ravel()
                for res_val in residual.values()
            ])
        else:
            residual_array = residual.ravel() if hasattr(residual, 'ravel') else np.array([residual])
            print(f'\nResidual shape: {residual_array.shape}')
        print(f'Residual norm: {np.linalg.norm(residual_array):.6e}')
        print(f'Residual max: {np.max(np.abs(residual_array)):.6e}')
    else:
        print('\n  WARNING: Residuals not found in problem dict')
        residual_array = None
    
    # Export to .mat file for comparison
    print(f'\nstate0 keys: {list(state0.keys())}')
    
    export_data = {
        'dt': dt,
        'grid_cells': model.G['cells']['num'],
        'well_names': [w['name'] for w in active_wells],
        'well_cells': [w.get('cells', []) for w in active_wells],
        'well_WI': [w.get('WI', []) for w in active_wells],
    }
    
    # Add state variables
    if 'pressure' in state0:
        export_data['state0_pressure'] = state0['pressure']
    if 'sW' in state0:
        export_data['state0_sW'] = state0['sW']
    if 'sG' in state0:
        export_data['state0_sG'] = state0['sG']
    if 's' in state0:
        export_data['state0_s'] = state0['s']
    
    if jac_dense is not None:
        export_data['jacobian'] = jac_dense
    if residual_array is not None:
        export_data['residual'] = residual_array
    
    output_file = 'spe1_cgnet_first_step.mat'
    sio.savemat(output_file, export_data)
    print(f'\n✓ Exported to {output_file}')
    
    print('\n' + '='*60)
    print('SUCCESS: SPE1 first step completed!')
    print('='*60)
    
except Exception as e:
    print(f'\n✗ ERROR: {e}')
    import traceback
    traceback.print_exc()
