#!/usr/bin/env python
"""
Analyze SPE1 first step results from PRSTCore.
Compare Jacobian structure, residual norms, and well coupling.
"""
import numpy as np
import scipy.io as sio
from pathlib import Path
import sys

def load_mat_file(filepath):
    """Load .mat file safely"""
    try:
        return sio.loadmat(filepath, squeeze_me=True)
    except Exception as e:
        print(f'ERROR loading {filepath}: {e}')
        return None

def analyze_jacobian(jac):
    """Analyze Jacobian matrix properties"""
    print('\n========== Jacobian Analysis ==========')
    print(f'Shape: {jac.shape}')
    print(f'Type: {type(jac).__name__}')
    
    # Convert to dense if sparse
    if hasattr(jac, 'todense'):
        jac_dense = jac.todense()
    else:
        jac_dense = jac
    
    jac_dense = np.asarray(jac_dense)
    nnz = np.count_nonzero(jac_dense)
    
    print(f'Non-zeros: {nnz}')
    print(f'Density: {nnz / jac_dense.size * 100:.4f}%')
    print(f'Frobenius norm: {np.linalg.norm(jac_dense):.6e}')
    print(f'2-norm (largest singular value): {np.max(np.linalg.svd(jac_dense, compute_uv=False)):.6e}')
    print(f'Condition number: {np.linalg.cond(jac_dense):.6e}')
    
    # Block structure analysis
    # For 3-phase black oil: [nc*3 equations] = [water, oil, gas] + well equations
    nc = 300  # SPE1 has 300 cells
    nc_eq = nc * 3  # 3 conservation equations per cell
    n_well_eq = jac_dense.shape[0] - nc_eq  # Remaining rows are well equations
    
    print(f'\nBlock structure:')
    print(f'  Grid cells: {nc}')
    print(f'  Grid equations: {nc_eq} (3 per cell)')
    print(f'  Well equations: {n_well_eq}')
    print(f'  Total equations: {jac_dense.shape[0]}')
    
    # Diagonal structure
    diag_entries = np.diag(jac_dense)
    print(f'\nDiagonal properties:')
    print(f'  Non-zero diagonals: {np.count_nonzero(diag_entries)}')
    print(f'  Min |diag|: {np.min(np.abs(diag_entries[diag_entries != 0])):.6e}')
    print(f'  Max |diag|: {np.max(np.abs(diag_entries)):.6e}')
    print(f'  Mean |diag|: {np.mean(np.abs(diag_entries[diag_entries != 0])):.6e}')
    
    return jac_dense, (nc, nc_eq, n_well_eq)

def analyze_residual(residual, eq_structure=None):
    """Analyze residual vector"""
    print('\n========== Residual Analysis ==========')
    print(f'Shape: {residual.shape}')
    print(f'Norm (2): {np.linalg.norm(residual):.6e}')
    print(f'Norm (1): {np.sum(np.abs(residual)):.6e}')
    print(f'Norm (inf): {np.max(np.abs(residual)):.6e}')
    print(f'Min value: {np.min(residual):.6e}')
    print(f'Max value: {np.max(residual):.6e}')
    
    # Component breakdown if structure known
    if eq_structure is not None:
        nc, nc_eq, n_well_eq = eq_structure
        print(f'\nComponent breakdown:')
        if nc_eq <= len(residual):
            grid_res = residual[:nc_eq]
            well_res = residual[nc_eq:] if n_well_eq > 0 else np.array([])
            
            print(f'  Grid residual norm: {np.linalg.norm(grid_res):.6e}')
            if len(well_res) > 0:
                print(f'  Well residual norm: {np.linalg.norm(well_res):.6e}')

def main():
    print('='*60)
    print('SPE1 First Step Results Analysis')
    print('='*60)
    
    # Load PRSTCore results
    cgnet_file = Path('spe1_cgnet_first_step.mat')
    if not cgnet_file.exists():
        print(f'ERROR: {cgnet_file} not found!')
        sys.exit(1)
    
    print(f'\nLoading PRSTCore results from {cgnet_file}...')
    cgnet_data = load_mat_file(str(cgnet_file))
    
    if cgnet_data is None:
        sys.exit(1)
    
    # Print available keys
    print(f'\nPRSTCore data keys: {list(cgnet_data.keys())}')
    
    # Analyze Jacobian
    if 'jacobian' in cgnet_data:
        jac_cgnet, eq_struct = analyze_jacobian(cgnet_data['jacobian'])
    else:
        print('WARNING: No Jacobian in PRSTCore data')
        jac_cgnet, eq_struct = None, None
    
    # Analyze Residual
    if 'residual' in cgnet_data:
        residual_cgnet = cgnet_data['residual'].ravel()
        analyze_residual(residual_cgnet, eq_struct)
    else:
        print('WARNING: No Residual in PRSTCore data')
    
    # State information
    print('\n========== Initial State ==========')
    if 'state0_pressure' in cgnet_data:
        p = cgnet_data['state0_pressure'].ravel()
        print(f'Pressure: min={np.min(p):.6e}, max={np.max(p):.6e}, mean={np.mean(p):.6e}')
    
    for key in ['state0_sW', 'state0_sG']:
        if key in cgnet_data:
            s = cgnet_data[key].ravel()
            print(f'{key}: min={np.min(s):.6e}, max={np.max(s):.6e}, mean={np.mean(s):.6e}')
    
    # Well information
    print('\n========== Wells ==========')
    if 'well_names' in cgnet_data:
        well_names = cgnet_data['well_names']
        well_cells = cgnet_data.get('well_cells', [])
        well_WI = cgnet_data.get('well_WI', [])
        
        if isinstance(well_names, str):
            well_names = [well_names]
        
        for i, name in enumerate(well_names):
            print(f'Well {i}: {name}')
            if isinstance(well_cells, (list, np.ndarray)) and len(well_cells) > i:
                cells = well_cells[i]
                if isinstance(cells, np.ndarray):
                    cells = cells.ravel()
                print(f'  Cells: {cells}')
            if isinstance(well_WI, (list, np.ndarray)) and len(well_WI) > i:
                wi = well_WI[i]
                if isinstance(wi, np.ndarray):
                    wi = wi.ravel()
                print(f'  WI: {wi}')
    
    # Timestep info
    if 'dt' in cgnet_data:
        dt = float(cgnet_data['dt'])
        print(f'\n========== Timestep ==========')
        print(f'dt = {dt:.6e} s = {dt/86400:.6f} days')
    
    print('\n' + '='*60)
    print('Analysis complete!')
    print('='*60)

if __name__ == '__main__':
    main()
