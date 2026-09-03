#!/usr/bin/env python3
"""
Analyze SPE1 results from PRSTCore and debug Jacobian structure
"""
import scipy.io as sio
import numpy as np
import os

# Load PRSTCore SPE1 results
cgnet_file = r'C:\Users\junji\Desktop\github\Cgnet\spe1_cgnet_first_step.mat'

if os.path.exists(cgnet_file):
    print(f"Loading {cgnet_file}...")
    data = sio.loadmat(cgnet_file, simplify_cells=True)
    
    print("\n=== PRSTCore SPE1 First Step Results ===\n")
    print("Keys in MAT file:")
    for key in sorted(data.keys()):
        if not key.startswith('__'):
            val = data[key]
            if isinstance(val, np.ndarray):
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}, nnz={np.count_nonzero(val) if val.dtype in [np.float32, np.float64, int] else 'N/A'}")
            else:
                print(f"  {key}: {type(val)}")
    
    print("\n=== Jacobian Analysis ===")
    if 'jacobian' in data:
        jac = data['jacobian']
        print(f"Jacobian shape: {jac.shape}")
        print(f"Jacobian dtype: {jac.dtype}")
        print(f"Jacobian nnz: {np.count_nonzero(jac)}")
        print(f"Jacobian density: {100*np.count_nonzero(jac)/(jac.shape[0]*jac.shape[1]):.6f}%")
        print(f"Jacobian norm: {np.linalg.norm(jac.toarray() if hasattr(jac, 'toarray') else jac):.6e}")
        
        # Check for block structure (should be 4 blocks: water, oil, gas, facility)
        if hasattr(jac, 'toarray'):
            jac_dense = jac.toarray()
        else:
            jac_dense = jac
        
        print(f"Jacobian min: {jac_dense.min():.6e}")
        print(f"Jacobian max: {jac_dense.max():.6e}")
    
    print("\n=== Residual Analysis ===")
    if 'residual' in data:
        res = data['residual'].flatten()
        print(f"Residual shape: {res.shape}")
        print(f"Residual dtype: {res.dtype}")
        print(f"Residual norm: {np.linalg.norm(res):.6e}")
        print(f"Residual min: {res.min():.6e}")
        print(f"Residual max: {res.max():.6e}")
        print(f"Residual mean: {np.mean(res):.6e}")
        print(f"Number of NaN: {np.sum(np.isnan(res))}")
        print(f"Number of Inf: {np.sum(np.isinf(res))}")
    
    print("\n=== Well Information ===")
    for key in ['well_names', 'well_cells', 'well_wi']:
        if key in data:
            val = data[key]
            print(f"{key}: {val}")
    
    print("\n=== State Information ===")
    if 'state0' in data:
        state = data['state0']
        if isinstance(state, dict):
            print(f"state0 keys: {list(state.keys())}")
        else:
            print(f"state0 type: {type(state)}")
    
    print("\nFile size:", os.path.getsize(cgnet_file) / 1e6, "MB")
else:
    print(f"File not found: {cgnet_file}")
    print("\nRunning SPE1 test script first...")
    os.system("cd C:\\Users\\junji\\Desktop\\github\\Cgnet && python scripts\\test_spe1_first_step.py")
    
    if os.path.exists(cgnet_file):
        print("\nRetrying analysis...")
        data = sio.loadmat(cgnet_file, simplify_cells=True)
        for key in sorted(data.keys()):
            if not key.startswith('__'):
                print(f"  {key}")
