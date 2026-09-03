"""Debug transmissibility calculation."""
import sys
sys.path.insert(0, '.')
import numpy as np
from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad

print('Loading SPE1...')
state0, model, schedule, solver = init_eclipse_problem_ad('examples/SpE1/BENCH_SPE1.DATA')

G = model.G
rock = model.rock
ops = model.operators

print(f'\n=== Grid Info ===')
print(f'cartDims: {G.get("cartDims")}')
nx, ny, nz = G.get("cartDims", [1, 1, 1])
print(f'nx={nx}, ny={ny}, nz={nz}, n_cells={nx*ny*nz}')

# Check grid spacing
if 'dx' in G and 'dy' in G and 'dz_layer' in G:
    dx = np.asarray(G['dx'])
    dy = np.asarray(G['dy'])
    dz = np.asarray(G['dz_layer'])
    print(f'\nGrid spacing (from G):')
    print(f'  dx shape: {dx.shape}, min={dx.min():.2f}, max={dx.max():.2f}')
    print(f'  dy shape: {dy.shape}, min={dy.min():.2f}, max={dy.max():.2f}')
    print(f'  dz shape: {dz.shape}, min={dz.min():.2f}, max={dz.max():.2f}')

print(f'\n=== Permeability ===')
perm = rock.get('perm')
print(f'perm type: {type(perm)}')
if perm is not None:
    perm_arr = np.asarray(perm)
    print(f'perm shape: {perm_arr.shape}')
    print(f'perm sample: {perm_arr.ravel()[:5]}')
else:
    print('perm is None')

print(f'\n=== Operators Transmissibility ===')
if ops and 'T' in ops:
    T = ops['T']
    print(f'T shape: {T.shape}')
    print(f'T min: {T.min():.6e}')
    print(f'T max: {T.max():.6e}')
    print(f'T non-zeros: {np.count_nonzero(T)}')
    print(f'First 10 T values: {T[:10]}')
    print(f'Non-zero T indices: {np.where(T != 0)[0][:10]}...' if np.any(T != 0) else 'All T are zero!')

# Try to manually recalculate T for the first few faces
if ops and 'N' in ops:
    N = ops['N']
    print(f'\n=== Manual T Recalculation ===')
    print(f'First 5 N pairs:')
    
    # Get permeabilities
    perm_arr = np.asarray(rock.get('perm', np.ones(nx*ny*nz)))
    if perm_arr.ndim == 1:
        kx_arr = perm_arr.ravel()
    else:
        kx_arr = perm_arr[:, 0]
    
    dx_vals = np.asarray(G.get('dx', np.ones(nx)))
    dy_vals = np.asarray(G.get('dy', np.ones(ny)))
    dz_vals = np.asarray(G.get('dz_layer', np.ones(nz)))
    
    for f in range(min(5, N.shape[0])):
        c1 = N[f, 0] - 1
        c2 = N[f, 1] - 1
        k1 = kx_arr[c1]
        k2 = kx_arr[c2]
        
        # Get i index for dx lookup: i = (c % (nx*nz)) // nz
        i1 = (c1 % (nx * nz)) // nz
        i2 = (c2 % (nx * nz)) // nz
        
        # Try to get distance
        if i1 < len(dx_vals) and i2 < len(dx_vals):
            dist = 0.5 * (dx_vals[i1] + dx_vals[i2])
        else:
            dist = 1.0  # fallback
        
        denom = k1 + k2
        khe = 2.0 * k1 * k2 / denom if denom > 0 else 0
        A = dy_vals[0] * dz_vals[0] if len(dy_vals) > 0 and len(dz_vals) > 0 else 1.0  # simplified
        T_calc = khe * A / dist if dist > 0 else 0
        
        print(f'  Face {f}: c1={c1}, c2={c2}, i1={i1}, i2={i2}, k1={k1:.2f}, k2={k2:.2f}, dist={dist:.2f}, T_calc={T_calc:.6e}')
        if 'T' in ops and f < len(ops['T']):
            print(f'           Actual T: {ops["T"][f]:.6e}')

# Check if dx/dy/dz are defined correctly
print(f'\n=== Grid dimension consistency check ===')
if 'dx' in G and 'dy' in G and 'dz_layer' in G:
    dx_len = len(np.asarray(G['dx']))
    dy_len = len(np.asarray(G['dy']))
    dz_len = len(np.asarray(G['dz_layer']))
    print(f'dx length: {dx_len} (expected nx={nx})')
    print(f'dy length: {dy_len} (expected ny={ny})')
    print(f'dz length: {dz_len} (expected nz={nz})')
