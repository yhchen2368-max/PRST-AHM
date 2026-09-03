"""Debug: Check grid type and operators initialization for SPE1."""
import sys
sys.path.insert(0, '.')
from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad

print('Loading SPE1...')
state0, model, schedule, solver = init_eclipse_problem_ad('examples/SpE1/BENCH_SPE1.DATA')

print(f'\n=== Grid Structure ===')
G = model.G
print(f'Grid type: {G.get("type")}')
print(f'Grid keys: {list(G.keys())[:10]}')
print(f'Cells: {G["cells"]["num"]}')

# Check what kind of grid it is
if 'cells' in G:
    print(f'  cells.num: {G["cells"]["num"]}')
if 'cartDims' in G:
    print(f'  cartDims: {G["cartDims"]}')
if 'xfaces' in G:
    print(f'  xfaces: {G["xfaces"][:5]}... ({len(G["xfaces"])} total)')
if 'faces' in G:
    print(f'  faces: {type(G["faces"])}, {len(G["faces"])} total')
if 'type' in G:
    print(f'  type: {G["type"]}')

print(f'\n=== Model Operators ===')
ops = model.operators
if ops is None:
    print('operators is None')
else:
    print(f'operators keys: {list(ops.keys())}')
    for key in ops:
        val = ops[key]
        if hasattr(val, 'shape'):
            print(f'  {key}: shape={val.shape}')
        else:
            print(f'  {key}: {type(val)}')

# Check if operators were initialized
print(f'\n=== Trying to initialize operators ===')
if ops is None or 'N' not in ops:
    from PRSTCore.ad_core.operators import setup_operators
    rock = model.rock
    print(f'Rock: {rock}')
    ops_new = setup_operators(G, rock)
    print(f'New operators keys: {list(ops_new.keys())}')
    for key in ops_new:
        val = ops_new[key]
        if hasattr(val, 'shape'):
            print(f'  {key}: shape={val.shape}')
        elif hasattr(val, '__len__'):
            print(f'  {key}: len={len(val)}')
        else:
            print(f'  {key}: {type(val)}')
    
    # Check N and T specifically
    if 'N' in ops_new:
        N = ops_new['N']
        print(f'\nNetwork matrix N: {N.shape if hasattr(N, "shape") else len(N)} entries')
        if hasattr(N, '__len__') and len(N) > 0:
            print(f'First 3 pairs: {N[:3]}')
    if 'T' in ops_new:
        T = ops_new['T']
        print(f'Transmissibility T: {T.shape if hasattr(T, "shape") else len(T)} entries')
        if hasattr(T, '__len__') and len(T) > 0:
            print(f'First 5 values: {T[:5] if hasattr(T, "__getitem__") else "N/A"}')

# Check rock properties
print(f'\n=== Rock Properties ===')
print(f'rock type: {type(model.rock)}')
if isinstance(model.rock, dict):
    print(f'rock keys: {list(model.rock.keys())[:10]}')
    if 'poro' in model.rock:
        poro = model.rock['poro']
        print(f'  poro: {len(poro) if hasattr(poro, "__len__") else poro}')
    if 'perm_x' in model.rock:
        perm = model.rock['perm_x']
        print(f'  perm_x: shape={perm.shape if hasattr(perm, "shape") else len(perm)}')
