"""Debug ZCORN and dz computation."""
import sys
sys.path.insert(0, '.')
import numpy as np
from PRSTCore.deckformat.deckinput import read_eclipse_deck, convert_deck_units

deck = read_eclipse_deck('examples/SpE1/BENCH_SPE1.DATA')
deck = convert_deck_units(deck)

g = deck.get('GRID', {})
rs = deck.get('RUNSPEC', {})
dims = rs.get('cartDims') or g.get('cartDims')
nx, ny, nz = dims

print(f'Grid dimensions: nx={nx}, ny={ny}, nz={nz}')
print(f'Expected cells: {nx*ny*nz}')
print(f'Expected ZCORN size: {nx*ny*nz*8}')

if 'ZCORN' in g:
    zcorn = g['ZCORN']
    zcorn_arr = np.asarray(zcorn, dtype=float)
    print(f'\nZCORN raw size: {zcorn_arr.size}')
    print(f'ZCORN sample values: {zcorn_arr[:20]}')
    
    # Try reshape
    try:
        if zcorn_arr.size % 8 == 0:
            zcorn_arr_8 = zcorn_arr.reshape(-1, 8)
            print(f'ZCORN reshaped to (-1, 8): {zcorn_arr_8.shape}')
            
            # Compute thickness per cell
            thickness = np.max(zcorn_arr_8, axis=1) - np.min(zcorn_arr_8, axis=1)
            print(f'Thickness array shape: {thickness.shape}')
            print(f'Thickness sample: {thickness[:10]}')
            print(f'Thickness min: {thickness.min()}, max: {thickness.max()}')
            
            # Reshape thickness to (nx, ny, nz)
            if nx * ny * nz == thickness.size:
                try:
                    thickness_reshaped = thickness.reshape((nx, ny, nz), order='F')
                    print(f'Thickness reshaped to ({nx}, {ny}, {nz}) with order=F: {thickness_reshaped.shape}')
                except Exception as e:
                    print(f'Failed to reshape with order=F: {e}')
                    thickness_reshaped = thickness.reshape((nx, ny, nz))
                    print(f'Thickness reshaped to ({nx}, {ny}, {nz}) with order=C: {thickness_reshaped.shape}')
                
                # Compute dz per layer
                dz = np.mean(thickness_reshaped, axis=(0, 1))
                print(f'\nDZ per layer (from thickness average):')
                print(f'  dz shape: {dz.shape}')
                print(f'  dz values: {dz}')
            else:
                print(f'ERROR: thickness size {thickness.size} != {nx*ny*nz}')
    except Exception as e:
        print(f'ERROR in reshape/compute: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()

if 'COORD' in g:
    coord = g['COORD']
    coord_arr = np.asarray(coord, dtype=float)
    print(f'\nCOORD raw size: {coord_arr.size}')
    print(f'Expected COORD size: {(nx+1)*(ny+1)*6} (6 per grid node)')
    print(f'COORD sample values: {coord_arr[:20]}')
    
    # Try to extract Z coordinates
    if coord_arr.size % 3 == 0:
        coord_3 = coord_arr.reshape(-1, 3)
        print(f'COORD reshaped to (-1, 3): {coord_3.shape}')
        print(f'Z range in COORD: min={coord_3[:, 2].min()}, max={coord_3[:, 2].max()}')
