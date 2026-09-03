"""Analyze per-cell ZCORN structure."""
import sys
sys.path.insert(0, '.')
import numpy as np
from PRSTCore.deckformat.deckinput import read_eclipse_deck, convert_deck_units

deck = read_eclipse_deck('examples/SpE1/BENCH_SPE1.DATA')
deck = convert_deck_units(deck)

g = deck.get('GRID', {})
rs = deck.get('RUNSPEC', {})
nx, ny, nz = rs.get('cartDims')

if 'ZCORN' in g:
    zcorn = np.asarray(g['ZCORN'], dtype=float)
    print(f'Grid: {nx}x{ny}x{nz}')
    print(f'ZCORN shape: {zcorn.shape}')
    
    # Reshape to (cells, 8)
    zcorn_8 = zcorn.reshape(-1, 8)
    print(f'Reshaped: {zcorn_8.shape}')
    
    # Examine first few cells
    print(f'\nFirst 5 cells ZCORN (8 corners per cell):')
    for i in range(5):
        print(f'  Cell {i}: {zcorn_8[i]}')
        print(f'    min={zcorn_8[i].min()}, max={zcorn_8[i].max()}, thickness={zcorn_8[i].max()-zcorn_8[i].min()}')
    
    # Examine cells in different layers
    print(f'\nCell indices for different layers:')
    print(f'  Layer 0 (top): cells 0-{nx*ny-1}')
    print(f'  Layer 1 (middle): cells {nx*ny}-{2*nx*ny-1}')
    print(f'  Layer 2 (bottom): cells {2*nx*ny}-{3*nx*ny-1}')
    
    # Check middle and bottom layer
    print(f'\nCell at start of each layer:')
    for k in range(nz):
        cell_idx = k * nx * ny
        print(f'  Layer {k}, cell {cell_idx}: {zcorn_8[cell_idx]}')
        print(f'    thickness={zcorn_8[cell_idx].max()-zcorn_8[cell_idx].min()}')
    
    # Check thickness per layer
    thickness = zcorn_8.max(axis=1) - zcorn_8.min(axis=1)
    thickness_reshaped = thickness.reshape((nx, ny, nz), order='F')
    
    print(f'\nThickness per layer (averaged):')
    for k in range(nz):
        dz_k = thickness_reshaped[:,:,k].mean()
        print(f'  Layer {k}: dz={dz_k}')
