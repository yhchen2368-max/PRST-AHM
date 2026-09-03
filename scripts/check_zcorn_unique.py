"""Check ZCORN unique values."""
import sys
sys.path.insert(0, '.')
import numpy as np
from PRSTCore.deckformat.deckinput import read_eclipse_deck, convert_deck_units

deck = read_eclipse_deck('examples/SpE1/BENCH_SPE1.DATA')
deck = convert_deck_units(deck)

g = deck.get('GRID', {})
if 'ZCORN' in g:
    zcorn = g['ZCORN']
    zcorn_arr = np.asarray(zcorn, dtype=float)
    
    unique_vals = np.unique(zcorn_arr)
    print(f'ZCORN unique values: {unique_vals}')
    print(f'Number of unique values: {len(unique_vals)}')
    
    if len(unique_vals) == 1:
        print(f'\n*** ERROR: ZCORN has only 1 unique value! Expected multiple Z levels ***')
        print(f'Likely cause: ZCORN not properly initialized from SPE1.GRDECL')
    else:
        print(f'\nZCORN values look correct (multiple unique values)')
        print(f'Min: {unique_vals.min()}, Max: {unique_vals.max()}')
        print(f'First 20 values: {zcorn_arr[:20]}')
        print(f'Last 20 values: {zcorn_arr[-20:]}')
