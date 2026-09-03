"""Debug: inspect wells dict before and after completion processing."""
import sys
sys.path.insert(0, '.')
from PRSTCore.deckformat.deckinput import read_eclipse_deck, convert_deck_units
from PRSTCore.deckformat.grid.init_eclipse_grid import init_eclipse_grid
from PRSTCore.deckformat.params.rock.init_eclipse_rock import init_eclipse_rock

deck_path = 'examples/SPE9/SPE9_CP.DATA'
deck = read_eclipse_deck(deck_path)
deck = convert_deck_units(deck)
G = init_eclipse_grid(deck)
rock = init_eclipse_rock(deck)

# Manually extract schedule
sched = deck.get('SCHEDULE', {})
print('WELSPECS records:', len(sched.get('WELSPECS', [])))
print('COMPDAT records:', len(sched.get('COMPDAT', [])))

# Check first COMPDAT entry structure
if sched.get('COMPDAT'):
    comp0 = sched['COMPDAT'][0]
    print(f'COMPDAT[0]: {comp0[:8] if len(comp0) >= 8 else comp0}')

# Check if G and rock have necessary fields
print(f'\nG.cartDims: {G.get("cartDims")}')
print(f'G.cells.num: {G.get("cells", {}).get("num")}')
print(f'G.cells.indexMap: {len(G.get("cells", {}).get("indexMap", []))} entries')
print(f'rock.perm shape: {rock.get("perm", [[]]).__class__.__name__}')
