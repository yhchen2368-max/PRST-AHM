"""Profile NORNE initialization step by step."""
import time, sys, numpy as np
sys.path.insert(0, '.')
sys.setrecursionlimit(10000)

steps = []
t0 = time.time()

# 1. Deck read
from PRSTCore.deckformat.deckinput import read_eclipse_deck, convert_deck_units
d = convert_deck_units(read_eclipse_deck('examples/Norne/Norne_simplified/NORNE_ATW2013.DATA'))
steps.append(('deck', time.time() - t0)); t0 = time.time()

# 2. Rock
from PRSTCore.deckformat.params.rock.init_eclipse_rock import init_eclipse_rock
rock = init_eclipse_rock(d)
steps.append(('rock', time.time() - t0)); t0 = time.time()

# 3. Grid
from PRSTCore.deckformat.grid.init_eclipse_grid import init_eclipse_grid
G = init_eclipse_grid(d)
steps.append(('grid', time.time() - t0)); t0 = time.time()

# 4. Fluid
from PRSTCore.ad_core.initialization.init_deck_adi_fluid import init_deck_adi_fluid
fluid = init_deck_adi_fluid(d)
steps.append(('fluid', time.time() - t0)); t0 = time.time()

# 5. Model
from PRSTCore.ad_core.models.generic_black_oil_model import make_generic_black_oil_model
model = make_generic_black_oil_model(G, rock, fluid)
steps.append(('model', time.time() - t0)); t0 = time.time()

# 6. Operators
from PRSTCore.ad_core.operators import setup_operators
ops = setup_operators(G, rock)
setattr(model, 'operators', ops)
steps.append(('operators', time.time() - t0)); t0 = time.time()

# 7. Porevolume
try:
    if isinstance(G, dict) and 'cell_volumes' in G and 'poro' in rock:
        pv = np.asarray(G['cell_volumes'], dtype=float) * np.asarray(rock['poro']).ravel()
        model.porevolume = pv
except Exception:
    pass
steps.append(('pv', time.time() - t0)); t0 = time.time()

# 8. State
nc = G.get('cells', {}).get('num', 1)
s = {'pressure': np.ones(nc)*1e7, 'saturation': np.zeros(nc), 'time': 0.0, 'wellSol': []}
steps.append(('state', time.time() - t0)); t0 = time.time()

# 9. Schedule
# (Schedule conversion - skip for now to see where the real bottleneck is)

for name, dt in steps:
    print(f'{name}: {dt:.2f}s')
print(f'G cells: {G.get("cells",{}).get("num","?")}, ops N: {ops.get("N",np.empty((0,))).shape}')
