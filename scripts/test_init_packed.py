from PRSTCore.ad_core.simulators.sim_runner.init_eclipse_packed_problem_ad import init_eclipse_packed_problem_ad
from PRSTCore.deckformat.deckinput.read_eclipse_deck import read_eclipse_deck
from PRSTCore.deckformat.grid.init_eclipse_grid import init_eclipse_grid
import pprint
try:
    deck = read_eclipse_deck('examples/SPE9/SPE9_CP.DATA')
    pprint.pprint({'RUNSPEC': deck.get('RUNSPEC'), 'GRID_keys': list(deck.get('GRID', {}).keys())})
    G = init_eclipse_grid(deck)
    pprint.pprint({'G_keys': list(G.keys()), 'cartDims': G.get('cartDims')})
    p = init_eclipse_packed_problem_ad('examples/SPE9/SPE9_CP.DATA')
    print('problem keys:', list(p.keys()))
    from PRSTCore.ad_core.simulators.sim_runner import simulate_packed_problem, get_packed_simulator_output
    ws, states = simulate_packed_problem(p)
    print('simulated steps:', len(states))
    wsols, st = get_packed_simulator_output(p)
    print('retrieved outputs:', len(wsols), len(st))
except Exception as e:
    import traceback
    traceback.print_exc()
