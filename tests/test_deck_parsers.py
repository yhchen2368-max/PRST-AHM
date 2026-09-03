from PRSTCore.deckformat.deckinput import read_eclipse_deck, initialize_deck
from PRSTCore.deckformat.deckinput.read_grid import read_grid
from PRSTCore.deckformat.deckinput.read_props import read_props
from PRSTCore.deckformat.deckinput.read_schedule import read_schedule

print('imports ok')
print('init', list(initialize_deck().keys()))
print('grid parse', read_grid('GRID\nDXV 1 2 3 /\nPERMX 10 20 30 /'))
print('props parse', read_props('ROCK 1 2 3 /\nDENSITY 1000 /'))
print('sched parse', read_schedule('WELSPECS well1 100 1 /\nTSTEP 1 /'))
