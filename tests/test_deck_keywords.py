from PRSTCore.deckformat.deckinput.read_grid import read_grid
from PRSTCore.deckformat.deckinput.read_props import read_props

print('test NNC')
grid = read_grid('GRID\nNNC 1 2 3 4 5 6 7 / 8 9 10 11 12 13 14 /')
print('NNC shape', grid.get('NNC').shape)

print('\ntest JFUNC')
grid2 = read_grid('GRID\nJFUNC J1 100 0 /')
print('JFUNC', grid2.get('JFUNC'))

print('\ntest ROCKTAB')
props = read_props('ROCKTAB 1 2 3 / 4 5 6 /')
print('ROCKTAB count', len(props.get('ROCKTAB')))
print('ROCKTAB[0]', props.get('ROCKTAB')[0])
