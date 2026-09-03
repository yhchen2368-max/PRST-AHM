"""Port of MRST example ``nearWellBoreModelingSim``: demonstration of the
generation of the necessary data structures passed to the AD simulators for
the NWM hybrid grid.

The generation involves several key processes:
  * Assemble the subgrids to get the global hybrid grid
  * Initialize the AD fluid
  * Make rocks from the subones
  * Compute the transmissibility and neighborship (including the NNC)
  * Setup the simulation model
  * Convert the simulation schedule
  * Define the initial state by equilibrium initialization

NOTE: this port mirrors the MRST script 1:1.  The surrounding deck /
AD-solver / plotting pipeline is required to execute it end to end.
"""

import numpy as np

from PRSTCore.nwm.models.NearWellboreModel import NearWellboreModel

# %% Load the subgrids (GC, GV, GW), the well info structure (well) and the
# input deck (deck) of example 'nearWellBoreModelingGrids'
# run nearWellBoreModelingGrids

# %% Define the NearWellboreModel
NWM = NearWellboreModel([GC, GV, GW], deck, well)

# %% Get the global hybrid grid
G = NWM.validateGlobalGrid()

# %% Initialize the AD fluid
fluid = NWM.setupFluid()

# %% Make rocks for the global grid
# -------------------------------------------------------------------------
# | Rock  | Grid | Source        | Permeability | Anisotropy |
# |       |      |               | coordinate   |            |
# | rockC | GC   | Input deck    | Local        | Yes        |
# | rockV | GV   | Interpolation | Global       | Yes        |
# |       |      | of rockC      |              |            |
# | rockW | GW   | User-defined  | Global       | No         |
# -------------------------------------------------------------------------
rockC = NWM.getCPGRockFromDeck()
rockV = NWM.getVOIRocksByInterp()

# Define a simple rock for the HW grid.  Each segment (layer) of the grid
# has uniform permeability and porosity.
nclayer = GW['cells']['num'] / GW['layers']['num']
milli = 1e-3
darcy = 9.869233e-13
permW = np.linspace(400, 500, GW['layers']['num']) * (milli * darcy)
permW = np.tile(permW, (int(nclayer), 1))
poroW = np.linspace(0.18, 0.2, GW['layers']['num'])
poroW = np.tile(poroW, (int(nclayer), 1))
rockW = {'perm': np.column_stack([permW.ravel(order='F')] * 3),
         'poro': poroW.ravel(order='F')}

# Assemble the subrocks to get the global one
rock = NWM.getGlobalRock([rockC, rockV, rockW])

# %% Compute the transmissibility and neighborship
# T = [TC; TV; TW], corresponding to G.faces.neighbors
T = NWM.getTransGloGrid(rock)

# Generate the NNC: compute the intersection relations between subgrids
intXn = NWM.computeIntxnRelation()
nnc = NWM.generateNonNeighborConn(intXn, rock, T)
T_all, N_all = NWM.assembleTransNeighbors(T, nnc)

# %% Setup the simulation model
model = NWM.setupSimModel(rock, T_all, N_all)

# %% Convert the simulation schedule
schedule = NWM.getSimSchedule(model, refDepthFrom='deck')

# %% Get the initial state by equilibrium initialization
initState = NWM.getInitState(model)

# %% Run the simulation (requires simulateScheduleAD)
# wellSols, states, report = simulateScheduleAD(initState, model, schedule)
