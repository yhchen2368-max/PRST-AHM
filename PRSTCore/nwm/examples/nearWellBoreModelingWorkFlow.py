"""Port of MRST example ``nearWellBoreModelingWorkFlow``: complete workflow
of the near-wellbore modeling (NWM) method, including building the volume
of interest (VOI) grid, building the horizontal-well (HW) grid, generating
the data structures passed to the AD simulators, running the simulation
(``simulateScheduleAD``, already ported as
:func:`PRSTCore.ad_core.simulators.simulate_schedule_ad.simulate_schedule_ad`),
and plotting the resulting saturation distribution and production curves.

Note on ``gridType='Voronoi'``: see the ``nearWellBoreModelingGrids.py``
module docstring -- DistMesh's initial sampling is randomized, so the
reconstruction is retried on failure. The persistent failure that used to
force a ``gridType='triangular'`` fallback here was a mistranslation of
MATLAB's column-major ``find`` in ``generateVOIGridNodes.VoronoiPts``, not
an unlucky draw.
"""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from PRSTCore.deckformat.deckinput.convert_deck_units import convert_deck_units
from PRSTCore.deckformat.deckinput.read_eclipse_deck import read_eclipse_deck
from PRSTCore.gridprocessing.compute_geometry import compute_geometry
from PRSTCore.gridprocessing.process_grdecl import process_grdecl

from PRSTCore.ad_core.simulators.simulate_schedule_ad import simulate_schedule_ad
from PRSTCore.ad_core.solvers import MUMPSSolverAD, NonLinearSolver, check_mumps
from PRSTCore.nwm.models.HorWellRegion import HorWellRegion
from PRSTCore.nwm.models.NearWellboreModel import NearWellboreModel
from PRSTCore.nwm.models.VolumeOfInterest import VolumeOfInterest
from PRSTCore.visualization import plot_cell_data, plot_grid

# NWM.data/trajectory.mat are the MRST module's own example data
# (mrst-2026a/modules/nwm/data/); reused here rather than duplicated.
DATA_DIR = Path(__file__).resolve().parents[3] / 'mrst-2026a' / 'modules' / 'nwm' / 'data'
OUT_DIR = Path(__file__).resolve().parent / 'output'
OUT_DIR.mkdir(exist_ok=True)


def savefig(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def reconstruct_with_retry(VOI, WR, layerRf, *, max_attempts=3, **kwargs):
    """DistMesh's initial sampling is randomized; retry on an unlucky draw
    (see module docstring).  A persistent failure is raised rather than
    quietly substituting a different grid type."""
    last = None
    for attempt in range(1, max_attempts + 1):
        try:
            return VOI.ReConstructToUnstructuredGrid(WR, layerRf, **kwargs)
        except Exception as exc:
            last = exc
            print(f'  reconstruction attempt {attempt} failed ({exc}); retrying...')
    raise RuntimeError(
        f'VOI reconstruction (gridType={kwargs.get("gridType")!r}) did not '
        f'succeed after {max_attempts} attempts') from last


# %% Read the ECLIPSE input deck
fn = str(DATA_DIR / 'NWM.data')
deck = read_eclipse_deck(fn)
deck = convert_deck_units(deck)

# %% Build the background Corner-point grid (CPG)
GC = compute_geometry(process_grdecl(deck['GRID']))

# %% Define the basic information of the horizontal well (HW)
pW = loadmat(str(DATA_DIR / 'trajectory.mat'))['pW']
ns = pW.shape[0] - 1
well = {
    'name': 'PROD',
    'trajectory': pW,
    'segmentNum': ns,
    'radius': 0.15 * np.ones(ns + 1),
    'skinFactor': np.zeros(ns),
    'openedSegs': np.arange(1, ns + 1),
}

# %% Build the layered unstructured VOI grid
pbdy = np.array([[240, 50], [160, 80], [120, 160], [150, 205], [230, 170],
                 [280, 90]], dtype=float)
nextra = [1, 1]
VOI = VolumeOfInterest(GC, well, pbdy, nextra)

VOI.maxWellSegLength2D()
WR = {'ly': 15, 'ny': 10, 'na': 5}
VOI.volumeLayerNumber()
layerRf = [2, 2, 2, 2]
GV = reconstruct_with_retry(VOI, WR, layerRf, multiplier=0.2, maxIter=500, gridType='Voronoi')

# %% Build the layered radial HW grid
regionIndices = [3, 8, 2, 7]
HW = HorWellRegion(GV, well, regionIndices)
radPara = {'gridType': 'gradual', 'boxRatio': [0.6, 0.6],
           'nRadCells': [7, 2], 'pDMult': 10, 'offCenter': True}
GW = HW.ReConstructToRadialGrid(radPara)

# %% Generate the data structures for the AD simulator
NWM = NearWellboreModel([GC, GV, GW], deck, well)

# Define a simple rock for the HW grid
nclayer = GW['cells']['num'] / GW['layers']['num']
milli = 1e-3
darcy = 9.869233e-13
permW = np.linspace(400, 500, GW['layers']['num']) * (milli * darcy)
permW = np.tile(permW, (int(nclayer), 1))
poroW = np.linspace(0.18, 0.2, GW['layers']['num'])
poroW = np.tile(poroW, (int(nclayer), 1))
rockW = {'perm': np.column_stack([permW.ravel(order='F')] * 3),
         'poro': poroW.ravel(order='F')}

# Get the MRST data structures for the NWM grid
G, rock, fluid, model, schedule, initState = NWM.packedSimData(rockW)

# %% Run the simulation
# The NWM hybrid grid's non-neighbor connections (subgrid NNCs) give the
# Jacobian a less regular sparsity pattern than a plain structured grid;
# MUMPS's direct sparse solve sidesteps needing an iterative solver tuned
# for that pattern, when the optional python-mumps backend is available.
nonlinear_solver = None
if check_mumps():
    print(' -- Using MUMPSSolverAD as the linear solver')
    nonlinear_solver = NonLinearSolver(linearSolver=MUMPSSolverAD())
print(' -- Running the simulation (simulate_schedule_ad)')
well_sols, states, report = simulate_schedule_ad(initState, model, schedule,
                                                  nonlinear_solver=nonlinear_solver,
                                                  return_report=True)
print(f'  simulated {len(states)} report steps')

# %% Plot the saturation distribution at the final report step
sw_final = np.asarray(states[-1]['s'])
sw_final = sw_final[:, 0] if sw_final.ndim > 1 else sw_final
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plot_cell_data(G, sw_final, ax=ax)
ax.set_title('Water saturation distribution (final report step)')
savefig(fig, '11_saturation_distribution.png')

# %% Plot the production dynamics (well rates / bhp vs time)
time_days = np.cumsum(np.asarray(schedule['step']['val'], dtype=float)) / 86400.0
well_names = [w.get('name', f'W{i}') for i, w in enumerate(schedule['control'][0]['W'])]
nwell = len(well_names)

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 8))
for wi, name in enumerate(well_names):
    qWs = np.array([ws[wi].get('qWs', np.nan) for ws in well_sols])
    qOs = np.array([ws[wi].get('qOs', np.nan) for ws in well_sols])
    bhp = np.array([ws[wi].get('bhp', np.nan) for ws in well_sols])
    axes[0].plot(time_days, qWs, marker='.', label=name)
    axes[1].plot(time_days, qOs, marker='.', label=name)
    axes[2].plot(time_days, bhp / 1e5, marker='.', label=name)
axes[0].set_ylabel('qWs [m^3/s]')
axes[1].set_ylabel('qOs [m^3/s]')
axes[2].set_ylabel('bhp [bar]')
axes[2].set_xlabel('Time [days]')
axes[0].set_title('Production dynamics')
axes[0].legend(fontsize=7)
fig.tight_layout()
savefig(fig, '12_production_dynamics.png')

print(f'\nAll figures written to {OUT_DIR}')
