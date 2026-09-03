"""Port of MRST example ``nearWellBoreModelingGrids``: demonstration of the
grids in the near-wellbore modeling (NWM) method.

The global grid consists of three subgrids adopting different gridding
strategies:

    | Grid | Description | Type                 | Constructor               |
    |------|-------------|----------------------|---------------------------|
    | GC   | Background  | Corner-point or      | process_grdecl +          |
    |      | grid        | Cartesian            | compute_geometry          |
    | GV   | VOI grid    | Unstructured,        | tessellationGrid +        |
    |      |             | Vertically layered   | makeLayeredGridNWM        |
    | GW   | HW grid     | Structured, Radial,  | buildRadialGrid +         |
    |      |             | Horizontally layered | makeLayeredGridNWM        |

Headless (matplotlib ``Agg``): every plot MRST would show interactively is
instead saved as a PNG under ``OUT_DIR``.

Note on ``gridType='Voronoi'``: DistMesh's initial point sampling is
randomized, matching MRST's own algorithm (MRST has no fixed seed here
either), so the reconstruction is retried on failure.

This retry used to also fall back to ``gridType='triangular'``, on the
belief that ``addEmpCells``' boundary-face path tracer simply failed for
unlucky draws. It did not: ``generateVOIGridNodes.VoronoiPts`` mistranslated
MATLAB's column-major two-output ``find`` when mapping well-boundary nodes
onto Voronoi vertices, which scrambled that mapping and left ``addEmpCells``
with far more gaps than it can close. (The earlier experiment that replayed
a captured failing ``(p, t, bnW)`` triple through MRST's own ``addEmpCells``
and saw the same failure was sound, but its input had already been corrupted
by that bug.) With the mapping fixed the Voronoi reconstruction succeeds on
the first attempt, so no grid-type fallback is used -- a failure here should
surface rather than silently build a grid MRST's example does not build.
"""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from PRSTCore.deckformat.deckinput.read_eclipse_deck import read_eclipse_deck
from PRSTCore.deckformat.deckinput.convert_deck_units import convert_deck_units
from PRSTCore.gridprocessing.compute_geometry import compute_geometry
from PRSTCore.gridprocessing.process_grdecl import process_grdecl

from PRSTCore.nwm.models.HorWellRegion import HorWellRegion
from PRSTCore.nwm.models.VolumeOfInterest import VolumeOfInterest
from PRSTCore.visualization import plot_grid

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
pW = loadmat(str(DATA_DIR / 'trajectory.mat'))['pW']  # (well points)
ns = pW.shape[0] - 1  # Number of well segments
well = {
    'name': 'PROD',
    'trajectory': pW,
    'segmentNum': ns,
    'radius': 0.15 * np.ones(ns + 1),
    'skinFactor': np.zeros(ns),
    'openedSegs': np.arange(1, ns + 1),
}

# %% Define the volume of interest (VOI)
pbdy = np.array([[240, 50],
                 [160, 80],
                 [120, 160],
                 [150, 205],
                 [230, 170],
                 [280, 90]], dtype=float)

# The VOI is vertically expanded by extra layers
nextra = [1, 1]

VOI = VolumeOfInterest(GC, well, pbdy, nextra)

geoV = VOI.allInfoOfVolume()
savefig(VOI.plotVolumeCells(geoV), '01_voi_cells.png')
savefig(VOI.plotVolumeLayerFaces(geoV), '02_voi_layer_faces.png')
savefig(VOI.plotVolumeBoundaries(geoV), '03_voi_boundaries.png')

# %% Build the layered unstructured VOI grid
VOI.maxWellSegLength2D()
WR = {'ly': 15, 'ny': 10, 'na': 5}
VOI.volumeLayerNumber()
layerRf = [2, 2, 2, 2]

savefig(VOI.plot2DWRSubGrid(WR), '04_2d_wr_subgrid.png')

GV = reconstruct_with_retry(VOI, WR, layerRf, multiplier=0.2, maxIter=500,
                            gridType='Voronoi')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
rng = np.random.default_rng(0)
for layer in range(1, GV['layers']['num'] + 1):
    plot_grid(GV, np.flatnonzero(GV['cells']['layers'] == layer), ax=ax, facecolor=rng.random(3))
ax.set_title('Layers of the VOI grid')
savefig(fig, '05_voi_grid_layers.png')

# %% Build the layered radial HW grid
regionIndices = [3, 8, 2, 7]
HW = HorWellRegion(GV, well, regionIndices)
geoW = HW.allInfoOfRegion()
savefig(HW.showWellRegionInVOIGrid(showWellRgionCells=True), '06_hw_region_in_voi.png')
savefig(HW.plotRegionCells(geoW), '07_hw_region_cells.png')
savefig(HW.plotRegionLayerFaces(geoW), '08_hw_region_layer_faces.png')

radPara = {'gridType': 'gradual',
           'boxRatio': [0.6, 0.6],
           'nRadCells': [7, 2],
           'pDMult': 10,
           'offCenter': True}
GW = HW.ReConstructToRadialGrid(radPara)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plot_grid(GW, ax=ax, facecolor='0.7')
ax.set_title('Radial HW grid')
savefig(fig, '09_hw_radial_grid.png')

# %% Build the layered unstructured VOI grid (triangular variant)
GV_tri = reconstruct_with_retry(VOI, WR, layerRf, multiplier=0.2, maxIter=500, gridType='triangular')
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
rng = np.random.default_rng(1)
for layer in range(1, GV_tri['layers']['num'] + 1):
    plot_grid(GV_tri, np.flatnonzero(GV_tri['cells']['layers'] == layer), ax=ax, facecolor=rng.random(3))
ax.set_title('Layers of the VOI grid (triangular variant)')
savefig(fig, '10_voi_grid_layers_triangular.png')

print(f'\nAll figures written to {OUT_DIR}')
