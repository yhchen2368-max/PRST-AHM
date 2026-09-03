"""End-to-end workflow test for the ported nwm module:
CPG -> VOI grid -> HW grid -> global hybrid grid assembly.

This exercises VolumeOfInterest, generateVOIGridNodes (triangular),
HorWellRegion, generateHWGridNodes (gradual + pureCircular),
buildRadialGrid, makeLayeredGridNWM, NearWellboreModel grid assembly and
the rock/transmissibility machinery (up to the AD-solver wiring).
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np

from PRSTCore.gridprocessing.cart_grid import cart_grid
from PRSTCore.gridprocessing.compute_geometry import compute_geometry

from PRSTCore.nwm.models.VolumeOfInterest import VolumeOfInterest
from PRSTCore.nwm.models.HorWellRegion import HorWellRegion
from PRSTCore.nwm.models.NearWellboreModel import NearWellboreModel

# --- Background Cartesian CPG (10 x 6 x 4)
GC = compute_geometry(cart_grid([10, 6, 4], [100.0, 60.0, 40.0]))
assert GC['cells']['num'] == 10 * 6 * 4
assert 'cartDims' in GC

# --- Well: horizontal along x in layer k=2 (0-based), at y ~ 30, z ~ 22.5
k = 1
yW = 30.0
zW = 22.5
xs = np.linspace(12.0, 88.0, 5)
pW = np.column_stack([xs, np.full_like(xs, yW), np.full_like(xs, zW)])
ns = len(pW) - 1
well = {
    'name': 'PROD',
    'trajectory': pW,
    'segmentNum': ns,
    'radius': 0.5 * np.ones(ns + 1),
    'skinFactor': np.zeros(ns),
    'openedSegs': np.arange(1, ns + 1),
}

# --- VOI boundary (polygon around the well in xy)
pbdy = np.array([[5, 10], [95, 10], [95, 50], [5, 50]], dtype=float)
nextra = [1, 1]

VOI = VolumeOfInterest(GC, well, pbdy, nextra)
geoV = VOI.allInfoOfVolume()
assert len(geoV['cells']) == len(geoV['KIndices'])
print('allInfoOfVolume OK; layers:', geoV['KIndices'].tolist())

# --- Build the layered unstructured VOI grid (triangular)
WR = {'ly': 8.0, 'ny': 6, 'na': 5}
layerRf = [2, 2, 2]
GV = VOI.ReConstructToUnstructuredGrid(WR, layerRf, multiplier=0.2,
                                       maxIter=200, gridType='triangular')
assert GV['cells']['num'] == GV['surfGrid']['cells']['num'] * GV['layers']['num']
assert GV['griddim'] == 3
assert np.all(np.isfinite(GV['faces']['areas'])) and np.all(GV['faces']['areas'] > 0)
assert np.all(np.isfinite(GV['cells']['volumes'])) and np.all(GV['cells']['volumes'] > 0)
print(f'VOI grid OK: nc={GV["cells"]["num"]}, nz={GV["layers"]["num"]}')

# --- Build the layered radial HW grid (gradual)
regionIndices = [2, 5, 2, 4]   # 1-based: 1 < ymin < ymax < ny, 1 < zmin < zmax < nz
HW = HorWellRegion(GV, well, regionIndices)
geoW = HW.allInfoOfRegion()
assert len(geoW['cells']) == ns + 1
print('allInfoOfRegion OK')

radPara = {'gridType': 'gradual', 'boxRatio': [0.6, 0.6],
           'nRadCells': [4, 2], 'pDMult': 10, 'offCenter': True}
GW = HW.ReConstructToRadialGrid(radPara)
assert GW['radDims'][0] == len(geoW['bdyNodes'][0])
assert GW['cells']['num'] > 0
print(f'HW grid (gradual) OK: radDims={GW["radDims"]}')

# --- pureCircular variant
radPara2 = {'gridType': 'pureCircular', 'maxRadius': 1.0, 'nRadCells': 4}
GW2 = HW.ReConstructToRadialGrid(radPara2)
assert GW2['radDims'][0] == len(geoW['bdyNodes'][0])
print(f'HW grid (pureCircular) OK: radDims={GW2["radDims"]}')

# --- NearWellboreModel grid assembly (skip the AD fluid)
class _NWMNoFluid(NearWellboreModel):
    def setupFluid(self):
        return {}

deck = {
    'RUNSPEC': {'OIL': True, 'WATER': True, 'GAS': False,
                'VAPOIL': False, 'DISGAS': False, 'cartDims': [10, 6, 4]},
    'GRID': {'cartDims': [10, 6, 4], 'DX': 10.0 * np.ones(240),
             'DY': 10.0 * np.ones(240), 'DZ': 10.0 * np.ones(240),
             'TOPS': np.zeros(60)},
    'REGIONS': {},
    'SOLUTION': {'EQUIL': [[2600, 2550, 30, 30, 1, 0]]},
    'SCHEDULE': {'control': [{'W': []}], 'step': {'val': [1.0], 'control': [0]}},
}

NWM = _NWMNoFluid([GC, GV, GW], deck, well)
G = NWM.validateGlobalGrid()
assert G['cells']['num'] == (GC['cells']['num'] - len(geoV['cells']) * 1
                             + GV['cells']['num'] - np.concatenate(geoW['cells']).size
                             + GW['cells']['num'])
assert np.all(np.unique(G['cells']['grdID']) == [1, 2, 3])
assert np.all(np.unique(G['faces']['grdID']) == [1, 2, 3])
print(f'Global hybrid grid OK: nc={G["cells"]["num"]}, nf={G["faces"]["num"]}')

# --- Cell maps consistency
NWM.checkCellMaps()
NWM.checkFaceMaps()

# --- Rock assembly (needs deck rock); use a homogeneous rock for all
rockC = {'perm': np.tile(np.array([100.0, 100.0, 10.0]), (GC['cells']['num'], 1)),
         'poro': np.full(GC['cells']['num'], 0.2),
         'ntg': np.ones(GC['cells']['num'])}
rockV = {'perm': np.tile(np.array([100.0, 100.0, 10.0]), (GV['cells']['num'], 1)),
         'poro': np.full(GV['cells']['num'], 0.2),
         'ntg': np.ones(GV['cells']['num'])}
rockW = {'perm': np.tile(np.array([100.0, 100.0, 10.0]), (GW['cells']['num'], 1)),
         'poro': np.full(GW['cells']['num'], 0.2),
         'ntg': np.ones(GW['cells']['num'])}
rock = NWM.getGlobalRock([rockC, rockV, rockW])
assert rock['perm'].shape == (G['cells']['num'], 3)
sub = NWM.assignSubRocks(rock)
assert len(sub) == 3
print('Rock assembly OK')

# --- Transmissibility of the global grid (linear part; radial HW path
# requires the skin factors from the deck well -- not set here, so use the
# linear computeTrans directly)
T = NWM.getTransGloGrid(rock)
assert T.shape == (G['faces']['num'],)
assert np.all(np.isfinite(T[T > 0])) and np.all(T >= 0)
print(f'getTransGloGrid OK: {T.size} faces')

# --- Intersection relations + NNC (non-matching boundaries only here)
intXn = NWM.computeIntxnRelation()
print('computeIntxnRelation OK')

print('END-TO-END WORKFLOW TEST PASSED')
