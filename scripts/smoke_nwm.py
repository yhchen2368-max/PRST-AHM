"""Smoke test for the ported nwm module (exercise the core geometry pipeline)."""
import numpy as np

from PRSTCore.nwm._core import (tessellationGrid, computeGeometry, removeCells,
                                gridCellNodes, gridFaceNodes, gridLogicalIndices)
from PRSTCore.nwm.gridding.buildRadialGrid import buildRadialGrid
from PRSTCore.nwm.gridding.makeLayeredGridNWM import makeLayeredGridNWM
from PRSTCore.nwm.gridding.makeConnListFromMat import makeConnListFromMat
from PRSTCore.nwm.gridding.pointsSingleWellNode import pointsSingleWellNode
from PRSTCore.nwm.gridding.getConnListAndBdyNodeWR2D import getConnListAndBdyNodeWR2D
from PRSTCore.nwm.gridding.assembleGrids import assembleGrids
from PRSTCore.nwm.gridding.radCartHybridGrid import radCartHybridGrid
from PRSTCore.nwm.trans.computeRadTransFactor import computeRadTransFactor
from PRSTCore.nwm.utils.tri_area import tri_area
from PRSTCore.nwm.utils.computeCentroids import computeCentroids
from PRSTCore.nwm.utils.circleCross import circleCross
from PRSTCore.nwm.utils.polyintersect import polyintersect
from PRSTCore.nwm.utils.computePD import computePD

# --- 1. tessellationGrid: 2x2 quad mesh from connectivity
p = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [2., 0.], [2., 1.]])
t = [np.array([0, 1, 3, 2]), np.array([1, 4, 5, 3])]
G = tessellationGrid(p, t)
G = computeGeometry(G)
assert G['cells']['num'] == 2 and G['faces']['num'] == 7
assert G['nodes']['num'] == 6
assert np.allclose(G['faces']['neighbors'][1], [0, 1])  # shared face (1,3)
assert G['faces']['neighbors'][0, 1] == -1              # boundary
assert np.allclose(G['cells']['centroids'][0], [0.5, 0.5])
print('tessellationGrid + computeGeometry OK')

# --- 2. buildRadialGrid + computeRadTransFactor (MRST example)
nA, nR, rW, rM = 40, 10, 2.0, 10.0
th = np.linspace(0, 2 * np.pi, nA + 1)[:-1]
r = np.logspace(np.log10(rW), np.log10(rM), nR + 1)
Rg, TH = np.meshgrid(r, th)
px, py = TH * 0 + Rg * np.cos(TH), Rg * np.sin(TH)
pr = np.column_stack([px.ravel(order='F'), py.ravel(order='F')])
GR, tR = buildRadialGrid(pr, nA, nR)
assert GR['radDims'] == [nA, nR]
ft = computeRadTransFactor(GR, np.zeros(2), 0.0)
assert ft.shape[0] == nA * nR * 4
assert np.all(np.isfinite(ft)) and np.all(ft > 0)
print('buildRadialGrid + computeRadTransFactor OK')

# --- 3. makeConnListFromMat + pointsSingleWellNode + getConnListAndBdyNodeWR2D
pW = np.array([[0., 0., 0.], [10., 0., 0.], [20., 0., 0.], [30., 0., 0.]])
ps = pointsSingleWellNode(pW, 15.0, 10, 5, 1)
assert ps['cart'].shape[0] == 11
tC_wr, t_wr, bn_wr, bnC_wr = getConnListAndBdyNodeWR2D([ps] * 4, 10, 5)
assert len(t_wr) > 0 and len(bn_wr) > 0 and len(bnC_wr) > 0
print('pointsSingleWellNode + getConnListAndBdyNodeWR2D OK')

# --- 4. makeLayeredGridNWM: extrude a 2x2 grid to 3 layers
G2 = tessellationGrid(p, t)
nz = 3
pSurfs = [np.column_stack([G2['nodes']['coords'], k * np.ones(G2['nodes']['num'])])
          for k in range(nz + 1)]
G3 = makeLayeredGridNWM(G2, pSurfs)
assert G3['cells']['num'] == 2 * nz
assert G3['faces']['num'] == 7 * nz + (nz + 1) * 2
assert G3['griddim'] == 3 and G3['layers']['num'] == nz
assert np.all(G3['faces']['neighbors'][G3['faces']['neighbors'] >= 0].max() < G3['cells']['num'])
print('makeLayeredGridNWM OK')

# --- 5. assembleGrids
Ga = assembleGrids([G, G2])
assert Ga['cells']['num'] == G['cells']['num'] + G2['cells']['num']
assert np.all(np.unique(Ga['cells']['grdID']) == [1, 2])
print('assembleGrids OK')

# --- 6. removeCells + gridLogicalIndices
Gr, cellmap, facemap, nodemap = removeCells(G, [0])
assert Gr['cells']['num'] == 1
assert cellmap[0] == 1  # new cell 0 -> old cell 1
print('removeCells OK')

# --- 7. radCartHybridGrid (MRST example)
from PRSTCore.gridprocessing.cart_grid import cart_grid
GC = cart_grid([20, 20], [200, 200])
GC = computeGeometry(GC)
ij = gridLogicalIndices(GC)
CI = np.flatnonzero((ij[0] >= 10) & (ij[0] <= 14) & (ij[1] >= 10) & (ij[1] <= 14))
pCI = GC['cells']['centroids'][CI]
pW2 = 0.5 * np.array([pCI[:, 0].min() + pCI[:, 0].max(),
                      pCI[:, 1].min() + pCI[:, 1].max()])
HG, tH = radCartHybridGrid(GC, CI, 0.2, 16.0, 10, pW2)
assert HG['cells']['num'] > 0
print('radCartHybridGrid OK')

# --- 8. utils
assert abs(tri_area(np.array([0., 0.]), np.array([1., 0.]), np.array([0., 1.])) - 0.5) < 1e-12
assert np.allclose(computeCentroids(np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])),
                   [0.5, 0.5])
pc = circleCross(0, 0, 1, 1, 0, 1)
assert np.allclose(np.sum(pc ** 2, axis=1), 1.0)  # both points on both circles
# Closed polygons are required (first point repeated at the end)
xs = np.array([0., 2., 2., 0., 0.]); ys = np.array([0., 0., 2., 2., 0.])
xd = np.array([-1., 1., 1., -1., -1.]); yd = np.array([1., 1., -1., -1., 1.])
xr, yr = polyintersect(xs, ys, xd, yd)
pts = np.column_stack([xr, yr])
assert len(pts) >= 2
assert np.all(np.isin(pts, [[1., 0.], [0., 1.]]).all(axis=1))
print('polyintersect intersections:', pts.tolist())
assert computePD(1.0, 1.0, 10.0, 10.0, 5.0, 5.0) > 0
print('utils OK')

print('ALL SMOKE TESTS PASSED')
