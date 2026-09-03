"""Port of MRST ``VolumeOfInterest``: class for the volume of interest (VOI)
in the Corner-point grid (CPG) or Cartesian grid, which generates the
geometrical information of the CPG/Cartesian grid in the VOI and constructs
the unstructured VOI grid."""

from __future__ import annotations

import numpy as np

from .._core import (computeGeometry, gridFaceNodes, gridLogicalIndices,
                     inpolygon, mcolon, mergeOptions, tessellationGrid,
                     uniqueStable)
from ..gridding.generateVOIGridNodes import generateVOIGridNodes
from ..gridding.getConnListAndBdyNodeWR2D import getConnListAndBdyNodeWR2D
from ..gridding.makeLayeredGridNWM import makeLayeredGridNWM
from ..gridding.pointsSingleWellNode import pointsSingleWellNode
from ..utils.dispInfo import dispInfo
from ..utils.sortPtsCounterClockWise import sortPtsCounterClockWise
from ..utils.tabulate_NWM import tabulate_NWM
from .. import _deps


class VolumeOfInterest:
    """Class for the volume of interest (VOI) in the Corner-point grid (CPG)
    or Cartesian grid.

    Properties
    ----------
    CPG : dict
        CPG or Cartesian grid structure.
    well : dict
        Structure of well information.
    boundary : ndarray
        2D VOI boundary specified by the polygon.
    extraLayers : ndarray
        Extra layers above and below the layers where the well is located.
    """

    def __init__(self, G, well, pbdy, nextra, **kwargs):
        self.CPG = G
        self.well = well
        self.boundary = np.asarray(pbdy, dtype=float)
        self.extraLayers = np.asarray(nextra, dtype=int)
        self.plotVolumeBoundaries(1, plotClippedBoundary=False)
        # All well points should be located inside the boundary
        pW = np.asarray(well['trajectory'], dtype=float)
        if not np.all(inpolygon(pW[:, 0], pW[:, 1], pbdy[:, 0], pbdy[:, 1])):
            raise ValueError('Well points outside the boundary are defected, '
                             'please enlarge the boundary')

    def logicalIndices(self, c=None):
        """Get the logical indices of the CPG (list of 0-based arrays)."""
        return gridLogicalIndices(self.CPG, c)

    def layerFaceIndicator(self):
        """Find the face indicator of the layered dimension (typically 'Z')."""
        G = self.CPG
        if ('cartDims' in G) and np.asarray(G['cells']['faces']).ndim == 2 \
                and np.asarray(G['cells']['faces']).shape[1] == 2:
            return [5, 6]
        else:
            return [np.nan, np.nan]

    def logicalToArray(self, ijk):
        """Convert logical indices to array indices (0-based)."""
        I, J, K = self.logicalIndices()
        c = np.array([np.flatnonzero((I == ijk[x, 0]) & (J == ijk[x, 1])
                                     & (K == ijk[x, 2]))[0]
                      for x in range(ijk.shape[0])])
        return c

    def allInfoOfVolume(self):
        """Get all information of the volume (cells, faces, nodes, boundary
        nodes/faces, box cells/faces, ...)."""
        packed = {}
        packed['cells'] = self.cellsOfVolume()
        packed['faces'] = self.facesOfVolume(packed['cells'])
        packed['nodes'] = self.nodesOfVolume(packed['faces'])
        bn, bf = self.boundaryInfoOfVolume(packed['faces'])
        packed['bdyNodes'] = bn
        packed['bdyFaces'] = bf
        packed['boxCells'] = self.boxCellsOfVolume()
        packed['boxFaces'] = self.facesOfVolume(packed['boxCells'])
        packed['PeacemanCells'] = self.PeacemanWellCells()
        packed['clippedBoundary'] = [self.CPG['nodes']['coords'][n, :2]
                                     for n in bn]
        packed['KIndices'] = self.kIndicesFromExtraLayers()
        return packed

    def PeacemanWellCells(self):
        """Find the well cells of the Peaceman well model (requires the
        'wellpaths' module, not yet ported)."""
        pW = np.asarray(self.well['trajectory'], dtype=float)
        wph = _deps.makeSingleWellpath(pW)
        return _deps.findWellPathCells(self.CPG, wph)

    def ijIndicesFromBoundary(self):
        """Get the i and j indices of the volume from the defined 2D
        boundary."""
        pbdy = self.boundary
        # All well points should be located inside the boundary
        pW = np.asarray(self.well['trajectory'], dtype=float)
        assert np.all(inpolygon(pW[:, 0], pW[:, 1], pbdy[:, 0], pbdy[:, 1])), \
            'Well points outside the boundary were defected, try to enlarge ' \
            'the boundary'
        I, J, K = self.logicalIndices()
        # Find the VOI cells per layer
        c = [None] * (int(np.max(K)) + 1)
        for k in range(int(np.min(K)), int(np.max(K)) + 1):
            cktol = np.flatnonzero(K == k)
            xy = self.CPG['cells']['centroids'][cktol, :2]
            in_ = inpolygon(xy[:, 0], xy[:, 1], pbdy[:, 0], pbdy[:, 1])
            c[k] = cktol[in_]
        # Combine the cells and extract the logical indices
        ij = np.concatenate([np.column_stack([I[cc], J[cc]])
                             for cc in c if len(cc) > 0])
        ij = np.unique(ij, axis=0)
        # Remove 'bad' ij (appears only once)
        tabi = tabulate_NWM(ij[:, 0])
        badi = np.isin(ij[:, 0], tabi[tabi[:, 1] == 1, 0])
        tabj = tabulate_NWM(ij[:, 1])
        badj = np.isin(ij[:, 1], tabj[tabj[:, 1] == 1, 0])
        ij = ij[~(badi | badj), :]
        return ij

    def kIndicesFromExtraLayers(self):
        """Get the grid layer indices from the extra layers and the layers
        occupied by the well."""
        nex = self.extraLayers
        _, _, K = self.logicalIndices()
        wc = self.PeacemanWellCells()
        kwc = K[wc]
        kmin = np.min(kwc) - nex[0]
        kmax = np.max(kwc) + nex[1]
        k = np.arange(kmin, kmax + 1)
        k = k[(k <= np.max(K)) & (k >= np.min(K))]
        return k

    def cellsOfVolume(self):
        """Get all cells of the volume (one array per layer)."""
        k = self.kIndicesFromExtraLayers()
        return [self.getCellsSingleLayer(kk) for kk in k]

    def getCellsSingleLayer(self, k):
        """Get the layer-``k`` cells inside the defined 2D polygon."""
        ij = self.ijIndicesFromBoundary()
        ijk = np.column_stack([ij, np.full(len(ij), k)])
        return self.logicalToArray(ijk)

    def facesOfVolume(self, c):
        """Get all layer-faces of the volume."""
        indicator = self.layerFaceIndicator()
        f = [self.getLayerFacesFromCells(cc, indicator[0]) for cc in c]
        f0 = self.getLayerFacesFromCells(c[-1], indicator[1])
        return f + [f0]

    def getLayerFacesFromCells(self, c, indicator):
        """Get the layer-faces of cell ``c`` in a single layer."""
        G = self.CPG
        facePos = G['cells']['facePos']
        rows = mcolon(facePos[c], facePos[c + 1] - 1)
        cf = G['cells']['faces'][rows, 0]
        dire = G['cells']['faces'][rows, 1]
        f = cf[dire == indicator]
        return f

    def nodesOfVolume(self, f):
        """Get all nodes of the layer-faces of the volume."""
        return [self.getNodesFromFaces(ff) for ff in f]

    def getNodesFromFaces(self, f):
        """Get the nodes of the layer-faces on a single surface."""
        return [gridFaceNodes(self.CPG, ff)[0] for ff in f]

    def boxCellsOfVolume(self):
        """Get all box cells of the volume."""
        k = self.kIndicesFromExtraLayers()
        return [self.getBoxCellsSingleLayer(kk) for kk in k]

    def getBoxCellsSingleLayer(self, k):
        """Get the layer-``k`` box cells (the defined 2D boundary is located
        inside the box)."""
        I, J, K = self.logicalIndices()
        ij = self.ijIndicesFromBoundary()
        en = 3
        imin = np.min(ij[:, 0]) - en
        imax = np.max(ij[:, 0]) + en
        jmin = np.min(ij[:, 1]) - en
        jmax = np.max(ij[:, 1]) + en
        boxc = np.flatnonzero((I >= imin) & (I <= imax) & (J >= jmin)
                              & (J <= jmax) & (K == k))
        return boxc

    def boundaryInfoOfVolume(self, f):
        """Get all boundary nodes and layer-faces of the volume."""
        results = [self.getBoundaryInfoSingleSurface(ff) for ff in f]
        bn = [r[0] for r in results]
        bf = [r[1] for r in results]
        return bn, bf

    def getBoundaryInfoSingleSurface(self, f):
        """Get the boundary information of the faces on a single surface:
        returns ``(bn, bf)`` - sorted boundary nodes and boundary faces."""
        G = self.CPG
        n = [gridFaceNodes(G, ff)[0] for ff in f]
        n = sortPtsCounterClockWise(G['nodes']['coords'][:, :2], n)
        assert all(len(x) == 4 for x in n)
        # Build a local grid g to find the boundary nodes of G
        nd = np.concatenate(n)
        nu, _, ic = np.unique(nd, return_index=True, return_inverse=True)
        t = ic.reshape(-1, 4)
        p = G['nodes']['coords'][nu, :2]
        g = tessellationGrid(p, t)
        g = computeGeometry(g)
        Ng = g['faces']['neighbors']
        # Boundary faces of g, sorted, counter-clockwise
        bfg = np.flatnonzero(~np.all(Ng >= 0, axis=1))
        bfg = sortPtsCounterClockWise(g['faces']['centroids'], [bfg])[0]
        # Nodes of bf, also boundary nodes of g
        bfng, pos = gridFaceNodes(g, bfg)
        assert np.all(np.diff(pos) == 2)
        bfng = bfng.reshape(-1, 2)
        # Boundary nodes of the VOI in g, sorted, counter-clockwise
        bng = [bfng[r, ~np.isin(bfng[r], bfng[r - 1])]
               for r in range(1, bfng.shape[0] - 1)]
        idx = np.isin(bfng[0], bfng[1])
        bng = np.concatenate([bfng[0, ~idx], bfng[0, idx]] + bng)
        # Boundary nodes of the VOI in G, sorted, counter-clockwise
        bn = nu[bng]
        # Preparations for building the Voronoi grid:
        # boundary cells in g
        bcg = np.maximum(Ng[bfg, 0], Ng[bfg, 1])
        bcg = uniqueStable(bcg)
        N = np.column_stack([bcg, np.concatenate([bcg[1:], bcg[:1]])])
        cc = np.full(len(bcg), -1, dtype=np.int64)
        for ii in range(N.shape[0]):
            c1 = Ng[np.any(Ng == N[ii, 0], axis=1), :]
            c1 = np.unique(c1)
            c1 = c1[(c1 >= 0) & (c1 != N[ii, 0])]
            c2 = Ng[np.any(Ng == N[ii, 1], axis=1), :]
            c2 = np.unique(c2)
            c2 = c2[(c2 >= 0) & (c2 != N[ii, 1])]
            inter = np.intersect1d(c1, c2)
            if inter.size:
                cc[ii] = inter[0]
        # Insert the 'Z' cells (MATLAB: bcg = [bcg, cc]'; bcg = bcg(:); --
        # the transpose before the column-major ravel interleaves bcg/cc
        # element-by-element: [bcg(1),cc(1),bcg(2),cc(2),...], not block-
        # order; ravel(order='C') on the (m,2) stack gives that directly).
        bcg = np.column_stack([bcg, cc]).ravel(order='C')
        bcg = bcg[bcg != -1]
        # Boundary faces of the VOI in g (cell index of g equals the face
        # index of the VOI face)
        bf = f[bcg]
        assert len(bf) == len(np.unique(bf)), \
            'Isolate boundary faces are detected, please redefine the ' \
            'boundary polygon'
        return bn, bf

    def prepareWellRegionNodes2D(self, WR):
        """Prepare the 2D well-region (WR) nodes: the WR is composed of a
        Cartesian region and two half-radial regions in the xy plane."""
        pW = np.asarray(self.well['trajectory'], dtype=float)
        nx = self.well['segmentNum']
        ny = np.atleast_1d(np.asarray(WR['ny'], dtype=int))
        for i in range(len(ny)):
            if ny[i] % 2 == 1:
                print(f'ny must be an even number, ny({i})+1 [{ny[i] + 1}] is '
                      'used instead')
                ny[i] = ny[i] + 1
        ly = np.atleast_1d(np.asarray(WR['ly'], dtype=float))
        na = WR['na']

        # Generate the WR points corresponding to all well nodes
        p = [pointsSingleWellNode(pW, ly, ny, na, ii) for ii in range(nx + 1)]
        pall = np.vstack([np.vstack([pp['cart'] for pp in p]),
                          np.vstack([pp['rad'] for pp in p])])
        pbdy = self.boundary
        assert np.all(inpolygon(pall[:, 0], pall[:, 1],
                                pbdy[:, 0], pbdy[:, 1])), \
            'Points outside the boundary were detected, please reduce the ' \
            'size of the Cartesian region'

        # Get the connectivity list and boundary nodes of the WR
        t, tC, bn, bnC = getConnListAndBdyNodeWR2D(p, int(np.sum(ny)), na)

        # Assign data to WR
        WR = dict(WR)
        WR['points'] = p
        WR['connlist'] = t
        WR['connlistC'] = tC
        WR['bdnodes'] = bn
        WR['bdnodesC'] = bnC
        WR['cartDims'] = [nx, int(np.sum(ny))]
        return WR

    def ReConstructToUnstructuredGrid(self, WR, layerRf, **kwargs):
        """Reconstruct the CPG in the VOI to a layered unstructured grid.

        The open-source triangle generator 'DistMesh' is used to obtain
        high-quality triangles; the scaled edge length function is
        ``h(p) = max(multiplier*d(p) + lIB, lOB)``.
        """
        print(' -- Reconstructing the CPG to unstructured VOI grid')
        opt = mergeOptions({'multiplier': 0.2, 'maxIter': 500,
                            'gridType': 'triangular'}, **kwargs)

        # Geometrical info of the VOI
        packed = self.allInfoOfVolume()
        if 'points' not in WR:
            WR = self.prepareWellRegionNodes2D(WR)

        # Generate the nodes for the unstructured grid
        pSurfs, t, bdyID = generateVOIGridNodes(self.CPG, packed, WR,
                                                layerRf, opt)

        # Construct the 2D grid
        p = pSurfs[0][:, :2]
        t = sortPtsCounterClockWise(p, t)
        gV = tessellationGrid(p, t)
        gV['nodes']['boundary'] = bdyID
        gV['cartDims'] = WR['cartDims']

        # Extrude the 2D grid to the 3D grid
        GV = makeLayeredGridNWM(gV, pSurfs, connectivity=t)
        layerRf = np.atleast_1d(np.asarray(layerRf, dtype=int))
        GV['layers']['refinement'] = layerRf[:len(packed['cells'])]
        GV['parentInfo'] = packed

        dispInfo(GV)
        return GV

    # -- plotting (demonstration aids) -------------------------------------

    def plotVolumeBoundaries(self, packed, **kwargs):
        """Plot the user-defined boundary and the clipped boundary."""
        opt = mergeOptions({'plotClippedBoundary': True}, **kwargs)
        pW = np.asarray(self.well['trajectory'], dtype=float)
        pB = self.boundary
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(pW[:, 0], pW[:, 1], pW[:, 2], 's-', label='Well trajectory')
        pBClose = np.vstack([pB, pB[0]])
        ax.plot(pBClose[:, 0], pBClose[:, 1],
                pW[0, 2] * np.ones(len(pBClose)), 'o-',
                label='Specified VOI boundary')
        if opt['plotClippedBoundary'] and not isinstance(packed, int):
            pBClipped = self.CPG['nodes']['coords'][packed['bdyNodes'][0], :]
            ax.plot(pBClipped[:, 0], pBClipped[:, 1], pBClipped[:, 2], 'g.-',
                    label='Clipped VOI boundary')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend()
        return fig

    def plotVolumeCells(self, packed):
        """Plot the cells inside the volume."""
        import matplotlib.pyplot as plt

        from PRSTCore.visualization import plot_grid

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        rng = np.random.default_rng(0)
        for cells in packed['cells']:
            plot_grid(self.CPG, cells, ax=ax, facecolor=rng.random(3))
        ax.set_title('Cells inside the volume')
        return fig

    def plotVolumeLayerFaces(self, packed):
        """Plot the layer-faces of the volume."""
        import matplotlib.pyplot as plt

        from PRSTCore.visualization import plot_faces

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        rng = np.random.default_rng(0)
        for faces in packed['faces']:
            plot_faces(self.CPG, faces, ax=ax, facecolor=rng.random(3), colorbar=False)
        ax.set_title('Layer-faces of the volume')
        return fig

    def plot2DWRSubGrid(self, WR):
        """Plot the subgrid of the 2D well region."""
        import matplotlib.pyplot as plt

        from .._core import tessellationGrid
        from PRSTCore.visualization import plot_grid

        if 'points' not in WR:
            WR = self.prepareWellRegionNodes2D(WR)
        pWR = np.vstack([np.vstack([pp['cart'] for pp in WR['points']]),
                         np.vstack([pp['rad'] for pp in WR['points']])])
        gWR = tessellationGrid(pWR, WR['connlist'])
        pW = np.asarray(self.well['trajectory'], dtype=float)

        fig, ax = plt.subplots()
        ax.plot(pW[:, 0], pW[:, 1], 'rs-', label='Well trajectory')
        ax.legend()
        plot_grid(gWR, ax=ax, facecolor='none')
        ax.set_title('2D well region subgrid')
        return fig

    def maxWellSegLength2D(self):
        """Display the maximum 2D length of the well segments."""
        pW = np.asarray(self.well['trajectory'], dtype=float)[:, :2]
        L = np.diff(pW, axis=0)
        L = np.sqrt(np.sum(L ** 2, axis=1))
        print(f'    Info : The maximum well-segment length in 2D is '
              f'{np.max(L):.2f}')

    def volumeLayerNumber(self):
        """Display the number of volume layers."""
        k = self.kIndicesFromExtraLayers()
        print(f'    Info : The number of VOI layers is {len(k)} (', end='')
        print(' '.join(str(kk) for kk in k), end='')
        print(')')
