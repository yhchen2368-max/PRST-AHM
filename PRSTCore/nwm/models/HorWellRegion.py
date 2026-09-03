"""Port of MRST ``HorWellRegion``: class for the horizontal-well (HW) region
in the volume-of-interest (VOI) grid, which generates the geometrical
information of the VOI grid and constructs the radial HW grid."""

from __future__ import annotations

import numpy as np

from .._core import gridCellFaces, gridFaceNodes, mergeOptions
from ..gridding.buildRadialGrid import buildRadialGrid
from ..gridding.generateHWGridNodes import generateHWGridNodes
from ..gridding.makeLayeredGridNWM import makeLayeredGridNWM


class HorWellRegion:
    """Class for the horizontal-well (HW) region in the volume-of-interest
    (VOI) grid.

    Properties
    ----------
    GVOI : dict
        Layered unstructured VOI grid.
    regionIndices : ndarray
        Logical indices of the HW region in the VOI grid.
    well : dict
        Structure of well information.
    """

    def __init__(self, G, well, regionIndices, **kwargs):
        # The HW grid is built inside the Cartesian region of the VOI grid,
        # and the logical indices of the HW region are specified:
        #   regionIndices: [ymin, ymax, zmin, zmax]  (1 < ymin < ymax < ny,
        #                                             1 < zmin < zmax < nz)
        self.GVOI = G
        self.well = well
        self.regionIndices = np.asarray(regionIndices, dtype=int)
        self.checkRegionIndices()

    def checkRegionIndices(self):
        """Check whether the region indices exceed the Cartesian-region
        dimension of the VOI grid."""
        nx, ny, nz = self.assignCartDimsOfVOIGrid()
        assert self.well['segmentNum'] == nx
        ymin, ymax, zmin, zmax = self.assignRegionIndices()
        assert 1 < ymin < ymax < ny, \
            '1 < Indices(1) < Indices(2) < %2.0f is not satisfied' % ny
        assert 1 < zmin < zmax < nz, \
            '1 < Indices(3) < Indices(4) < %2.0f is not satisfied' % nz

    def allInfoOfRegion(self):
        """Get all information of the region (cells, layer-faces, nodes of
        layer-faces, boundary nodes and indices of vertices)."""
        packed = {}
        packed['cells'] = self.cellsOfRegion()
        packed['nodes'] = self.nodesOfRegion()
        packed['faces'] = self.facesOfRegion(packed['cells'], packed['nodes'])
        packed['bdyNodes'] = self.bdyNodesOfRegion()
        packed['vertexID'] = self.IDOfFourVertices()
        return packed

    def cellsOfRegion(self):
        """Get the cells of the HW region in the VOI grid.

        ``regionIndices``/``assignRegionIndices`` stay 1-based (matching
        MRST's literal ``[ymin, ymax, zmin, zmax]`` convention, as set by
        callers), but ``GVOI``'s own cell/node arrays are 0-based
        (PRSTCore's grid convention) -- every formula below is MRST's
        exact 1-based arithmetic, with a final ``- 1`` to convert the
        resulting (validly-computed) 1-based cell number to a 0-based
        Python index. Column *selectors* (``c_layers``, used to index into
        a NumPy array) are separately converted to 0-based via ``- 1``
        where they are used, since that is an array-indexing operation,
        not a cell-number value.
        """
        G = self.GVOI
        nx, ny, nz = self.assignCartDimsOfVOIGrid()
        ymin, ymax, zmin, zmax = self.assignRegionIndices()
        c_y = np.arange(ymin, ymax + 1)
        c_yz = np.tile(c_y, (nz, 1)).T
        c_yz = c_yz + np.arange(nz)[None, :] * (G['cells']['num'] // nz)
        c_layers = np.arange(zmin, zmax + 1) - 1
        c_yz = c_yz[:, c_layers]
        c_yz = c_yz.ravel(order='F')
        c = [c_yz + x * ny - 1 for x in range(nx)]
        return c

    def nodesOfRegion(self):
        """Get the nodes of the layer-faces of the HW region in the VOI
        grid."""
        nx, ny, _ = self.assignCartDimsOfVOIGrid()
        n_yz = self.getNodesSingleSurface()
        n = [n_yz.ravel(order='F') + x * (ny + 1) for x in range(nx + 1)]
        return n

    def facesOfRegion(self, c, n):
        """Get the layer-faces of the HW region in the VOI grid."""
        G = self.GVOI

        def faceFun(c0):
            return gridCellFaces(G, c0)[0]

        def nodeFun(f):
            return gridFaceNodes(G, f)[0]

        f = [None] * len(n)
        c = c + [c[-1]]
        for k in range(len(f)):
            ck = c[k]
            fk = [faceFun(cc) for cc in ck]
            f[k] = np.zeros(len(ck), dtype=np.int64)
            for j in range(len(ck)):
                fkj = fk[j]
                nds = [nodeFun(ff) for ff in fkj]
                idx = [np.all(np.isin(nd, n[k])) for nd in nds]
                f[k][j] = fkj[np.array(idx, dtype=bool)][0]
        return f

    def bdyNodesOfRegion(self):
        """Get the boundary nodes of the HW region in the VOI grid."""
        nx, ny, _ = self.assignCartDimsOfVOIGrid()
        n_yz = self.getNodesSingleSurface()
        bn_yz = np.concatenate([n_yz[0, :],
                                n_yz[1:-1, -1],
                                n_yz[-1, ::-1],
                                n_yz[-2:0:-1, 0]])
        bn = [bn_yz + x * (ny + 1) for x in range(nx + 1)]
        return bn

    def IDOfFourVertices(self):
        """Get the indices of the four vertices in the boundary nodes."""
        n_yz = self.getNodesSingleSurface()
        vx = np.array([n_yz[0, 0], n_yz[0, -1], n_yz[-1, -1], n_yz[-1, 0]])
        bn_yz = np.concatenate([n_yz[0, :],
                                n_yz[1:-1, -1],
                                n_yz[-1, ::-1],
                                n_yz[-2:0:-1, 0]])
        vxID = np.flatnonzero(np.isin(bn_yz, vx))
        return vxID

    def cartCellsOfVOIGrid(self):
        """Get the cells of the VOI grid in the Cartesian region. See
        :meth:`cellsOfRegion`'s docstring for the indexing convention."""
        G = self.GVOI
        nx, ny, nz = self.assignCartDimsOfVOIGrid()
        c = np.arange(1, ny + 1)[:, None] + (G['cells']['num'] // nz) * np.arange(nz)[None, :]
        c = c.ravel(order='F')
        c = [c + (i - 1) * ny - 1 for i in range(1, nx + 1)]
        return c

    def getNodesSingleSurface(self):
        """Get the nodes of the HW region in the VOI grid on a single
        surface. See :meth:`cellsOfRegion`'s docstring for the 1-based
        MRST-arithmetic / final-``-1`` conversion convention used here."""
        G = self.GVOI
        _, _, nz = self.assignCartDimsOfVOIGrid()
        ymin, ymax, zmin, zmax = self.assignRegionIndices()
        n_y = np.arange(ymin, ymax + 2)
        n_yz = np.tile(n_y, (nz + 1, 1)).T
        n_yz = n_yz + np.arange(nz + 1)[None, :] * (G['nodes']['num'] // (nz + 1))
        n_layers = np.arange(zmin - 1, zmax + 1)
        n_yz = n_yz[:, n_layers]
        return n_yz.T - 1

    def assignRegionIndices(self):
        """Assign the region indices."""
        Ind = self.regionIndices
        return Ind[0], Ind[1], Ind[2], Ind[3]

    def assignCartDimsOfVOIGrid(self):
        """Assign the Cartesian dimensions of the VOI grid."""
        G = self.GVOI
        G2 = G['surfGrid']
        return int(G2['cartDims'][0]), int(G2['cartDims'][1]), int(G['layers']['num'])

    def ReConstructToRadialGrid(self, radPara):
        """Reconstruct the VOI grid in the HW region to a layered radial grid.

        Two types of grid lines are provided:
          'pureCircular' : the radial grid lines are pure circular
                           (requires 'maxRadius' and 'nRadCells')
          'gradual'      : the radial grid lines vary from the circular line
                           to the rectangular line of a specified box
                           gradually (requires 'boxRatio', 'nRadCells',
                           'pDMult' and 'offCenter')
        """
        print(' -- Reconstructing the VOI grid to radial HW grid')

        # Geometrical info of the HW region
        packed = self.allInfoOfRegion()
        pSurfs, pSurfXY, wellbores = \
            generateHWGridNodes(self.GVOI, packed, self.well, radPara)

        # Construct the 2D grid
        p = pSurfXY[0]
        nA = np.unique([len(x) for x in packed['bdyNodes']])
        assert len(nA) == 1
        nA = int(nA[0])
        nRadCells = np.atleast_1d(np.asarray(radPara['nRadCells'], dtype=int))
        gW, t = buildRadialGrid(p, nA, int(np.sum(nRadCells)))
        gW['nodes']['boundary'] = np.arange(gW['nodes']['num'] - nA,
                                            gW['nodes']['num'])

        # Rewrite the radial dimensions, to be compatible with
        # 'computeRadTransFactor'
        gridType = radPara['gridType']
        if gridType == 'pureCircular':
            # The outer-most radial cells are not 'real radial cells'
            gW['radDims'] = [nA, int(nRadCells[0] - 1), 1]
        elif gridType == 'gradual':
            # The radial cells outside the box are not 'real radial cells'
            gW['radDims'] = [nA] + [int(x) for x in nRadCells]
        else:
            raise ValueError(f'Unknown radial grid type: {gridType}')

        # Extrude the 2D grid to the 3D grid
        GW = makeLayeredGridNWM(gW, pSurfs, connectivity=t)
        GW['radDims'] = list(gW['radDims']) + [GW['layers']['num']]
        GW['layers']['refinement'] = np.ones(GW['layers']['num'], dtype=np.int64)
        GW['wellbores'] = wellbores
        GW['layers']['coordsXY'] = pSurfXY
        GW['parentInfo'] = packed
        return GW

    def showWellRegionInVOIGrid(self, **kwargs):
        """Show the well region in the VOI grid."""
        import matplotlib.pyplot as plt

        from PRSTCore.visualization import plot_grid

        opt = mergeOptions({'showWellRgionCells': True}, **kwargs)
        G = self.GVOI
        pW = np.asarray(self.well['trajectory'], dtype=float)
        cCart = self.cartCellsOfVOIGrid()
        cReg = self.cellsOfRegion()
        cRes = cCart[0][~np.isin(cCart[0], cReg[0])]

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        rng = np.random.default_rng(0)
        plot_grid(G, cRes, ax=ax, facecolor=rng.random(3))
        if opt['showWellRgionCells']:
            plot_grid(G, cReg[0], ax=ax, facecolor=rng.random(3))
            ax.plot([pW[0, 0]], [pW[0, 1]], [pW[0, 2]], 'rs', markersize=10)
        return fig

    def plotRegionCells(self, packed):
        """Plot the cells inside the HW region."""
        import matplotlib.pyplot as plt

        from PRSTCore.visualization import plot_grid

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        rng = np.random.default_rng(0)
        for cells in packed['cells']:
            plot_grid(self.GVOI, cells, ax=ax, facecolor=rng.random(3))
        ax.set_title('Cells inside the region')
        return fig

    def plotRegionLayerFaces(self, packed):
        """Plot the layer-faces of the HW region."""
        import matplotlib.pyplot as plt

        from PRSTCore.visualization import plot_faces

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        rng = np.random.default_rng(0)
        for faces in packed['faces']:
            plot_faces(self.GVOI, faces, ax=ax, facecolor=rng.random(3), colorbar=False)
        ax.set_title('Layer-faces of the region')
        return fig
