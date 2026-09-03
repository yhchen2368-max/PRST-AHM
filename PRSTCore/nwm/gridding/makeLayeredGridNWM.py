"""Port of MRST ``makeLayeredGridNWM``: extrude a 2D grid to a layered 3D
grid according to the topology of the 2D grid and the provided surface point
sets (topologically aligned in the layered direction)."""

import numpy as np

from .._core import computeGeometry, gridCellNodes, mergeOptions
from ..utils.sortPtsCounterClockWise import sortPtsCounterClockWise


def getNodesOfcell(G):
    """Sorted (counter-clockwise) node list of each cell of the 2D grid ``G``."""
    cn, pos = gridCellNodes(G, np.arange(G['cells']['num']))
    cn = [cn[pos[c]:pos[c + 1]] for c in range(G['cells']['num'])]
    cn = sortPtsCounterClockWise(G['nodes']['coords'][:, :2], cn)
    return cn


def makeLayeredGridNWM(g, p, **kwargs):
    """Extrude the 2D grid ``g`` to a layered 3D grid.

    Parameters
    ----------
    g : dict
        The 2D grid to be extruded.
    p : list of ndarray
        Points of all surfaces (``nz+1`` sets of ``np x 3``), topologically
        aligned in the layered direction.
    connectivity : list of 1D arrays, optional
        Connectivity list (nodes of cells) for ``g``; required if the 2D
        grid is not on the xy plane.

    Returns
    -------
    G : dict
        Valid 2.5D layered grid structure (with geometry).
    """
    opt = mergeOptions({'connectivity': None}, **kwargs)

    p = list(p)
    # All sets of points should have 3 columns and the same number of rows
    assert all(np.asarray(x).ndim == 2 and np.asarray(x).shape[1] == 3 for x in p)
    np_pts = np.unique([np.asarray(x).shape[0] for x in p])
    assert len(np_pts) == 1, 'Point sets must have the same number of rows'
    np_ = int(np_pts[0])

    # Cell number in the layered direction
    nz = len(p) - 1
    nc_g = g['cells']['num']
    nf_g = g['faces']['num']

    G = {}
    # G.cells
    G['cells'] = {}
    G['cells']['num'] = nc_g * nz

    ncf_g = np.diff(g['cells']['facePos'])
    # surface-major: for each layer (nz), all 2D cells -- matches the cell
    # numbering (layer-major, see cells.layers) and the cf data below (and
    # MATLAB's repmat(ncf_g + 2, 1, nz) + (:)).
    ncf = np.tile(ncf_g + 2, nz)
    G['cells']['facePos'] = np.concatenate([[0], np.cumsum(ncf)])

    cf_g = [g['cells']['faces'][g['cells']['facePos'][c]:g['cells']['facePos'][c + 1], 0]
            for c in range(nc_g)]
    cf = [[None] * nz for _ in range(nc_g)]
    dire = [[None] * nz for _ in range(nc_g)]
    for k in range(nz):
        fXY = [x + k * nf_g for x in cf_g]
        fZ = [np.array([nf_g * nz + k * nc_g + x, nf_g * nz + (k + 1) * nc_g + x],
                       dtype=np.int64) for x in range(nc_g)]
        for c in range(nc_g):
            cf[c][k] = np.concatenate([fXY[c], fZ[c]])
            dire[c][k] = np.concatenate([np.ones(len(fXY[c]), dtype=np.int64), [5, 6]])
    cf = np.concatenate([cf[c][k] for k in range(nz) for c in range(nc_g)])
    dire = np.concatenate([dire[c][k] for k in range(nz) for c in range(nc_g)])
    G['cells']['faces'] = np.column_stack([cf, dire])

    layers = np.repeat(np.arange(1, nz + 1), nc_g)
    G['cells']['layers'] = layers

    # G.faces
    G['faces'] = {}
    G['faces']['num'] = nf_g * nz + (nz + 1) * nc_g
    assert G['faces']['num'] == int(G['cells']['faces'][:, 0].max()) + 1

    G['faces']['surfaces'] = np.zeros(G['faces']['num'], dtype=np.int64)
    surfaces = np.repeat(np.arange(1, nz + 2), nc_g)
    G['faces']['surfaces'][nf_g * nz:] = surfaces
    nlayerF = int(np.count_nonzero(G['faces']['surfaces'] == 0))
    layerF = np.repeat(np.arange(1, nz + 1), nlayerF // nz)
    layerF = np.concatenate([layerF,
                             np.zeros(int(np.count_nonzero(G['faces']['surfaces'] > 0)),
                                      dtype=np.int64)])
    assert len(layerF) == G['faces']['num']
    G['faces']['layers'] = layerF

    nfn1 = np.full(nf_g * nz, 4, dtype=np.int64)
    if opt['connectivity'] is None:
        cn_g = getNodesOfcell(g)
    else:
        cn_g = opt['connectivity']
    if isinstance(cn_g, np.ndarray):
        cn_g = [np.asarray(row, dtype=np.int64).ravel() for row in cn_g]
    else:
        cn_g = [np.asarray(x, dtype=np.int64).ravel() for x in cn_g]

    nfn2 = np.array([len(x) for x in cn_g], dtype=np.int64)
    # surface-major: for each surface (nz+1), all cells -- matches the fn2
    # node data below (and MATLAB's repmat(..., 1, nz+1) + (:)).
    nfn2 = np.tile(nfn2, nz + 1)
    nfn = np.concatenate([nfn1, nfn2])
    G['faces']['nodePos'] = np.concatenate([[0], np.cumsum(nfn)])

    fn_g = [g['faces']['nodes'][g['faces']['nodePos'][f]:g['faces']['nodePos'][f + 1]]
            for f in range(nf_g)]
    fn1 = []
    for k in range(nz):
        for f in range(nf_g):
            x = fn_g[f]
            fn1.append(np.concatenate([x[[0, 1]] + k * np_, x[[1, 0]] + (k + 1) * np_]))
    fn1 = np.concatenate(fn1)

    fn2 = []
    for k in range(nz + 1):
        for c in range(nc_g):
            fn2.append(cn_g[c] + k * np_)
    fn2 = np.concatenate(fn2)

    G['faces']['nodes'] = np.concatenate([fn1, fn2])

    # G.nodes
    G['nodes'] = {}
    G['nodes']['num'] = np_ * (nz + 1)
    G['nodes']['coords'] = np.vstack(p)

    # G.type / G.griddim / G.layers / G.surfGrid
    G['type'] = list(g.get('type', [])) + ['makeLayeredGridNWM']
    G['griddim'] = 3
    G['layers'] = {'num': nz}
    G['surfGrid'] = g

    # G.faces.neighbors
    G = computeGeometry(G, findNeighbors=True)
    return G
