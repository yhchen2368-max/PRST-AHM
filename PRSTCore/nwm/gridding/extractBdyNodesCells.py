"""Port of MRST ``extractBdyNodesCells``: a 2D version of
``VolumeOfInterest.getBoundaryInfoSingleSurface`` which extracts the sorted
boundary nodes and cells (counter-clockwise) of an inner continuous region
('region of interest') specified by cells ``cI``."""

import numpy as np

from .._core import (computeGeometry, gridCellNodes, gridFaceNodes,
                     mergeOptions, tessellationGrid)
from ..utils.sortPtsCounterClockWise import sortPtsCounterClockWise


def extractBdyNodesCells(G, cI, **kwargs):
    """Extract the sorted boundary nodes and cells (counter-clockwise) of a
    region of interest specified by cells ``cI`` in the 2D grid ``G``.

    Returns ``(bdNodes, bdCells)``.
    """
    opt = mergeOptions({'plotResults': True}, **kwargs)
    cI = np.asarray(cI, dtype=np.int64).ravel()

    # Build a local grid 'g' from nodes (connectivity list) of cI
    n = [gridCellNodes(G, c)[0] for c in cI]
    n = sortPtsCounterClockWise(G['nodes']['coords'][:, :2], n)
    assert all(len(x) == 4 for x in n)
    n = np.concatenate(n)
    nu, _, ic = np.unique(n, return_index=True, return_inverse=True)
    p = G['nodes']['coords'][nu, :2]
    t = ic.reshape(-1, 4)
    g = tessellationGrid(p, t)
    g = computeGeometry(g)

    # Get boundary faces of g, sorted, counter-clockwise
    N = g['faces']['neighbors']
    bf = np.flatnonzero(~np.all(N >= 0, axis=1))
    bf = sortPtsCounterClockWise(g['faces']['centroids'], [bf])[0]

    # Get boundary nodes of g
    bfn, pos = gridFaceNodes(g, bf)
    assert np.all(np.diff(pos) == 2)
    bfn = bfn.reshape(-1, 2)
    bn = [bfn[r, ~np.isin(bfn[r], bfn[r - 1])] for r in range(1, bfn.shape[0] - 1)]
    idx = np.isin(bfn[0], bfn[1])
    bn = np.concatenate([bfn[0, ~idx], bfn[0, idx]] + bn)

    # Get boundary nodes of G in the ROI
    bdNodes = nu[bn]  # sorted, counter-clockwise

    # Get boundary cells of g
    bc = np.maximum(N[bf, 0], N[bf, 1])  # the interior cell of each bdy face
    bc = _unique_stable(bc)              # some cells appear twice
    # Insert the 'Z' cells
    N0 = np.column_stack([bc, np.concatenate([bc[1:], bc[:1]])])
    zc = np.full(len(bc), -1, dtype=np.int64)
    for ii in range(len(N0)):
        c1 = N[np.any(N == N0[ii, 0], axis=1), :]
        c1 = np.unique(c1)
        c1 = c1[(c1 >= 0) & (c1 != N0[ii, 0])]
        c2 = N[np.any(N == N0[ii, 1], axis=1), :]
        c2 = np.unique(c2)
        c2 = c2[(c2 >= 0) & (c2 != N0[ii, 1])]
        inter = np.intersect1d(c1, c2)
        if inter.size:
            zc[ii] = inter[0]
    # MATLAB: bc = [bc, zc]'; bc = bc(:); -- the transpose before the
    # column-major ravel interleaves bc/zc element-by-element
    # ([bc(1),zc(1),bc(2),zc(2),...]), not block-order; ravel(order='C')
    # on the (m,2) stack gives that directly.
    bc = np.column_stack([bc, zc]).ravel(order='C')
    bc = bc[bc != -1]

    # Get boundary cells of G in the ROI
    bdCells = cI[bc]
    if len(bc) != len(np.unique(bc)):
        raise ValueError('Isolate boundary cells are detected, please '
                         'redefine the boundary polygon')

    if opt['plotResults']:
        print('   (plotResults is ignored in the Python port)')
    return bdNodes, bdCells


def _unique_stable(a):
    a = np.asarray(a)
    _, idx = np.unique(a, return_index=True)
    return a[np.sort(idx)]
