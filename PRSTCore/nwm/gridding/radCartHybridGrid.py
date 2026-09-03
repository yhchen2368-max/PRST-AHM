"""Port of MRST ``radCartHybridGrid``: build a hybrid grid by gluing a
radial grid in the near-well region to the Cartesian grid elsewhere in the
reservoir."""

import numpy as np

from .._core import computeGeometry, gridCellNodes, removeCells, tessellationGrid
from ..utils.sortPtsClockWise import sortPtsClockWise
from .buildRadialGrid import buildRadialGrid
from .extractBdyNodesCells import extractBdyNodesCells


def radCartHybridGrid(GC, CI, rW, rM, nR, pW):
    """Build the hybrid grid by gluing the radial grid in the near-well
    region to the Cartesian grid elsewhere in the reservoir.

    Parameters
    ----------
    GC : dict
        The Cartesian grid structure.
    CI : array_like
        Cells inside the well region.
    nR : int
        Number of cells in the radial direction.
    rW : float
        The minimum radius (wellbore radius).
    rM : float
        The maximum radius.
    pW : array_like
        The well point coordinates (2D).

    Returns
    -------
    G : dict
        Valid hybrid grid definition.
    t : list of 1D arrays
        Connectivity list of the hybrid grid.
    """
    pW = np.asarray(pW, dtype=float)

    # Get the sorted boundary nodes of the region (counter-clockwise)
    bn, _ = extractBdyNodesCells(GC, CI, plotResults=False)

    # The angular dimension and grid angles are determined by the boundary
    # nodes to conform with the Cartesian grid
    nA = len(bn)

    # Compute the angles
    pbn = GC['nodes']['coords'][bn]
    pbn0 = pbn - pW
    th = np.arctan2(pbn0[:, 1], pbn0[:, 0])

    # Get the grid radii
    r = np.logspace(np.log10(rW), np.log10(rM), nR + 1)

    # Get the radial grid points
    R_, TH = np.meshgrid(r, th)
    px = R_ * np.cos(TH)
    py = R_ * np.sin(TH)
    pR = np.column_stack([px.ravel(order='F'), py.ravel(order='F')]) + pW

    # The boundary points are the outermost angular points
    pR = np.vstack([pR, pbn])

    # Build the radial grid
    GR, tR = buildRadialGrid(pR, nA, nR + 1)

    # Points and connectivity list of the radial grid
    pR = GR['nodes']['coords']

    # Remove the cells inside the well region first
    GC_Rem, _, _, mapn = removeCells(GC, CI)
    pC = GC_Rem['nodes']['coords']

    # Get the indices of boundary nodes in GC_Rem
    bn = np.array([np.flatnonzero(mapn == n)[0] for n in bn])

    # Merge the common nodes (boundary nodes).  The boundary node indices in
    # pC are replaced by the ones in pR.
    nNo = np.arange(len(pC), dtype=np.int64)
    # The non-boundary nodes indices
    idx = ~np.isin(nNo, bn)
    nNo[idx] = np.arange(np.count_nonzero(idx)) + len(pR)
    pC = pC[idx]
    # The boundary nodes in pR are the last nA nodes
    nNo[bn] = len(pR) - nA + np.arange(nA)

    # Map the connectivity list of GC_Rem
    cnC, pos = gridCellNodes(GC_Rem, np.arange(GC_Rem['cells']['num']))
    cnC = nNo[cnC]
    tC = [cnC[pos[c]:pos[c + 1]] for c in range(GC_Rem['cells']['num'])]

    # Assemble the points and connectivity lists
    p = np.vstack([pR, pC])

    # Sort tC to the same direction as tR (clockwise)
    tC = sortPtsClockWise(p, tC)
    t = tR + tC

    # Build the hybrid grid
    G = tessellationGrid(p, t)
    G = computeGeometry(G)
    G['subGrids'] = [GR, GC_Rem]
    return G, t
