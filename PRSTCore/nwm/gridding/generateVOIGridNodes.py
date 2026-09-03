"""Port of MRST ``generateVOIGridNodes``: generate the 3D points of all
surfaces of the volume of interest (VOI) and the connectivity list
corresponding to the 2D planar points."""

import numpy as np

from .._core import (ddiff, delaunayn, dpoly, griddata,
                     gridCellNodes, gridFaceNodes, inpolygon, mergeOptions,
                     removeShortEdges, tessellationGrid, voronoin)
from ..utils.circleCross import circleCross
from ..utils.euclideanDistance import euclideanDistance
from ..utils.sortPtsCounterClockWise import sortPtsCounterClockWise
from .passToDistmesh import passToDistmesh


def generateVOIGridNodes(GC, packed, WR, layerRf, opt):
    """Generate the 3D points of all surfaces of the volume of interest (VOI)
    and the connectivity list corresponding to the 2D planar points.

    Returns ``(pSurfs, t, bdyID)``.
    """
    # Reference surface
    refSurf = (len(packed['nodes']) + 1) // 2 - 1
    gridType = opt['gridType']
    if gridType == 'triangular':
        generator = triangularPts
    elif gridType == 'Voronoi':
        generator = VoronoiPts
    else:
        raise ValueError(f'generateVOIGridNodes: Unknown grid type: {gridType}')

    # Get 2D points and connectivity list
    p, t, bdyID = generator(GC, packed, WR, refSurf, opt)

    # Get surface points
    pSurfs = getSurfacePoints(GC, packed, layerRf, refSurf, p, bdyID)
    return pSurfs, t, bdyID


def triangularPts(G, packed, WR, refSurf, opt):
    """Generate the 2D points, connectivity list and boundary nodes using the
    'triangular' (DistMesh + Delaunay) approach."""
    # Assign VOI data
    bnV = packed['bdyNodes'][refSurf]

    # Assign WR data
    pW = np.vstack([np.vstack([pp['cart'] for pp in WR['points']]),
                    np.vstack([pp['rad'] for pp in WR['points']])])
    tW = WR['connlist']
    bnW = WR['bdnodes']

    # Get boundary points
    pib = pW[bnW, :]                       # Inner boundary points
    pob = G['nodes']['coords'][bnV, :2]    # Outer boundary points

    # Generate basic points from 'DistMesh'
    pdis, fd = passToDistmesh(pib, pob, opt['multiplier'], opt['maxIter'])

    # Delaunay triangulation
    delTri = delaunayn(pdis)

    # Remove cells inside the WR
    tol = 0.1
    tV = np.asarray(delTri, dtype=np.int64)
    pVmid = (pdis[tV[:, 0]] + pdis[tV[:, 1]] + pdis[tV[:, 2]]) / 3
    in_ = fd(pVmid) < -tol
    tV = tV[in_, :]

    # Add the WR points and connectivity list
    npib = pib.shape[0]
    npob = pob.shape[0]
    nIDIB = tV >= npib          # 0-based: inner-boundary nodes are 0..npib-1
    npWR = pW.shape[0]
    tV = tV.copy()
    tV[nIDIB] = tV[nIDIB] - npib + npWR
    tV[~nIDIB] = bnW[tV[~nIDIB]]
    t = tW + [np.asarray(row, dtype=np.int64) for row in tV]
    p = np.vstack([pW, pdis[npib:, :]])
    bdyID = npWR + np.arange(npob)
    return p, t, bdyID


def VoronoiPts(G, packed, WR, refSurf, opt):
    """Generate the 2D points, connectivity list and boundary nodes using the
    'Voronoi' (PEBI) approach."""
    # Get outer boundary points and auxiliary points of VOI
    fV = packed['faces'][refSurf]
    bfV = packed['bdyFaces'][refSurf]
    boxfV = packed['boxFaces'][refSurf]
    bnV = packed['bdyNodes'][refSurf]
    pob = G['nodes']['coords'][bnV, :2]
    pob2 = G['faces']['centroids'][bfV, :2]
    boxfV = boxfV[~np.isin(boxfV, fV)]
    pauxV = G['faces']['centroids'][boxfV, :2]  # box face centroids

    # Get inner boundary points and auxiliary points of WR
    pW = np.vstack([np.vstack([pp['cart'] for pp in WR['points']]),
                    np.vstack([pp['rad'] for pp in WR['points']])])
    tW = WR['connlist']
    bnW = WR['bdnodes']
    pib = pW[bnW, :]
    pIn, pOut, R = computeAuxPts(pW, bnW, 0.23)
    pib2 = pOut
    pauxW = pIn

    # Generate basic points from 'DistMesh'
    pdis, _fd = passToDistmesh(pib2, pob2, opt['multiplier'], opt['maxIter'],
                               pIBRadius=R)

    # Get Voronoi points and connectivity list
    pall = np.vstack([pdis, pauxV, pauxW])
    pVor, tVor = voronoin(pall)

    # Clip the diagram
    fdI = lambda p: dpoly(p, np.vstack([pib, pib[0]]))
    fdO = lambda p: dpoly(p, np.vstack([pob, pob[0]]))
    fd = lambda p: ddiff(fdO(p), fdI(p))
    tol1 = 0.1
    tol2 = 0.1
    try:
        pVor, tVor = clipDiagram(pVor, tVor, fd, tol1, tol2)
    except Exception:
        pVor, tVor = clipDiagram2(pVor, tVor, fd, tol1, tol2)

    # Add WR points, and map the connectivity list again
    npWR = pW.shape[0]
    D2 = euclideanDistance(pVor, pib)
    # MATLAB ``[IBID, ~] = find(bsxfun(@eq, D2, min(D2)))`` walks the logical
    # matrix in *column-major* order, so IBID(j) is the Voronoi vertex
    # closest to the j-th inner-boundary point -- one entry per column, in
    # column order, which is what pairs it with bnW(j) below.  Collapsing
    # over columns with ``any(..., axis=1)`` instead returns the row indices
    # in ascending order and deduplicated, which bears no relation to the
    # pib ordering: every well-boundary node then adopts a different node's
    # Voronoi vertex, and the cells around the well stop matching its
    # boundary faces.
    IBID = np.argmin(D2, axis=0)
    if np.unique(IBID).size != IBID.size:
        raise RuntimeError(
            'Two inner-boundary points share their nearest Voronoi vertex; '
            'the well-boundary node mapping would be ambiguous.')
    map1 = np.flatnonzero(~np.isin(np.arange(pVor.shape[0]), IBID))
    pVor = pVor[map1, :]
    old_to_new = {int(old): npWR + new for new, old in enumerate(map1)}
    map2 = np.column_stack([bnW, IBID])
    old_to_wr = {int(old): int(wr) for wr, old in zip(bnW, IBID)}
    tVor_new = []
    for x in tVor:
        mapped = []
        for xi in x:
            xi = int(xi)
            if xi in old_to_new:
                mapped.append(old_to_new[xi])
            elif xi in old_to_wr:
                mapped.append(old_to_wr[xi])
        if mapped:
            tVor_new.append(np.unique(mapped))
    t = tW + tVor_new
    p = np.vstack([pW, pVor])
    D3 = euclideanDistance(p, pob)
    # Same column-major ``find`` contract as IBID above.  getSurfacePoints
    # assigns ``pfull(bdyID, :) = GC.nodes.coords(bnV{k}, :)`` positionally,
    # so bdyID(j) must be the point nearest the j-th outer-boundary node;
    # an ascending, deduplicated row list hands every boundary node another
    # node's coordinates.
    bdyID = np.argmin(D3, axis=0)

    # Add empty cells
    p, t = addEmpCells(p, t, bnW)
    return p, t, bdyID


def getSurfacePoints(GC, packed, layerRf, refSurf, p, bdyID):
    """Generate the 3D points of all surfaces of the VOI grid."""
    bnV = packed['bdyNodes']
    nV = packed['nodes']
    np_ = p.shape[0]
    inID = np.flatnonzero(~np.isin(np.arange(np_), bdyID))
    players = [None] * len(nV)

    # Get points of each VOI surface
    for k in range(len(nV)):
        xi = p[inID, 0]
        yi = p[inID, 1]
        # Shift the inner points
        xcRef = np.mean(GC['nodes']['coords'][bnV[refSurf], 0])
        ycRef = np.mean(GC['nodes']['coords'][bnV[refSurf], 1])
        xcNow = np.mean(GC['nodes']['coords'][bnV[k], 0])
        ycNow = np.mean(GC['nodes']['coords'][bnV[k], 1])
        xi = xi + (xcNow - xcRef)
        yi = yi + (ycNow - ycRef)
        # Interpolate the Z-coords of inner points
        n = np.unique(np.concatenate(nV[k]))
        x = GC['nodes']['coords'][n, 0]
        y = GC['nodes']['coords'][n, 1]
        z = GC['nodes']['coords'][n, 2]
        zi = griddata(x, y, z, xi, yi)
        pfull = np.zeros((np_, 3))
        pfull[np.ix_(inID, [0, 1])] = np.column_stack([xi, yi])
        pfull[inID, 2] = zi
        pfull[bdyID, :] = GC['nodes']['coords'][bnV[k], :]
        players[k] = pfull

    # Add points of refined surfaces
    for k in range(len(players) - 1):
        pUpp = players[k]
        pBot = players[k + 1]
        scal = np.linspace(0, 1, layerRf[k] + 1)
        scal = scal[1:-1]
        prefine = [pUpp + s * (pBot - pUpp) for s in scal]
        prefine = np.vstack(prefine)
        players[k] = np.vstack([players[k], prefine])

    players = np.vstack(players)
    nlayer = len(players) // np_
    players = [players[(L - 1) * np_:L * np_] for L in range(1, nlayer + 1)]
    return players


def clipDiagram(pVor, tVor, fd, tol1, tol2):
    """Remove points outside the region and points too close to each other."""
    # Remove points outside the region
    in_ = np.flatnonzero((fd(pVor) < tol1) & np.all(np.isfinite(pVor), axis=1))
    map_ = np.column_stack([np.arange(len(in_)), in_])
    pVor = pVor[in_, :]

    # Remove conflict points (points too close to each other)
    D = euclideanDistance(pVor, pVor)
    D = np.triu(D)

    removed, reserved = np.nonzero(D < tol2)
    ii = removed < reserved
    removed = removed[ii]
    reserved = reserved[ii]
    if len(removed):
        map_[removed, 0] = map_[reserved, 0]
        idx = np.flatnonzero(~np.isin(np.arange(pVor.shape[0]), removed))
        pVor = pVor[idx, :]
        newpos = {int(old): i for i, old in enumerate(idx)}
        map_[:, 0] = np.array([newpos[int(x)] for x in map_[:, 0]])

    # Map the connectivity list
    tVor_new = []
    for x in tVor:
        sel = map_[np.isin(map_[:, 1], x), 0]
        sel = np.unique(sel)
        if len(sel) > 3:
            tVor_new.append(sel)
    return pVor, tVor_new


def clipDiagram2(pVor, tVor, fd, tol1, tol2):
    """Fallback clipping of the Voronoi diagram (based on cell centroids)."""
    # Remove points outside the region
    cCenter = np.vstack([np.mean(pVor[t, :], axis=0) for t in tVor])
    in_ = fd(cCenter) < tol1
    t = [tVor[i] for i in np.flatnonzero(in_)]
    n = np.unique(np.concatenate(t))
    p = pVor[n, :]
    t = [np.flatnonzero(np.isin(n, tt)) for tt in t]
    t = sortPtsCounterClockWise(p, t)
    g = tessellationGrid(p, t)
    g = removeShortEdges(g, tol2)
    pVor = g['nodes']['coords']
    tVor = [gridCellNodes(g, c)[0] for c in range(g['cells']['num'])]
    return pVor, tVor


def computeAuxPts(p, bn, m0):
    """Compute the inner/outer auxiliary points (and their radius) of the WR
    boundary used as Voronoi sites."""
    pib = np.vstack([p[bn, :], p[bn[0], :]])
    n = len(bn)
    e2n = np.column_stack([np.arange(n), np.concatenate([np.arange(1, n), [0]])])
    # Compute the radius
    edges = np.column_stack([bn, np.concatenate([bn[1:], bn[:1]])])
    L = p[edges[:, 0], :] - p[edges[:, 1], :]
    L = np.sqrt(np.sum(L ** 2, axis=1))
    do = True
    m = m0
    while do:
        if m > 0.5:
            throwError(L)
        try:
            m = m + 0.02
            R = (np.concatenate([[L[0]], L[:n - 1]]) + np.concatenate([[L[n - 1]], L[1:]])) * m
            pIn = np.zeros((edges.shape[0], 2))
            pOut = np.zeros((edges.shape[0], 2))
            for i in range(edges.shape[0]):
                x1, y1 = p[edges[i, 0], 0], p[edges[i, 0], 1]
                x2, y2 = p[edges[i, 1], 0], p[edges[i, 1], 1]
                r1, r2 = R[e2n[i, 0]], R[e2n[i, 1]]
                pCross = circleCross(x1, y1, r1, x2, y2, r2)
                in_ = inpolygon(pCross[:, 0], pCross[:, 1], pib[:, 0], pib[:, 1])
                in_idx = np.flatnonzero(in_)
                if len(in_idx) == 1:
                    pIn[i, :] = pCross[in_idx[0], :]
                    pOut[i, :] = pCross[~in_, :][0, :]
                else:
                    raise ValueError('circleCross: ambiguous intersection')
            do = False
        except Exception:
            do = True
    return pIn, pOut, R


def addEmpCells(p, t, bnW):
    """Add empty cells which appear during the generation of the Voronoi
    grid."""
    t = sortPtsCounterClockWise(p, t)
    G = tessellationGrid(p, t)
    fn, pos = gridFaceNodes(G, np.arange(G['faces']['num']))
    fn = fn.reshape(-1, 2)
    assert np.all(np.diff(pos) == 2)
    assert np.all(fn[:, 1] > fn[:, 0])
    bnW = np.concatenate([bnW, bnW[:1]])
    fW = np.zeros(len(bnW) - 1, dtype=np.int64)
    for i in range(len(bnW) - 1):
        n = np.sort(bnW[i:i + 2])
        fW[i] = np.flatnonzero(np.all(fn == n, axis=1))[0]
    bf = np.flatnonzero(~np.all(G['faces']['neighbors'] >= 0, axis=1))
    bf = bf[~np.isin(bf, fW)]
    bfn = fn[bf, :]

    for i in range(len(fW)):
        f = fW[i]
        n = np.sort(bnW[i:i + 2])
        n1, n3 = n[0], n[1]
        if not np.all(G['faces']['neighbors'][f, :] >= 0):
            f2 = bf[np.any(bfn == n1, axis=1)]
            FM = []
            NM = []
            for j in range(len(f2)):
                n2 = fn[f2[j], :]
                n2 = n2[n2 != n1][0]
                NM.append([n2])
                FM.append([f2[j]])

            for j in range(len(f2)):
                while True:
                    fm = bf[np.any(bfn == NM[j][-1], axis=1)]
                    fm = fm[fm != FM[j][-1]][0]
                    nm = fn[fm, :]
                    nm = nm[nm != NM[j][-1]][0]
                    NM[j].append(nm)
                    FM[j].append(fm)
                    if np.any(bnW == nm):
                        break
            idx = [x[-1] == n3 for x in NM]
            NM = NM[idx.index(True)]
            t = t + [np.concatenate([[n1], np.array(NM)])]
    return p, t


def throwError(L):
    raise ValueError(
        'Cannot generate appropriate Voronoi sites, please \n'
        '   (1) Increase the resolution of well trajectory (add more well points) \n'
        'Or (2) Use the grid type \'triangular\' instead \n'
        'Or (3) Increase the value of \'WR.ly\', the suggested value is '
        f'{1.1 * np.max(L):.0f} (may require to enlarge the VOI boundary)')
