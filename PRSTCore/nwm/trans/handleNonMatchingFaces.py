"""Port of MRST ``handleNonMatchingFaces``: compute the intersection relation
of a surface shared by two grids ``G1`` and ``G2``.

The surface is divided into a set of subfaces due to the different face
geometries of ``G1`` and ``G2``.  The fallback polygon-intersection routine
``polyintersect`` (Gerben J. de Boer) is used directly in place of the
Mapping-Toolbox ``polyxpoly``.
"""

import numpy as np

from .._core import mergeOptions, inpolygon, polyarea
from ..utils.computeCentroids import computeCentroids
from ..utils.convertTo3DPlane import convertTo3DPlane
from ..utils.convertToXYPlane import convertToXYPlane
from ..utils.polyintersect import polyintersect
from ..utils.sortPtsCounterClockWise import sortPtsCounterClockWise


def _as_node_array(fn):
    if isinstance(fn, (list, tuple)):
        return np.asarray(fn[0], dtype=np.int64)
    return np.asarray(fn, dtype=np.int64)


def getSortedFaceNodes(G, f, isSorted):
    """Sorted nodes (0-based) of face ``f``; sorts counter-clockwise when
    the stored node order is not trusted."""
    fn = G['faces']['nodes'][G['faces']['nodePos'][f]:G['faces']['nodePos'][f + 1]]
    if not isSorted:
        pts1 = G['nodes']['coords']
        tmp = np.zeros(3)
        norZ = G['faces']['normals'][f] / G['faces']['areas'][f]
        pts1, _, _, _, _ = convertToXYPlane(pts1, fn, tmp, normalZ=norZ)
        fn = sortPtsCounterClockWise(pts1[:, :2], [fn])[0]
    return fn


def addEdgePoints(p):
    """Add points along the edges of the polygon specified by ``p``."""
    N = 20
    pEdge = []
    for i in range(p.shape[0] - 1):
        p1 = p[i]
        p2 = p[i + 1]
        d = np.linspace(0, 1, N + 2)
        pEdge.append(np.column_stack([p1[0] + d * (p2[0] - p1[0]),
                                      p1[1] + d * (p2[1] - p1[1])]))
    pEdge = np.vstack(pEdge)
    return np.vstack([p, pEdge])


def IntxnRelationSingleFace(G1, f1, fn1, G2, faces2, fnodes2):
    """Intersection relation of the single face ``f1`` of ``G1`` against the
    face set ``faces2`` of ``G2``."""
    fn1 = _as_node_array(fn1)
    # 1. Coordinate transformation
    nor_z = G1['faces']['normals'][f1] / G1['faces']['areas'][f1]
    pts1 = G1['nodes']['coords']
    pts2 = G2['nodes']['coords']
    pts1, pts2, T, R, _ = convertToXYPlane(pts1, fn1, pts2, normalZ=None)
    # Closed polygon of face f1
    p1 = pts1[np.concatenate([fn1, fn1[:1]]), :2]

    # 2. Find f2 fully located inside f1
    allIn = np.array([np.all(inpolygon(pts2[x, 0], pts2[x, 1], p1[:, 0], p1[:, 1]))
                      for x in fnodes2])
    if np.any(allIn):
        f2 = faces2[allIn]
        R_In = np.column_stack([f2, G2['faces']['areas'][f2],
                                G2['faces']['centroids'][f2],
                                G2['faces']['normals'][f2]])
    else:
        R_In = np.empty((0, 8))

    # 3. Find faces2 intersecting with f1
    pEdge = addEdgePoints(p1)
    iX = np.array([np.any(inpolygon(pEdge[:, 0], pEdge[:, 1], pts2[x, 0], pts2[x, 1]))
                   for x in fnodes2])
    iX = np.flatnonzero(iX)
    if iX.size == 0:
        if R_In.shape[0] == 0:
            return np.empty((0, 9))
        return np.column_stack([np.full(R_In.shape[0], f1), R_In])

    f2_l = [None] * len(iX)
    areas = [None] * len(iX)
    f_nor = [None] * len(iX)
    f_ctd = [None] * len(iX)
    for kk, k in enumerate(iX):
        fn2 = fnodes2[k]
        p2 = pts2[np.concatenate([fn2, fn2[:1]]), :2]
        In2 = inpolygon(p1[:, 0], p1[:, 1], p2[:, 0], p2[:, 1])
        p1In2 = p1[In2, :]
        In1 = inpolygon(p2[:, 0], p2[:, 1], p1[:, 0], p1[:, 1])
        p2In1 = p2[In1, :]

        # Polygon intersection (polyintersect replaces polyxpoly)
        xi, yi = polyintersect(p1[:, 0], p1[:, 1], p2[:, 0], p2[:, 1])
        xyi = np.unique(np.column_stack([xi, yi]), axis=0)
        xi = xyi[:, 0]
        yi = xyi[:, 1]

        p = np.unique(np.vstack([p1In2, p2In1, np.column_stack([xi, yi])]), axis=0)
        # Remove subfaces whose areas are small compared with areas of both
        # f1 and f2
        minRatio = 0.0001
        if p.shape[0] > 2:
            tmp = sortPtsCounterClockWise(p, [np.arange(p.shape[0])])
            p = p[tmp[0], :]
            ppoly = np.vstack([p, p[0]])
            area = polyarea(ppoly[:, 0], ppoly[:, 1])
            ratio = [area / G1['faces']['areas'][f1],
                     area / G2['faces']['areas'][faces2[k]]]
            if all(r > minRatio for r in ratio):
                f2_l[kk] = faces2[k]
                areas[kk] = area
                f_nor[kk] = area * nor_z
                pmid = computeCentroids(p)
                pmid = np.append(pmid, pts1[fn1[0], 2])
                pmid = convertTo3DPlane(pmid, T, R)
                f_ctd[kk] = pmid

    # Remove empty faces
    idx = [x is not None for x in areas]
    f2_l = [f2_l[i] for i in range(len(iX)) if idx[i]]
    areas = [areas[i] for i in range(len(iX)) if idx[i]]
    f_ctd = [f_ctd[i] for i in range(len(iX)) if idx[i]]
    f_nor = [f_nor[i] for i in range(len(iX)) if idx[i]]

    if areas:
        R_Ixn = np.column_stack([np.asarray(f2_l),
                                 np.asarray(areas),
                                 np.vstack(f_ctd),
                                 np.vstack(f_nor)])
    else:
        R_Ixn = np.empty((0, 8))

    # 4. Combine the relations
    if R_In.shape[0] == 0 and R_Ixn.shape[0] == 0:
        return np.empty((0, 9))
    relation_f = np.vstack([R_In, R_Ixn])
    f2 = relation_f[:, 0]
    _, ia = np.unique(f2, return_index=True)
    relation_f = relation_f[ia, :]
    relation_f = np.column_stack([np.full(relation_f.shape[0], f1), relation_f])
    return relation_f


def handleNonMatchingFaces(G1, faces1, G2, faces2, **kwargs):
    """Compute the intersection relation of a surface shared by ``G1`` and
    ``G2``.

    Parameters
    ----------
    G1, G2 : dict
        Grids sharing a surface.
    faces1, faces2 : array_like
        Surface face sets from ``G1`` and ``G2``, constituting the same 3D
        surface (continuous and complete).
    isfaceNodesSorted : bool, optional
        Whether the nodes of faces stored at ``G.faces.nodes`` are sorted
        (for both grids).  Default: False.

    Returns
    -------
    relation : ndarray, ``n x 9``
        Column 1 - Face of G1; column 2 - Face of G2; column 3 - Areas of
        intersection subfaces; columns 4-6 - Centroids; columns 7-9 -
        Normals of the subfaces.
    """
    opt = mergeOptions({'isfaceNodesSorted': False}, **kwargs)
    faces1 = np.asarray(faces1, dtype=np.int64).ravel()
    faces2 = np.asarray(faces2, dtype=np.int64).ravel()

    # Nodes of faces
    faceNodes1 = [getSortedFaceNodes(G1, f, opt['isfaceNodesSorted']) for f in faces1]
    faceNodes2 = [getSortedFaceNodes(G2, f, opt['isfaceNodesSorted']) for f in faces2]

    # Get the intersection relation
    relation = [IntxnRelationSingleFace(G1, f, fn, G2, faces2, faceNodes2)
                for f, fn in zip(faces1, faceNodes1)]
    # ``np.vstack`` handles a mix of empty (0, 9) and non-empty (k, 9)
    # entries fine; the only case actually needing a guard is an empty
    # *list* (faces1 itself empty). Gating on ``relation[0].size`` instead
    # silently discarded every later face's intersection data whenever the
    # first face in faces1 happened to have none -- a real bug that could
    # leave subgrids without any NNC connection between them.
    relation = np.vstack(relation) if relation else np.empty((0, 9))
    if relation.size == 0:
        relation = np.empty((0, 9))

    idx = ~np.isin(faces2, relation[:, 1])
    if np.any(idx):
        sub = np.flatnonzero(idx)
        relation_r = [IntxnRelationSingleFace(G2, f, faceNodes2[i], G1, faces1, faceNodes1)
                      for i, f in zip(sub, faces2[sub])]
        relation_r = np.vstack(relation_r) if relation_r else np.empty((0, 9))
        if relation_r.size:
            f1 = relation_r[:, 1]
            f2 = relation_r[:, 0]
            relation_r[:, 0] = f1
            relation_r[:, 1] = f2
            relation = np.vstack([relation, relation_r])
    return relation
