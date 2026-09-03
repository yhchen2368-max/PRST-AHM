"""Internal core-grid helpers used by the near-wellbore-model (nwm) port.

A 1:1 port, preserving the MRST function names, of the small set of MRST
*core* grid/geometry routines that the ``nwm`` module relies on but which
are not (yet) exposed by the public PRSTCore packages.  This module is an
implementation detail of ``PRSTCore.nwm`` and not part of its public API.

Conventions (identical to the rest of PRSTCore, differing from MRST):

  * cells / faces / nodes are 0-based;
  * ``G['faces']['neighbors']`` uses ``-1`` (not ``0``) to mark "no cell
    on this side" (a boundary face);
  * ``G['cells']['facePos']`` / ``G['faces']['nodePos']`` start at 0 and
    use 0-based half-face / half-face-node positions.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# General array helpers (MRST core/utils)
# ---------------------------------------------------------------------------

def mergeOptions(prm, *pairs, **kwargs):
    """Port of MRST ``merge_options``.

    ``prm`` is a dict of defaults; the remaining positional ``pairs`` are
    ``'key', value, 'key', value, ...`` (or ``**kwargs``) overriding the
    defaults.  Unknown keys raise a ``ValueError`` (as MRST warns and skips,
    but for a port failing loudly is preferable).
    """
    out = dict(prm)
    items = list(kwargs.items())
    if len(pairs) % 2 != 0:
        raise ValueError('mergeOptions: option list must be key/value pairs')
    for i in range(0, len(pairs), 2):
        items.append((pairs[i], pairs[i + 1]))
    for key, val in items:
        if key not in out:
            raise ValueError(f"mergeOptions: unknown option '{key}'")
        out[key] = val
    return out


def rldecode(A, n, dim=0):
    """Port of MRST ``rldecode`` along axis ``dim`` (default 0)."""
    A = np.asarray(A)
    n = np.asarray(n, dtype=np.int64)
    if n.size == 1:
        n = np.full(A.shape[dim], int(n), dtype=np.int64)
    if n.size != A.shape[dim]:
        raise ValueError('rldecode: repeat counts must match size along dim')
    reps = [1] * A.ndim
    reps[dim] = -1
    return np.repeat(A, n, axis=dim).reshape(reps if False else (-1,) if A.ndim == 1 else A.shape[:dim] + (int(n.sum()),) + A.shape[dim + 1:])


def mcolon(lo, hi):
    """Port of MRST ``mcolon(lo, hi)``: concatenation of ``lo(i):hi(i)``."""
    lo = np.asarray(lo, dtype=np.int64).ravel()
    hi = np.asarray(hi, dtype=np.int64).ravel()
    if lo.size == 0:
        return np.empty(0, dtype=np.int64)
    lens = hi - lo + 1
    if np.any(lens < 0):
        raise ValueError('mcolon: negative range length')
    total = int(lens.sum())
    idx = np.arange(total, dtype=np.int64)
    starts = np.repeat(lo, lens)
    group_starts = np.repeat(np.cumsum(lens) - lens, lens)
    return starts + (idx - group_starts)


def tabulate(u):
    """Port of MRST ``tabulate`` (via ``accumarray(u, 1)``)."""
    u = np.asarray(u, dtype=np.int64).ravel()
    v = np.bincount(u, minlength=int(u.max()) + 1 if u.size else 0)
    return np.column_stack([np.arange(1, v.size + 1, dtype=np.int64), v])


def uniqueStable(a):
    """Port of MRST ``uniqueStable``: unique values preserving first order."""
    a = np.asarray(a)
    _, idx = np.unique(a, return_index=True)
    return a[np.sort(idx)]


def removeShortEdges(G, tol=0.0):
    """Self-contained port of MRST ``removeShortEdges`` (used by the Voronoi
    grid fallback path in ``generateVOIGridNodes``).

    Nodes joined by an edge with ``||edge|| < tol`` are collapsed to a
    single node at the (component) mean position; faces and cells that
    collapse as a consequence are removed.  The grid is rebuilt as a 2D
    tessellation grid.  Returns ``(H, cellmap)``.
    """
    G = dict(G)
    nodePos = np.asarray(G['faces']['nodePos'], dtype=np.int64)
    fnodes = np.asarray(G['faces']['nodes'], dtype=np.int64)
    x = np.asarray(G['nodes']['coords'], dtype=float)
    griddim = int(G.get('griddim', 2))

    # Build the (undirected) edge list: for each face, edges (n_i, n_{i+1})
    # with the last node of the face wrapped back to the first.
    num = np.diff(nodePos)
    face_ends = nodePos[1:] - 1
    face_starts = nodePos[:-1]
    E1 = fnodes
    E2 = np.empty(fnodes.size, dtype=np.int64)
    for k in range(num.size):
        E2[face_starts[k]:face_ends[k]] = fnodes[face_starts[k] + 1:face_ends[k] + 1]
        E2[face_ends[k]] = fnodes[face_starts[k]]
    E = np.column_stack([E1, E2])

    L = np.sqrt(np.sum((x[E[:, 0]] - x[E[:, 1]]) ** 2, axis=1))
    short = np.flatnonzero(L < tol)
    if short.size == 0:
        return G, np.arange(G['cells']['num'], dtype=np.int64)

    # Union-find over the short-edge graph
    parent = np.arange(x.shape[0], dtype=np.int64)

    def _find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for e in E[short]:
        _union(int(e[0]), int(e[1]))

    comp = np.array([_find(n) for n in range(x.shape[0])], dtype=np.int64)
    comps = uniqueStable(comp)
    mean = np.zeros((x.shape[0], x.shape[1]), dtype=float)
    counts = np.zeros(x.shape[0], dtype=float)
    np.add.at(mean, comp, x)
    np.add.at(counts, comp, 1.0)
    mean = mean / counts[:, None]
    new_coords = mean[comps]
    newid = {int(c): k for k, c in enumerate(comps)}
    map_ = np.array([newid[int(c)] for c in comp], dtype=np.int64)

    # Rebuild the cells from the original cell-node lists
    cn, pos = gridCellNodes(G)
    cleaned = []
    cellmap = []
    for c in range(G['cells']['num']):
        nodes = map_[cn[pos[c]:pos[c + 1]]]
        nodes = np.unique(nodes)
        if nodes.size >= 3:
            cleaned.append(nodes)
            cellmap.append(c)
    if not cleaned:
        raise ValueError('removeShortEdges: all cells collapsed')
    H = tessellationGrid(new_coords, cleaned)
    H['type'] = list(G.get('type', [])) + ['removeShortEdges']
    return H, np.array(cellmap, dtype=np.int64)


def _mapExcluding(indices):
    """Port of MRST private ``mapExcluding`` (0-based version).

    Returns ``m`` where ``m[i]`` is the new (0-based) index of old entity
    ``i``, or ``-1`` if entity ``i`` was excluded/removed.
    """
    ind = ~np.asarray(indices, dtype=bool)
    m = np.cumsum(ind) - 1
    m[~ind] = -1
    return m


# ---------------------------------------------------------------------------
# Geometry helpers (MRST core, used by the nwm utils/gridding)
# ---------------------------------------------------------------------------

def inpolygon(x, y, xv, yv):
    """Port of MATLAB ``inpolygon`` (points on the boundary count as inside).

    Vectorized over points the same way its sibling ``_pointsInside`` is
    (looping only over the -- typically far fewer -- polygon edges): the
    original per-point-per-edge Python double loop made every ``inpolygon``
    call, and anything that calls it per DistMesh iteration, cost
    O(n_points * n_vertices) interpreted Python operations.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    xv = np.asarray(xv, dtype=float).ravel()
    yv = np.asarray(yv, dtype=float).ravel()
    nv = xv.size
    inside = np.zeros(x.shape, dtype=bool)
    on = np.zeros(x.shape, dtype=bool)
    with np.errstate(divide='ignore', invalid='ignore'):
        for j in range(nv):
            x1, y1 = xv[j], yv[j]
            x2, y2 = xv[(j + 1) % nv], yv[(j + 1) % nv]
            cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
            on |= ((np.minimum(x1, x2) - 1e-12 <= x) & (x <= np.maximum(x1, x2) + 1e-12) &
                   (np.minimum(y1, y2) - 1e-12 <= y) & (y <= np.maximum(y1, y2) + 1e-12) &
                   (np.abs(cross) <= 1e-9 * (1 + abs(x2 - x1) + abs(y2 - y1))))
            mask = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
            inside ^= mask
    return on | inside


def _pointsInside(x, y, xv, yv):
    """Strictly-inside test (crossing number), used by the DistMesh helpers."""
    x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    xv = np.asarray(xv, dtype=float).ravel()
    yv = np.asarray(yv, dtype=float).ravel()
    nv = xv.size
    inside = np.zeros(x.shape, dtype=bool)
    with np.errstate(divide='ignore', invalid='ignore'):
        for j in range(nv):
            x1, y1 = xv[j], yv[j]
            x2, y2 = xv[(j + 1) % nv], yv[(j + 1) % nv]
            mask = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
            inside ^= mask
    return inside


def polyarea(x, y):
    """Port of MATLAB ``polyarea`` (shoelace formula)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def faceNormals(p):
    """Port of the private ``faceNormals`` used by ``convertToXYPlane``."""
    p1, p2, p3 = p[0], p[1], p[2]
    a = (p2[1] - p1[1]) * (p3[2] - p1[2]) - (p3[1] - p1[1]) * (p2[2] - p1[2])
    b = (p2[2] - p1[2]) * (p3[0] - p1[0]) - (p3[2] - p1[2]) * (p2[0] - p1[0])
    c = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
    nor = np.array([a, b, c], dtype=float)
    return nor / np.linalg.norm(nor)


def griddata(x, y, z, xq, yq, method='linear'):
    """Port of MATLAB ``griddata`` (default 'linear')."""
    from scipy.interpolate import griddata as _griddata
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    xq = np.asarray(xq, dtype=float)
    yq = np.asarray(yq, dtype=float)
    return _griddata((x, y), z, (xq, yq), method=method)


def delaunayn(p):
    """Port of MATLAB ``delaunayn`` (2D triangulation)."""
    from scipy.spatial import Delaunay
    tri = Delaunay(np.asarray(p, dtype=float))
    return tri.simplices


def voronoin(p, options=None):
    """Port of MATLAB ``voronoin`` for the finite-vertex case.

    scipy's ``Voronoi`` only stores finite vertices, so the returned
    vertex array contains no ``Inf`` rows (the caller must not rely on
    them; the ``clipDiagram``-style handling already drops them).
    """
    from scipy.spatial import Voronoi
    v = Voronoi(np.asarray(p, dtype=float))
    cells = [np.asarray(r, dtype=np.int64) for r in v.regions]
    return v.vertices, cells


# ---------------------------------------------------------------------------
# DistMesh signed distance functions (module 'upr' -> 'distmesh')
# ---------------------------------------------------------------------------

def dsegment(p, p1, p2):
    """Signed-ish distance from points ``p`` to segment ``[p1, p2]``."""
    p = np.asarray(p, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    v = p2 - p1
    w = p - p1
    c1 = np.sum(w * v, axis=1)
    c2 = float(np.dot(v, v))
    d = np.where(c1 <= 0, np.linalg.norm(p - p1, axis=1),
                 np.where(c1 >= c2, np.linalg.norm(p - p2, axis=1),
                          np.linalg.norm(p - (p1 + (c1 / c2)[:, None] * v), axis=1)))
    return d


def dpoly(p, pv):
    """Port of DistMesh ``dpoly``: signed distance to a closed polygon."""
    p = np.asarray(p, dtype=float)
    pv = np.asarray(pv, dtype=float)
    d = np.full(p.shape[0], np.inf)
    for j in range(pv.shape[0] - 1):
        d = np.minimum(d, dsegment(p, pv[j], pv[j + 1]))
    inside = _pointsInside(p[:, 0], p[:, 1], pv[:, 0], pv[:, 1])
    d = np.where(inside, -d, d)
    return d


def ddiff(d1, d2):
    """Port of DistMesh ``ddiff`` (set difference)."""
    return np.maximum(d1, -d2)


def drectangle(p, x1, x2, y1, y2):
    """Port of DistMesh ``drectangle``."""
    p = np.asarray(p, dtype=float)
    return -np.minimum(np.minimum(np.minimum(-y1 + p[:, 1], y2 - p[:, 1]),
                                  -x1 + p[:, 0]), x2 - p[:, 0])


def dcircle(p, xc, yc, r):
    """Port of DistMesh ``dcircle``."""
    p = np.asarray(p, dtype=float)
    return np.sqrt((p[:, 0] - xc) ** 2 + (p[:, 1] - yc) ** 2) - r


def huniform(p):
    """Port of DistMesh ``huniform``."""
    p = np.asarray(p, dtype=float)
    return np.ones(p.shape[0])


# ---------------------------------------------------------------------------
# Grid topology / geometry (MRST core/gridprocessing + core/utils/gridtools)
# ---------------------------------------------------------------------------

def tessellationGrid(p, t):
    """Port of MRST ``tessellationGrid.m`` (2D grids only).

    ``p`` is an ``n x 2`` node coordinate array; ``t`` is either a 2D array
    (one fixed-length polygon per row) or a list of 1D arrays of node ids
    (variable-length polygons).  Node ids are 0-based.
    """
    p = np.asarray(p, dtype=float)
    assert p.ndim == 2 and p.shape[1] == 2, 'tessellationGrid: 2D points required'
    if isinstance(t, np.ndarray) and t.ndim == 2:
        n = np.full(t.shape[0], t.shape[1], dtype=np.int64)
        t_list = [np.asarray(row, dtype=np.int64).ravel() for row in t]
    else:
        t_list = [np.asarray(x, dtype=np.int64).ravel() for x in t]
        n = np.array([x.size for x in t_list], dtype=np.int64)
    nc = len(t_list)

    # Validate node references (mirrors MRST's tessellationGrid asserts)
    all_nodes = np.concatenate(t_list) if t_list else np.empty(0, dtype=np.int64)
    if all_nodes.size:
        assert all_nodes.max() < p.shape[0], \
            'Tessellation list T references invalid points (too large)'
        assert all_nodes.min() >= 0, \
            'Tessellation list T references invalid points (negative)'

    # Build the ordered edges, assign a face id to each unique unordered pair
    face_map = {}
    face_nodes = []
    cells_faces = np.empty(int(n.sum()), dtype=np.int64)
    side = np.empty(int(n.sum()), dtype=np.int64)  # 0: first node is min, 1: first is max
    k = 0
    for nodes in t_list:
        nn = nodes.size
        for j in range(nn):
            a, b = int(nodes[j]), int(nodes[(j + 1) % nn])
            lo, hi = (a, b) if a < b else (b, a)
            key = (lo, hi)
            fid = face_map.get(key)
            if fid is None:
                fid = len(face_nodes)
                face_map[key] = fid
                face_nodes.append((lo, hi))
            cells_faces[k] = fid
            side[k] = 0 if a == lo else 1
            k += 1

    nf = len(face_nodes)
    G = {
        'cells': {
            'num': nc,
            'facePos': np.concatenate([[0], np.cumsum(n)]),
            'indexMap': np.arange(nc, dtype=np.int64),
            'faces': cells_faces.reshape(-1, 1),
        },
        'faces': {
            'num': nf,
            'nodePos': np.arange(0, 2 * nf + 1, 2, dtype=np.int64),
            'nodes': np.array([x for pair in face_nodes for x in pair], dtype=np.int64),
        },
        'nodes': {'num': p.shape[0], 'coords': p},
        'type': ['tessellationGrid'],
        'griddim': 2,
    }

    # Neighbourship (0-based, -1 = boundary)
    cellNo = rldecode(np.arange(nc, dtype=np.int64), np.diff(G['cells']['facePos']))
    neighbors = np.full((nf, 2), -1, dtype=np.int64)
    for k in range(cells_faces.size):
        neighbors[cells_faces[k], side[k]] = cellNo[k]
    G['faces']['neighbors'] = neighbors

    # Uniquify nodes
    h = np.zeros(p.shape[0], dtype=bool)
    h[G['faces']['nodes']] = True
    if not h.all():
        ucoords, _, mapn = np.unique(p, axis=0, return_index=True, return_inverse=True)
        G['nodes']['coords'] = ucoords
        G['nodes']['num'] = ucoords.shape[0]
        G['faces']['nodes'] = mapn[G['faces']['nodes']]
    return G


def gridCellFaces(G, c=None):
    """Port of MRST ``gridCellFaces``: ``(cf, pos)``, 0-based."""
    if c is None:
        c = np.arange(G['cells']['num'], dtype=np.int64)
    c = np.asarray(c, dtype=np.int64).ravel()
    facePos = G['cells']['facePos']
    nf = facePos[c + 1] - facePos[c]
    rows = mcolon(facePos[c], facePos[c + 1] - 1)
    cf = G['cells']['faces'][rows, 0]
    pos = np.concatenate([[0], np.cumsum(nf)])
    return cf, pos


def gridCellNodes(G, c=None, unique=True):
    """Port of MRST ``gridCellNodes``: ``(n, pos)``, 0-based."""
    if c is None:
        c = np.arange(G['cells']['num'], dtype=np.int64)
    c = np.asarray(c, dtype=np.int64).ravel()
    facePos = G['cells']['facePos']
    nodePos = G['faces']['nodePos']
    nf = facePos[c + 1] - facePos[c]
    cellno = rldecode(np.arange(c.size, dtype=np.int64), nf)
    nnode = np.diff(nodePos)
    rows = mcolon(facePos[c], facePos[c + 1] - 1)
    cf = G['cells']['faces'][rows, 0]
    ni = mcolon(nodePos[cf], nodePos[cf + 1] - 1)
    W_cell = rldecode(cellno, nnode[cf])
    W_node = G['faces']['nodes'][ni]
    if unique:
        order = np.lexsort((W_node, W_cell))
        sc, sn = W_cell[order], W_node[order]
        keep = np.ones(sc.size, dtype=bool)
        keep[1:] = (sc[1:] != sc[:-1]) | (sn[1:] != sn[:-1])
        W_cell, W_node = sc[keep], sn[keep]
    counts = np.bincount(W_cell, minlength=c.size)
    pos = np.concatenate([[0], np.cumsum(counts)])
    return W_node, pos


def gridFaceNodes(G, f):
    """Port of MRST ``gridFaceNodes``: ``(n, pos)``, 0-based."""
    f = np.asarray(f, dtype=np.int64).ravel()
    nodePos = G['faces']['nodePos']
    nnode = np.diff(nodePos)
    ni = mcolon(nodePos[f], nodePos[f + 1] - 1)
    pos = np.concatenate([[0], np.cumsum(nnode[f])])
    return G['faces']['nodes'][ni], pos


def _findNeighbors(G):
    """Port of MRST ``computeGeometry>findNeighbors`` (0-based, -1 boundary)."""
    cellNo = rldecode(np.arange(G['cells']['num'], dtype=np.int64),
                      np.diff(G['cells']['facePos']))
    cf = G['cells']['faces'][:, 0]
    j = np.argsort(cf, kind='stable')
    cellfaces = cf[j]
    cellNo = cellNo[j]
    hf = np.flatnonzero(cellfaces[:-1] == cellfaces[1:])
    N = np.full((G['faces']['num'], 2), -1, dtype=np.int64)
    N[cellfaces[hf], 0] = cellNo[hf]
    N[cellfaces[hf + 1], 1] = cellNo[hf + 1]
    isboundary = np.ones(cellNo.size, dtype=bool)
    isboundary[hf] = False
    isboundary[hf + 1] = False
    N[cellfaces[isboundary], 0] = cellNo[isboundary]
    return N


def _averageCoordinates(n, c, w=None):
    """Port of MRST private ``averageCoordinates``.

    Returns ``(average, no)`` where ``no`` is the (0-based) group index of
    each row of ``c``.
    """
    n = np.asarray(n, dtype=np.int64)
    c = np.asarray(c, dtype=float)
    if c.ndim == 1:
        c = c.reshape(-1, 1)
    if w is None:
        w = np.ones(c.shape[0], dtype=float)
    else:
        w = np.full(c.shape[0], float(w), dtype=float)
    no = rldecode(np.arange(n.size, dtype=np.int64), n)
    caug = np.column_stack([c, np.ones(c.shape[0], dtype=float)])
    sums = np.zeros((n.size, caug.shape[1]), dtype=float)
    np.add.at(sums, no, caug * w[:, None])
    wsum = sums[:, -1]
    with np.errstate(invalid='ignore', divide='ignore'):
        avg = sums[:, :-1] / wsum[:, None]
    return avg, no


def findNormalDirections(G):
    """Port of MRST ``computeGeometry>findNormalDirections``."""
    griddim = int(G['griddim'])
    nfn = np.diff(G['faces']['nodePos'])
    fcenters, _ = _averageCoordinates(nfn, G['nodes']['coords'][G['faces']['nodes']])
    ncf = np.diff(G['cells']['facePos'])
    ccenters, cellno = _averageCoordinates(ncf, fcenters[G['cells']['faces'][:, 0]])
    cf = G['cells']['faces'][:, 0]
    if griddim == 2:
        edges = G['faces']['nodes'].reshape(-1, 2)
        n1 = G['nodes']['coords'][edges[:, 0]]
        n2 = G['nodes']['coords'][edges[:, 1]]
        L = n2 - n1
        n = np.column_stack([L[:, 1], -L[:, 0]])
        v1 = fcenters[cf] - ccenters[cellno]
        a = np.sum(v1 * n[cf], axis=1)
    else:
        n1 = G['nodes']['coords'][G['faces']['nodes'][G['faces']['nodePos'][:-1]]]
        n2 = G['nodes']['coords'][G['faces']['nodes'][G['faces']['nodePos'][:-1] + 1]]
        v1 = fcenters[cf] - ccenters[cellno]
        v2 = n1[cf] - fcenters[cf]
        v3 = n2[cf] - n1[cf]
        a = np.sum(np.cross(v1, v2) * v3, axis=1)
    sgn = 2 * (G['faces']['neighbors'][cf, 0] == cellno) - 1
    i = np.bincount(cf, weights=a * sgn, minlength=G['faces']['num']) < 0
    neighbors = G['faces']['neighbors'].copy()
    neighbors[i] = neighbors[i][:, [1, 0]]
    G['faces']['neighbors'] = neighbors
    return G


def computeGeometry(G, findNeighbors=False):
    """Port of MRST ``computeGeometry`` (grid geometry + optional neighbour
    detection).  Reuses :func:`PRSTCore.gridprocessing.compute_geometry` for
    the actual geometry computation."""
    from PRSTCore.gridprocessing.compute_geometry import compute_geometry as _cg
    G = dict(G)
    if findNeighbors or 'neighbors' not in G.get('faces', {}):
        G['faces']['neighbors'] = _findNeighbors(G)
        G = findNormalDirections(G)
    return _cg(G)


def gridLogicalIndices(G, c=None):
    """Port of MRST ``gridLogicalIndices``: returns a list of 0-based logical
    index arrays, one per entry of ``G['cartDims']``."""
    if c is None:
        gcell = np.asarray(G['cells']['indexMap']).ravel()
    else:
        gcell = np.asarray(G['cells']['indexMap'])[np.asarray(c, dtype=np.int64).ravel()]
    ijk = np.unravel_index(gcell, tuple(int(x) for x in G['cartDims']), order='F')
    return [np.asarray(x, dtype=np.int64) for x in ijk]


def removeCells(G, cells):
    """Port of MRST ``removeCells``.

    Returns ``(H, cellmap, facemap, nodemap)`` where the maps are the
    new -> old (0-based) index lists (``cellmap[i]`` is the old cell id of
    new cell ``i``).
    """
    cells = np.asarray(cells, dtype=np.int64).ravel()
    nc = G['cells']['num']
    if cells.size == 0:
        return (G, np.arange(nc, dtype=np.int64),
                np.arange(G['faces']['num'], dtype=np.int64),
                np.arange(G['nodes']['num'], dtype=np.int64))

    G = dict(G)
    G['cells'] = dict(G['cells'])
    G['faces'] = dict(G['faces'])
    G['nodes'] = dict(G['nodes'])

    ind = np.zeros(nc, dtype=bool)
    ind[cells] = True
    cellmap = _mapExcluding(ind)  # old -> new (0-based), -1 removed

    # remove and renumber cells in cellFaces
    numFaces = np.diff(G['cells']['facePos'])
    keep_cf = ~rldecode(ind, numFaces)
    cells_faces = G['cells']['faces'][keep_cf].copy()

    # alter cell numbering in faces.neighbors
    neighbors = G['faces']['neighbors'].copy()
    pos = neighbors >= 0
    neighbors[pos] = cellmap[neighbors[pos]]

    # alter cells
    numFaces = numFaces[~ind]
    G['cells']['num'] = nc - cells.size
    G['cells']['facePos'] = np.concatenate([[0], np.cumsum(numFaces)])
    if 'indexMap' in G['cells']:
        G['cells']['indexMap'] = G['cells']['indexMap'][~ind]
    if 'global' in G['cells']:
        G['cells']['global'] = G['cells']['global'][~ind]

    # new numbering of faces
    ind_f = np.all(neighbors == -1, axis=1)
    facemap = _mapExcluding(ind_f)

    if 'nodes' in G['faces']:
        numNodes = np.diff(G['faces']['nodePos'])
        keep_fn = ~rldecode(ind_f, numNodes)
        faces_nodes = G['faces']['nodes'][keep_fn]
        numNodes = numNodes[~ind_f]

    cells_faces[:, 0] = facemap[cells_faces[:, 0]]
    if np.any(cells_faces[:, 0] < 0):
        raise ValueError('In removeCells: Too many faces removed!')
    G['cells']['faces'] = cells_faces

    typ = G.get('type', [])
    if any('computeGeometry' in t for t in typ):
        G['cells']['centroids'] = G['cells']['centroids'][~ind]
        G['cells']['volumes'] = G['cells']['volumes'][~ind]
        G['faces']['areas'] = G['faces']['areas'][~ind_f]
        G['faces']['centroids'] = G['faces']['centroids'][~ind_f]
        G['faces']['normals'] = G['faces']['normals'][~ind_f]

    G['faces']['neighbors'] = neighbors[~ind_f]
    if 'nodes' in G['faces']:
        G['faces']['nodes'] = faces_nodes
        G['faces']['nodePos'] = np.concatenate([[0], np.cumsum(numNodes)])
    if 'tag' in G['faces']:
        G['faces']['tag'] = G['faces']['tag'][~ind_f]
    G['faces']['num'] = G['faces']['num'] - int(ind_f.sum())

    if 'nodes' in G['faces']:
        ind_n = np.ones(G['nodes']['num'], dtype=bool)
        ind_n[G['faces']['nodes']] = False
        nodemap = _mapExcluding(ind_n)
        G['nodes']['coords'] = G['nodes']['coords'][~ind_n]
        G['nodes']['num'] = int(G['nodes']['num'] - ind_n.sum())
        G['faces']['nodes'] = nodemap[G['faces']['nodes']]
        if np.any(G['faces']['nodes'] < 0):
            raise ValueError('In removeCells: Too many nodes removed!')
    else:
        nodemap = np.arange(G['nodes']['num'], dtype=np.int64)

    G['type'] = list(typ) + ['removeCells']
    return (G,
            np.flatnonzero(cellmap >= 0),
            np.flatnonzero(facemap >= 0),
            np.flatnonzero(nodemap >= 0))
