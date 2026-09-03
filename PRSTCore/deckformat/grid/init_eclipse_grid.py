"""Initialize a simple MRST-like grid structure from an ECLIPSE deck.

This is a pragmatic, lightweight Python counterpart to MRST's
`initEclipseGrid.m`. It supports basic tensor grids defined by `DXV/DYV/DZV`
and a fallback for corner-point (`ZCORN`/`COORD`) grids by returning
raw arrays for downstream processing.

The returned structure `G` is a dict with keys depending on grid type.
"""

import numpy as np


def _make_tensor_grid(dx, dy, dz, actnum=None, z_offset=0.0):
    """Build the tensorGrid/extractSubgrid data used by MRST.

    Cell arrays and their active-cell mapping use MATLAB's I/J/K ordering
    (I contiguous, then J, then K).
    """
    dx = np.asarray(dx, dtype=float).ravel()
    dy = np.asarray(dy, dtype=float).ravel()
    dz = np.asarray(dz, dtype=float).ravel()
    nx, ny, nz = len(dx), len(dy), len(dz)
    nfull = nx * ny * nz
    xfaces = np.concatenate(([0.0], np.cumsum(dx)))
    yfaces = np.concatenate(([0.0], np.cumsum(dy)))
    zfaces = float(z_offset) + np.concatenate(([0.0], np.cumsum(dz)))
    xc = 0.5 * (xfaces[:-1] + xfaces[1:])
    yc = 0.5 * (yfaces[:-1] + yfaces[1:])
    zc = 0.5 * (zfaces[:-1] + zfaces[1:])
    Xc, Yc, Zc = np.meshgrid(xc, yc, zc, indexing='ij')
    centroids = np.column_stack((Xc.ravel(order='F'), Yc.ravel(order='F'),
                                 Zc.ravel(order='F')))
    dx3 = np.broadcast_to(dx[:, None, None], (nx, ny, nz)).ravel(order='F')
    dy3 = np.broadcast_to(dy[None, :, None], (nx, ny, nz)).ravel(order='F')
    dz3 = np.broadcast_to(dz[None, None, :], (nx, ny, nz)).ravel(order='F')
    dimensions = np.column_stack((dx3, dy3, dz3))
    volumes = np.prod(dimensions, axis=1)
    if actnum is None:
        active = np.ones(nfull, dtype=bool)
    else:
        active = np.asarray(actnum, dtype=int).ravel().astype(bool)
        if active.size != nfull:
            raise ValueError('ACTNUM length does not match cartDims')
    index_map = np.flatnonzero(active)
    cart_to_active = np.full(nfull, -1, dtype=int)
    cart_to_active[index_map] = np.arange(index_map.size, dtype=int)
    return {
        'type': 'tensor', 'xfaces': xfaces, 'yfaces': yfaces, 'zfaces': zfaces,
        'cell_centers': (Xc, Yc, Zc), 'cartDims': [nx, ny, nz],
        'ACTNUM': active.astype(np.int32), 'cart_to_active': cart_to_active,
        'cell_volumes': volumes[index_map], 'cell_dimensions': dimensions[index_map],
        'cells': {'indexMap': index_map, 'num': int(index_map.size),
                  'centroids': centroids[index_map],
                  'volumes': volumes[index_map],
                  'dimensions': dimensions[index_map]},
    }


def _build_corner_point_nodes(coord, zcorn, nx, ny, nz):
    """Port MRST ``buildCornerPtNodes`` for straight ECLIPSE pillars."""
    shape = (2 * nx, 2 * ny, 2 * nz)
    pillars = np.arange((nx + 1) * (ny + 1), dtype=int).reshape((nx + 1, ny + 1), order='F')
    p1 = pillars[:nx, :ny]
    p2 = pillars[1:, :ny]
    p3 = pillars[:nx, 1:]
    p4 = pillars[1:, 1:]

    line_ix = np.empty(shape, dtype=int)
    repeated = lambda p: np.broadcast_to(p[:, :, None], (nx, ny, 2 * nz))
    line_ix[0::2, 0::2, :] = repeated(p1)
    line_ix[1::2, 0::2, :] = repeated(p2)
    line_ix[0::2, 1::2, :] = repeated(p3)
    line_ix[1::2, 1::2, :] = repeated(p4)

    lines = np.asarray(coord, dtype=float).reshape((-1, 6))
    line_data = lines[line_ix.ravel(order='F')]
    z = np.asarray(zcorn, dtype=float).ravel(order='F')
    z_top = line_data[:, 2]
    z_bottom = line_data[:, 5]
    dz = z_bottom - z_top
    # MRST treats coincident pillar endpoints as vertical to avoid 0/0.
    t = np.zeros_like(z)
    regular = np.abs(dz) >= 100.0 * np.finfo(float).eps
    t[regular] = (z[regular] - z_top[regular]) / dz[regular]
    x = line_data[:, 0] + t * (line_data[:, 3] - line_data[:, 0])
    y = line_data[:, 1] + t * (line_data[:, 4] - line_data[:, 1])
    return (
        x.reshape(shape, order='F'),
        y.reshape(shape, order='F'),
        z.reshape(shape, order='F'),
    )


def _corner_point_cell_geometry(X, Y, Z):
    """Return MRST ``computeGeometry`` centroids/volumes for CP nodes.

    The previous diagonal-six-tetrahedra shortcut is exact for a rectangular
    cell but not for Norne's warped corner-point cells.  This is the 3-D
    face-triangulation construction in ``computeGeometry.m``: face centres
    are area weighted, a cell centre is their arithmetic mean, and the
    sub-triangle tetrahedra provide the final volume-weighted centroid.
    """
    corners = [
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
        (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
    ]
    vertices = np.stack(
        [
            np.stack(
                (X[ox::2, oy::2, oz::2], Y[ox::2, oy::2, oz::2], Z[ox::2, oy::2, oz::2]),
                axis=-1,
            )
            for ox, oy, oz in corners
        ],
        axis=-2,
    )

    full_vertices = vertices.reshape((-1, 8, 3), order='F')
    # The order only establishes a cyclic polygon for each quadrilateral;
    # its sign is made outward below, exactly as cfsign does in MRST.
    face_nodes = np.asarray([
        (0, 2, 3, 1), (4, 5, 7, 6),
        (0, 4, 6, 2), (1, 3, 7, 5),
        (0, 1, 5, 4), (2, 6, 7, 3),
    ], dtype=int)
    face_vertices = full_vertices[:, face_nodes, :]
    p_center = np.mean(face_vertices, axis=2)
    a = face_vertices
    b = np.roll(face_vertices, -1, axis=2)
    sub_normal = np.cross(b - a, p_center[:, :, None, :] - a) / 2.0
    sub_area = np.linalg.norm(sub_normal, axis=-1)
    sub_centroid = (a + b + p_center[:, :, None, :]) / 3.0
    face_area = np.sum(sub_area, axis=2)
    # ``processGRDECL`` removes collapsed faces before ``computeGeometry``.
    # The natural ECLIPSE corner ordering above already gives the same
    # outward orientation for the six nominal faces, so it is important not
    # to infer a direction from a centroid: for a collapsed wedge that
    # heuristic can reverse a perfectly valid top face.  Exclude zero-area
    # faces from the source ``cCenter = sum(faceCentroids)/#faces`` average,
    # exactly as the post-processGRDECL topology does.
    valid_face = face_area > 0.0
    safe_face_area = np.where(valid_face, face_area, 1.0)
    face_centroid = (np.sum(sub_area[:, :, :, None] * sub_centroid, axis=2) /
                     safe_face_area[:, :, None])
    n_valid_faces = np.maximum(np.sum(valid_face, axis=1), 1)
    c_center = (np.sum(face_centroid * valid_face[:, :, None], axis=1) /
                n_valid_faces[:, None])
    out_normal = sub_normal
    relative_subcentroid = sub_centroid - c_center[:, None, None, :]
    tetra_volume = np.sum(relative_subcentroid * out_normal, axis=-1) / 3.0
    volumes = np.sum(tetra_volume, axis=(1, 2))
    safe_volumes = np.where(volumes > 0.0, volumes, 1.0)
    centroid = c_center + (
        np.sum(tetra_volume[:, :, :, None] * (0.75 * relative_subcentroid), axis=(1, 2)) /
        safe_volumes[:, None]
    )

    def mean_edge_length(edge_pairs):
        return np.mean(
            [np.linalg.norm(vertices[..., b, :] - vertices[..., a, :], axis=-1) for a, b in edge_pairs],
            axis=0,
        )

    dx_ijk = mean_edge_length(((0, 1), (2, 3), (4, 5), (6, 7)))
    dy_ijk = mean_edge_length(((0, 2), (1, 3), (4, 6), (5, 7)))
    dz_ijk = mean_edge_length(((0, 4), (1, 5), (2, 6), (3, 7)))

    centroids = centroid
    dimensions = np.column_stack((dx_ijk.ravel(order='F'), dy_ijk.ravel(order='F'), dz_ijk.ravel(order='F')))
    return centroids, volumes, dimensions


def _cp_face_polygon(pa, pb, nodes):
    """Return MRST ``computeFaceGeometry``'s polygon for two CP sides.

    ``pa`` and ``pb`` contain the four corner-node numbers of the two
    sides, ordered bottom-left, bottom-right, top-left, top-right.  The
    implementation is deliberately scalar: fault stacks in real decks are
    sparse, while keeping this close to the MATLAB routine makes the
    handling of crossing layer boundaries and pinch-outs auditable.
    """
    pa = np.asarray(pa, dtype=int)
    pb = np.asarray(pb, dtype=int)
    if np.array_equal(pa, pb):
        return pa.tolist()

    az = nodes[pa, 2]
    bz = nodes[pb, 2]
    chosen = pa.copy()
    chosen[:2][az[:2] < bz[:2]] = pb[:2][az[:2] < bz[:2]]
    chosen[2:][az[2:] > bz[2:]] = pb[2:][az[2:] > bz[2:]]

    # Positions 1, 3, 5, 7 in processGRDECL's J vector.  ``None`` has
    # the same role as NaN there.
    corners = [chosen[0], None, chosen[1], None,
               chosen[3], None, chosen[2], None]
    if chosen[0] == chosen[3]:
        corners[0] = None
    if chosen[1] == chosen[2]:
        corners[2] = None

    def intersection(a0, a1, b0, b1):
        za0, za1 = nodes[a0, 2], nodes[a1, 2]
        zb0, zb1 = nodes[b0, 2], nodes[b1, 2]
        den = (za1 - za0) - (zb1 - zb0)
        if den == 0.0:
            return None
        t = (zb0 - za0) / den
        point = nodes[a0] + t * (nodes[a1] - nodes[a0])
        # The MATLAB implementation uses unique() only within this call.
        # Reusing an existing point when it is bitwise identical preserves
        # the topology while avoiding needless numerical copies.
        same = np.flatnonzero(np.all(nodes == point, axis=1))
        if same.size:
            return int(same[0])
        nonlocal_nodes.append(point)
        return nodes.shape[0] + len(nonlocal_nodes) - 1

    # New intersection points cannot be appended until all references have
    # been generated, otherwise the node array used by intersection changes
    # mid-call.  The caller appends ``new_nodes`` afterwards.
    nonlocal_nodes = []
    pairs = ((pa[0], pa[1], pb[0], pb[1]),  # bottom-bottom -> p2
             (pa[2], pa[3], pb[2], pb[3]),  # top-top       -> p6
             (pa[0], pa[1], pb[2], pb[3]),  # bottom-top    -> p4/p8
             (pa[2], pa[3], pb[0], pb[1]))  # top-bottom    -> p4/p8
    intersections = [None] * 4
    for n, (a0, a1, b0, b1) in enumerate(pairs):
        if (nodes[a0, 2] - nodes[b0, 2]) * (nodes[a1, 2] - nodes[b1, 2]) < 0.0:
            intersections[n] = intersection(a0, a1, b0, b1)

    corners[1] = intersections[0]
    corners[5] = intersections[1]
    if intersections[2] is not None:
        if az[0] > bz[2]:
            corners[0], corners[6], corners[7] = None, None, intersections[2]
        else:
            corners[2], corners[4], corners[3] = None, None, intersections[2]
    if intersections[3] is not None:
        if bz[0] > az[2]:
            corners[0], corners[6], corners[7] = None, None, intersections[3]
        else:
            corners[2], corners[4], corners[3] = None, None, intersections[3]

    # A temporary negative number identifies an intersection node.  It is
    # converted by the caller after adding the new coordinates to ``nodes``.
    first_new = nodes.shape[0]
    converted = []
    for node in corners:
        if node is None:
            continue
        converted.append(node)
    return converted, np.asarray(nonlocal_nodes, dtype=float).reshape((-1, 3)), first_new


def _cp_polygon(pa, pb, nodes):
    """``_cp_face_polygon`` with a compact, append-safe node interface."""
    pa = np.asarray(pa, dtype=int)
    pb = np.asarray(pb, dtype=int)
    if np.array_equal(pa, pb):
        return pa.tolist(), nodes
    az, bz = nodes[pa, 2], nodes[pb, 2]
    choose = pa.copy()
    lower = az[:2] < bz[:2]
    upper = az[2:] > bz[2:]
    choose[np.flatnonzero(lower)] = pb[:2][lower]
    choose[2 + np.flatnonzero(upper)] = pb[2:][upper]
    out = [choose[0], None, choose[1], None, choose[3], None, choose[2], None]
    if choose[0] == choose[3]: out[0] = None
    if choose[1] == choose[2]: out[2] = None

    def cross_line(a0, a1, b0, b1):
        za0, za1 = nodes[a0, 2], nodes[a1, 2]
        zb0, zb1 = nodes[b0, 2], nodes[b1, 2]
        if (za0-zb0)*(za1-zb1) >= 0.0:
            return None
        t = (zb0-za0) / ((za1-za0)-(zb1-zb0))
        point = nodes[a0] + t*(nodes[a1]-nodes[a0])
        # A strict z-crossing cannot coincide with either endpoint.  MRST
        # de-duplicates intersections within one fault-stack batch; their
        # shared coordinates (rather than global node identity) are what
        # matters for geometry, so append directly here.
        return nodes.shape[0], np.vstack((nodes, point))

    new = [None]*4
    for ix, args in enumerate(((pa[0],pa[1],pb[0],pb[1]),
                               (pa[2],pa[3],pb[2],pb[3]),
                               (pa[0],pa[1],pb[2],pb[3]),
                               (pa[2],pa[3],pb[0],pb[1]))):
        val = cross_line(*args)
        if val is not None:
            new[ix], nodes = val
    out[1], out[5] = new[0], new[1]
    if new[2] is not None:
        if az[0] > bz[2]: out[0], out[6], out[7] = None, None, new[2]
        else:              out[2], out[4], out[3] = None, None, new[2]
    if new[3] is not None:
        if bz[0] > az[2]: out[0], out[6], out[7] = None, None, new[3]
        else:              out[2], out[4], out[3] = None, None, new[3]
    return [int(x) for x in out if x is not None], nodes


def _cp_overlap(a1, a2, b1, b2):
    """MRST's ``doIntersect`` for the two pillar-line faces."""
    overlap = lambda x1, x2, y1, y2: max(x1, y1) < min(x2, y2)
    return (overlap(a1[0], a2[0], b1[0], b2[0]) or
            overlap(a1[1], a2[1], b1[1], b2[1]) or
            (a1[0]-b1[0])*(a1[1]-b1[1]) < 0.0 or
            (a2[0]-b2[0])*(a2[1]-b2[1]) < 0.0)


def _cp_general_topology(X, Y, Z, active):
    """Process a faulted corner-point grid like MRST ``processGRDECL``.

    In contrast to the historical six-face shortcut, this creates the
    actual (possibly split) fault faces before evaluating geometry.  The
    topology is intentionally retained on the returned grid: TPFA needs
    the same split connections as MRST's ``computeTrans``.
    """
    nx, ny, nz = (X.shape[0]//2, X.shape[1]//2, X.shape[2]//2)
    minz, maxz = float(np.min(Z)), float(np.max(Z))
    zplane = np.zeros_like(Z[:, :, 0])
    Xe = np.concatenate((X[:, :, :1], X[:, :, :1], X, X[:, :, -1:], X[:, :, -1:]), axis=2)
    Ye = np.concatenate((Y[:, :, :1], Y[:, :, :1], Y, Y[:, :, -1:], Y[:, :, -1:]), axis=2)
    Ze = np.concatenate((minz-2.0+zplane[:, :, None], minz-1.0+zplane[:, :, None],
                         Z, maxz+1.0+zplane[:, :, None], maxz+2.0+zplane[:, :, None]), axis=2)
    nk = nz + 2
    ae = np.concatenate((np.ones((nx, ny, 1), dtype=bool), active,
                         np.ones((nx, ny, 1), dtype=bool)), axis=2)

    # MATLAB unique([Z,Y,X], 'rows'), hence the deliberately non-standard
    # sort key and Fortran reshaping below.
    zyx = np.column_stack((Ze.ravel(order='F'), Ye.ravel(order='F'), Xe.ravel(order='F')))
    unique_zyx, inverse = np.unique(zyx, axis=0, return_inverse=True)
    nodes = unique_zyx[:, ::-1].copy()
    P = inverse.reshape(Ze.shape, order='F')
    B = np.arange(nx*ny*nk, dtype=int).reshape((nx, ny, nk), order='F')
    faces, neighbors, face_tags = [], [], []

    def append(nodes_for_face, ca, cb, tag):
        if ca < 0 and cb < 0:
            return
        # Two coincident nodes is a zero-area pinch face.  This is the same
        # early elimination performed by findFaces/computeFaceGeometry.
        if len(nodes_for_face) < 3:
            return
        pts = nodes[np.asarray(nodes_for_face, dtype=int)]
        pc = np.mean(pts, axis=0)
        normal = np.sum(np.cross(np.roll(pts, -1, axis=0)-pts, pc-pts), axis=0)/2.0
        if np.linalg.norm(normal) <= 1.0e-18:
            return
        faces.append([int(q) for q in nodes_for_face])
        neighbors.append((int(ca), int(cb)))
        face_tags.append(int(tag))

    def cell_id(i, j, k):
        return int(B[i, j, k]) if ae[i, j, k] else -1

    def face_nodes(i, j, k, side):
        # Natural CP ordering, cyclic for each nominal side.
        q = (2*i, 2*j, 2*k)
        if side == 'imin':  return [P[q[0],q[1],q[2]], P[q[0],q[1]+1,q[2]], P[q[0],q[1]+1,q[2]+1], P[q[0],q[1],q[2]+1]]
        if side == 'imax':  return [P[q[0]+1,q[1],q[2]], P[q[0]+1,q[1]+1,q[2]], P[q[0]+1,q[1]+1,q[2]+1], P[q[0]+1,q[1],q[2]+1]]
        if side == 'jmin':  return [P[q[0],q[1],q[2]], P[q[0],q[1],q[2]+1], P[q[0]+1,q[1],q[2]+1], P[q[0]+1,q[1],q[2]]]
        if side == 'jmax':  return [P[q[0],q[1]+1,q[2]], P[q[0],q[1]+1,q[2]+1], P[q[0]+1,q[1]+1,q[2]+1], P[q[0]+1,q[1]+1,q[2]]]
        if side == 'kmin':  return [P[q[0],q[1],q[2]], P[q[0]+1,q[1],q[2]], P[q[0]+1,q[1]+1,q[2]], P[q[0],q[1]+1,q[2]]]
        return [P[q[0],q[1],q[2]+1], P[q[0],q[1]+1,q[2]+1], P[q[0]+1,q[1]+1,q[2]+1], P[q[0]+1,q[1],q[2]+1]]

    def add_pillar_direction(axis):
        nonlocal nodes
        ni, nj = (nx-1, ny) if axis == 0 else (nx, ny-1)
        for i in range(ni):
            for j in range(nj):
                if axis == 0:
                    aseq = np.column_stack((P[2*i+1, 2*j, :], P[2*i+1, 2*j+1, :]))
                    bseq = np.column_stack((P[2*i+2, 2*j, :], P[2*i+2, 2*j+1, :]))
                    ca = lambda k: cell_id(i, j, k)
                    cb = lambda k: cell_id(i+1, j, k)
                    aside, bside = 'imax', 'imin'
                else:
                    aseq = np.column_stack((P[2*i, 2*j+1, :], P[2*i+1, 2*j+1, :]))
                    bseq = np.column_stack((P[2*i, 2*j+2, :], P[2*i+1, 2*j+2, :]))
                    ca = lambda k: cell_id(i, j, k)
                    cb = lambda k: cell_id(i, j+1, k)
                    aside, bside = 'jmax', 'jmin'
                regular = np.array_equal(aseq, bseq)
                if regular:
                    for k in range(nk):
                        append(face_nodes(i, j, k, aside), ca(k), cb(k), axis+1)
                    continue

                # findFaults removes all point rows belonging to inactive
                # cells *before* looking for overlaps.  Keeping those rows
                # as explicit void intervals subtly creates extra lateral
                # faces at Norne's inactive boundary.
                aid = np.asarray([ca(k) for k in range(nk)], dtype=int)
                bid = np.asarray([cb(k) for k in range(nk)], dtype=int)
                akeep = aid >= 0
                bkeep = bid >= 0
                aseq = np.vstack([aseq[2*k:2*k+2] for k in np.flatnonzero(akeep)])
                bseq = np.vstack([bseq[2*k:2*k+2] for k in np.flatnonzero(bkeep)])
                aid, bid = aid[akeep], bid[bkeep]

                # Direct scalar version of findConnections.  Odd intervals
                # are the ZCORN gaps and therefore have the void cell -1.
                jleft = jright = 0
                for ka in range(aseq.shape[0]-1):
                    ida = aid[ka//2] if ka % 2 == 0 else -1
                    za1, za2 = nodes[aseq[ka], 2], nodes[aseq[ka+1], 2]
                    kb = min(jleft, jright)
                    # This is findConnections' monotone stack walk.  A
                    # faulted Norne interface has ~48 ZCORN levels, so it
                    # avoids an otherwise quadratic all-layer search.
                    while kb < bseq.shape[0]-1 and np.any(nodes[bseq[kb], 2] < za2):
                        idb = bid[kb//2] if kb % 2 == 0 else -1
                        zb1, zb2 = nodes[bseq[kb], 2], nodes[bseq[kb+1], 2]
                        if ida >= 0 or idb >= 0:
                            if (not np.all(za1 == za2) and not np.all(zb1 == zb2)
                                    and _cp_overlap(za1, za2, zb1, zb2)):
                                pa = [aseq[ka,0], aseq[ka,1], aseq[ka+1,0], aseq[ka+1,1]]
                                pb = [bseq[kb,0], bseq[kb,1], bseq[kb+1,0], bseq[kb+1,1]]
                                polygon, nodes = _cp_polygon(pa, pb, nodes)
                                # processGRDECL swaps I/J to reuse the
                                # pillar routine, then reverses all J face
                                # nodes to restore its physical orientation.
                                if axis == 1:
                                    polygon = polygon[::-1]
                                append(polygon, ida, idb, axis+1)
                        if zb1[0] < za2[0]:
                            jleft = kb
                        if zb1[1] < za2[1]:
                            jright = kb
                        kb += 1

    add_pillar_direction(0)
    # Regular external I boundaries.
    for j in range(ny):
        for k in range(nk):
            append(face_nodes(0, j, k, 'imin'), -1, cell_id(0, j, k), 1)
            append(face_nodes(nx-1, j, k, 'imax'), cell_id(nx-1, j, k), -1, 1)
    add_pillar_direction(1)
    for i in range(nx):
        for k in range(nk):
            append(face_nodes(i, 0, k, 'jmin'), -1, cell_id(i, 0, k), 2)
            append(face_nodes(i, ny-1, k, 'jmax'), cell_id(i, ny-1, k), -1, 2)

    # findVerticalFaces: coalesce consecutive, identical horizontal faces.
    for i in range(nx):
        for j in range(ny):
            seq = []
            for p in range(2*nk):
                seq.append(tuple((P[2*i,2*j,p], P[2*i+1,2*j,p],
                                  P[2*i+1,2*j+1,p], P[2*i,2*j+1,p])))
            cseq = [-1]*(2*nk+1)
            for k in range(nk):
                cseq[2*k+1] = cell_id(i, j, k)
            p = 0
            while p < len(seq):
                q = p + 1
                while q < len(seq) and seq[q] == seq[p]:
                    q += 1
                append(list(seq[p]), cseq[p], cseq[q], 3)
                p = q

    # Remove auxiliary/inactive cells exactly as processGRDECL does.  All
    # original active cells retain Eclipse Cartesian (Fortran) ordering.
    keep_ext = []
    index_map = []
    for k in range(1, nz+1):
        for j in range(ny):
            for i in range(nx):
                if active[i, j, k-1]:
                    keep_ext.append(int(B[i,j,k]))
                    index_map.append(i + nx*(j + ny*(k-1)))
    ext_to_active = np.full(nx*ny*nk, -1, dtype=int)
    ext_to_active[np.asarray(keep_ext, dtype=int)] = np.arange(len(keep_ext), dtype=int)
    final_faces, final_neighbors, final_tags = [], [], []
    for f, (ca, cb), tag in zip(faces, neighbors, face_tags):
        aa = ext_to_active[ca] if ca >= 0 else -1
        bb = ext_to_active[cb] if cb >= 0 else -1
        if aa >= 0 or bb >= 0:
            final_faces.append(f)
            final_neighbors.append((aa, bb))
            final_tags.append(tag)
    return (nodes, final_faces, np.asarray(final_neighbors, dtype=int),
            np.asarray(final_tags, dtype=int), np.asarray(index_map, dtype=int))


def _cp_geometry_from_topology(nodes, face_nodes, neighbors, ncell, nominal_centers=None):
    """MRST ``computeGeometry`` for the topology emitted above."""
    cell_faces = [[] for _ in range(ncell)]
    face_centers, face_normals, face_areas = [], [], []
    oriented_neighbors = np.asarray(neighbors, dtype=int).copy()
    oriented_nodes = [list(f) for f in face_nodes]
    for fi, (f, pair) in enumerate(zip(oriented_nodes, oriented_neighbors)):
        points = nodes[np.asarray(f, dtype=int)]
        pcenter = np.mean(points, axis=0)
        nxt = np.roll(points, -1, axis=0)
        subnormal = np.cross(nxt-points, pcenter-points)/2.0
        subarea = np.linalg.norm(subnormal, axis=1)
        normal = np.sum(subnormal, axis=0)
        # ``computeGeometry`` defines area as the sum of its fan-triangle
        # areas, not the length of their resultant normal.  The distinction
        # matters on Norne's warped horizontal faces.
        area = float(np.sum(subarea))
        if area <= 0.0:
            # Keep face numbering aligned with topology.  This can only be
            # a collapsed pinch face and contributes zero to the cell
            # tetrahedra, exactly as the geometry kernel's zero normal.
            centroid = pcenter
        else:
            centroid = np.sum(((points+nxt+pcenter)/3.0)*subarea[:,None], axis=0)/np.sum(subarea)
        a, b = pair
        # Face-node order is established by processGRDECL: normals point
        # from neighbor 1 to neighbor 2.  Re-inferring it from centroids is
        # not valid for warped/non-convex CP cells (and changes MRST's
        # signed fan-triangle volume).  ``nominal_centers`` is retained in
        # the signature for compatibility with the former lightweight path.
        face_centers.append(centroid)
        face_normals.append(normal)
        face_areas.append(area)
        if a >= 0: cell_faces[a].append((fi, 1.0))
        if b >= 0: cell_faces[b].append((fi, -1.0))

    # No face is expected to be degenerate after processGRDECL's filters;
    # retain indexing defensively in case a pathological deck has one.
    face_centers = np.asarray(face_centers, dtype=float)
    face_normals = np.asarray(face_normals, dtype=float)
    face_areas = np.asarray(face_areas, dtype=float)
    volumes = np.zeros(ncell, dtype=float)
    centroids = np.zeros((ncell, 3), dtype=float)
    for c, linked in enumerate(cell_faces):
        if not linked:
            continue
        ids = np.asarray([fi for fi, _ in linked], dtype=int)
        ccenter = np.mean(face_centers[ids], axis=0)
        volume = 0.0
        relcentroid = np.zeros(3, dtype=float)
        for fi, sign in linked:
            f = oriented_nodes[fi]
            points = nodes[np.asarray(f, dtype=int)]
            pcenter = np.mean(points, axis=0)
            nxt = np.roll(points, -1, axis=0)
            subnormal = np.cross(nxt-points, pcenter-points)/2.0
            # face_geom3d records a sign for each fan triangle.  A warped
            # polygon can have a locally reversed triangle; treating all
            # of them as outward was the remaining source of incorrect
            # CP volumes/centres after the fault topology was restored.
            f_normal = np.sum(subnormal, axis=0)
            subnormal *= np.sign(np.sum(subnormal*f_normal, axis=1))[:, None] * sign
            subcentroid = (points+nxt+pcenter)/3.0
            rel = subcentroid-ccenter
            tv = np.sum(rel*subnormal, axis=1)/3.0
            volume += float(np.sum(tv))
            relcentroid += np.sum(tv[:,None]*(0.75*rel), axis=0)
        volumes[c] = volume
        centroids[c] = ccenter + relcentroid/max(volume, 1e-300)
    return (centroids, volumes, oriented_nodes, oriented_neighbors,
            face_centers, face_normals, face_areas)


def _cp_mex_point_lists(coord, zcorn, active, nx, ny, nz):
    """Port ``finduniquepoints`` from MRST's ``processgrid_mex``.

    The MEX processor, used by ``initEclipseProblemAD`` by default, keeps
    every distinct depth on a pillar.  This differs materially from the
    MATLAB processor at faults: a lateral face may legitimately contain
    more than four vertices along its two pillar edges.
    """
    zcorn = np.asarray(zcorn, dtype=float).reshape((2*nx, 2*ny, 2*nz), order='F')
    active = np.asarray(active, dtype=bool)
    lists = np.empty((2*nx, 2*ny, 2*nz+2), dtype=np.int64)
    node_coords = []
    pillar_depths = {}
    lines = np.asarray(coord, dtype=float).reshape((-1, 6))
    sentinel_min = np.iinfo(np.int64).min
    sentinel_max = np.iinfo(np.int64).max

    for jp in range(ny+1):
        for ip in range(nx+1):
            depths = []
            for jy in (2*jp-1, 2*jp):
                for ix in (2*ip-1, 2*ip):
                    if 0 <= ix < 2*nx and 0 <= jy < 2*ny:
                        for kz in range(2*nz):
                            if active[ix//2, jy//2, kz//2]:
                                depths.append(float(zcorn[ix, jy, kz]))
            # Every pillar has an active boundary point in a valid grid.
            values = np.unique(np.asarray(depths, dtype=float))
            pillar = ip + (nx+1)*jp
            start = len(node_coords)
            line = lines[pillar]
            dz = line[5] - line[2]
            for z in values:
                t = 0.0 if dz == 0.0 else (z-line[2])/dz
                # Keep C's arithmetic order for bit-level agreement with
                # processgrid_mex/interpolate_pillar.
                node_coords.append(((1.0-t)*line[0] + t*line[3],
                                    (1.0-t)*line[1] + t*line[4], z))
            pillar_depths[pillar] = (values, start)

    for jy in range(2*ny):
        for ix in range(2*nx):
            pillar = ((ix+1)//2) + (nx+1)*((jy+1)//2)
            values, start = pillar_depths[pillar]
            row = lists[ix, jy]
            row[0], row[-1] = sentinel_min, sentinel_max
            for kz in range(2*nz):
                if not active[ix//2, jy//2, kz//2]:
                    row[kz+1] = row[kz]
                else:
                    z = zcorn[ix, jy, kz]
                    # MEX's tolerance is zero in the normal AD path.
                    pos = int(np.searchsorted(values, z, side='left'))
                    row[kz+1] = start + pos
    return np.asarray(node_coords, dtype=float), lists


def _cp_mex_intersection(nodes, ids):
    """Port ``approximate_intersection_pt`` from processgrid_mex."""
    a0, a1, b0, b1 = (int(q) for q in ids)
    p0, p1, p2, p3 = nodes[[a0, a1, b0, b1]]
    den = (p1[2]-p0[2]) - (p3[2]-p2[2])
    alpha = 0.0 if abs(den) == 0.0 else (p2[2]-p0[2])/den
    z = p0[2]*(1.0-alpha) + p1[2]*alpha
    # The two points on a horizontal line of the bilinear surface.
    beta1 = (p2[2]-z)/(p2[2]-p0[2])
    beta2 = (z-p0[2])/(p2[2]-p0[2])
    x1 = p0[:2]*beta1 + p2[:2]*beta2
    beta1 = (z-p3[2])/(p1[2]-p3[2])
    beta2 = (p1[2]-z)/(p1[2]-p3[2])
    x2 = p1[:2]*beta1 + p3[:2]*beta2
    xy = x1*(1.0-alpha) + x2*alpha
    return np.array((xy[0], xy[1], z), dtype=float)


def _cp_mex_topology(coord, zcorn, active, nx, ny, nz):
    """Faithful topology port of MRST's default ``processgrid_mex``."""
    nodes, plist = _cp_mex_point_lists(coord, zcorn, active, nx, ny, nz)
    sentinel_min = np.iinfo(np.int64).min
    sentinel_max = np.iinfo(np.int64).max
    faces, neighbors, tags, intersections = [], [], [], []
    nline = 2*nz + 2

    def meaningful(a1, b1, i, j):
        return not ((a1[i] == sentinel_min and b1[j] == sentinel_min) or
                    (a1[i+1] == sentinel_max and b1[j+1] == sentinel_max))

    def intersects(a1, a2, b1, b2, i, j):
        return (max(a1[i], b1[j]) < min(a1[i+1], b1[j+1]) or
                max(a2[i], b2[j]) < min(a2[i+1], b2[j+1]) or
                ((a1[i] > b1[j] and a2[i] < b2[j]) or
                 (a1[i] < b1[j] and a2[i] > b2[j])) or
                ((a1[i+1] > b1[j+1] and a2[i+1] < b2[j+1]) or
                 (a1[i+1] < b1[j+1] and a2[i+1] > b2[j+1])))

    def topology(a1, a2, b1, b2, i, j, cross):
        mask = [-1]*8
        mask[0] = b1[j+1] if a1[i+1] > b1[j+1] else a1[i+1]
        mask[2] = b2[j+1] if a2[i+1] > b2[j+1] else a2[i+1]
        mask[4] = a2[i] if a2[i] > b2[j] else b2[j]
        mask[6] = a1[i] if a1[i] > b1[j] else b1[j]
        if mask[0] == mask[6]: mask[6] = -1
        if mask[2] == mask[4]: mask[4] = -1
        mask[1], mask[5] = cross[3], cross[0]
        if cross[1] >= 0:
            if a1[i] > b1[j+1]: mask[0], mask[6], mask[7] = -1, -1, cross[1]
            else:                mask[2], mask[4], mask[3] = -1, -1, cross[1]
        if cross[2] >= 0:
            if a1[i+1] < b1[j]: mask[0], mask[6], mask[7] = -1, -1, cross[2]
            else:                mask[2], mask[4], mask[3] = -1, -1, cross[2]
        return [int(q) for q in mask[::-1] if q >= 0]

    def process_pair(a1, a2, b1, b2, base_a, base_b, tag):
        nonlocal nodes
        itop = np.full(nline, -1, dtype=int)
        ibottom = np.full(nline, -1, dtype=int)
        k1 = k2 = 0
        j = 0
        for i in range(nline-1):
            if a1[i] == a1[i+1] and a2[i] == a2[i+1]:
                continue
            while j < nline-1 and (b1[j] < a1[i+1] or b2[j] < a2[i+1]):
                if b1[j] == b1[j+1] and b2[j] == b2[j+1]:
                    itop[j+1] = itop[j]
                    j += 1
                    continue
                if intersects(a1, a2, b1, b2, i, j):
                    ca = (i-1)//2 if i % 2 else -1
                    cb = (j-1)//2 if j % 2 else -1
                    if (a1[i] == b1[j] and a1[i+1] == b1[j+1] and
                            a2[i] == b2[j] and a2[i+1] == b2[j+1]):
                        if meaningful(a1, b1, i, j) and (ca >= 0 or cb >= 0):
                            poly = [int(a1[i]), int(a2[i])]
                            poly.extend(range(int(a2[i])+1, int(a2[i+1])+1))
                            poly.extend(range(int(a1[i+1]), int(a1[i]), -1))
                            faces.append(poly)
                            neighbors.append((base_a(ca) if ca >= 0 else -1,
                                              base_b(cb) if cb >= 0 else -1))
                            tags.append(tag)
                    else:
                        if ((a1[i+1] > b1[j+1] and a2[i+1] < b2[j+1]) or
                                (a1[i+1] < b1[j+1] and a2[i+1] > b2[j+1])):
                            itop[j+1] = len(nodes)
                            intersections.append((int(a1[i+1]), int(a2[i+1]),
                                                  int(b1[j+1]), int(b2[j+1])))
                            nodes = np.vstack((nodes, _cp_mex_intersection(nodes, intersections[-1])))
                        else:
                            itop[j+1] = -1
                        cross = (int(ibottom[j]), int(ibottom[j+1]),
                                 int(itop[j]), int(itop[j+1]))
                        if meaningful(a1, b1, i, j) and (ca >= 0 or cb >= 0):
                            faces.append(topology(a1, a2, b1, b2, i, j, cross))
                            neighbors.append((base_a(ca) if ca >= 0 else -1,
                                              base_b(cb) if cb >= 0 else -1))
                            tags.append(tag)
                if b1[j] < a1[i+1]: k1 = j
                if b2[j] < a2[i+1]: k2 = j
                j += 1
            itop, ibottom = ibottom, itop
            itop.fill(-1)
            j = min(k1, k2)

    # Constant-I faces (tag 1) and constant-J faces (tag 2), following
    # process_vertical_faces/igetvectors exactly.
    for j in range(ny):
        for i in range(nx+1):
            im, ip = max(1, 2*i)-1, min(2*nx, 2*i+1)-1
            jm, jp = 2*j, 2*j+1
            process_pair(plist[im,jm], plist[im,jp], plist[ip,jm], plist[ip,jp],
                         lambda k, ii=i, jj=j: -1 if ii-1 < 0 else (ii-1)+nx*(jj+ny*k),
                         lambda k, ii=i, jj=j: -1 if ii >= nx else ii+nx*(jj+ny*k), 1)
    for j in range(ny+1):
        for i in range(nx):
            im, ip = 2*i, 2*i+1
            jm, jp = max(1, 2*j)-1, min(2*ny, 2*j+1)-1
            # Clockwise rotation in process_vertical_faces(direction=1).
            process_pair(plist[ip,jm], plist[im,jm], plist[ip,jp], plist[im,jp],
                         lambda k, ii=i, jj=j: -1 if jj-1 < 0 else ii+nx*((jj-1)+ny*k),
                         lambda k, ii=i, jj=j: -1 if jj >= ny else ii+nx*(jj+ny*k), 2)

    # Constant-K faces, including the MEX collapsed-cell rule.
    valid = np.zeros(nx*ny*nz, dtype=bool)
    for j in range(ny):
        for i in range(nx):
            c0, c1 = plist[2*i,2*j], plist[2*i,2*j+1]
            c2, c3 = plist[2*i+1,2*j], plist[2*i+1,2*j+1]
            previous = -1
            for k in range(1, 2*nz+1):
                collapsed = (c0[k] == c0[k+1] and c1[k] == c1[k+1] and
                             c2[k] == c2[k+1] and c3[k] == c3[k+1])
                if collapsed:
                    continue
                poly = [int(c0[k]), int(c2[k]), int(c3[k]), int(c1[k])]
                if k % 2:
                    cell = i + nx*(j + ny*((k-1)//2))
                    valid[cell] = True
                    faces.append(poly); neighbors.append((previous, cell)); tags.append(3)
                    previous = cell
                elif previous >= 0:
                    faces.append(poly); neighbors.append((previous, -1)); tags.append(3)
                    previous = -1

    old_to_new = np.full(nx*ny*nz, -1, dtype=int)
    index_map = np.flatnonzero(valid)
    old_to_new[index_map] = np.arange(index_map.size, dtype=int)
    final_faces, final_neighbors, final_tags = [], [], []
    for f, (a, b), tag in zip(faces, neighbors, tags):
        aa = old_to_new[a] if a >= 0 else -1
        bb = old_to_new[b] if b >= 0 else -1
        if aa >= 0 or bb >= 0:
            final_faces.append(f); final_neighbors.append((aa, bb)); final_tags.append(tag)
    return (nodes, final_faces, np.asarray(final_neighbors, dtype=int),
            np.asarray(final_tags, dtype=int), index_map)


def _corner_point_grid(coord, zcorn, nx, ny, nz, actnum_raw):
    """Build the corner-point grid dict from raw COORD/ZCORN arrays.

    ``coord`` is the flat COORD array (6 values per pillar); ``zcorn`` is
    the flat ZCORN array in MATLAB/Fortran order (``2*nx`` by ``2*ny`` by
    ``2*nz``), exactly the raw ECLIPSE keyword layout.  ``actnum_raw`` is
    the raw ACTNUM array (or ``None`` for an all-active grid).  Shared by
    genuine ``ZCORN``/``COORD`` decks and by the block-centred (DX/DY/DZ +
    TOPS) path, which synthesizes an equivalent COORD/ZCORN pair.
    """
    coord = np.asarray(coord, dtype=float).ravel()
    zcorn = np.asarray(zcorn, dtype=float).ravel(order='F')
    expected_coord = 6 * (nx + 1) * (ny + 1)
    expected_zcorn = 8 * nx * ny * nz
    if coord.size != expected_coord or zcorn.size != expected_zcorn:
        raise ValueError(
            'Invalid corner-point dimensions: expected %d COORD and %d ZCORN values, got %d and %d'
            % (expected_coord, expected_zcorn, coord.size, zcorn.size)
        )

    X, Y, Z = _build_corner_point_nodes(coord, zcorn, nx, ny, nz)
    # ``processGRDECL`` splits faulted logical sides into their actual
    # polygonal faces.  This is indispensable for Norne: treating every
    # logical block as a six-face hexahedron shifts centroids by metres
    # at faults and consequently changes hydrostatic initialization.
    nfull = nx * ny * nz

    # Keep the original eight physical corner points for every logical
    # cell.  ``computeTrans.m`` derives the one-sided transmissibility
    # from the actual face normal and face centroid, so cell dimensions
    # alone are not sufficient on a tilted corner-point grid.
    corners = [
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
        (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
    ]
    corner_vertices = np.stack(
        [
            np.stack(
                (X[ox::2, oy::2, oz::2],
                 Y[ox::2, oy::2, oz::2],
                 Z[ox::2, oy::2, oz::2]),
                axis=-1,
            )
            for ox, oy, oz in corners
        ],
        axis=-2,
    ).reshape((-1, 8, 3), order='F')

    actnum = np.asarray(actnum_raw if actnum_raw is not None else np.ones(nfull), dtype=int).ravel().astype(bool)
    if actnum.size != nfull:
        raise ValueError('ACTNUM length does not match cartDims')
    active_cartesian = np.flatnonzero(actnum)
    cart_to_active = np.full(nfull, -1, dtype=int)
    cart_to_active[active_cartesian] = np.arange(active_cartesian.size, dtype=int)

    nodes, topology_faces, topology_neighbors, topology_tags, topology_index_map = _cp_mex_topology(
        coord, zcorn, actnum.reshape((nx, ny, nz), order='F'), nx, ny, nz
    )
    if not np.array_equal(topology_index_map, active_cartesian):
        raise RuntimeError('Corner-point topology changed Eclipse active-cell ordering')

    # The nominal centres are used solely to establish face-normal
    # direction.  Volumes/centres below are fully topology based.
    nominal_centers, _, dimensions = _corner_point_cell_geometry(X, Y, Z)
    centroids, volumes, topology_faces, topology_neighbors, face_centroids, face_normals, face_areas = \
        _cp_geometry_from_topology(
            nodes, topology_faces, topology_neighbors, active_cartesian.size,
            nominal_centers[active_cartesian],
        )

    # Per-axis widths are retained for the regular TPFA shortcut.  Cell
    # vectors remain authoritative for nonuniform corner-point grids.
    dim_ijk = dimensions.reshape((nx, ny, nz, 3), order='F')
    return {
        'type': 'corner_point',
        'cartDims': [nx, ny, nz],
        'COORD': coord.reshape((-1, 6)),
        'ZCORN': zcorn.reshape((2 * nx, 2 * ny, 2 * nz), order='F'),
        'dx': np.mean(dim_ijk[:, :, :, 0], axis=(1, 2)),
        'dy': np.mean(dim_ijk[:, :, :, 1], axis=(0, 2)),
        'dz_layer': np.mean(dim_ijk[:, :, :, 2], axis=(0, 1)),
        'cart_to_active': cart_to_active,
        'cell_volumes': volumes,
        'cell_dimensions': dimensions[active_cartesian],
        'corner_vertices': corner_vertices[active_cartesian],
        'faces': {
            'nodes': topology_faces,
            'neighbors': topology_neighbors,
            'centroids': face_centroids,
            'normals': face_normals,
            'areas': face_areas,
            'tag': topology_tags,
        },
        'nodes': {'coords': nodes},
        'cells': {
            'indexMap': active_cartesian,
            'num': int(active_cartesian.size),
            'centroids': centroids,
            'volumes': volumes,
            'dimensions': dimensions[active_cartesian],
        },
    }


def _dx_dy_dz_tops_to_corner_point(dx, dy, dz, tops, nx, ny, nz):
    """Synthesize an equivalent COORD/ZCORN pair for a block-centred
    (DX/DY/DZ + TOPS) grid whose per-column TOPS is not constant.

    ECLIPSE's block-centred grid format lets every column have its own top
    depth; MRST's ``initEclipseGrid.m`` only supports the constant-TOPS
    special case because it builds a simple ``tensorGrid`` with a single
    ``depthz`` value per node, which cannot represent per-column depth
    offsets.  The corner-point (COORD/ZCORN) representation has no such
    restriction -- each cell already owns eight independent corners -- so
    a block-centred grid is converted to the equivalent corner-point grid
    with vertical pillars: cell (i, j, k)'s top/bottom face depths are
    ``TOPS(i, j) + cumsum(DZ(i, j, :))``, identical for all four corners
    of that face since a block-centred cell has no internal dip.

    ``dx``/``dy``/``dz`` are the per-axis cell sizes (length nx/ny/nz,
    already reduced from any cell-wise DX/DY/DZ input); ``tops`` is the
    per-column top depth (length nx*ny, ECLIPSE's I-fastest-then-J order).
    """
    dx = np.asarray(dx, dtype=float).ravel()
    dy = np.asarray(dy, dtype=float).ravel()
    dz = np.asarray(dz, dtype=float).ravel()
    tops = np.asarray(tops, dtype=float).ravel()

    xfaces = np.concatenate(([0.0], np.cumsum(dx)))
    yfaces = np.concatenate(([0.0], np.cumsum(dy)))

    # Column-wise cumulative layer-boundary depth: shape (nx, ny, nz + 1).
    tops_ij = tops.reshape((nx, ny), order='F')
    layer_bottom = tops_ij[:, :, None] + np.cumsum(dz)[None, None, :]
    z_bounds = np.concatenate([tops_ij[:, :, None], layer_bottom], axis=2)

    # COORD: one vertical pillar per (nx+1) x (ny+1) node, I fastest then J.
    z_top_pillar = float(z_bounds.min()) - 1.0
    z_bot_pillar = float(z_bounds.max()) + 1.0
    px, py = np.meshgrid(xfaces, yfaces, indexing='ij')
    coord = np.empty(((nx + 1) * (ny + 1), 6), dtype=float)
    coord[:, 0] = px.ravel(order='F')
    coord[:, 1] = py.ravel(order='F')
    coord[:, 2] = z_top_pillar
    coord[:, 3] = px.ravel(order='F')
    coord[:, 4] = py.ravel(order='F')
    coord[:, 5] = z_bot_pillar

    # ZCORN: shape (2*nx, 2*ny, 2*nz); every cell's four corners along a
    # given face share the same depth (flat top/bottom, no internal dip).
    zcorn = np.empty((2 * nx, 2 * ny, 2 * nz), dtype=float)
    top_layer = z_bounds[:, :, :-1]
    bot_layer = z_bounds[:, :, 1:]
    for oz, layer in ((0, top_layer), (1, bot_layer)):
        for ox in (0, 1):
            for oy in (0, 1):
                zcorn[ox::2, oy::2, oz::2] = layer

    return coord.ravel(), zcorn.ravel(order='F')


def init_eclipse_grid(deck, mapAxes=False, removeZeroPV=False, useMex=False, **kwargs):
    """Construct a minimal grid structure from `deck`.

    Parameters
    ----------
    deck : dict
        Deck as returned by `read_eclipse_deck`/`read_grid`.
    mapAxes, removeZeroPV, useMex : ignored (kept for API parity)

    Returns
    -------
    dict
        Minimal grid-like dict. Examples:
          - tensor grid: {'type':'tensor','xfaces':..., 'yfaces':..., 'zfaces':..., 'cell_centers':(Xc,Yc,Zc), 'cartDims': [...]}
          - corner point: {'type':'corner_point','ZCORN':..., 'COORD':..., 'cartDims': [...]} 
    """
    if 'GRID' not in deck:
        raise ValueError('Deck has no GRID section')

    g = deck.get('GRID', {})
    rs = deck.get('RUNSPEC', {})
    dims = rs.get('cartDims') or g.get('cartDims')

    # Corner-point / GRDECL.  The coordinate construction below is a direct
    # translation of MRST core/gridprocessing/buildCornerPtNodes.m: ZCORN
    # is a 2*nx by 2*ny by 2*nz array in MATLAB/Fortran order, not a
    # sequence of independent eight-value cells.
    if 'ZCORN' in g and 'COORD' in g:
        if dims is None or len(dims) < 3:
            raise ValueError('Corner-point grid requires cartDims')
        nx, ny, nz = (int(dims[0]), int(dims[1]), int(dims[2]))
        return _corner_point_grid(g['COORD'], g['ZCORN'], nx, ny, nz, g.get('ACTNUM'))

    # Tensor-grid: DXV/DYV/DZV or DX/DY/DZ
    # Prefer DXV/DYV/DZV
    if 'DXV' in g and 'DYV' in g and 'DZV' in g:
        dx = np.asarray(g['DXV'], dtype=float)
        dy = np.asarray(g['DYV'], dtype=float)
        dz = np.asarray(g['DZV'], dtype=float)

        xfaces = np.concatenate(([0.0], np.cumsum(dx)))
        yfaces = np.concatenate(([0.0], np.cumsum(dy)))
        zfaces = np.concatenate(([0.0], np.cumsum(dz)))

        return _make_tensor_grid(dx, dy, dz, g.get('ACTNUM'))

    # Fallback: if DX/DY/DZ present as full arrays in block
    if 'DX' in g and 'DY' in g and 'DZ' in g:
        dx = np.asarray(g['DX'], dtype=float)
        dy = np.asarray(g['DY'], dtype=float)
        dz = np.asarray(g['DZ'], dtype=float)
        if dims is not None and len(dims) >= 3:
            nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
            nc = nx * ny * nz

            def _axis_from_cellwise(arr, axis):
                arr = np.asarray(arr, dtype=float).ravel()
                if arr.size == 1:
                    if axis == 'x':
                        return np.full(nx, float(arr[0]), dtype=float)
                    if axis == 'y':
                        return np.full(ny, float(arr[0]), dtype=float)
                    return np.full(nz, float(arr[0]), dtype=float)
                if axis == 'x' and arr.size == nx:
                    return arr.copy()
                if axis == 'y' and arr.size == ny:
                    return arr.copy()
                if axis == 'z' and arr.size == nz:
                    return arr.copy()
                if arr.size == nc:
                    # ECLIPSE box data is ordered I fastest, then J, then K.
                    # A cell-wise DX/DY/DZ array is usually still separable
                    # per axis (e.g. SPE9's DZ is genuinely a per-layer
                    # thickness, just repeated nx*ny times per the ECLIPSE
                    # "600*20 600*15 ..." box-fill syntax); collapsing it to
                    # one grid-wide average instead of extracting the real
                    # per-axis profile silently gives every cell the same
                    # size.  Only fall back to averaging when the array
                    # truly varies along the other two axes too.
                    arr3 = arr.reshape((nx, ny, nz), order='F')
                    if axis == 'x':
                        if np.allclose(arr3, arr3[:, :1, :1]):
                            return arr3[:, 0, 0].copy()
                        return arr3.mean(axis=(1, 2))
                    if axis == 'y':
                        if np.allclose(arr3, arr3[:1, :, :1]):
                            return arr3[0, :, 0].copy()
                        return arr3.mean(axis=(0, 2))
                    if np.allclose(arr3, arr3[:1, :1, :]):
                        return arr3[0, 0, :].copy()
                    return arr3.mean(axis=(0, 1))
                # Last-resort fallback: average to a constant per axis.
                if axis == 'x':
                    return np.full(nx, float(np.mean(arr)), dtype=float)
                if axis == 'y':
                    return np.full(ny, float(np.mean(arr)), dtype=float)
                return np.full(nz, float(np.mean(arr)), dtype=float)

            dx = _axis_from_cellwise(dx, 'x')
            dy = _axis_from_cellwise(dy, 'y')
            dz = _axis_from_cellwise(dz, 'z')

        z_offset = 0.0
        if 'TOPS' in g:
            tops = np.asarray(g['TOPS'], dtype=float).ravel()
            nxy = nx * ny
            if tops.size and np.all(tops[:min(nxy, tops.size)] == tops[0]):
                z_offset = float(tops[0])
            elif tops.size >= nxy:
                # MRST's initEclipseGrid.m only handles the constant-TOPS
                # special case (a single tensorGrid depthz value); a
                # per-column TOPS surface has no such restriction once
                # expressed as a corner-point grid, since every cell there
                # already owns eight independent corners.  See
                # ``_dx_dy_dz_tops_to_corner_point``.
                coord, zcorn = _dx_dy_dz_tops_to_corner_point(dx, dy, dz, tops[:nxy], nx, ny, nz)
                return _corner_point_grid(coord, zcorn, nx, ny, nz, g.get('ACTNUM'))
        return _make_tensor_grid(dx, dy, dz, g.get('ACTNUM'), z_offset=z_offset)

    raise NotImplementedError('Grid type not supported by init_eclipse_grid yet')
