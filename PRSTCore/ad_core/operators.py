"""TPFA discrete operators used by the AD reservoir models.

The cell ordering and transmissibility reduction follow MRST's
``setupOperatorsTPFA -> getFaceTransmissibility`` path: Cartesian I is the
fastest index, then J, then K, and two one-sided half transmissibilities
are reduced harmonically across each internal interface.
"""

import numpy as _np


def setup_operators(G, rock):
    """Build internal neighbours, transmissibilities and pore volumes.

    ``G`` may be a tensor grid or a straight-pillar corner-point grid made
    by :func:`init_eclipse_grid`.  Inactive Cartesian cells are skipped,
    while the returned N indices remain one-based to match MRST's operator
    convention and the existing model assembly code.
    """
    if not isinstance(G, dict) or G.get('type') not in {'tensor', 'corner_point'}:
        return {}

    nx, ny, nz = (int(value) for value in G['cartDims'])
    nfull = nx * ny * nz
    cells = G.get('cells', {})

    has_topology = (G.get('type') == 'corner_point' and
                    G.get('faces', {}).get('neighbors') is not None)
    if G.get('type') == 'corner_point':
        cart_to_active = _np.asarray(G.get('cart_to_active'), dtype=int).ravel()
        dimensions_active = _np.asarray(cells.get('dimensions', G.get('cell_dimensions')), dtype=float)
    else:
        cart_to_active = _build_cart_to_active(G, nfull)
        dimensions_active = _tensor_dimensions(G, cart_to_active)

    if cart_to_active.size != nfull:
        raise ValueError('cart_to_active must contain one entry per Cartesian cell')
    active_cartesian = _np.flatnonzero(cart_to_active >= 0)
    nactive = active_cartesian.size
    if dimensions_active.shape != (nactive, 3):
        raise ValueError('cell dimensions must have shape (number of active cells, 3)')

    permeability = _permeability_tensor(rock, nactive)
    dims_full = _np.zeros((nfull, 3), dtype=float)
    perm_full = _np.zeros((nfull, 3), dtype=float)
    dims_full[active_cartesian] = dimensions_active
    perm_full[active_cartesian] = permeability

    # MATLAB uses column-major [I, J, K] storage for all deck/grid vectors.
    cell_map = cart_to_active.reshape((nx, ny, nz), order='F')

    if has_topology:
        # General processGRDECL topology (including split fault faces).
        # This is the same one-sided htrans_xyz reduction used by MRST's
        # computeTrans/getFaceTransmissibility sequence.
        N, T, A = _topological_corner_point_interfaces(
            G['faces'], cells['centroids'], permeability,
            None if rock is None else rock.get('ntg'),
            None if rock is None else rock.get('multipliers'),
            cells.get('indexMap'),
            None if rock is None else rock.get('faultdata'),
            G['cartDims'],
        )
    elif G.get('type') == 'corner_point':
        # Direct port of getFaceTransmissibility -> computeTrans/htrans_xyz
        # for the regular (non-faulted) logical connections represented by
        # this lightweight grid.  In particular, no Cartesian-area shortcut
        # is valid for SPE9's tilted pillars.
        corner_vertices = _np.asarray(G.get('corner_vertices'), dtype=float)
        N, T, A = _corner_point_interfaces(
            cell_map, corner_vertices, cells['centroids'], permeability,
        )
    else:
        dims = dims_full.reshape((nx, ny, nz, 3), order='F')
        perm = perm_full.reshape((nx, ny, nz, 3), order='F')

        neighbors = []
        transmissibilities = []
        areas = []
        for axis in range(3):
            N_axis, T_axis, A_axis = _directional_interfaces(cell_map, dims, perm, axis)
            if N_axis.size:
                neighbors.append(N_axis)
                transmissibilities.append(T_axis)
                areas.append(A_axis)

        if neighbors:
            N = _np.vstack(neighbors)
            T = _np.concatenate(transmissibilities)
            A = _np.concatenate(areas)
        else:
            N = _np.empty((0, 2), dtype=int)
            T = _np.empty((0,), dtype=float)
            A = _np.empty((0,), dtype=float)

    volumes = cells.get('volumes', G.get('cell_volumes'))
    if volumes is None:
        volumes = _np.prod(dimensions_active, axis=1)
    pv = _np.asarray(volumes, dtype=float).ravel()
    if pv.size != nactive:
        raise ValueError('cell volume vector does not match active cells')

    # 'oneBased' states the N convention outright rather than leaving the
    # consumer to infer it from the index values (see
    # GenericBlackOilModel._internal_connections): setup_operators_tpfa and
    # the nwm hybrid-grid assembly both emit 0-based N instead.
    return {'N': N, 'T': T, 'areas': A, 'pv': pv, 'oneBased': True}


def _build_cart_to_active(G, nfull):
    if 'cart_to_active' in G:
        return _np.asarray(G['cart_to_active'], dtype=int).ravel()
    actnum = _np.asarray(G.get('ACTNUM', _np.ones(nfull)), dtype=int).ravel().astype(bool)
    if actnum.size != nfull:
        raise ValueError('ACTNUM length does not match cartDims')
    out = _np.full(nfull, -1, dtype=int)
    out[_np.flatnonzero(actnum)] = _np.arange(_np.count_nonzero(actnum), dtype=int)
    return out


def _tensor_dimensions(G, cart_to_active):
    xfaces = _np.asarray(G['xfaces'], dtype=float)
    yfaces = _np.asarray(G['yfaces'], dtype=float)
    zfaces = _np.asarray(G['zfaces'], dtype=float)
    dx, dy, dz = _np.diff(xfaces), _np.diff(yfaces), _np.diff(zfaces)
    nx, ny, nz = len(dx), len(dy), len(dz)
    # I/J/K array then flatten in MRST's natural (Fortran) order.
    dx3 = _np.broadcast_to(dx[:, None, None], (nx, ny, nz)).ravel(order='F')
    dy3 = _np.broadcast_to(dy[None, :, None], (nx, ny, nz)).ravel(order='F')
    dz3 = _np.broadcast_to(dz[None, None, :], (nx, ny, nz)).ravel(order='F')
    full = _np.column_stack((dx3, dy3, dz3))
    return full[_np.flatnonzero(cart_to_active >= 0)]


def _permeability_tensor(rock, nactive):
    raw = None if rock is None else rock.get('perm')
    if raw is None:
        return _np.ones((nactive, 3), dtype=float)
    arr = _np.asarray(raw, dtype=float)
    if arr.ndim == 0:
        return _np.full((nactive, 3), float(arr), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape((-1, 1))
    if arr.shape[0] != nactive:
        if arr.shape[0] == 1:
            arr = _np.repeat(arr, nactive, axis=0)
        else:
            raise ValueError('rock permeability vector does not match active cells')
    if arr.shape[1] == 1:
        arr = _np.repeat(arr, 3, axis=1)
    elif arr.shape[1] == 2:
        arr = _np.column_stack((arr, arr[:, 1]))
    return arr[:, :3]


def _directional_interfaces(cell_map, dimensions, permeability, axis):
    """Construct one direction of interfaces and MRST-style TPFA T."""
    if axis == 0:
        left, right = cell_map[:-1, :, :], cell_map[1:, :, :]
        dl, dr = dimensions[:-1, :, :, 0], dimensions[1:, :, :, 0]
        # Use the arithmetic average of the two physical face areas.
        area_l = dimensions[:-1, :, :, 1] * dimensions[:-1, :, :, 2]
        area_r = dimensions[1:, :, :, 1] * dimensions[1:, :, :, 2]
        kl, kr = permeability[:-1, :, :, 0], permeability[1:, :, :, 0]
    elif axis == 1:
        left, right = cell_map[:, :-1, :], cell_map[:, 1:, :]
        dl, dr = dimensions[:, :-1, :, 1], dimensions[:, 1:, :, 1]
        area_l = dimensions[:, :-1, :, 0] * dimensions[:, :-1, :, 2]
        area_r = dimensions[:, 1:, :, 0] * dimensions[:, 1:, :, 2]
        kl, kr = permeability[:, :-1, :, 1], permeability[:, 1:, :, 1]
    else:
        left, right = cell_map[:, :, :-1], cell_map[:, :, 1:]
        dl, dr = dimensions[:, :, :-1, 2], dimensions[:, :, 1:, 2]
        area_l = dimensions[:, :, :-1, 0] * dimensions[:, :, :-1, 1]
        area_r = dimensions[:, :, 1:, 0] * dimensions[:, :, 1:, 1]
        kl, kr = permeability[:, :, :-1, 2], permeability[:, :, 1:, 2]

    active = (left >= 0) & (right >= 0)
    if not _np.any(active):
        return (_np.empty((0, 2), dtype=int), _np.empty((0,), dtype=float), _np.empty((0,), dtype=float))

    a = 0.5 * (area_l[active] + area_r[active])
    # computeTrans produces one-sided T=k*A/(distance-to-face), then
    # getFaceTransmissibility applies 1/(1/T1 + 1/T2).  This is not the
    # same as applying a harmonic k to an averaged cell distance when the
    # two cells have different widths/permeabilities.
    t_left = kl[active] * a / _np.maximum(0.5 * dl[active], 1.0e-30)
    t_right = kr[active] * a / _np.maximum(0.5 * dr[active], 1.0e-30)
    inv_t = _np.zeros_like(t_left)
    good = (t_left > 0.0) & (t_right > 0.0)
    inv_t[good] = 1.0 / t_left[good] + 1.0 / t_right[good]
    t = _np.zeros_like(t_left)
    t[good] = 1.0 / inv_t[good]

    N = _np.column_stack((left[active] + 1, right[active] + 1)).astype(int, copy=False)
    return N, t, a


def _corner_point_interfaces(cell_map, corner_vertices, cell_centers, permeability):
    """Port logical CP-face TPFA connections from MRST's ``computeTrans``.

    MRST constructs face normals through ``computeGeometry.m`` and then
    evaluates ``C' * K * N / (C' * C)`` for every cell face in
    ``computeTrans.m``.  The grid supported here has six nominal faces per
    logical cell, so their face-node order is the one used by
    ``computeCpGeometry.m``.  The connection iteration order reproduces
    ``processGRDECL``'s internal-face ordering: K, then I, then J.
    """
    corner_vertices = _np.asarray(corner_vertices, dtype=float)
    cell_centers = _np.asarray(cell_centers, dtype=float)
    permeability = _np.asarray(permeability, dtype=float)
    nactive = permeability.shape[0]
    if corner_vertices.shape != (nactive, 8, 3):
        raise ValueError('corner_vertices must have shape (number of active cells, 8, 3)')
    if cell_centers.shape != (nactive, 3):
        raise ValueError('corner-point cell centers must have shape (number of active cells, 3)')

    # Node order follows processGRDECL's preserved cpnodes, and these are
    # the six nominal faces from computeCpGeometry.m.
    face_nodes = (
        (0, 2, 6, 4), (1, 5, 7, 3),
        (0, 4, 5, 1), (2, 3, 7, 6),
        (0, 1, 3, 2), (4, 6, 7, 5),
    )
    # For an I/J/K interface: left cell's max face, right cell's min face.
    face_pairs = ((1, 0), (3, 2), (5, 4))

    neighbors = []
    transmissibilities = []
    areas = []
    for axis, (left_face, right_face) in enumerate(face_pairs):
        if axis == 0:
            left, right = cell_map[:-1, :, :], cell_map[1:, :, :]
        elif axis == 1:
            left, right = cell_map[:, :-1, :], cell_map[:, 1:, :]
        else:
            left, right = cell_map[:, :, :-1], cell_map[:, :, 1:]

        # ``processGRDECL`` traverses these connections with K fastest,
        # then I, then J.  This also reproduces model.operators.N exactly.
        left = _np.transpose(left, (2, 0, 1)).ravel(order='F')
        right = _np.transpose(right, (2, 0, 1)).ravel(order='F')
        active = (left >= 0) & (right >= 0)
        left, right = left[active], right[active]
        if left.size == 0:
            continue

        hleft, area = _corner_point_half_trans(
            corner_vertices[left], cell_centers[left], permeability[left],
            face_nodes[left_face],
        )
        hright, _ = _corner_point_half_trans(
            corner_vertices[right], cell_centers[right], permeability[right],
            face_nodes[right_face],
        )
        # getFaceTransmissibility.m: 1 ./ accumarray(face, 1 ./ hT)
        t = 1.0 / (1.0 / hleft + 1.0 / hright)
        neighbors.append(_np.column_stack((left + 1, right + 1)).astype(int, copy=False))
        transmissibilities.append(t)
        areas.append(area)

    if not neighbors:
        return (_np.empty((0, 2), dtype=int), _np.empty((0,), dtype=float),
                _np.empty((0,), dtype=float))
    return _np.vstack(neighbors), _np.concatenate(transmissibilities), _np.concatenate(areas)


def _corner_point_half_trans(vertices, cell_centers, permeability, node_ids):
    """MRST ``computeGeometry`` face primitive plus ``htrans_xyz``."""
    points = vertices[:, node_ids, :]
    # computeGeometry.m / face_geom3d: face centre is the area-weighted
    # centroid of the fan triangles.  For the nominal SPE9 quad faces the
    # fan has equal-area triangles, and coincides with this arithmetic
    # centre (also the direct CP-face definition used by MRST).
    face_center = _np.mean(points, axis=1)
    normal = _np.zeros_like(face_center)
    for index in range(4):
        normal += _np.cross(
            points[:, (index + 1) % 4, :] - points[:, index, :],
            face_center - points[:, index, :],
        ) / 2.0

    c = face_center - cell_centers
    outward = _np.sum(normal * c, axis=1) < 0.0
    normal[outward] *= -1.0
    # computeTrans.m htrans_xyz: C'*K*N/(C'*C), followed by fixNegative.
    htrans = _np.sum(c * permeability * normal, axis=1) / _np.sum(c * c, axis=1)
    htrans = _np.abs(htrans)
    return htrans, _np.linalg.norm(normal, axis=1)


def _topological_corner_point_interfaces(faces, cell_centers, permeability,
                                         ntg=None, multipliers=None, index_map=None,
                                         faultdata=None, cart_dims=None):
    """TPFA connections for a fault-aware ``processGRDECL`` topology."""
    neighbors = _np.asarray(faces['neighbors'], dtype=int)
    centroids = _np.asarray(faces['centroids'], dtype=float)
    normals = _np.asarray(faces['normals'], dtype=float)
    areas = _np.asarray(faces['areas'], dtype=float)
    internal = (neighbors[:, 0] >= 0) & (neighbors[:, 1] >= 0)
    if not _np.any(internal):
        return (_np.empty((0, 2), dtype=int), _np.empty((0,), dtype=float),
                _np.empty((0,), dtype=float))
    a, b = neighbors[internal, 0], neighbors[internal, 1]
    normal = normals[internal]
    ca = centroids[internal] - cell_centers[a]
    cb = centroids[internal] - cell_centers[b]
    ha = _np.sum(ca*permeability[a]*normal, axis=1)/_np.sum(ca*ca, axis=1)
    hb = _np.sum(cb*permeability[b]*(-normal), axis=1)/_np.sum(cb*cb, axis=1)
    # MRST's computeTrans calls fixNegative() before the harmonic reduction.
    ha, hb = _np.abs(ha), _np.abs(hb)
    # computeTrans applies NTG to each half transmissibility on I/J faces
    # (but never to K faces) before setupOperatorsTPFA performs the
    # harmonic reduction.  Norne contains low-NTG layers where this is a
    # two-order-of-magnitude effect.
    if ntg is not None:
        values = _np.asarray(ntg, dtype=float).ravel()
        if values.size == 1:
            values = _np.full(cell_centers.shape[0], values[0], dtype=float)
        if values.size == cell_centers.shape[0]:
            tag = _np.asarray(faces.get('tag', _np.full(neighbors.shape[0], 3)), dtype=int)[internal]
            lateral = tag != 3
            ha[lateral] *= values[a[lateral]]
            hb[lateral] *= values[b[lateral]]
    face_mult = _np.ones(ha.size, dtype=float)
    if multipliers:
        tag = _np.asarray(faces.get('tag', _np.full(neighbors.shape[0], 3)), dtype=int)[internal]
        active_map = (_np.arange(cell_centers.shape[0], dtype=int) if index_map is None
                      else _np.asarray(index_map, dtype=int).ravel())
        # computeTranMult maps Eclipse's positive direction multiplier to
        # the first (outgoing) half-face and the trailing-hyphen variant to
        # the second (incoming) half-face.  InitEclipseRock spells the
        # latter with an underscore.
        spec = {
            'x': (1, True), 'x_': (1, False),
            'y': (2, True), 'y_': (2, False),
            'z': (3, True), 'z_': (3, False),
        }
        for name, raw in multipliers.items():
            item = spec.get(str(name).lower())
            if item is None:
                continue
            direction, first = item
            values = _np.asarray(raw, dtype=float).ravel()
            if values.size == 1:
                values = _np.full(cell_centers.shape[0], values[0], dtype=float)
            elif values.size != cell_centers.shape[0]:
                if values.size > int(active_map.max(initial=-1)):
                    values = values[active_map]
                else:
                    values = _np.pad(values, (0, max(0, cell_centers.shape[0]-values.size)),
                                     constant_values=1.0)[:cell_centers.shape[0]]
            chosen = tag == direction
            if first:
                face_mult[chosen] *= values[a[chosen]]
            else:
                face_mult[chosen] *= values[b[chosen]]
    good = (ha > 0.0) & (hb > 0.0)
    t = _np.zeros_like(ha)
    t[good] = 1.0/(1.0/ha[good] + 1.0/hb[good])
    # Rock MULTX/Y/Z fields are accumulated by MRST into a face multiplier
    # and applied after harmonic reduction (unlike NTG, which is a genuine
    # half-face scaling in computeTrans).
    t *= face_mult
    if faultdata and cart_dims is not None:
        # ``getFaceTransmissibility`` obtains MULTFLT through
        # ``processFaults`` and applies it to the *face* transmissibility,
        # after the two one-sided values have been reduced harmonically.
        # This is intentionally distinct from MULTX/Y/Z above.
        t *= _fault_face_multipliers(
            faces, internal, faultdata, cart_dims, index_map,
            cell_centers.shape[0],
        )
    return _np.column_stack((a+1, b+1)), t, areas[internal]


def _fault_face_multipliers(faces, internal, faultdata, cart_dims, index_map,
                            nactive):
    """Port MRST ``processFaults``/``expand_fault_multipliers``.

    ECLIPSE FAULTS records select a Cartesian cell box and one cardinal
    cell face.  ``processGRDECL`` stores the positive I/J/K side as the
    first face neighbour, hence X/Y/Z select column one and X-/Y-/Z-
    select column two.  Multiple records for a named fault form a union;
    multipliers from distinct overlapping faults are multiplied.
    """
    records = faultdata.get('faults', []) if isinstance(faultdata, dict) else []
    mult_records = faultdata.get('multflt', []) if isinstance(faultdata, dict) else []
    if not records or not mult_records:
        return _np.ones(int(_np.count_nonzero(internal)), dtype=float)

    # MATLAB's processFaults only retains named faults with finite MULTFLT.
    mult = {}
    for row in mult_records:
        if len(row) >= 2:
            try:
                value = float(row[1])
            except (TypeError, ValueError):
                continue
            if _np.isfinite(value):
                mult[str(row[0]).lower()] = value
    if not mult:
        return _np.ones(int(_np.count_nonzero(internal)), dtype=float)

    nx, ny, nz = (int(v) for v in cart_dims)
    nfull = nx*ny*nz
    active_map = (_np.arange(nactive, dtype=int) if index_map is None
                  else _np.asarray(index_map, dtype=int).ravel())
    full_to_active = _np.full(nfull, -1, dtype=int)
    usable = (active_map >= 0) & (active_map < nfull)
    full_to_active[active_map[usable]] = _np.flatnonzero(usable)

    neighbors = _np.asarray(faces['neighbors'], dtype=int)
    tags = _np.asarray(faces.get('tag', _np.zeros(neighbors.shape[0])), dtype=int)
    result = _np.ones(neighbors.shape[0], dtype=float)
    grouped = {}
    for row in records:
        if len(row) >= 8:
            grouped.setdefault(str(row[0]).lower(), []).append(row)

    direction = {
        'x': (1, 0), 'i': (1, 0), 'x+': (1, 0), 'i+': (1, 0),
        'x-': (1, 1), 'i-': (1, 1),
        'y': (2, 0), 'j': (2, 0), 'y+': (2, 0), 'j+': (2, 0),
        'y-': (2, 1), 'j-': (2, 1),
        'z': (3, 0), 'k': (3, 0), 'z+': (3, 0), 'k+': (3, 0),
        'z-': (3, 1), 'k-': (3, 1),
    }
    for name, rows in grouped.items():
        value = mult.get(name)
        if value is None:
            continue
        on_fault = _np.zeros(neighbors.shape[0], dtype=bool)
        for row in rows:
            try:
                i1, i2, j1, j2, k1, k2 = (int(float(v)) for v in row[1:7])
            except (TypeError, ValueError):
                continue
            side = direction.get(str(row[7]).strip().lower())
            if side is None:
                continue
            axis, column = side
            # Equivalent to fault_cells + act(c) in processFaults.m.
            ii = _np.arange(max(i1, 1), min(i2, nx) + 1, dtype=int)
            jj = _np.arange(max(j1, 1), min(j2, ny) + 1, dtype=int)
            kk = _np.arange(max(k1, 1), min(k2, nz) + 1, dtype=int)
            if not (ii.size and jj.size and kk.size):
                continue
            full = ((ii[:, None, None] - 1)
                    + (jj[None, :, None] - 1)*nx
                    + (kk[None, None, :] - 1)*nx*ny).ravel()
            cells = full_to_active[full]
            cells = cells[cells >= 0]
            if cells.size:
                on_fault |= ((tags == axis)
                             & _np.isin(neighbors[:, column], cells,
                                        assume_unique=False))
        result[on_fault] *= value
    return result[internal]
