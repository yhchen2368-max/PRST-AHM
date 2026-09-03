"""Port of MRST ``computeWellIndex.m`` (core/utils).

The Peaceman well index. A well's rate is proportional to it, so an
error here scales every rate the well ever produces -- and it will not
announce itself, because the simulation still converges, to a well that
flows the wrong amount.

    WI = 2*pi*Kh / (ln(re/rw) + skin)

with the equivalent radius accounting for the block's shape *and* its
permeability anisotropy:

    re = 2*wc * sqrt(d1^2*sqrt(k2/k1) + d2^2*sqrt(k1/k2))
              / ( (k2/k1)^(1/4) + (k1/k2)^(1/4) )

For an isotropic square block this collapses to the familiar
``re = 0.28*dx``; the general form is what makes it right for the long
thin blocks a real deck is full of.

``d1``, ``d2`` are the block's extents *across* the well and ``ell`` is
the extent *along* it, so which of dx/dy/dz plays which role depends on
the perforation direction. Net-to-gross scales the perforated length.

**What this replaces.** PRSTCore computed the well index from the cube
root of the cell volume, taking ``h = vol**(1/3)`` and ``re = 0.2*h``
with a comment calling it a crude estimate. For SPE1's 1000x1000x20 ft
blocks that height is out by a factor of five, and the two errors
partly cancel to leave the index 6% off MRST's -- close enough to look
plausible and wrong enough to matter.
"""

import numpy as _np

#: Peaceman constants against the block aspect ratio, for the mixed
#: finite-element inner products. The two-point product uses 0.14 flat.
_WELL_CONSTANT_TABLE = _np.array([
    [1, 0.292], [2, 0.278], [3, 0.262], [4, 0.252], [5, 0.244],
    [8, 0.231], [9, 0.229], [16, 0.220], [17, 0.219], [32, 0.213],
    [33, 0.213], [64, 0.210], [65, 0.210],
])

_TPF = ('ip_tpf', 'ip_quasitpf')
_RT = ('ip_rt', 'ip_simple', 'ip_quasirt')


def compute_well_index(G, rock, radius, cells, Dir='z', Skin=None, Kh=None,
                       InnerProduct='ip_tpf', cellDims=None, Subset=None):
    """Return the well index for each perforated cell.

    Parameters
    ----------
    G, rock
        Grid and rock; ``rock['perm']`` may be isotropic, diagonal or
        full-tensor, and ``rock['ntg']`` scales the perforated length.
    radius : float or array
        Well bore radius per perforation.
    cells : array
        Perforated cells, 0-based.
    Dir : str
        Perforation direction, ``'x'``, ``'y'`` or ``'z'``; one letter
        for all or one per perforation.
    Skin : array, optional
    Kh : array, optional
        Permeability-thickness. A negative entry means "compute it".
    cellDims : tuple of arrays, optional
        ``(dx, dy, dz)`` per perforation, when the caller already knows
        them. Otherwise taken from the grid.
    """
    cells = _np.atleast_1d(_np.asarray(cells, dtype=int)).ravel()
    nc = cells.size
    skin = _np.zeros(nc) if Skin is None else _broadcast(Skin, nc)
    kh = _np.full(nc, -1.0) if Kh is None else _broadcast(Kh, nc)
    radius = _broadcast(radius, nc)

    d1, d2, ell, k1, k2 = _connection_dimensions(G, rock, cells, Dir,
                                                 cellDims)

    with _np.errstate(divide='ignore', invalid='ignore'):
        k21 = _np.where(k1 != 0, k2 / k1, 0.0)
        k12 = _np.where(k2 != 0, k1 / k2, 0.0)
    k21[~_np.isfinite(k21)] = 0.0
    k12[~_np.isfinite(k12)] = 0.0

    wc = _well_constant(d1, d2, InnerProduct)
    numerator = 2 * wc * _np.sqrt(d1 ** 2 * _np.sqrt(k21)
                                  + d2 ** 2 * _np.sqrt(k12))
    denominator = _nthroot(k21, 4) + _nthroot(k12, 4)
    with _np.errstate(divide='ignore', invalid='ignore'):
        re = numerator / denominator
    re[~_np.isfinite(re)] = 0.0

    ke = _np.sqrt(k1 * k2)
    missing = kh < 0
    griddim = int(G.get('griddim', 3)) if isinstance(G, dict) else 3
    kh = kh.copy()
    kh[missing] = (ell[missing] * ke[missing]) if griddim > 2 else ke[missing]

    with _np.errstate(divide='ignore', invalid='ignore'):
        wi = 2 * _np.pi * kh / (_np.log(re / radius) + skin)

    _check_peaceman(wi, re, radius)
    return wi if Subset is None else wi[_np.asarray(Subset, dtype=int)]


def _connection_dimensions(G, rock, cells, direction, cellDims):
    """Port of ``connection_dimensions``.

    Which extent is 'across' and which is 'along' depends on the
    perforation direction: a vertical well sees dx and dy across it and
    dz along it, and a well drilled in x sees dy and dz across.
    """
    dx, dy, dz = _geometric_dimensions(G, cells, cellDims)
    k = _extract_permeability(rock, cells)

    ntg = _np.ones(cells.size)
    if isinstance(rock, dict) and rock.get('ntg') is not None:
        values = _np.atleast_1d(_np.asarray(rock['ntg'],
                                            dtype=float)).ravel()
        if values.size > cells.max():
            ntg = values[cells]

    direction = _np.asarray([str(d).lower() for d in
                             (list(direction) if len(str(direction)) > 1
                              else [str(direction)] * cells.size)])
    if direction.size == 1:
        direction = _np.repeat(direction, cells.size)

    d1 = _np.zeros(cells.size)
    d2 = _np.zeros(cells.size)
    ell = _np.zeros(cells.size)
    k1 = _np.zeros(cells.size)
    k2 = _np.zeros(cells.size)

    for axis, (a, b, along, ka, kb) in {
            'x': (dy, dz, dx, k[:, 1], k[:, 2]),
            'y': (dx, dz, dy, k[:, 0], k[:, 2]),
            'z': (dx, dy, dz, k[:, 0], k[:, 1])}.items():
        ci = direction == axis
        if not _np.any(ci):
            continue
        d1[ci], d2[ci], ell[ci] = a[ci], b[ci], along[ci]
        k1[ci], k2[ci] = ka[ci], kb[ci]
        if axis == 'z':
            # Net-to-gross reduces the perforated length, and MRST
            # applies it on the vertical branch only.
            ell[ci] = ntg[ci] * ell[ci]

    return d1, d2, ell, k1, k2


def _geometric_dimensions(G, cells, cellDims):
    """Cell extents, from the caller or from the grid's bounding boxes."""
    if cellDims is not None:
        dx, dy, dz = (_np.atleast_1d(_np.asarray(v, dtype=float)).ravel()
                      for v in cellDims)
        if dx.size > cells.size:
            dx, dy, dz = dx[cells], dy[cells], dz[cells]
        return dx, dy, dz

    dims = _cell_dims(G, cells)
    if dims is not None:
        return dims
    raise ValueError('Cannot determine cell dimensions for the well index; '
                     'pass cellDims explicitly')


def _cell_dims(G, cells):
    """Per-cell extents from a Cartesian grid, or from node bounding
    boxes when one is available."""
    if not isinstance(G, dict):
        return None

    cart = G.get('cartDims')
    nodes = G.get('nodes')
    faces = G.get('faces')

    if nodes is not None and faces is not None and \
            'nodePos' in (faces or {}):
        boxes = _bounding_boxes(G, cells)
        if boxes is not None:
            return boxes

    if cart is not None and 'cells' in G:
        volumes = _np.atleast_1d(_np.asarray(
            G['cells'].get('volumes', []), dtype=float)).ravel()
        if volumes.size:
            # A last resort, and a poor one: it assumes a cube. Kept
            # only so a grid with no geometry still returns something,
            # and it is exactly what this port exists to stop relying on.
            h = volumes[cells] ** (1.0 / 3.0)
            return h, h.copy(), h.copy()
    return None


def _bounding_boxes(G, cells):
    """The axis-aligned extent of each cell, from its nodes."""
    try:
        coords = _np.asarray(G['nodes']['coords'], dtype=float)
        face_nodes = _np.asarray(G['faces']['nodes'], dtype=int).ravel()
        node_pos = _np.asarray(G['faces']['nodePos'], dtype=int).ravel()
        cell_faces = _np.asarray(G['cells']['faces'], dtype=int)
        if cell_faces.ndim > 1:
            cell_faces = cell_faces[:, 0]
        face_pos = _np.asarray(G['cells']['facePos'], dtype=int).ravel()
    except (KeyError, TypeError):
        return None

    out = _np.zeros((cells.size, 3))
    for i, cell in enumerate(cells):
        faces = cell_faces[face_pos[cell]:face_pos[cell + 1]]
        nodes = _np.concatenate([face_nodes[node_pos[f]:node_pos[f + 1]]
                                 for f in faces]) if faces.size else None
        if nodes is None or nodes.size == 0:
            return None
        pts = coords[nodes, :]
        out[i, :] = pts.max(axis=0) - pts.min(axis=0)
    return out[:, 0], out[:, 1], out[:, 2]


def _extract_permeability(rock, cells):
    """Port of ``extract_permeability``: the diagonal, whatever the
    storage. One column is isotropic, three is diagonal, six is a full
    tensor whose diagonal entries are 0, 3 and 5."""
    perm = _np.atleast_2d(_np.asarray(rock['perm'], dtype=float))
    if perm.shape[0] == 1 and perm.shape[1] > 3:
        perm = perm.T
    values = perm[cells, :]
    ncol = values.shape[1]
    if ncol == 1:
        return _np.repeat(values, 3, axis=1)
    if ncol == 2:
        return _np.column_stack([values[:, 0], values[:, 1], values[:, 1]])
    if ncol == 3:
        return values
    if ncol == 6:
        return values[:, [0, 3, 5]]
    if ncol == 9:
        return values[:, [0, 4, 8]]
    raise ValueError('Unexpected permeability with %d columns' % ncol)


def _well_constant(d1, d2, inner_product):
    """Port of ``wellConstant``.

    The two-point product uses a flat 0.14 -- and 2*0.14 = 0.28 is where
    the familiar Peaceman constant comes from. The mixed products
    interpolate a table on the block's aspect ratio instead.
    """
    if inner_product in _TPF:
        return _np.full(_np.size(d1), 0.14)
    if inner_product in _RT:
        with _np.errstate(divide='ignore', invalid='ignore'):
            ratio = _np.maximum(_np.round(d1 / d2), _np.round(d2 / d1))
        ratio[~_np.isfinite(ratio)] = 1.0
        return _np.interp(ratio, _WELL_CONSTANT_TABLE[:, 0],
                          _WELL_CONSTANT_TABLE[:, 1])
    raise ValueError("Unknown inner product %r" % inner_product)


def _check_peaceman(wi, re, radius):
    """Port of ``check_peaceman_wi``: warn where the model is misapplied.

    Peaceman's derivation assumes the well bore is much smaller than the
    block. A radius approaching the equivalent radius makes the
    logarithm small and the index enormous, and past it the sign flips
    -- a well that produces backwards.
    """
    import warnings

    bad = _np.asarray(re) <= _np.asarray(radius)
    if _np.any(bad):
        warnings.warn(
            'Peaceman well index is not valid where the bore radius '
            'reaches the equivalent radius (%d perforation(s)): the '
            'index is negative or unbounded there.'
            % int(_np.count_nonzero(bad)), RuntimeWarning)


def _broadcast(value, n):
    values = _np.atleast_1d(_np.asarray(value, dtype=float)).ravel()
    return _np.full(n, float(values[0])) if values.size == 1 else values


def _nthroot(x, n):
    x = _np.asarray(x, dtype=float)
    return _np.sign(x) * _np.abs(x) ** (1.0 / n)
