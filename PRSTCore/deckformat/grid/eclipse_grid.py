"""Port of MRST ``initEclipseGrid.m`` (model-io/deckformat/grid).

Builds a **complete** MRST grid from a deck's GRID section, by dispatching
to the same constructors MRST does:

======================  =========================================
``COORD`` + ``ZCORN``   :func:`process_grdecl`
``DXV``/``DYV``/``DZV`` :func:`tensor_grid`, then ``extract_subgrid``
``DX``/``DY``/``DZ``    run-length encoded to the above
======================  =========================================

The distinction that matters is *complete*. The older
:mod:`init_eclipse_grid` describes itself as "a pragmatic, lightweight
counterpart" and produces a different structure per branch -- a
Cartesian deck came back with ``xfaces``/``yfaces``/``zfaces`` and no
``faces`` at all, a corner-point deck with ``faces`` but no
``cells.faces``. Neither carries the cell-to-face topology
(``cells.faces``, ``cells.facePos``, ``griddim``) that every MRST
geometry routine indexes through, so ``computeTrans``,
``computeWellIndex`` and the permeability parameter's
``perm2directionalTrans`` could not run on a deck-derived grid at all --
they were ported faithfully and had nothing to be called on.

The pieces used here are themselves ports, and ``process_grdecl`` is
checked against MRST on SPE9's corner-point grid and Norne's faulted one
(``tests/test_process_grdecl_mrst_parity.py``).
"""

import numpy as _np

from PRSTCore.gridprocessing.compute_geometry import compute_geometry
from PRSTCore.gridprocessing.extract_subgrid import extract_subgrid
from PRSTCore.gridprocessing.process_grdecl import process_grdecl
from PRSTCore.gridprocessing.tensor_grid import tensor_grid


def init_eclipse_grid(deck, mapAxes=False, removeZeroPV=False, useMex=False,
                      computeGeometry=True, minPoreVolume=None, **kwargs):
    """Construct an MRST grid from ``deck``.

    ``computeGeometry`` is not MRST's -- there the caller writes
    ``computeGeometry(initEclipseGrid(deck))`` and always does. It is
    exposed here so a caller that only wants the topology can say so.

    ``removeZeroPV`` drops cells that can hold nothing, as MRST's option of
    the same name does.  ``minPoreVolume`` raises that bar to a volume: it
    is ECLIPSE's MINPV, which the deck may also state itself (MINPV, or
    MINPVV per cell).  MRST reads both keywords and then never applies
    them, so a deck that relies on MINPV to shed its dead cells arrives
    with them still in place.
    """
    grid = deck.get('GRID') or {}
    runspec = deck.get('RUNSPEC') or {}
    dims = runspec.get('cartDims') or grid.get('cartDims')

    if 'COORD' in grid and 'ZCORN' in grid and dims is not None:
        G = process_grdecl(dict(grid, cartDims=_np.asarray(dims, dtype=int)))
    elif _is_delta_grid(grid):
        G = _tensor_from_deltas(grid, dims)
    else:
        raise ValueError('Grid not implemented: GRID has neither '
                         'COORD/ZCORN nor DX/DY/DZ')

    actnum = grid.get('ACTNUM')
    if actnum is not None and not ('COORD' in grid and 'ZCORN' in grid):
        # processGRDECL applies ACTNUM itself; the tensor branch does not.
        G = extract_subgrid(G, _np.asarray(actnum).ravel() > 0)

    # Geometry first, because a pore-volume threshold needs cell volumes
    # whenever the deck does not state PORV outright.  ``remove_cells``
    # carries volumes and centroids through, so filtering afterwards costs
    # nothing but the cells that go.
    if computeGeometry:
        G = compute_geometry(G)

    threshold = _pore_volume_threshold(grid, minPoreVolume)
    if removeZeroPV or threshold is not None:
        G = _remove_small_pore_volume(G, grid, dims, threshold)

    return G


def _is_delta_grid(grid):
    """``is_delta_grid``: either the vector or the full-array form."""
    return (all(k in grid for k in ('DX', 'DY', 'DZ'))
            or all(k in grid for k in ('DXV', 'DYV', 'DZV')))


def _tensor_from_deltas(grid, dims):
    """The tensor branch, including ``getDeltas``' run-length check.

    A block-centred deck states one thickness per cell; MRST insists the
    result really is a tensor product -- that DX repeats identically
    down J and K, and so on -- rather than quietly taking the first row.
    """
    nx, ny, nz = (int(dims[0]), int(dims[1]), int(dims[2]))
    dxv = _deltas(grid, 'X', (nx, ny, nz), (1, 2))
    dyv = _deltas(grid, 'Y', (nx, ny, nz), (0, 2))
    dzv = _deltas(grid, 'Z', (nx, ny, nz), (0, 1))

    depthz = _depthz(grid, nx, ny)
    return tensor_grid(_np.cumsum(_np.r_[0.0, dxv]),
                       _np.cumsum(_np.r_[0.0, dyv]),
                       _np.cumsum(_np.r_[0.0, dzv]),
                       depthz=depthz)


def _deltas(grid, axis, shape, constant_axes):
    """One spacing vector along ``axis``.

    ``DXV`` gives it directly. ``DX`` gives one value per cell, which is
    only a tensor grid if it is constant along the other two axes --
    checked rather than assumed, as ``getDeltas`` does.
    """
    vector = grid.get('D%sV' % axis)
    if vector is not None:
        return _np.asarray(vector, dtype=float).ravel()

    nx, ny, nz = shape
    full = _np.asarray(grid['D%s' % axis], dtype=float).ravel()
    if full.size != nx * ny * nz:
        raise ValueError('D%s has %d values, expected %d'
                         % (axis, full.size, nx * ny * nz))
    block = full.reshape((nz, ny, nx))          # ECLIPSE order: X fastest

    for other in constant_axes:
        # 0 -> X, 1 -> Y, 2 -> Z, and the block is indexed (Z, Y, X).
        along = {0: 2, 1: 1, 2: 0}[other]
        first = _np.take(block, 0, axis=along)
        if not _np.allclose(block, _np.expand_dims(first, along)):
            raise ValueError('Only tensor-grid supported: D%s varies along '
                             'the %s axis' % (axis, 'XYZ'[other]))

    keep = {0: 2, 1: 1, 2: 0}['XYZ'.index(axis)]
    return _np.moveaxis(block, keep, 0).reshape(block.shape[keep], -1)[:, 0]


def _depthz(grid, nx, ny):
    """Node depths for the top surface.

    MRST's ``tensorGrid`` accepts one ``depthz`` per top-surface node (a
    ``(nx+1) x (ny+1)`` array), so a varying top surface is expressible.
    Its ``initEclipseGrid`` only builds ``depthz`` from a *constant* TOPS
    and errors otherwise; this port goes one step further: a varying TOPS
    (one top depth per column) is interpolated to the nodes by averaging
    the adjacent column tops, which is exactly the geometry ``tensorGrid``
    describes.  A deck that states ``DEPTHZ`` directly is used as-is.
    """
    if 'DEPTHZ' in grid:
        return _np.asarray(grid['DEPTHZ'], dtype=float).ravel()
    tops = grid.get('TOPS')
    if tops is None:
        return _np.zeros((nx + 1) * (ny + 1))
    tops = _np.asarray(tops, dtype=float).ravel()[:nx * ny]
    if tops.size and _np.allclose(tops, tops[0]):
        return _np.full((nx + 1) * (ny + 1), float(tops[0]))
    # Varying TOPS: build one depth per top node by averaging the tops of
    # the columns around it.  tops is in ECLIPSE order (X fastest), so
    # ``tops.reshape((ny, nx))[j, i]`` is column (i, j).
    tops_ij = tops.reshape((ny, nx)).T          # [i, j]
    depthz = _np.zeros((nx + 1, ny + 1))
    count = _np.zeros((nx + 1, ny + 1))
    # Node (I, J) is shared by columns i in {I-1, I} and j in {J-1, J}.
    depthz[0:nx, 0:ny] += tops_ij
    depthz[1:nx + 1, 0:ny] += tops_ij
    depthz[0:nx, 1:ny + 1] += tops_ij
    depthz[1:nx + 1, 1:ny + 1] += tops_ij
    count[0:nx, 0:ny] += 1
    count[1:nx + 1, 0:ny] += 1
    count[0:nx, 1:ny + 1] += 1
    count[1:nx + 1, 1:ny + 1] += 1
    depthz = depthz / _np.maximum(count, 1)
    return depthz.ravel(order='F')


def _pore_volume_threshold(grid, override=None):
    """The MINPV volume a cell must reach to stay active, or ``None``.

    ``override`` wins over the deck, so a deck that states no MINPV -- and
    then relies on the simulator's own default to shed cells it marked
    active but gave no pore volume -- can still be run.  A per-cell MINPVV
    is returned as an array.
    """
    if override is not None:
        return float(override)
    per_cell = grid.get('MINPVV')
    if per_cell is not None:
        return _np.asarray(per_cell, dtype=float).ravel()
    scalar = grid.get('MINPV')
    if scalar is not None:
        return float(_np.asarray(scalar, dtype=float).ravel()[0])
    return None


def _cell_pore_volumes(G, grid, dims):
    """Pore volume of every cell the grid still has, or ``None``.

    ``PORV`` states it outright.  Otherwise it is the geometric volume
    times porosity, net-to-gross and any pore-volume multiplier -- which
    needs ``compute_geometry`` to have run.
    """
    total = int(_np.prod(_np.asarray(dims, dtype=int)))
    index = _np.asarray(G['cells'].get('indexMap',
                                       _np.arange(G['cells']['num'])),
                        dtype=int).ravel()

    def full_box(key):
        values = grid.get(key)
        if values is None:
            return None
        values = _np.asarray(values, dtype=float).ravel()
        return values[index] if values.size == total else None

    porv = full_box('PORV')
    if porv is not None:
        return porv

    volumes = G.get('cells', {}).get('volumes')
    if volumes is None:
        return None
    pv = _np.asarray(volumes, dtype=float).ravel()
    for key in ('PORO', 'NTG', 'MULTPV'):
        factor = full_box(key)
        if factor is not None:
            pv = pv * factor
    return pv


def _remove_small_pore_volume(G, grid, dims, threshold=None):
    """Drop cells that cannot hold enough fluid to carry an equation.

    A cell with no pore volume contributes a conservation equation that is
    identically zero, so the Jacobian comes out structurally singular and
    the linear solver stops before its first iteration -- on T142, 242115
    of 433104 ACTNUM-active cells, and 462541 of 866211 empty rows.

    Falls back to the old test on the individual factors when the pore
    volume itself cannot be formed (no PORV and no geometry).
    """
    pv = _cell_pore_volumes(G, grid, dims)
    if pv is None:
        total = int(_np.prod(_np.asarray(dims, dtype=int)))
        mask = _np.ones(total, dtype=bool)
        for key in ('PORO', 'PORV', 'NTG', 'MULTPV'):
            values = grid.get(key)
            if values is not None:
                values = _np.asarray(values, dtype=float).ravel()
                if values.size == total:
                    mask &= values > 0
        index = _np.asarray(G['cells'].get('indexMap',
                                           _np.arange(G['cells']['num'])),
                            dtype=int).ravel()
        return extract_subgrid(G, mask[index])

    if threshold is None:
        keep = pv > 0.0
    else:
        # ECLIPSE deactivates a cell whose pore volume is *below* MINPV.
        if isinstance(threshold, _np.ndarray):
            index = _np.asarray(G['cells'].get('indexMap',
                                               _np.arange(G['cells']['num'])),
                                dtype=int).ravel()
            limit = threshold[index] if threshold.size > pv.size else threshold
        else:
            limit = threshold
        keep = pv >= limit
    if keep.all():
        return G
    return extract_subgrid(G, keep)
