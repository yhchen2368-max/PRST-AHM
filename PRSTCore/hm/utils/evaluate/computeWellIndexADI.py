"""Port of MRST ``computeWellIndexADI.m`` (mrst-2026a/hm/utils/evaluate).

The differentiable counterpart of ``core/utils/computeWellIndex.m``: the
same Peaceman productivity index, but written so the permeability may be a
``SparseADI`` so that WI carries a derivative with respect to the tuned
permeability during history matching.

Two departures from the non-differentiable original, both present in the
MATLAB and kept here:

* every non-finite intermediate (``k2/k1``, ``k1/k2``, ``re``, and finally
  ``WI`` itself) is zeroed rather than propagated, so a zero-permeability
  perforation yields ``WI = 0`` instead of a NaN;
* NTG multiplies the connection length ``ell`` -- but only on the branch
  that ran last, see :func:`_connection_dimensions`.
"""

import numpy as _np

from PRSTCore.ad_core.adi import SparseADI as _SparseADI
from PRSTCore.ad_core.adi import ad_select as _ad_select

# wellConstant for the 'ip_tpf'/'ip_quasitpf' inner products.
_WELL_CONSTANT_TPF = 0.14

# Interpolation table for the mixed inner products (ratio -> constant).
_WELL_CONSTANT_TABLE = _np.array([
    [1, 0.292], [2, 0.278], [3, 0.262], [4, 0.252], [5, 0.244],
    [8, 0.231], [9, 0.229], [16, 0.220], [17, 0.219], [32, 0.213],
    [33, 0.213], [64, 0.210], [65, 0.210],
], dtype=float)


def computeWellIndexADI(G, rock, radius, cells, Dir='z', Subset=None,
                        cellDims=None, InnerProduct='ip_tpf', Skin=None,
                        Kh=None):
    """Return the Peaceman well index for each entry of ``cells``."""
    cells = _np.atleast_1d(_np.asarray(cells, dtype=int)).ravel()
    nc = cells.size
    Skin = _np.zeros(nc) if Skin is None else _np.atleast_1d(
        _np.asarray(Skin, dtype=float)).ravel()
    Kh = _np.full(nc, -1.0) if Kh is None else _np.atleast_1d(
        _np.asarray(Kh, dtype=float)).ravel()
    radius = _np.atleast_1d(_np.asarray(radius, dtype=float)).ravel()
    if radius.size == 1:
        radius = _np.full(nc, radius[0])

    d1, d2, ell, k1, k2 = _connection_dimensions(G, rock, cells, radius,
                                                 Dir, cellDims)

    # A zero-permeability perforation makes these ratios infinite and the
    # equivalent radius NaN; the MATLAB zeroes each in turn rather than
    # propagating, which is what _zero_nonfinite reproduces. The warnings
    # the intermediates raise on the way are therefore expected.
    with _np.errstate(divide='ignore', invalid='ignore'):
        k21 = _zero_nonfinite(k2 / k1)
        k12 = _zero_nonfinite(k1 / k2)

        wc = _wellConstant(d1, d2, InnerProduct)
        re1 = 2.0 * wc * ((d1 ** 2) * k21 ** 0.5 + (d2 ** 2) * k12 ** 0.5) ** 0.5
        re2 = k21 ** 0.25 + k12 ** 0.25
        re = _zero_nonfinite(re1 / re2)

    ke = (k1 * k2) ** 0.5

    # Kh < 0 marks "not supplied" -- fill it from the connection length.
    griddim = int(G.get('griddim', 3)) if isinstance(G, dict) else 3
    fill = Kh < 0.0
    Kh_out = _promote(Kh, ke)
    if _np.any(fill):
        replacement = (ell * ke) if griddim > 2 else ke
        Kh_out = _put(Kh_out, fill, replacement)

    with _np.errstate(divide='ignore', invalid='ignore'):
        A = _log(re / radius)
        WI = 2.0 * _np.pi * Kh_out / (A + Skin)
    bad = ~_np.isfinite(_value(WI)) | ~_np.isfinite(_value(A))
    if _np.any(bad):
        WI = _put(WI, bad, 0.0)

    _check_peaceman_wi(WI, re, radius)

    if Subset is not None:
        subset = _np.asarray(Subset)
        WI = WI[subset] if not isinstance(WI, _SparseADI) else WI[
            _np.flatnonzero(subset) if subset.dtype == bool else subset]
    return WI


def _connection_dimensions(G, rock, cells, radius, Dir, cellDims):
    """Port of the local ``connection_dimensions``.

    ``d1``/``d2`` are the two cross-flow extents, ``ell`` the connection
    length, ``k1``/``k2`` the two cross-flow permeabilities.
    """
    dx, dy, dz = _geometric_dimensions(G, cells, cellDims)
    k = _extract_permeability(rock, cells)

    ntg = _np.ones(cells.size, dtype=float)
    if isinstance(rock, dict) and rock.get('ntg') is not None:
        values = _np.asarray(rock['ntg'], dtype=float).ravel()
        if values.size == 1:
            ntg = _np.full(cells.size, values[0])
        elif values.size > int(cells.max(initial=-1)):
            ntg = values[cells]

    welldir = _np.asarray([str(d).lower() for d in _np.atleast_1d(Dir)])
    if welldir.size == 1:
        welldir = _np.repeat(welldir, cells.size)

    n = cells.size
    d1 = _np.zeros(n)
    d2 = _np.zeros(n)
    ell = _np.zeros(n)
    k1 = _zeros_like_any(k, n)
    k2 = _zeros_like_any(k, n)

    last = None
    for axis, (a, b, l, i1, i2) in {
        'x': (dy, dz, dx, 1, 2),
        'y': (dx, dz, dy, 0, 2),
        'z': (dx, dy, dz, 0, 1),
    }.items():
        ci = welldir == axis
        if not _np.any(ci):
            continue
        d1[ci] = a[ci]
        d2[ci] = b[ci]
        ell[ci] = l[ci]
        k1 = _put(k1, ci, k[i1])
        k2 = _put(k2, ci, k[i2])
        last = ci

    # The MATLAB applies NTG as `ell(ci) = ntg(ci).*ell(ci)` *after* the
    # three branches, so `ci` is whichever mask was assigned last -- the
    # 'z' mask when any z perforation exists, otherwise 'y', otherwise 'x'.
    # Reproduced rather than corrected.
    if last is not None:
        ell[last] = ntg[last] * ell[last]
    return d1, d2, ell, k1, k2


def _geometric_dimensions(G, cells, cellDims):
    """Port of ``geometric_dimensions``: explicit dims, else cellDims(G)."""
    if cellDims is not None:
        dims = _np.atleast_2d(_np.asarray(cellDims, dtype=float))
        nc_grid = int(G['cells']['num']) if isinstance(G, dict) else dims.shape[0]
        if dims.shape[0] == nc_grid:
            sel = dims[cells, :]
        elif dims.shape[0] == cells.size:
            sel = dims
        else:
            raise ValueError(
                "Input 'cellDims' does neither match number of grid cells "
                'nor number of well cells.')
        return sel[:, 0], sel[:, 1], sel[:, 2]

    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import \
        _cell_bounding_box_dims
    nc_grid = int(G['cells']['num'])
    dims = _np.asarray(_cell_bounding_box_dims(G, nc_grid), dtype=float)
    return dims[cells, 0], dims[cells, 1], dims[cells, 2]


def _extract_permeability(rock, cells):
    """Return the three per-cell permeability components for ``cells``."""
    perm = rock['perm'] if isinstance(rock, dict) else rock.perm
    if isinstance(perm, (list, tuple)):
        cols = [p[cells] if not isinstance(p, _SparseADI) else p[cells]
                for p in perm]
    else:
        arr = _np.atleast_2d(_np.asarray(perm, dtype=float))
        if arr.shape[1] == 1:
            cols = [arr[cells, 0]] * 3
        elif arr.shape[1] == 2:
            cols = [arr[cells, 0], arr[cells, 1], arr[cells, 1]]
        else:
            cols = [arr[cells, 0], arr[cells, 1], arr[cells, 2]]
    while len(cols) < 3:
        cols.append(cols[-1])
    return cols


def _wellConstant(d1, d2, innerProduct):
    """Port of ``wellConstant``."""
    if innerProduct in ('ip_tpf', 'ip_quasitpf'):
        return _WELL_CONSTANT_TPF
    ratio = _np.maximum(_np.round(d1 / d2), _np.round(d2 / d1))
    return _np.interp(ratio, _WELL_CONSTANT_TABLE[:, 0],
                      _WELL_CONSTANT_TABLE[:, 1])


def _check_peaceman_wi(WI, re, radius):
    """Port of ``check_peaceman_wi``."""
    wi, r = _value(WI), _value(re)
    if _np.any(wi < 0):
        if _np.any(r < radius):
            raise ValueError('Equivalent radius in well model smaller than '
                             'well radius causing negative well index.')
        raise ValueError('Large negative skin factor causing negative well index.')


# --------------------------------------------------------------- helpers --

def _value(x):
    return x.val if isinstance(x, _SparseADI) else _np.asarray(x, dtype=float)


def _log(x):
    return x.log() if isinstance(x, _SparseADI) else _np.log(x)


def _zero_nonfinite(x):
    bad = ~_np.isfinite(_value(x))
    return _put(x, bad, 0.0) if _np.any(bad) else x


def _promote(plain, reference):
    """Promote a plain array to ADI when ``reference`` carries derivatives."""
    if isinstance(reference, _SparseADI) and not isinstance(plain, _SparseADI):
        return _SparseADI.constant(_np.asarray(plain, dtype=float),
                                   reference.nvar)
    return plain


def _zeros_like_any(values, n):
    for v in values:
        if isinstance(v, _SparseADI):
            return _SparseADI.constant(_np.zeros(n), v.nvar)
    return _np.zeros(n, dtype=float)


def _put(out, mask, values):
    """``out(mask) = values(mask)`` for both plain arrays and ADI."""
    mask = _np.asarray(mask)
    if isinstance(out, _SparseADI) or isinstance(values, _SparseADI):
        nvar = out.nvar if isinstance(out, _SparseADI) else values.nvar
        n = out.val.size if isinstance(out, _SparseADI) else _np.size(out)
        out = _promote(_np.broadcast_to(_np.asarray(_value(out)), (n,)).copy()
                       if not isinstance(out, _SparseADI) else out,
                       values if isinstance(values, _SparseADI) else out)
        if not isinstance(out, _SparseADI):
            out = _SparseADI.constant(_np.asarray(out, dtype=float), nvar)
        if not isinstance(values, _SparseADI):
            values = _SparseADI.constant(
                _np.broadcast_to(_np.asarray(values, dtype=float), (n,)).copy(),
                nvar)
        return _ad_select(mask, values, out)
    out = _np.array(out, dtype=float, copy=True)
    values = _np.broadcast_to(_np.asarray(values, dtype=float), out.shape)
    out[mask] = values[mask]
    return out
