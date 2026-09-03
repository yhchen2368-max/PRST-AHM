"""Python port of MRST's ``computeWellIndex.m`` (mrst-2026a/core/utils): the
classic Peaceman equivalent-radius well index formula, for the default
two-point-flux (``ip_tpf``) inner product.
"""

from __future__ import annotations

import numpy as _np


def compute_well_index(dx, dy, dz, kx, ky, kz, radius, *, direction="z", skin=0.0, kh=None,
                        griddim: int = 3) -> _np.ndarray:
    """Port of MRST ``computeWellIndex.m`` (``InnerProduct='ip_tpf'`` path,
    ``wellConstant`` = 0.14).

    Parameters
    ----------
    dx, dy, dz : array
        Per-completion cell extents along each axis.
    kx, ky, kz : array
        Per-completion permeability along each axis.
    radius : array or float
        Wellbore radius.
    direction : array of {'x','y','z'} or a single such value
        Well direction through each completion (perpendicular plane pair
        used for the equivalent radius).
    skin : array or float
        Skin factor (dimensionless).
    kh : array or None
        Explicit permeability-thickness override per completion; ``-1``
        (or, as here, omitted) computes it from ``ell*sqrt(k1*k2)``.
    """
    dx, dy, dz = (_np.atleast_1d(_np.asarray(a, dtype=float)) for a in (dx, dy, dz))
    kx, ky, kz = (_np.atleast_1d(_np.asarray(a, dtype=float)) for a in (kx, ky, kz))
    n = dx.size

    radius = _np.broadcast_to(_np.atleast_1d(_np.asarray(radius, dtype=float)), (n,)).copy()
    skin = _np.broadcast_to(_np.atleast_1d(_np.asarray(skin, dtype=float)), (n,)).copy()
    direction = _np.asarray(direction) if _np.ndim(direction) else _np.full(n, direction)
    direction = _np.broadcast_to(direction, (n,))

    d1 = _np.zeros(n)
    d2 = _np.zeros(n)
    ell = _np.zeros(n)
    k1 = _np.zeros(n)
    k2 = _np.zeros(n)

    ci = direction == "x"
    d1[ci], d2[ci], ell[ci], k1[ci], k2[ci] = dy[ci], dz[ci], dx[ci], ky[ci], kz[ci]
    ci = direction == "y"
    d1[ci], d2[ci], ell[ci], k1[ci], k2[ci] = dx[ci], dz[ci], dy[ci], kx[ci], kz[ci]
    ci = direction == "z"
    d1[ci], d2[ci], ell[ci], k1[ci], k2[ci] = dx[ci], dy[ci], dz[ci], kx[ci], ky[ci]

    with _np.errstate(divide="ignore", invalid="ignore"):
        k21 = _np.where(_np.isfinite(k2 / k1), k2 / k1, 0.0)
        k12 = _np.where(_np.isfinite(k1 / k2), k1 / k2, 0.0)

    wc = 0.14  # wellConstant(..., 'ip_tpf')
    re1 = 2.0 * wc * _np.sqrt(d1**2 * _np.sqrt(k21) + d2**2 * _np.sqrt(k12))
    re2 = k21 ** (1.0 / 4.0) + k12 ** (1.0 / 4.0)
    with _np.errstate(divide="ignore", invalid="ignore"):
        re = _np.where(_np.isfinite(re1 / re2), re1 / re2, 0.0)
    ke = _np.sqrt(k1 * k2)

    if kh is None:
        kh_arr = _np.full(n, -1.0)
    else:
        kh_arr = _np.broadcast_to(_np.atleast_1d(_np.asarray(kh, dtype=float)), (n,)).copy()
    override = kh_arr < 0
    kh_arr = kh_arr.copy()
    kh_arr[override] = (ell[override] * ke[override]) if griddim > 2 else ke[override]

    WI = 2.0 * _np.pi * kh_arr / (_np.log(re / radius) + skin)

    if _np.any(WI < 0):
        if _np.any(re < radius):
            raise ValueError("Equivalent radius in well model smaller than well radius: negative well index")
        raise ValueError("Large negative skin factor causing negative well index")

    return WI
