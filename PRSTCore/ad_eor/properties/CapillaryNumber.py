"""Port of MRST ``CapillaryNumber.m``.

Capillary number ``Nc = |v| / sigma(cs)`` used by
``SurfactantRelativePermeability`` to interpolate between miscible
(surfactant-desaturated) and immiscible relative permeability curves.

Only the ``'square'`` velocity-reconstruction method is ported as *active*
code: it is ``prop.vmeth``'s hardcoded default in the ``.m`` source, and the
well-contribution branch under it is itself commented out there
(``add_well_contrib = false; ...`` is dead code in ``CapillaryNumber.m``).
The ``'linear'`` method and its live well contribution are ported as
``computeWellContrib`` for completeness but are not wired into
``evaluateOnDomain`` here, matching current upstream behaviour.
"""

import numpy as _np


def CapillaryNumber(fluid, gradp, T, cs, sqVeloc):
    """``gradp``: pressure difference across each internal face
    (``s.Grad(p)``, i.e. ``p[c2] - p[c1]``); ``T``: face transmissibility;
    ``sqVeloc``: callable from :func:`ad_eor.utils.computeSqVelocTPFA`.
    Small gradients are floored at ``1e-8`` (same guard as the ``.m``
    source) to avoid ``log(Nc)`` blowing up in
    ``SurfactantRelativePermeability`` when a well limit switch flattens
    the pressure field for one step."""
    gradp = _np.asarray(gradp, dtype=float).ravel()
    tooSmall = _np.abs(gradp) < 1.0e-8
    gradp = (~tooSmall) * gradp + tooSmall * 1.0e-8
    v = -_np.asarray(T, dtype=float).ravel() * gradp

    veloc_sq = sqVeloc(v)
    abs_veloc = _np.sqrt(veloc_sq)
    sigma = fluid['ift'](cs)
    return abs_veloc / sigma


def computeWellContrib(G, W, p, pBH):
    """Port of the ``.m`` file's local ``computeWellContrib`` (used only by
    the ``'linear'`` velocity method's live well contribution)."""
    nperf = [len(_np.atleast_1d(w['cells'])) for w in W]
    perf2well = _np.repeat(_np.arange(len(W)), nperf)
    wc = _np.concatenate([_np.atleast_1d(w['cells']) for w in W]) if W else _np.zeros(0, dtype=int)
    pW = _np.asarray(p, dtype=float).ravel()[wc]
    pBHw = _np.asarray(pBH, dtype=float).ravel()[perf2well]
    Tw = _np.concatenate([_np.atleast_1d(w['WI']) for w in W]) if W else _np.zeros(0)

    welldir = []
    for w in W:
        d = w.get('dir')
        n = len(_np.atleast_1d(w['cells']))
        if _np.isscalar(d) or (hasattr(d, '__len__') and len(d) == 1):
            welldir.extend([d if _np.isscalar(d) else d[0]] * n)
        else:
            welldir.extend(list(d))
    welldir = _np.asarray(welldir)

    dx, dy, dz = _cell_dims(G, wc)
    thicknessWell = dz.copy()
    thicknessWell = _np.where(welldir == 'Y', dy, thicknessWell)
    thicknessWell = _np.where(welldir == 'X', dx, thicknessWell)

    rR = _np.concatenate([_np.atleast_1d(w['rR']) for w in W]) if W else _np.zeros(0)

    velocW = Tw * (pW - pBHw) / (2.0 * _np.pi * rR * thicknessWell)
    return velocW, wc


def _cell_dims(G, cells):
    cells = _np.asarray(cells, dtype=int)
    cart_dims = _np.asarray(G['cartDims'], dtype=int).ravel()
    centroids = _np.asarray(G['cells']['centroids'], dtype=float)
    dx = _np.full(cells.size, float(_np.ptp(centroids[:, 0])) / max(int(cart_dims[0]), 1))
    dy = _np.full(cells.size, float(_np.ptp(centroids[:, 1])) / max(int(cart_dims[1]), 1))
    dz = _np.full(cells.size, float(_np.ptp(centroids[:, 2])) / max(int(cart_dims[2]), 1) if cart_dims.size > 2 else 1.0)
    return dx, dy, dz
