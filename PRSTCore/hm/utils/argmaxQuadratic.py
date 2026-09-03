"""Port of MRST ``argmaxQuadratic.m`` (mrst-2026a/hm/utils).

Fits the quadratic through a value/derivative pair at ``p1.a`` and a value
at ``p2.a``, then returns the stationary point.  Used by the line searches
in ``hm/utils/optimizer``.
"""

import numpy as _np


def argmaxQuadratic(p1, p2):
    """Return ``(xOpt, poly)``.

    ``p1`` supplies ``a`` (abscissa), ``v`` (value) and ``dv`` (derivative);
    ``p2`` supplies ``a`` and ``v``.  Both may be dicts or attribute
    objects.  ``poly`` is in MATLAB ``polyval`` order (highest power
    first), expressed in the shifted coordinate ``x - p1.a``.

    The stationary point is ``-inf`` when the fit has no interior extremum
    or when it falls at or below ``p1.a``, matching the MATLAB guards.
    """
    a1, v1, dv1 = _get(p1, 'a'), _get(p1, 'v'), _get(p1, 'dv')
    a2, v2 = _get(p2, 'a'), _get(p2, 'v')

    # p1.a = 0; p2.a = p2.a - shift  (work in the shifted coordinate)
    shift = a1
    a = a2 - shift

    poly = _np.zeros(3, dtype=float)
    poly[1], poly[2] = dv1, v1
    if a == 0.0:
        # a^2 in the denominator below; an undefined leading coefficient.
        return -_np.inf, poly
    poly[0] = (v2 - a * dv1 - v1) / (a ** 2)

    # roots(polyder(poly)): the derivative of a quadratic is linear, so at
    # most one root -- and none at all when the quadratic degenerates.
    deriv = _np.polyder(poly)
    roots = _np.roots(deriv) if _np.any(deriv) else _np.zeros(0)
    roots = roots[_np.isreal(roots)].real if roots.size else roots

    if roots.size == 0:
        return -_np.inf, poly
    xe = float(roots[0])
    # MATLAB compares against p1.a, which is 0 in the shifted coordinate.
    if xe < 0.0:
        return -_np.inf, poly
    return xe + shift, poly


def _get(obj, key):
    if isinstance(obj, dict):
        return float(obj[key])
    return float(getattr(obj, key))
