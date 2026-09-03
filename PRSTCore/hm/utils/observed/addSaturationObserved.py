"""Port of MRST ``addSaturationObserved.m`` (mrst-2026a/hm/utils/observed).

Writes a logged saturation profile into the observed container, averaged
over each perforated cell's depth interval.

The log is a piecewise-constant function of depth: each row covers
``[top, bottom]`` and holds one value. It is *non-additive* -- a depth
outside every logged interval contributes zero rather than being
interpolated -- and each cell's value is the interval mean

    (1 / (bottom - top)) * integral(log(z) dz)  over the cell's interval.

A profile given in percent (any value above one) is rescaled to fractions.
"""

import numpy as _np

from .getCellFacesDepth import getCellFacesDepth

_PHASE_COLUMN = {'W': 'water', 'O': 'oil', 'G': 'gas'}


def addSaturationObserved(observed, time_sim, data, G, schedule, phNames):
    """Port of ``addSaturationObserved``."""
    W = schedule['control'][0]['W']
    wellnames = [w['name'] for w in W]
    nwells = len(W)
    nphase = len(phNames)

    # Every well starts with a zero profile of the right shape.
    for step in range(len(observed)):
        sols = observed[step].setdefault('wellsol',
                                         [{} for _ in range(nwells)])
        for w in range(nwells):
            ncells = _np.atleast_1d(_np.asarray(W[w]['cells'])).ravel().size
            sols[w]['sw'] = _np.zeros((ncells, nphase))

    for name, table in data:
        w = _well_index(wellnames, name)
        if w is None:
            continue
        top, bottom = getCellFacesDepth(G, W[w]['cells'])
        dates = _np.asarray(table['date'])
        for value in _np.unique(dates):
            ix = dates == value
            step = _step_for(time_sim, value)
            if step is None:
                continue
            for p, phase in enumerate(phNames):
                column = _PHASE_COLUMN.get(str(phase).upper())
                if column is None:
                    raise ValueError('Unsupported phase name: %s' % phase)
                s = _getDepthDependentNonAdditive(
                    {k: _np.asarray(v)[ix] for k, v in table.items()
                     if k != 'date'},
                    top, bottom, column)
                if _np.any(s > 1):
                    s = s / 100.0
                observed[step]['wellsol'][w]['sw'][:, p] = s
    return observed


def _getDepthDependentNonAdditive(table, top, bottom, column):
    """Interval mean of the logged profile over each cell."""
    nc = top.size
    assert nc == bottom.size
    h = _np.asarray(table[column], dtype=float).ravel()
    a = _np.asarray(table['top'], dtype=float).ravel()
    b = _np.asarray(table['bottom'], dtype=float).ravel()

    out = _np.zeros(nc)
    for i in range(nc):
        lo, hi = float(top[i]), float(bottom[i])
        if hi == lo:
            out[i] = float(_nonAdditivePieceWise(h, a, b, _np.array([lo]))[0])
            continue
        out[i] = _integrate_piecewise(h, a, b, lo, hi) / (hi - lo)
    return out


def _integrate_piecewise(h, a, b, lo, hi):
    """Exact integral of the piecewise-constant log over ``[lo, hi]``.

    MATLAB calls ``integral`` on the step function; integrating it exactly
    avoids the quadrature warnings a discontinuous integrand provokes and
    gives the same value.
    """
    total = 0.0
    for value, start, stop in zip(h, a, b):
        left = max(lo, float(start))
        right = min(hi, float(stop))
        if right > left:
            total += float(value) * (right - left)
    return total


def _nonAdditivePieceWise(h, a, b, x):
    """Port of ``nonAdditivePieceWise``: first interval containing ``x``."""
    out = _np.zeros(_np.size(x))
    for i, t in enumerate(_np.atleast_1d(x)):
        hits = _np.flatnonzero((t >= a) & (t <= b))
        if hits.size:
            out[i] = h[hits[0]]
    return out


def _well_index(wellnames, name):
    lowered = str(name).lower()
    for i, wn in enumerate(wellnames):
        if str(wn).lower() == lowered:
            return i
    return None


def _step_for(time_sim, value):
    matches = _np.flatnonzero(_np.asarray(time_sim) == value)
    return int(matches[0]) if matches.size else None
