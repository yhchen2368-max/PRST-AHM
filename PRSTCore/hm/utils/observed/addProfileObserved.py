"""Port of MRST ``addProfileObserved.m`` (mrst-2026a/hm/utils/observed).

Distributes a well's measured total rate over its perforations according
to a production-logging profile.

The log is *additive*: each row's value is spread over its depth interval
(divided by the interval width), so integrating over a cell recovers that
cell's share of the total. The shares are normalised to sum to one and
then scaled by the well's measured rate, so the perforation rates always
add back up to it.
"""

import numpy as _np

from .getCellFacesDepth import getCellFacesDepth
from ._dates import serial_dates as _serial_dates
from ._dates import step_for as _step_for

_RATE_FIELD = {'W': 'qWs', 'O': 'qOs', 'G': 'qGs'}


def addProfileObserved(observed, time_sim, data, G, schedule, phNames):
    """Port of ``addProfileObserved``."""
    W = schedule['control'][0]['W']
    wellnames = [w['name'] for w in W]
    nwells = len(W)
    nphase = len(phNames)

    for step in range(len(observed)):
        sols = observed[step].setdefault('wellSol', [])
        while len(sols) < nwells:
            sols.append({'name': wellnames[len(sols)]})
        for w in range(nwells):
            ncells = _np.atleast_1d(_np.asarray(W[w]['cells'])).ravel().size
            sols[w]['cqs'] = _np.zeros((ncells, nphase))

    for name, table in data:
        w = _well_index(wellnames, name)
        if w is None:
            continue
        top, bottom = getCellFacesDepth(G, W[w]['cells'])
        dates = _serial_dates(table['date'])
        for value in _np.unique(dates):
            ix = dates == value
            step = _step_for(time_sim, value)
            if step is None:
                continue
            subset = {k: _np.asarray(v)[ix] for k, v in table.items()
                      if k != 'date'}
            for p, phase in enumerate(phNames):
                field = _RATE_FIELD.get(str(phase).upper())
                if field is None:
                    raise ValueError('Unsupported phase name: %s' % phase)
                rate = float(observed[step]['wellSol'][w][field])
                ratio = getDepthDependentAdditive(
                    subset, top, bottom, 'cq%s' % str(phase).upper())
                with _np.errstate(divide='ignore', invalid='ignore'):
                    ratio = ratio / ratio.sum()
                observed[step]['wellSol'][w]['cqs'][:, p] = rate * ratio
    return observed


def getDepthDependentAdditive(table, top, bottom, column):
    """Integral of the additive log over each cell's depth interval.

    An absent column yields zeros, matching MATLAB's ``isfield`` guard.
    """
    nc = top.size
    assert nc == bottom.size
    out = _np.zeros(nc)
    if column not in table:
        return out
    h = _np.asarray(table[column], dtype=float).ravel()
    a = _np.asarray(table['top'], dtype=float).ravel()
    b = _np.asarray(table['bottom'], dtype=float).ravel()
    for i in range(nc):
        out[i] = _integrate_additive(h, a, b, float(top[i]), float(bottom[i]))
    return out


def _integrate_additive(h, a, b, lo, hi):
    """Exact integral of the additive step function over ``[lo, hi]``.

    ``additivePieceWise`` evaluates to ``h/(b - a)``, so integrating a full
    interval returns ``h`` itself -- the value is a quantity spread over
    the interval, not a density to be sampled. Integrating the step
    function exactly gives the same result as MATLAB's ``integral`` without
    the quadrature warnings a discontinuous integrand provokes.
    """
    total = 0.0
    for value, start, stop in zip(h, a, b):
        width = float(stop) - float(start)
        if width == 0:
            continue
        left = max(lo, float(start))
        right = min(hi, float(stop))
        if right > left:
            total += float(value) / width * (right - left)
    return total


def additivePieceWise(h, a, b, x):
    """Port of ``additivePieceWise``: ``h/(b-a)`` on the first interval
    containing ``x``, zero outside every interval."""
    out = _np.zeros(_np.size(x))
    for i, t in enumerate(_np.atleast_1d(x)):
        hits = _np.flatnonzero((t >= a) & (t <= b))
        if hits.size:
            j = hits[0]
            out[i] = h[j] / (b[j] - a[j])
    return out


def _well_index(wellnames, name):
    lowered = str(name).lower()
    for i, wn in enumerate(wellnames):
        if str(wn).lower() == lowered:
            return i
    return None

