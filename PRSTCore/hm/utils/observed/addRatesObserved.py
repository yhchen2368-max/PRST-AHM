"""Port of MRST ``addRatesObserved.m`` and ``addBhpObserved.m``
(mrst-2026a/hm/utils/observed).

Writes measured well rates / bottom-hole pressures into the observed
container at the report steps whose dates match.

Rates are signed by the well's own ``sign``, so a producer's measured
(positive) rate is stored negative, matching MRST's convention. Pressures
are read in MPa and stored in Pa.
"""

import numpy as _np

_DAY = 86400.0
_MPA = 1.0e6

_RATE_FIELD = {'W': 'qWs', 'O': 'qOs', 'G': 'qGs'}


def addRatesObserved(observed, time_sim, data, G, schedule, phNames):
    """Port of ``addRatesObserved``: measured surface rates, m^3/day -> m^3/s."""
    wellnames = [w['name'] for w in schedule['control'][0]['W']]

    for name, table in data:
        w = _well_index(wellnames, name)
        if w is None:
            continue
        dates = _np.asarray(table['date'])
        for value in _np.unique(dates):
            ix = dates == value
            step = _step_for(time_sim, value)
            if step is None:
                continue
            sol = observed[step]['wellSol'][w]
            sign = float(sol.get('sign', 1.0))
            for phase in phNames:
                field = _RATE_FIELD.get(str(phase).upper())
                if field is None:
                    raise ValueError('Unsupported phase name: %s' % phase)
                sol[field] = sign * _first(table[field][ix]) / _DAY
    return observed


def addBhpObserved(observed, time_sim, data, G, schedule, phNames=None):
    """Port of ``addBhpObserved``: measured bhp, MPa -> Pa."""
    wellnames = [w['name'] for w in schedule['control'][0]['W']]

    for name, table in data:
        w = _well_index(wellnames, name)
        if w is None:
            continue
        dates = _np.asarray(table['date'])
        for value in _np.unique(dates):
            ix = dates == value
            step = _step_for(time_sim, value)
            if step is None:
                continue
            observed[step]['wellSol'][w]['bhp'] = _first(table['bhp'][ix]) * _MPA
    return observed


def _well_index(wellnames, name):
    lowered = str(name).lower()
    for i, wn in enumerate(wellnames):
        if str(wn).lower() == lowered:
            return i
    return None


def _step_for(time_sim, value):
    """The report step whose date equals ``value``."""
    matches = _np.flatnonzero(_np.asarray(time_sim) == value)
    return int(matches[0]) if matches.size else None


def _first(values):
    arr = _np.atleast_1d(_np.asarray(values, dtype=float)).ravel()
    return float(arr[0]) if arr.size else 0.0
