"""Port of MRST ``addRatesObserved.m`` and ``addBhpObserved.m``
(mrst-2026a/hm/utils/observed).

Writes measured well rates / bottom-hole pressures into the observed
container at the report steps whose dates match.

Rates are signed by the well's own ``sign``, so a producer's measured
(positive) rate is stored negative, matching MRST's convention. Pressures
are read in MPa and stored in Pa.
"""

import numpy as _np

from ._dates import matlab_field_value as _matlab_field_value
from ._dates import serial_dates as _serial_dates
from ._dates import step_for as _step_for

_DAY = 86400.0
_MPA = 1.0e6

_RATE_FIELD = {'W': 'qWs', 'O': 'qOs', 'G': 'qGs'}
_INPUT_FIELD = {
    'W': (('qWs', 'water'), 1.0),
    'O': (('qOs', 'oil'), 1.0),
    # readProductionHistory documents its gas column in 10^4 m^3/day.
    'G': (('qGs',), 1.0),
}


def addRatesObserved(observed, time_sim, data, G, schedule, phNames):
    """Port of ``addRatesObserved``: measured surface rates, m^3/day -> m^3/s."""
    wellnames = [w['name'] for w in schedule['control'][0]['W']]

    for name, table in data:
        w = _well_index(wellnames, name)
        if w is None:
            continue
        dates = _serial_dates(table['date'])
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
                names, scale = _INPUT_FIELD[str(phase).upper()]
                source = next((name for name in names if name in table), None)
                if source is None and str(phase).upper() == 'G' and 'gas' in table:
                    source, scale = 'gas', 1.0e4
                if source is None:
                    raise KeyError(names[0])
                sol[field] = _matlab_field_value(
                    sign * _np.asarray(table[source], dtype=float)[ix]
                    * scale / _DAY)
    return observed


def addBhpObserved(observed, time_sim, data, G, schedule, phNames=None):
    """Port of ``addBhpObserved``: measured bhp, MPa -> Pa."""
    wellnames = [w['name'] for w in schedule['control'][0]['W']]

    for name, table in data:
        w = _well_index(wellnames, name)
        if w is None:
            continue
        dates = _serial_dates(table['date'])
        for value in _np.unique(dates):
            ix = dates == value
            step = _step_for(time_sim, value)
            if step is None:
                continue
            observed[step]['wellSol'][w]['bhp'] = _matlab_field_value(
                _np.asarray(table['bhp'], dtype=float)[ix] * _MPA)
    return observed


def _well_index(wellnames, name):
    lowered = str(name).lower()
    for i, wn in enumerate(wellnames):
        if str(wn).lower() == lowered:
            return i
    return None

