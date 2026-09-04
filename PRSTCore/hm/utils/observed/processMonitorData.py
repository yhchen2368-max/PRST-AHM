"""Port of MRST ``processMonitorData.m`` (mrst-2026a/hm/utils/observed).

Filters every measurement category down to wells the schedule actually
knows about, and returns the union of all measurement dates -- the report
times the simulation has to hit for the mismatch to be computable.

A measurement for an unknown well is dropped with a warning rather than
silently ignored, since it usually means a name mismatch between the
history file and the deck.

``rates``/``bhp`` may carry the sentinel ``'fromModel'``, meaning "use the
simulated values", and are then left alone.
"""

import warnings as _warnings
from copy import deepcopy as _deepcopy

import numpy as _np

from ._tables import matlab_datenum

_CATEGORIES = (
    ('rates', 'observed rates'),
    ('bhp', 'observed bhp'),
    ('profile', 'profile tested data'),
    ('tracer', 'tracer test data'),
    ('saturation', 'saturation test data'),
)

_FROM_MODEL = 'frommodel'


def processMonitorData(data, wellnames):
    """Return ``(data, time)``."""
    # A MATLAB struct has copy-on-write value semantics.  Filtering the
    # returned struct must not mutate the App's saved reader result.
    data = _deepcopy(data)
    known = {str(n).lower() for n in wellnames}
    times = []

    for key, description in _CATEGORIES:
        entries = data.get(key)
        if _is_empty(entries) or (key in ('rates', 'bhp') and
                                  _is_from_model(entries)):
            continue

        kept, category_times = [], []
        for entry in entries:
            name = _entry_name(entry)
            if str(name).lower() not in known:
                _warnings.warn(
                    'Well %s with %s has no schedule data. Skip this well.'
                    % (name, description), RuntimeWarning)
                continue
            kept.append(entry)
            category_times.extend(_entry_dates(key, entry))

        data[key] = kept
        times.extend(category_times)

    return data, _unique_sorted(times)


def _is_from_model(entries):
    """``strcmpi(data.rates, 'fromModel')`` -- the "use the model" sentinel."""
    return isinstance(entries, str) and entries.lower() == _FROM_MODEL


def _entry_name(entry):
    if isinstance(entry, dict):
        return entry.get('injector', entry.get('name'))
    return entry[0]


def _entry_dates(key, entry):
    """Every date the entry carries.

    A tracer record contributes both its injection date and each of its
    breakthrough sample dates.
    """
    if key == 'tracer' and isinstance(entry, dict):
        out = [entry['date']]
        output = _np.atleast_2d(_np.asarray(entry.get('output', []), dtype=object))
        if output.size:
            out.extend(list(output[:, 0]))
        return out
    table = entry[1] if not isinstance(entry, dict) else entry
    return list(_np.asarray(table['date'], dtype=object))


def _unique_sorted(values):
    return _np.asarray(sorted({matlab_datenum(v) for v in values}),
                       dtype=float)


def _is_empty(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value == ''
    try:
        return len(value) == 0
    except TypeError:
        return False
