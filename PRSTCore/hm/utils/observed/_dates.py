"""MATLAB-date helpers shared by FAHM's observation assemblers."""

import numpy as _np

from ._tables import matlab_datenum


def serial_dates(values):
    """Return a one-dimensional MATLAB-serial-day array."""
    raw = _np.asarray(values, dtype=object).ravel()
    return _np.asarray([matlab_datenum(value) for value in raw], dtype=float)


def step_for(time_sim, value):
    """Return the zero-based report step matching ``value`` exactly."""
    timeline = serial_dates(time_sim)
    target = matlab_datenum(value)
    matches = _np.flatnonzero(timeline == target)
    return int(matches[0]) if matches.size else None


def matlab_field_value(values):
    """Preserve MATLAB scalar versus repeated-row assignment semantics."""
    out = _np.asarray(values, dtype=float).ravel()
    if out.size == 1:
        return float(out[0])
    return _np.array(out, copy=True)
