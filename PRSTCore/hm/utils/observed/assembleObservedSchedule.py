"""FAHM Stage 7: merge observation dates and assemble ``observed``.

This is the executable boundary corresponding to ``FAHM.StartButtonPushed``
lines 1868--1913.  It intentionally stops before parameter construction and
objective/adjoint setup.
"""

from copy import deepcopy as _deepcopy
from dataclasses import dataclass

import numpy as _np

from ._tables import matlab_datenum as _matlab_datenum
from .addProfileObserved import addProfileObserved
from .addRatesObserved import addBhpObserved, addRatesObserved
from .addSaturationObserved import addSaturationObserved
from .addTracerObserved import addTracerObserved
from .getObservedFromSchedule import getObservedFromSchedule
from .processMonitorData import processMonitorData

_DAY = 86400.0


@dataclass(frozen=True)
class ObservedScheduleAssembly:
    """App state established by the observation-assembly part of Start."""

    deck: dict
    monitor_data: dict
    time_obs: _np.ndarray
    time_deck: _np.ndarray
    time_sim: _np.ndarray
    control_repeat: _np.ndarray
    schedule: dict
    observed: list
    phase_names: tuple


def assembleObservedSchedule(deck, model, monitor_data, *,
                             schedule_converter=None):
    """Return FAHM's refined deck, MRST schedule and observed cell array.

    ``deck`` is copied before refinement.  This gives the caller MATLAB-like
    value ownership: assigning the returned deck to the App does not mutate a
    saved pre-Start snapshot or the model's own deck.
    """
    if not isinstance(deck, dict) or not isinstance(deck.get('SCHEDULE'), dict):
        raise ValueError('FAHM Stage 7 requires deck.SCHEDULE')
    if model is None:
        raise ValueError('FAHM Stage 7 requires the Stage 5 model')

    work_deck = _deepcopy(deck)
    monitor = _complete_monitor_data(monitor_data)
    wellnames = _deck_well_names(work_deck)
    monitor, time_obs = processMonitorData(monitor, wellnames)

    if schedule_converter is None:
        from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import \
            _convert_deck_schedule_to_mrst
        schedule_converter = _convert_deck_schedule_to_mrst
    schedule = schedule_converter(
        model, work_deck, G=model.G, rock=model.rock)

    start = _start_datenum((work_deck.get('RUNSPEC') or {}).get('START'))
    step_seconds = _np.asarray(
        schedule.get('step', {}).get('val', []), dtype=float).ravel()
    if step_seconds.size == 0 or _np.any(step_seconds <= 0.0):
        raise ValueError('FAHM Stage 7 requires positive schedule steps')
    time_deck = start + _np.cumsum(step_seconds) / _DAY

    time_obs = _np.asarray(time_obs, dtype=float).ravel()
    if time_obs.size and (_np.any(time_obs <= start) or
                          _np.any(time_obs > time_deck[-1])):
        raise ValueError(
            'Observation dates must be after START and no later than the '
            'last deck report date')
    time_sim = _np.union1d(time_obs, time_deck)
    repeat = _control_repeat(time_sim, time_deck)

    if _np.any(repeat > 1):
        new_seconds = _np.diff(_np.concatenate([[start], time_sim])) * _DAY
        schedule = _refine_mrst_schedule(schedule, repeat, new_seconds)
        _refine_deck_schedule(work_deck['SCHEDULE'], repeat, new_seconds)

    observed = getObservedFromSchedule(schedule)
    phase_names = tuple(_phase_names(model))
    if _has_entries(monitor['rates']):
        observed = addRatesObserved(
            observed, time_sim, monitor['rates'], model.G, schedule,
            phase_names)
    if _has_entries(monitor['bhp']):
        observed = addBhpObserved(
            observed, time_sim, monitor['bhp'], model.G, schedule,
            phase_names)
    if _has_entries(monitor['profile']):
        observed = addProfileObserved(
            observed, time_sim, monitor['profile'], model.G, schedule,
            phase_names)
    if _has_entries(monitor['saturation']):
        observed = addSaturationObserved(
            observed, time_sim, monitor['saturation'], model.G, schedule,
            phase_names)
    if _has_entries(monitor['tracer']):
        observed, schedule = addTracerObserved(
            observed, time_sim, monitor['tracer'], model.G, schedule,
            phase_names)

    return ObservedScheduleAssembly(
        deck=work_deck,
        monitor_data=monitor,
        time_obs=_np.array(time_obs, copy=True),
        time_deck=_np.array(time_deck, copy=True),
        time_sim=_np.array(time_sim, copy=True),
        control_repeat=_np.array(repeat, dtype=int, copy=True),
        schedule=schedule,
        observed=observed,
        phase_names=phase_names,
    )


def _complete_monitor_data(data):
    data = _deepcopy(data or {})
    for name in ('rates', 'bhp', 'tracer', 'profile', 'saturation'):
        data.setdefault(name, [])
    return data


def _deck_well_names(deck):
    controls = (deck.get('SCHEDULE') or {}).get('control') or []
    if not controls:
        raise ValueError('FAHM Stage 7 requires deck schedule controls')
    rows = (controls[-1] or {}).get('WELSPECS') or []
    names = [str(row[0]) for row in rows if row]
    if not names:
        raise ValueError('FAHM Stage 7 requires WELSPECS in the last control')
    return names


def _start_datenum(value):
    if isinstance(value, str):
        from PRSTCore.deckformat.deckinput.schedule_control import \
            parse_eclipse_date
        parsed = parse_eclipse_date(value)
        if parsed is None:
            raise ValueError('Cannot parse RUNSPEC.START: %r' % value)
        # schedule_control uses Python ordinal; MATLAB datenum is +366.
        return float(parsed) + 366.0
    if value is None:
        raise ValueError('FAHM Stage 7 requires RUNSPEC.START')
    return _matlab_datenum(value)


def _control_repeat(time_sim, time_deck):
    positions = []
    for value in time_deck:
        hits = _np.flatnonzero(time_sim == value)
        if hits.size != 1:
            raise ValueError('A deck report date is missing from time_sim')
        positions.append(int(hits[0]))
    positions = _np.asarray(positions, dtype=int)
    return _np.diff(_np.concatenate([positions, [time_sim.size]]))


def _active_controls(schedule):
    step = schedule.get('step') or {}
    values = _np.asarray(step.get('val', []), dtype=float).ravel()
    mapping = _np.asarray(step.get('control', []), dtype=int).ravel()
    controls = schedule.get('control') or []
    if mapping.size != values.size:
        raise ValueError('schedule.step.control must map every report step')
    if mapping.size and (_np.any(mapping < 0) or _np.any(mapping >= len(controls))):
        raise IndexError('schedule.step.control contains an invalid control')
    return controls, mapping


def _repeat_active_controls(schedule, repeat):
    controls, mapping = _active_controls(schedule)
    if mapping.size != repeat.size:
        raise ValueError('control repeat count must match original report steps')
    out = []
    source_indices = []
    for step, count in enumerate(repeat):
        source = int(mapping[step])
        for _ in range(int(count)):
            out.append(_deepcopy(controls[source]))
            source_indices.append(source)
    return out, source_indices


def _refine_mrst_schedule(schedule, repeat, new_seconds):
    out = _deepcopy(schedule)
    controls, source_indices = _repeat_active_controls(out, repeat)
    out['control'] = controls
    out['step'] = {
        'val': _np.asarray(new_seconds, dtype=float),
        'control': _np.arange(len(controls), dtype=int),
    }
    for field in ('multipliers', 'multpv'):
        values = out.get(field)
        if isinstance(values, (list, tuple)) and values:
            out[field] = [_deepcopy(values[index]) for index in source_indices]
    return out


def _refine_deck_schedule(schedule, repeat, new_seconds):
    controls, _ = _repeat_active_controls(schedule, repeat)
    factor = float(schedule.get('_time_factor', _DAY))
    if factor <= 0.0:
        raise ValueError('SCHEDULE._time_factor must be positive')
    schedule['control'] = controls
    schedule['step'] = {
        'val': _np.asarray(new_seconds, dtype=float) / factor,
        'control': _np.arange(len(controls), dtype=int),
    }


def _phase_names(model):
    getter = getattr(model, 'getPhaseNames', None)
    if callable(getter):
        return [str(name) for name in getter()]
    flags = (('W', 'water'), ('O', 'oil'), ('G', 'gas'))
    names = [name for name, attr in flags if bool(getattr(model, attr, False))]
    if not names:
        raise ValueError('Cannot determine the active phase ordering')
    return names


def _has_entries(value):
    if value is None or isinstance(value, str):
        return False
    try:
        return len(value) > 0
    except TypeError:
        return False
