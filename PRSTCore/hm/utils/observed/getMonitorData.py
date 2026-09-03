"""Port of MRST ``getMonitorData.m`` (mrst-2026a/hm/utils/observed).

Assembles the complete observed-data container for a history match: well
rates and bhp (from files or from the schedule itself), plus optional
profile, saturation and tracer surveys, with the report steps refined so
every survey date is a step boundary.

The MATLAB has six defects in this file, five of which stop it running.
Each is noted at its site; the port implements the evident intent:

1. the Saturation block splits ``opt.Profile`` instead of
   ``opt.Saturation`` -- with Saturation given and Profile empty this
   errors immediately;
2. the saturation loop reads ``data.profile{i,2}.date`` instead of
   ``data.saturation{i,2}.date`` -- errors when no profile was supplied;
3. the tracer breakthrough loop indexes ``output(i,1)`` with the *record*
   index ``i`` where it means the *sample* index ``t``;
4. that same loop reuses ``t`` for both the tracer index and the sample
   counter, and assigns ``output(t,:)`` -- a row that still carries its
   date -- into a numeric slot;
5. the tracer warning prints ``data.tracer{i,'injector'}`` while the test
   above it uses ``data.tracer{i,1}``;
6. rates and bhp are read through ``getObservedFromFile(filename, time,
   {...})``, a three-argument call the two-argument ``getObservedFromFile.m``
   does not accept. The port calls the readers directly.
"""

import warnings as _warnings

import numpy as _np

from PRSTCore.hm.utils.controlIndex import control_index

from .addProfileObserved import getDepthDependentAdditive
from .addSaturationObserved import _getDepthDependentNonAdditive
from .getCellFacesDepth import getCellFacesDepth
from .readProductionHistory import readProductionHistory
from .readProfileTest import readProfileTest
from .readSaturationTest import readSaturationTest
from .readTracerTest import readTracerTest

_DAY = 86400.0
_MPA = 1.0e6
_GAS_RATE = 1.0e4 / _DAY      # 10^4 m^3/day
_SOURCE_RATE = 0.1 / _DAY

_RATE_FIELD = {'W': 'qWs', 'O': 'qOs', 'G': 'qGs'}


def getMonitorData(model, schedule, start, Rates='fromModel', BHP='fromModel',
                   Tracer=None, Profile=None, Saturation=None):
    """Return ``(observed, schedule, name_tra)``.

    ``start`` is the simulation start date; file options accept a single
    path or several separated by ``;``.
    """
    W = schedule['control'][-1]['W']
    nw = len(W)
    names = [w['name'] for w in W]
    num_cells = [_np.atleast_1d(_np.asarray(w['cells'])).ravel().size for w in W]

    dt_days = _np.asarray(schedule['step']['val'], dtype=float) / _DAY
    time = _dates_from(start, _np.cumsum(dt_days))

    data, flag = {}, {}
    survey_times = []

    data['profile'], flag['profile'], t_prf = _load_survey(
        Profile, readProfileTest, names, 'Profile tested')
    survey_times += t_prf

    data['saturation'], flag['saturation'], t_sat = _load_survey(
        # Defect 1: the MATLAB splits opt.Profile here.
        Saturation, readSaturationTest, names, 'Saturation tested')
    survey_times += t_sat

    data['tracer'], flag['tracer'], t_tra = _load_tracer(Tracer, names)
    survey_times += t_tra
    name_tra = _unique_stable([str(rec['name']) for rec in data['tracer']])

    data['rates'], flag['rates'] = _load_wellwise(Rates, 'Rates')
    data['bhp'], flag['bhp'] = _load_wellwise(BHP, 'BHP')

    # --- per-step well quantities -------------------------------------
    nstep = int(_np.asarray(schedule['step']['val']).size)
    per_step = {k: [] for k in ('status', 'type', 'val', 'sign',
                                'qWs', 'qOs', 'qGs', 'bhp')}
    for step in range(nstep):
        Ws = schedule['control'][control_index(
            schedule['step'], step, len(schedule['control']))]['W']
        sign = _np.asarray([float(w.get('sign', 0.0)) for w in Ws])
        per_step['type'].append([w.get('type') for w in Ws])
        per_step['val'].append(_np.asarray([w.get('val', 0.0) for w in Ws],
                                           dtype=float))
        per_step['sign'].append(sign)

        rates = _np.zeros((nw, 3))
        if flag['rates'] == 'fromModel':
            for j, field in enumerate(('qWs', 'qOs', 'qGs')):
                rates[:, j] = [float(_np.ravel(w.get(field, 0.0))[0]) for w in Ws]
        elif flag['rates'] == 'fromFile':
            rates = _wellwise_row(data['rates'], names, time[step],
                                  ('water', 'oil', 'gas'),
                                  (1.0 / _DAY, 1.0 / _DAY, _GAS_RATE))
        # A measured rate carries no sign; the schedule's does.
        rates = sign[:, None] * _np.abs(rates)
        per_step['qWs'].append(rates[:, 0])
        per_step['qOs'].append(rates[:, 1])
        per_step['qGs'].append(rates[:, 2])

        bhp = _np.zeros(nw)
        if flag['bhp'] == 'fromModel':
            bhp = _np.asarray([float(_np.ravel(w.get('bhp', 0.0))[0]) for w in Ws])
        elif flag['bhp'] == 'fromFile':
            bhp = _wellwise_row(data['bhp'], names, time[step],
                                ('bhp',), (_MPA,))[:, 0]
        per_step['bhp'].append(bhp)

        status = (_np.abs(rates).sum(axis=1) > 0) | (_np.abs(bhp) > 0)
        per_step['status'].append(status)

    # --- refine the schedule so every survey date is a step boundary ---
    time_sim = _union_dates(survey_times, time)
    schedule = _refine_schedule(schedule, start, time, time_sim, per_step)
    nstep = len(time_sim)

    phNames = list(model.getPhaseNames()) if hasattr(model, 'getPhaseNames') \
        else ['W', 'O']
    np_ = len(phNames)
    nt = len(name_tra)

    cqs = [[_np.zeros((n, np_)) for n in num_cells] for _ in range(nstep)]
    sW = [[_np.zeros(n) for n in num_cells] for _ in range(nstep)]
    tracer = [_np.zeros((nw, nt)) for _ in range(nstep)]

    if flag['tracer'] == 'fromFile':
        schedule = _apply_tracer(schedule, model, W, names, name_tra,
                                 data['tracer'], time_sim, tracer, np_, nt)

    if flag['profile'] == 'fromFile':
        _apply_profile(model, W, names, data['profile'], time_sim, cqs,
                       per_step, phNames)

    if flag['saturation'] == 'fromFile':
        _apply_saturation(model, W, names, data['saturation'], time_sim, sW)

    observed = []
    for step in range(nstep):
        observed.append({'wellSol': [
            {'name': names[w], 'status': bool(per_step['status'][step][w]),
             'type': per_step['type'][step][w],
             'val': per_step['val'][step][w],
             'sign': per_step['sign'][step][w],
             'bhp': per_step['bhp'][step][w],
             'qWs': per_step['qWs'][step][w],
             'qOs': per_step['qOs'][step][w],
             'qGs': per_step['qGs'][step][w],
             'cqs': cqs[step][w], 'sw': sW[step][w],
             'tracer': tracer[step][w, :]}
            for w in range(nw)]})

    return observed, schedule, name_tra


# ------------------------------------------------------------- loading --

def _split_paths(option):
    if not option:
        return []
    if isinstance(option, (list, tuple)):
        return list(option)
    return [p for p in str(option).split(';') if p]


def _load_survey(option, reader, names, description):
    """Read a survey and drop wells the schedule does not know."""
    if not option:
        return [], 'none', []
    entries = []
    for path in _split_paths(option):
        entries.extend(reader(path))

    known = {str(n).lower() for n in names}
    kept, times = [], []
    for name, table in entries:
        if str(name).lower() not in known:
            _warnings.warn('%s well %s has no schedule data. Skip this well.'
                           % (description, name), RuntimeWarning)
            continue
        kept.append((name, table))
        times.extend(list(_np.asarray(table['date'], dtype=object)))
    return kept, 'fromFile', times


def _load_tracer(option, names):
    if not option:
        return [], 'none', []
    records = []
    for path in _split_paths(option):
        records.extend(readTracerTest(path))

    known = {str(n).lower() for n in names}
    kept, times = [], []
    for rec in records:
        # Defect 5: the MATLAB's warning indexes a field the test does not.
        if str(rec['injector']).lower() not in known:
            _warnings.warn('Tracer tested well %s has no schedule data. '
                           'Skip this well.' % rec['injector'], RuntimeWarning)
            continue
        kept.append(rec)
        times.append(rec['date'])
        output = _np.atleast_2d(_np.asarray(rec.get('output', []), dtype=object))
        if output.size:
            times.extend(list(output[:, 0]))
    return kept, 'fromFile', times


def _load_wellwise(option, label):
    """Rates/bhp: a sentinel, a file list, or nothing."""
    if option is None or (isinstance(option, (list, tuple)) and not option):
        return [], 'none'
    if isinstance(option, str) and option.lower() == 'frommodel':
        return [], 'fromModel'
    if not option:
        return [], 'none'
    # Defect 6: the MATLAB calls a three-argument getObservedFromFile that
    # does not exist; the reader is called directly instead.
    entries = []
    for path in _split_paths(option):
        entries.extend(readProductionHistory(path))
    return entries, 'fromFile'


def _wellwise_row(entries, names, when, columns, scales):
    """The measured values for every well at date ``when``."""
    out = _np.zeros((len(names), len(columns)))
    lookup = {str(n).lower(): i for i, n in enumerate(names)}
    for name, table in entries:
        w = lookup.get(str(name).lower())
        if w is None:
            continue
        ix = _np.flatnonzero(_np.asarray(table['date'], dtype=object) == when)
        if ix.size == 0:
            continue
        for j, (column, scale) in enumerate(zip(columns, scales)):
            if column in table:
                out[w, j] = float(_np.asarray(table[column],
                                              dtype=float)[ix[0]]) * scale
    return out


# ------------------------------------------------------------ schedule --

def _refine_schedule(schedule, start, time, time_sim, per_step):
    """Insert the survey dates as step boundaries, repeating each step's
    controls and observed values across the steps it was split into."""
    if len(time_sim) == len(time):
        return schedule

    # Which original step each refined step belongs to.
    owner = []
    j = 0
    for value in time_sim:
        while j < len(time) - 1 and value > time[j]:
            j += 1
        owner.append(j)

    for key, values in per_step.items():
        per_step[key] = [values[o] for o in owner]

    controls = schedule['control']
    step_control = _np.asarray(schedule['step']['control'], dtype=int)
    schedule['control'] = [controls[int(step_control[o]) - 1] for o in owner]
    schedule['step'] = {
        'control': _np.arange(1, len(time_sim) + 1),
        'val': _np.asarray(_date_diffs(start, time_sim), dtype=float) * _DAY,
    }
    return schedule


# -------------------------------------------------------------- surveys --

def _apply_tracer(schedule, model, W, names, name_tra, records, time_sim,
                  tracer, np_, nt):
    for rec in records:
        w = _index(names, rec['injector'])
        c = _index(name_tra, rec['name'])
        if w is None or c is None:
            continue
        depth = _np.atleast_2d(_np.asarray(rec['depth'], dtype=float))
        virtual = {'top': depth[:, 0], 'bottom': depth[:, 1],
                   'cell': _np.ones(depth.shape[0])}
        top, bottom = getCellFacesDepth(model.G, W[w]['cells'])
        ratio = getDepthDependentAdditive(virtual, top, bottom, 'cell')

        cells = _np.atleast_1d(_np.asarray(W[w]['cells'], dtype=int)).ravel()
        vals = _SOURCE_RATE * _np.ones(cells.size) * ratio
        sat = _np.zeros(np_)
        sat[0] = 1.0
        src_tracer = _np.zeros((cells.size, nt))
        total = float(vals.sum())
        if total != 0:
            src_tracer[:, c] = float(rec['dosage']) / (total * _DAY)

        step = _step_for(time_sim, rec['date'])
        if step is not None:
            schedule['control'][step]['src'] = {
                'cell': cells, 'rate': vals,
                'sat': _np.tile(sat, (cells.size, 1)), 'tracer': src_tracer}

        # Defects 3 and 4: the MATLAB indexes output(i,1) with the record
        # index and assigns the whole row, date column included.
        output = _np.atleast_2d(_np.asarray(rec.get('output', []), dtype=object))
        producers = list(rec.get('producer', []))
        for row in output:
            step = _step_for(time_sim, row[0])
            if step is None:
                continue
            for k, producer in enumerate(producers):
                pw = _index(names, producer)
                if pw is not None:
                    tracer[step][pw, c] = float(row[k + 1])
    return schedule


def _apply_profile(model, W, names, entries, time_sim, cqs, per_step, phNames):
    for name, table in entries:
        w = _index(names, name)
        if w is None:
            continue
        top, bottom = getCellFacesDepth(model.G, W[w]['cells'])
        dates = _np.asarray(table['date'], dtype=object)
        for value in _unique_stable(list(dates)):
            step = _step_for(time_sim, value)
            if step is None:
                continue
            ix = dates == value
            subset = {k: _np.asarray(v)[ix] for k, v in table.items()
                      if k != 'date'}
            for p, phase in enumerate(phNames):
                field = _RATE_FIELD.get(str(phase).upper())
                if field is None:
                    raise ValueError('Unsupported phase name: %s' % phase)
                rate = float(per_step[field][step][w])
                ratio = getDepthDependentAdditive(
                    subset, top, bottom, 'cq%s' % str(phase).upper())
                total = ratio.sum()
                if total != 0:
                    ratio = ratio / total
                cqs[step][w][:, p] = rate * ratio


def _apply_saturation(model, W, names, entries, time_sim, sW):
    for name, table in entries:
        w = _index(names, name)
        if w is None:
            continue
        top, bottom = getCellFacesDepth(model.G, W[w]['cells'])
        # Defect 2: the MATLAB reads data.profile's dates here.
        dates = _np.asarray(table['date'], dtype=object)
        for value in _unique_stable(list(dates)):
            step = _step_for(time_sim, value)
            if step is None:
                continue
            ix = dates == value
            subset = {k: _np.asarray(v)[ix] for k, v in table.items()
                      if k != 'date'}
            sw = _getDepthDependentNonAdditive(subset, top, bottom, 'water')
            if _np.any(sw > 1):
                sw = sw / 100.0
            sW[step][w] = sw


# --------------------------------------------------------------- dates --

def _dates_from(start, offsets_in_days):
    import datetime as dt
    base = start
    if isinstance(base, dt.datetime):
        base = base.date()
    return _np.asarray([base + dt.timedelta(days=float(d))
                        for d in offsets_in_days], dtype=object)


def _date_diffs(start, dates):
    import datetime as dt
    base = start.date() if isinstance(start, dt.datetime) else start
    out, previous = [], base
    for value in dates:
        out.append((value - previous).days)
        previous = value
    return out


def _union_dates(survey_times, time):
    values = list(time) + list(survey_times)
    return _np.asarray(sorted(_unique_stable(values)), dtype=object)


def _step_for(time_sim, value):
    matches = _np.flatnonzero(_np.asarray(time_sim, dtype=object) == value)
    return int(matches[0]) if matches.size else None


def _index(names, target):
    lowered = str(target).lower()
    for i, n in enumerate(names):
        if str(n).lower() == lowered:
            return i
    return None


def _unique_stable(values):
    seen, out = set(), []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out
