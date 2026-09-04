"""Port of MRST ``getNormalizationFactors.m`` (mrst-2026a/hm/utils/observed).

Per-quantity weights that put the mismatch terms of a history-matching
objective on a comparable footing.

Each weight is ``total_time / integral(|q|)`` over the *producing* wells
(``sign == -1``) only, so a quantity that is typically large gets a small
weight. A quantity that never appears keeps a weight of zero, which zeroes
its contribution rather than dividing by nothing.
"""

import numpy as _np


def getNormalizationFactors(observed):
    """Return ``{'ww', 'wo', 'wg', 'wp', 'wt'}``."""
    if not observed:
        raise ValueError('observed must contain at least one report step')
    ns = len(observed)
    nw = len(observed[0]['wellSol'])
    if nw == 0:
        raise ValueError('observed wellSol must contain at least one well')

    dt = _timestep_lengths(observed, ns)

    flag = _np.zeros((nw, ns))
    qWs = _np.zeros((nw, ns))
    qOs = _np.zeros((nw, ns))
    qGs = _np.zeros((nw, ns))
    bhp = _np.zeros((nw, ns))
    cTs = _np.zeros((nw, ns))

    for step in range(ns):
        sol = observed[step]['wellSol']
        if len(sol) != nw:
            raise ValueError(
                'observed step %d has %d wells; expected %d'
                % (step, len(sol), nw))
        # MATLAB's helper multiplies *every* field by status, including
        # sign.  A shut producer therefore has flag 0 and is excluded by
        # ``flag == -1`` in getScaling.
        flag[:, step] = _vertcatIfPresent(sol, 'sign', nw)
        qWs[:, step] = _vertcatIfPresent(sol, 'qWs', nw)
        qOs[:, step] = _vertcatIfPresent(sol, 'qOs', nw)
        qGs[:, step] = _vertcatIfPresent(sol, 'qGs', nw)
        bhp[:, step] = _vertcatIfPresent(sol, 'bhp', nw)
        cTs[:, step] = _tracer_mean(sol, nw)

    return {
        'ww': _getScaling(qWs, dt, flag),
        'wo': _getScaling(qOs, dt, flag),
        'wg': _getScaling(qGs, dt, flag),
        'wp': _getScaling(bhp, dt, flag),
        'wt': _getScaling(cTs, dt, flag),
    }


def _timestep_lengths(observed, ns):
    """``dt`` directly, or differenced from cumulative ``time``."""
    if 'dt' in observed[0]:
        if any('dt' not in observed[s] for s in range(ns)):
            raise ValueError('dt must be present at every observed step')
        dt = _np.asarray([_one_scalar(observed[s]['dt'], 'dt')
                          for s in range(ns)])
    elif 'time' in observed[0]:
        if any('time' not in observed[s] for s in range(ns)):
            raise ValueError('time must be present at every observed step')
        T = _np.asarray([_one_scalar(observed[s]['time'], 'time')
                         for s in range(ns)])
        dt = _np.concatenate([[T[0]], _np.diff(T)])
    else:
        # In MATLAB ``dt`` is left undefined and the first getScaling call
        # fails.  Raise at the source of the malformed fixture instead of
        # silently inventing unit-length timesteps.
        raise ValueError('observed must provide dt or cumulative time')
    if not _np.all(_np.isfinite(dt)):
        raise ValueError('observed timestep values must be finite')
    return dt


def _getScaling(qs, dt, flag):
    """Port of the local ``getScaling``: ``t / integral(|q|)`` over producers."""
    w = 0.0
    t = 0.0
    for step in range(dt.size):
        ix = flag[:, step] == -1
        if _np.any(ix):
            qavg = float(_np.mean(_np.abs(qs[ix, step])))
            w += qavg * dt[step]
            t += dt[step]
    return t / w if w > 0 else 0.0


def _vertcatIfPresent(sol, field, nw):
    """Port of ``vertcatIfPresent``: absent or empty -> zeros.

    A present field is multiplied by ``status``, so a shut well
    contributes nothing.
    """
    present = [field in well and well[field] is not None for well in sol]
    if not any(present):
        return _np.zeros(nw)
    if not all(present):
        raise ValueError('%s must be present for every well or none' % field)
    raw = [_np.asarray(w[field], dtype=float).ravel() for w in sol]
    if all(value.size == 0 for value in raw):
        return _np.zeros(nw)
    values = _np.asarray([_one_scalar(value, field) for value in raw],
                         dtype=float)
    if values.size != nw:
        raise ValueError('%s has width %d; expected %d'
                         % (field, values.size, nw))
    status = _statuses(sol)
    return values * status


def _tracer_mean(sol, nw):
    """``mean(vertcatIfPresent(wellSol, 'tracer'), 2)``."""
    present = ['tracer' in well and well['tracer'] is not None
               for well in sol]
    if not any(present):
        return _np.zeros(nw)
    if not all(present):
        raise ValueError('tracer must be present for every well or none')
    rows = [_np.asarray(w['tracer'], dtype=float).ravel() for w in sol]
    widths = {row.size for row in rows}
    if len(widths) != 1:
        raise ValueError('tracer width must agree for every well')
    width = widths.pop()
    if width == 0:
        return _np.zeros(nw)
    values = _np.vstack(rows)
    if values.shape[0] != nw:
        raise ValueError('tracer row count does not match wells')
    values = values * _statuses(sol)[:, None]
    return _np.mean(values, axis=1)


def _statuses(sol):
    if any('status' not in well for well in sol):
        raise ValueError('status must be present for every well')
    return _np.asarray([float(bool(well['status'])) for well in sol])


def _one_scalar(value, name):
    arr = _np.atleast_1d(_np.asarray(value, dtype=float)).ravel()
    if arr.size != 1:
        raise ValueError('%s must contain exactly one scalar; got %d values'
                         % (name, arr.size))
    return float(arr[0])
