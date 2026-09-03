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
    ns = len(observed)
    nw = len(observed[0]['wellSol'])

    dt = _timestep_lengths(observed, ns)

    flag = _np.zeros((nw, ns))
    qWs = _np.zeros((nw, ns))
    qOs = _np.zeros((nw, ns))
    qGs = _np.zeros((nw, ns))
    bhp = _np.zeros((nw, ns))
    cTs = _np.zeros((nw, ns))

    for step in range(ns):
        sol = observed[step]['wellSol']
        flag[:, step] = _vertcatIfPresent(sol, 'sign', nw, mask_by_status=False)
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
        return _np.asarray([float(observed[s]['dt']) for s in range(ns)])
    if 'time' in observed[0]:
        T = _np.asarray([float(observed[s]['time']) for s in range(ns)])
        return _np.concatenate([[T[0]], _np.diff(T)])
    return _np.ones(ns)


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


def _vertcatIfPresent(sol, field, nw, mask_by_status=True):
    """Port of ``vertcatIfPresent``: absent or empty -> zeros.

    A present field is multiplied by ``status``, so a shut well
    contributes nothing.
    """
    if not sol or field not in sol[0]:
        return _np.zeros(nw)
    values = _np.asarray([_scalar(w.get(field)) for w in sol], dtype=float)
    if values.size == 0:
        return _np.zeros(nw)
    if mask_by_status:
        status = _np.asarray([float(bool(w.get('status', True))) for w in sol])
        values = values * status
    return values


def _tracer_mean(sol, nw):
    """``mean(vertcatIfPresent(wellSol, 'tracer'), 2)``."""
    if not sol or 'tracer' not in sol[0]:
        return _np.zeros(nw)
    out = _np.zeros(nw)
    for i, w in enumerate(sol):
        values = _np.atleast_1d(_np.asarray(w.get('tracer', 0.0), dtype=float)).ravel()
        out[i] = float(_np.mean(values)) if values.size else 0.0
        out[i] *= float(bool(w.get('status', True)))
    return out


def _scalar(value):
    if value is None:
        return 0.0
    arr = _np.atleast_1d(_np.asarray(value, dtype=float)).ravel()
    return float(arr[0]) if arr.size else 0.0
