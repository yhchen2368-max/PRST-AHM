"""Port of MRST ``wellSensitivitesOW.m`` (mrst-2026a/hm/utils/evaluate).

Not a mismatch at all: this returns a *single production quantity* summed
over the selected wells, so an adjoint sweep against it gives the
sensitivity of that quantity to the parameters.

``ProductionIndices`` picks which quantity -- ``'qWs'``, ``'qOs'`` or
``'bhp'`` -- and ``WellIndices`` which wells contribute.
"""

import numpy as _np

from .matchObservedLW import _expandToFull, _scalar

_QUANTITIES = ('qWs', 'qOs', 'bhp')


def wellSensitivitesOW(model, states, schedule, observed,
                       ProductionIndices='qWs', WellIndices=None,
                       ComputePartials=False, tStep=None, state=None,
                       from_states=True, mismatchSum=True):
    """Return the selected quantity, one entry per report step."""
    if ProductionIndices not in _QUANTITIES:
        raise ValueError('Unsupported ProductionIndices: %s' % ProductionIndices)

    dts = _np.asarray(schedule['step']['val'], dtype=float).ravel()
    if tStep is None:
        tSteps = _np.arange(dts.size)
        numSteps = dts.size
    else:
        tSteps = _np.atleast_1d(_np.asarray(tStep, dtype=int)).ravel()
        numSteps = 1

    obj = []
    for step in range(numSteps):
        sol_obs = observed[int(tSteps[step])]['wellSol']
        nw = len(sol_obs)

        if WellIndices is not None:
            matchCases = _np.zeros(nw, dtype=bool)
            matchCases[_np.asarray(WellIndices, dtype=int)] = True
        else:
            matchCases = _np.ones(nw, dtype=bool)

        status_obs = _np.asarray([bool(w.get('status', True)) for w in sol_obs])
        obs = {name: _vertcatIfPresent(sol_obs, name, nw)
               for name in _QUANTITIES}

        if ComputePartials:
            st = model.getStateAD(states[int(tSteps[step])], True) \
                if from_states else state
            sim = {name: model.FacilityModel.getProp(st, name)
                   for name in _QUANTITIES}
            status = _np.asarray([bool(w.get('status', True))
                                  for w in st['wellSol']])
        else:
            st = states[int(tSteps[step])]
            sim = {name: _vertcatIfPresent(st['wellSol'], name, nw)
                   for name in _QUANTITIES}
            status = _np.asarray([bool(w.get('status', True))
                                  for w in st['wellSol']])

        if not status.all() or not status_obs.all():
            sim['bhp'], obs['bhp'] = _expandToFull(
                sim['bhp'], obs['bhp'], status, status_obs, True)
            for name in ('qWs', 'qOs'):
                sim[name], obs[name] = _expandToFull(
                    sim[name], obs[name], status, status_obs, False)

        picked = matchCases * sim[ProductionIndices]
        obj.append(picked.sum() if mismatchSum else picked)

    return obj


def _vertcatIfPresent(sol, field, nw):
    status = _np.asarray([bool(w.get('status', True)) for w in sol])
    if not sol or field not in sol[0]:
        return _np.zeros(int(_np.count_nonzero(status)))
    values = _np.asarray([_scalar(w.get(field)) for w in sol], dtype=float)
    assert values.size == nw
    return values[status]
