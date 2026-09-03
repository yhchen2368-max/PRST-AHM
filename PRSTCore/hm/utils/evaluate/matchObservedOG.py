"""Port of MRST ``matchObservedOG.m`` (mrst-2026a/hm/utils/evaluate).

Gas-rate / oil-rate / bhp mismatch objective -- the gas-condensate
counterpart of :mod:`matchObservedLW`::

    obj(step) = dt/(T*nMatched) * sum( (wg*(qGs - qGs_obs))^2
                                     + (wo*(qOs - qOs_obs))^2
                                     + (wp*(bhp - bhp_obs))^2 )

Three differences from the liquid/water-cut version beyond the phases:

* the weights are **per well**, not scalars -- a well with no measured
  rate for a phase gets weight zero for it, so it contributes nothing
  instead of dividing by zero;
* the bhp weight is zero for a well with neither gas nor oil measured;
* ``matchWellIndices`` selects an explicit well subset, as an alternative
  to ``matchOnlyProducers``.

Because the weights are per well, they too must be scattered back to the
full well list when either side has a shut well (``expandWeightsToFull``).
"""

import numpy as _np
import scipy.sparse as _sp

from .matchObservedLW import _concat, _expandToFull, _scalar, _sum


def matchObservedOG(model, states, schedule, observed, GasRateWeight=None,
                    OilRateWeight=None, BHPWeight=None, ComputePartials=False,
                    tStep=None, state=None, from_states=True,
                    matchOnlyProducers=False, matchWellIndices=None,
                    mismatchSum=True, accumulateWells=None,
                    accumulateTypes=None):
    """Return one mismatch entry per requested report step."""
    dts = _np.asarray(schedule['step']['val'], dtype=float).ravel()
    totTime = float(dts.sum())

    if tStep is None:
        tSteps = _np.arange(dts.size)
        numSteps = dts.size
    else:
        tSteps = _np.atleast_1d(_np.asarray(tStep, dtype=int)).ravel()
        numSteps = 1
        dts = dts[tSteps]

    obj = []
    for step in range(numSteps):
        sol_obs = observed[int(tSteps[step])]['wellSol']
        nw = len(sol_obs)

        if matchOnlyProducers:
            matchCases = _np.asarray([float(w.get('sign', 0.0))
                                      for w in sol_obs]) < 0
        elif matchWellIndices is not None:
            matchCases = _np.zeros(nw, dtype=bool)
            matchCases[_np.asarray(matchWellIndices, dtype=int)] = True
        else:
            matchCases = _np.ones(nw, dtype=bool)

        status_obs = _np.asarray([bool(w.get('status', True)) for w in sol_obs])
        qGs_obs = _vertcatIfPresent(sol_obs, 'qGs', nw)
        qOs_obs = _vertcatIfPresent(sol_obs, 'qOs', nw)
        bhp_obs = _vertcatIfPresent(sol_obs, 'bhp', nw)

        wg, wo, wp = _getWeights(qGs_obs, qOs_obs, bhp_obs,
                                 GasRateWeight, OilRateWeight, BHPWeight)

        if ComputePartials:
            st = model.getStateAD(states[int(tSteps[step])], True) \
                if from_states else state
            qGs = model.FacilityModel.getProp(st, 'qGs')
            qOs = model.FacilityModel.getProp(st, 'qOs')
            bhp = model.FacilityModel.getProp(st, 'bhp')
            status = _np.asarray([bool(w.get('status', True))
                                  for w in st['wellSol']])
        else:
            st = states[int(tSteps[step])]
            qGs = _vertcatIfPresent(st['wellSol'], 'qGs', nw)
            qOs = _vertcatIfPresent(st['wellSol'], 'qOs', nw)
            bhp = _vertcatIfPresent(st['wellSol'], 'bhp', nw)
            status = _np.asarray([bool(w.get('status', True))
                                  for w in st['wellSol']])

        if not status.all() or not status_obs.all():
            bhp, bhp_obs = _expandToFull(bhp, bhp_obs, status, status_obs, True)
            qGs, qGs_obs = _expandToFull(qGs, qGs_obs, status, status_obs, False)
            qOs, qOs_obs = _expandToFull(qOs, qOs_obs, status, status_obs, False)
            wg, wo, wp = _expandWeightsToFull(wg, wo, wp, status_obs)

        dt = float(dts[step])
        nmatch = int(_np.count_nonzero(matchCases))
        fac = dt / (totTime * max(nmatch, 1))

        terms = [(wg * matchCases * (qGs - qGs_obs)) ** 2,
                 (wo * matchCases * (qOs - qOs_obs)) ** 2,
                 (wp * matchCases * (bhp - bhp_obs)) ** 2]

        if mismatchSum:
            obj.append(fac * sum(_sum(t) for t in terms))
            continue

        mm = [fac * t for t in terms]
        if accumulateTypes is None:
            tmp = mm
        else:
            pt = _np.atleast_1d(_np.asarray(accumulateTypes, dtype=int)).ravel()
            tmp = [0] * int(pt.max())
            for k in range(3):
                if pt[k] > 0:
                    tmp[pt[k] - 1] = tmp[pt[k] - 1] + mm[k]
        if accumulateWells is not None:
            pw = _np.atleast_1d(_np.asarray(accumulateWells, dtype=int)).ravel()
            keep = _np.flatnonzero(pw)
            M = _sp.csr_matrix((_np.ones(keep.size), (pw[keep] - 1, keep)),
                               shape=(int(pw.max()), pw.size))
            tmp = [M @ x for x in tmp]
        obj.append(_concat(tmp))

    return obj


def _getWeights(qGs, qOs, bhp, wg, wo, wp):
    """Port of ``getWeights``: reciprocal per-well rate, zero where absent."""
    rg, ro = _np.abs(qGs), _np.abs(qOs)

    if wg is None:
        wg = _np.zeros(qGs.size)
        ix = rg != 0
        wg[ix] = 1.0 / rg[ix]
    else:
        wg = _np.full(qGs.size, float(wg))

    if wo is None:
        wo = _np.zeros(qOs.size)
        ix = ro != 0
        wo[ix] = 1.0 / ro[ix]
    else:
        wo = _np.full(qOs.size, float(wo))

    if wp is None:
        dp = float(_np.max(bhp) - _np.min(bhp)) if bhp.size else 0.0
        wp = _np.zeros(bhp.size)
        # A well with neither phase measured gets no pressure weight either.
        ix = (rg != 0) | (ro != 0)
        if dp != 0:
            wp[ix] = 1.0 / dp
    else:
        wp = _np.full(bhp.size, float(wp))

    return wg, wo, wp


def _expandWeightsToFull(wg, wo, wp, status_obs):
    """Port of ``expandWeightsToFull``: per-well weights follow their wells."""
    out = []
    for w in (wg, wo, wp):
        tmp = _np.zeros(status_obs.size)
        tmp[status_obs] = _np.asarray(w, dtype=float).ravel()
        out.append(tmp)
    return tuple(out)


def _vertcatIfPresent(sol, field, nw):
    status = _np.asarray([bool(w.get('status', True)) for w in sol])
    if not sol or field not in sol[0]:
        return _np.zeros(int(_np.count_nonzero(status)))
    values = _np.asarray([_scalar(w.get(field)) for w in sol], dtype=float)
    assert values.size == nw
    return values[status]
