"""Port of MRST ``matchObservedOW.m``
(autodiff/optimization/objectives).

The mismatch a water-oil history match minimises: a weighted sum of
squared differences in water rate, oil rate and bottom-hole pressure,
one term per well per report step, scaled by that step's share of the
total time and by the number of wells being matched.

This is the objective ``hm/test/HistoryMatching.m`` uses. It is the
three-term sibling of :mod:`matchObservedOWG` -- the gas rate is absent,
not weighted to zero -- and MRST keeps them as separate files because
they are separate objectives, so this stays a separate module too. The
shared machinery (weights, the well-status expansion, the accumulation
options) comes from that one rather than being written twice.

The default weights are the reciprocal of what is being matched, so the
three terms are commensurable without the caller choosing units: water
and oil *share* ``1/sum(|qWs| + |qOs|)`` -- their errors trade off
against one another and against nothing else -- while bhp takes
``1/(max - min)`` of the observed pressures.
"""

import numpy as _np

from .matchObservedOWG import (_expandToFull, _getWeights, _scalar,
                               _vertcatIfPresent)


def matchObservedOW(model, states, schedule, observed, WaterRateWeight=None,
                    OilRateWeight=None, BHPWeight=None,
                    ComputePartials=False, tStep=None, state=None,
                    from_states=True, matchOnlyProducers=False,
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
        matchCases = (_np.asarray([float(w.get('sign', 0.0))
                                   for w in sol_obs]) < 0) \
            if matchOnlyProducers else _np.ones(nw, dtype=bool)

        qWs_obs = _vertcatIfPresent(sol_obs, 'qWs', nw)
        qOs_obs = _vertcatIfPresent(sol_obs, 'qOs', nw)
        bhp_obs = _vertcatIfPresent(sol_obs, 'bhp', nw)
        status_obs = _np.asarray([bool(w.get('status', True))
                                  for w in sol_obs])

        # The gas argument is what makes this the three-term objective:
        # passing zeros leaves ``wg`` at zero and keeps the water/oil
        # shared denominator exactly as ``matchObservedOW``'s own
        # getWeights computes it.
        ww, wo, _wg, wp = _getWeights(qWs_obs, qOs_obs,
                                      _np.zeros_like(qOs_obs), bhp_obs,
                                      WaterRateWeight, OilRateWeight, 0.0,
                                      BHPWeight)

        if ComputePartials:
            st = model.getStateAD(states[int(tSteps[step])], True) \
                if from_states else state
            qWs = model.FacilityModel.getProp(st, 'qWs')
            qOs = model.FacilityModel.getProp(st, 'qOs')
            bhp = model.FacilityModel.getProp(st, 'bhp')
        else:
            st = states[int(tSteps[step])]
            qWs = _vertcatIfPresent(st['wellSol'], 'qWs', nw)
            qOs = _vertcatIfPresent(st['wellSol'], 'qOs', nw)
            bhp = _vertcatIfPresent(st['wellSol'], 'bhp', nw)
        status = _np.asarray([bool(w.get('status', True))
                              for w in st['wellSol']])

        if not status.all() or not status_obs.all():
            bhp, bhp_obs = _expandToFull(bhp, bhp_obs, status, status_obs,
                                         True)
            qWs, qWs_obs = _expandToFull(qWs, qWs_obs, status, status_obs,
                                         False)
            qOs, qOs_obs = _expandToFull(qOs, qOs_obs, status, status_obs,
                                         False)

        dt = float(dts[step])
        nmatch = int(_np.count_nonzero(matchCases))
        fac = dt / (totTime * max(nmatch, 1))

        terms = [(ww * matchCases * (qWs - qWs_obs)) ** 2,
                 (wo * matchCases * (qOs - qOs_obs)) ** 2,
                 (wp * matchCases * (bhp - bhp_obs)) ** 2]

        if mismatchSum:
            obj.append(fac * sum(_scalar(t) for t in terms))
            continue

        mm = [fac * t for t in terms]
        if accumulateTypes is None:
            tmp = mm
        else:
            pt = _np.atleast_1d(_np.asarray(accumulateTypes,
                                            dtype=int)).ravel()
            tmp = [0] * int(pt.max())
            for k in range(3):
                if pt[k] > 0:
                    tmp[pt[k] - 1] = tmp[pt[k] - 1] + mm[k]
        if accumulateWells is not None:
            import scipy.sparse as _sp
            pw = _np.atleast_1d(_np.asarray(accumulateWells,
                                            dtype=int)).ravel()
            keep = _np.flatnonzero(pw)
            M = _sp.csr_matrix((_np.ones(keep.size), (pw[keep] - 1, keep)),
                               shape=(int(pw.max()), pw.size))
            tmp = [M @ x for x in tmp]
        obj.append(_concat(tmp))

    return obj


def _concat(items):
    from PRSTCore.ad_core.adi import SparseADI
    if any(hasattr(x, 'val') for x in items):
        return SparseADI.concat(items)
    return _np.concatenate([_np.atleast_1d(_np.asarray(x)).ravel()
                            for x in items])
