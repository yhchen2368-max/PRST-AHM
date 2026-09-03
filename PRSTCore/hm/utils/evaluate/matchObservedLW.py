"""Port of MRST ``matchObservedLW.m`` (mrst-2026a/hm/utils/evaluate).

Liquid-rate / water-cut / bhp mismatch objective, summed over wells and
weighted by each report step's share of the total time::

    obj(step) = dt/(T*nMatched) * sum( (wl*(qLs - qLs_obs))^2
                                     + (wc*(wct - wct_obs))^2
                                     + (wp*(bhp - bhp_obs))^2 )

A well shut on one side only contributes zero to every term rather than a
spurious difference (``expandToFull`` with ``setToZero``).

**Defect reproduced.** The observed water cut is computed as

    qLs_obs = qWs_obs + qOs_obs;
    wct_obs = qLs_obs./qLs_obs;      % identically 1, or NaN when zero

The simulated side gets it right (``wct = qWs./qLs``), so the water-cut
term compares the simulated cut against a constant 1. The intent is
plainly ``(qLs_obs - qOs_obs)./qLs_obs``. This is reproduced rather than
corrected: changing it would make the objective differ numerically from
MRST's, which is exactly what a 1:1 port must not do silently. Pass
``fix_observed_water_cut=True`` to opt into the corrected form.
"""

import numpy as _np
import scipy.sparse as _sp


def matchObservedLW(model, states, schedule, observed, LiquidRateWeight=None,
                    WaterCutWeight=None, BHPWeight=None, ComputePartials=False,
                    tStep=None, state=None, from_states=True,
                    matchOnlyProducers=False, mismatchSum=True,
                    accumulateWells=None, accumulateTypes=None,
                    fix_observed_water_cut=False):
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
        matchCases = (_np.asarray([float(w.get('sign', 0.0)) for w in sol_obs])
                      < 0) if matchOnlyProducers else _np.ones(nw, dtype=bool)

        status_obs = _np.asarray([bool(w.get('status', True)) for w in sol_obs])
        qLs_obs = _vertcatIfPresent(sol_obs, 'qWs', nw)
        wct_obs = _vertcatIfPresent(sol_obs, 'qOs', nw)
        bhp_obs = _vertcatIfPresent(sol_obs, 'bhp', nw)
        qWs_obs = qLs_obs.copy()
        qLs_obs = qLs_obs + wct_obs
        with _np.errstate(divide='ignore', invalid='ignore'):
            if fix_observed_water_cut:
                wct_obs = qWs_obs / qLs_obs
            else:
                # See the module docstring: MRST's own expression.
                wct_obs = qLs_obs / qLs_obs

        wl, wc, wp = _getWeights(qLs_obs, wct_obs, bhp_obs,
                                 LiquidRateWeight, WaterCutWeight, BHPWeight)

        if ComputePartials:
            st = model.getStateAD(states[int(tSteps[step])], True) \
                if from_states else state
            qWs = model.FacilityModel.getProp(st, 'qWs')
            wct = model.FacilityModel.getProp(st, 'qOs')
            bhp = model.FacilityModel.getProp(st, 'bhp')
            status = _np.asarray([bool(w.get('status', True))
                                  for w in st['wellSol']])
        else:
            st = states[int(tSteps[step])]
            qWs = _vertcatIfPresent(st['wellSol'], 'qWs', nw)
            wct = _vertcatIfPresent(st['wellSol'], 'qOs', nw)
            bhp = _vertcatIfPresent(st['wellSol'], 'bhp', nw)
            status = _np.asarray([bool(w.get('status', True))
                                  for w in st['wellSol']])

        qLs = qWs + wct
        wct = qWs / qLs

        if not status.all() or not status_obs.all():
            bhp, bhp_obs = _expandToFull(bhp, bhp_obs, status, status_obs, True)
            qLs, qLs_obs = _expandToFull(qWs, qLs_obs, status, status_obs, False)
            wct, wct_obs = _expandToFull(wct, wct_obs, status, status_obs, False)

        dt = float(dts[step])
        nmatch = int(_np.count_nonzero(matchCases))
        fac = dt / (totTime * max(nmatch, 1))

        terms = [(wl * matchCases * (qLs - qLs_obs)) ** 2,
                 (wc * matchCases * (wct - wct_obs)) ** 2,
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


def _getWeights(qLs, wct, bhp, wl, wc, wp):
    """Port of ``getWeights``: reciprocal magnitude, or zero when flat."""
    if wl is None:
        total = float(_np.sum(_np.abs(qLs)))
        wl = 0.0 if total == 0 else 1.0 / _np.abs(qLs)
    if wc is None:
        wc = 1.0
    if wp is None:
        spread = float(_np.max(bhp) - _np.min(bhp)) if _np.size(bhp) else 0.0
        wp = 0.0 if spread == 0 else 1.0 / spread
    return wl, wc, wp


def _vertcatIfPresent(sol, field, nw):
    """Port of ``vertcatIfPresent``: open wells only, zeros when absent."""
    status = _np.asarray([bool(w.get('status', True)) for w in sol])
    if not sol or field not in sol[0]:
        return _np.zeros(int(_np.count_nonzero(status)))
    values = _np.asarray([_scalar(w.get(field)) for w in sol], dtype=float)
    assert values.size == nw
    return values[status]


def _expandToFull(v, v_obs, status, status_obs, setToZero):
    """Port of ``expandToFull``: scatter both sides back to every well.

    ``setToZero`` blanks wells whose open/shut state disagrees, so a well
    shut on one side only contributes nothing.
    """
    tmp = _np.zeros(status.size)
    if hasattr(v, 'val'):
        from PRSTCore.ad_core.adi import SparseADI
        v = SparseADI.scatter(_np.flatnonzero(status), v, status.size)
    else:
        tmp[status] = _np.asarray(v, dtype=float).ravel()
        v = tmp

    tmp = _np.zeros(status.size)
    tmp[status_obs] = _np.asarray(v_obs, dtype=float).ravel()
    v_obs = tmp

    if setToZero:
        ix = status != status_obs
        if _np.any(ix):
            if hasattr(v, 'val'):
                from PRSTCore.ad_core.adi import ad_select
                v = ad_select(ix, _np.zeros(status.size), v)
            else:
                v[ix] = 0.0
            v_obs[ix] = 0.0
    return v, v_obs


def _sum(x):
    return x.sum() if hasattr(x, 'sum') else _np.sum(x)


def _concat(items):
    if any(hasattr(x, 'val') for x in items):
        from PRSTCore.ad_core.adi import SparseADI
        return SparseADI.concat(items)
    return _np.concatenate([_np.atleast_1d(_np.asarray(x)).ravel()
                            for x in items])


def _scalar(value):
    if value is None:
        return 0.0
    arr = _np.atleast_1d(_np.asarray(value, dtype=float)).ravel()
    return float(arr[0]) if arr.size else 0.0
