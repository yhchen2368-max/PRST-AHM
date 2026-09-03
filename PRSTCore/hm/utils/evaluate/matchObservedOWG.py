"""Port of MRST ``matchObservedOWG.m``
(mrst-2026a/autodiff/optimization/objectives).

Three-phase rate and bhp mismatch::

    obj(step) = dt/(T*nMatched) * sum( (ww*(qWs - qWs_obs))^2
                                     + (wo*(qOs - qOs_obs))^2
                                     + (wg*(qGs - qGs_obs))^2
                                     + (wp*(bhp - bhp_obs))^2 )

Unlike :mod:`matchObservedOG`, whose weights are per well, these are
**scalars** covering every well at once, and water and oil share one
normaliser::

    rw = sum(|qWs_obs| + |qOs_obs|)      ww = wo = 1/rw
    rg = sum(|qGs_obs|)                  wg = 1/rg
    wp = 1/(max(bhp_obs) - min(bhp_obs))

Sharing ``rw`` between water and oil is deliberate: it makes the two
liquid terms commensurate with each other, so a field producing mostly
water does not have its oil mismatch scaled up out of proportion. A
phase whose observed rates are all zero gets weight zero rather than a
division by zero, and likewise the pressure term when every bhp is equal.

**Defect: matchOnlyProducers cannot be used.** The MATLAB reads

    matchCases = (vertcat(sol.sign) < 0);

but ``sol`` is never assigned in this function -- the observed container
is ``sol_obs``. Passing ``matchOnlyProducers`` therefore raises in MATLAB
rather than selecting producers. Since the branch cannot run at all, this
port fills in the evident target (``sol_obs.wellSol``) rather than
reproducing an error; the intent is unambiguous, and reproducing it would
leave the option permanently unusable.
"""

import numpy as _np
import scipy.sparse as _sp

from .matchObservedLW import _concat, _sum


def matchObservedOWG(model, states, schedule, observed, WaterRateWeight=None,
                     OilRateWeight=None, GasRateWeight=None, BHPWeight=None,
                     ComputePartials=False, tStep=None, state=None,
                     from_states=True, matchOnlyProducers=False,
                     mismatchSum=True, accumulateWells=None,
                     accumulateTypes=None, return_match_map=False):
    """Return one mismatch entry per requested report step.

    With ``return_match_map`` the second output is MRST's ``matchMap``,
    which that function always leaves empty.
    """
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
    matchMap = None
    for step in range(numSteps):
        sol_obs = observed[int(tSteps[step])]['wellSol']
        nw = len(sol_obs)

        if matchOnlyProducers:
            # See the module docstring: MRST reads an undefined `sol` here.
            matchCases = _np.asarray([float(w.get('sign', 0.0))
                                      for w in sol_obs]) < 0
        else:
            matchCases = _np.ones(nw, dtype=bool)

        qWs_obs = _vertcatIfPresent(sol_obs, 'qWs', nw)
        qOs_obs = _vertcatIfPresent(sol_obs, 'qOs', nw)
        qGs_obs = _vertcatIfPresent(sol_obs, 'qGs', nw)
        bhp_obs = _vertcatIfPresent(sol_obs, 'bhp', nw)
        status_obs = _np.asarray([bool(w.get('status', True))
                                  for w in sol_obs])

        ww, wo, wg, wp = _getWeights(qWs_obs, qOs_obs, qGs_obs, bhp_obs,
                                     WaterRateWeight, OilRateWeight,
                                     GasRateWeight, BHPWeight)

        if ComputePartials:
            st = model.getStateAD(states[int(tSteps[step])], True) \
                if from_states else state
            qWs = model.FacilityModel.getProp(st, 'qWs')
            qOs = model.FacilityModel.getProp(st, 'qOs')
            qGs = model.FacilityModel.getProp(st, 'qGs')
            bhp = model.FacilityModel.getProp(st, 'bhp')
            assert hasattr(qWs, 'val'), 'ComputePartials requires AD state'
        else:
            st = states[int(tSteps[step])]
            qWs = _vertcatIfPresent(st['wellSol'], 'qWs', nw)
            qOs = _vertcatIfPresent(st['wellSol'], 'qOs', nw)
            qGs = _vertcatIfPresent(st['wellSol'], 'qGs', nw)
            bhp = _vertcatIfPresent(st['wellSol'], 'bhp', nw)
            assert not hasattr(qWs, 'val')
        status = _np.asarray([bool(w.get('status', True))
                              for w in st['wellSol']])

        if not status.all() or not status_obs.all():
            bhp, bhp_obs = _expandToFull(bhp, bhp_obs, status, status_obs,
                                         True)
            qWs, qWs_obs = _expandToFull(qWs, qWs_obs, status, status_obs,
                                         False)
            qOs, qOs_obs = _expandToFull(qOs, qOs_obs, status, status_obs,
                                         False)
            qGs, qGs_obs = _expandToFull(qGs, qGs_obs, status, status_obs,
                                         False)

        dt = float(dts[step])
        fac = dt / (totTime * max(int(_np.count_nonzero(matchCases)), 1))

        mm = [(ww * matchCases * (qWs - qWs_obs)) ** 2,
              (wo * matchCases * (qOs - qOs_obs)) ** 2,
              (wg * matchCases * (qGs - qGs_obs)) ** 2,
              (wp * matchCases * (bhp - bhp_obs)) ** 2]

        if mismatchSum:
            obj.append(fac * sum(_sum(t) for t in mm))
            continue

        mm = [fac * t for t in mm]
        if accumulateTypes is None:
            tmp = mm
        else:
            pt = _np.atleast_1d(_np.asarray(accumulateTypes,
                                            dtype=int)).ravel()
            tmp = [0] * int(pt.max())
            for k in range(4):
                if pt[k] > 0:
                    tmp[pt[k] - 1] = tmp[pt[k] - 1] + mm[k]
        if accumulateWells is not None:
            pw = _np.atleast_1d(_np.asarray(accumulateWells,
                                            dtype=int)).ravel()
            keep = _np.flatnonzero(pw)
            M = _sp.csr_matrix((_np.ones(keep.size), (pw[keep] - 1, keep)),
                               shape=(int(pw.max()), pw.size))
            tmp = [M @ x for x in tmp]
        obj.append(_concat(tmp))

    return (obj, matchMap) if return_match_map else obj


def _getWeights(qWs, qOs, qGs, bhp, ww, wo, wg, wp):
    """Port of ``getWeights``: scalar weights, water and oil sharing one.

    Each defaults to the reciprocal of its phase's total observed
    magnitude, or zero when that total is zero.
    """
    rw = float(_np.sum(_np.abs(qWs) + _np.abs(qOs)))
    rg = float(_np.sum(_np.abs(qGs)))

    if ww is None:
        ww = 0.0 if float(_np.sum(_np.abs(qWs))) == 0 else 1.0 / rw
    if wo is None:
        wo = 0.0 if float(_np.sum(_np.abs(qOs))) == 0 else 1.0 / rw
    if wg is None:
        wg = 0.0 if float(_np.sum(_np.abs(qGs))) == 0 else 1.0 / rg
    if wp is None:
        dp = float(_np.max(bhp) - _np.min(bhp)) if _np.size(bhp) else 0.0
        wp = 0.0 if dp == 0 else 1.0 / dp
    return float(ww), float(wo), float(wg), float(wp)


def _vertcatIfPresent(sol, field, nw):
    """Port of ``vertcatIfPresent``: open wells only, zeros when absent.

    This variant asserts the field covers every well; unlike
    matchObservedOG's it has no empty-vector shortcut.
    """
    status = _np.asarray([bool(w.get('status', True)) for w in sol])
    if not sol or field not in sol[0]:
        return _np.zeros(int(_np.count_nonzero(status)))
    values = _np.asarray([_scalar(w.get(field)) for w in sol], dtype=float)
    assert values.size == nw, 'field %r does not cover all %d wells' % (field,
                                                                        nw)
    return values[status]


def _expandToFull(v, v_obs, status, status_obs, setToZero):
    """Port of ``expandToFull``: scatter both sides over every well."""
    if hasattr(v, 'val'):
        from PRSTCore.ad_core.adi import SparseADI
        v = SparseADI.scatter(_np.flatnonzero(status), v, status.size)
    else:
        tmp = _np.zeros(status.size)
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


def _scalar(value):
    if value is None:
        return 0.0
    arr = _np.atleast_1d(_np.asarray(value, dtype=float)).ravel()
    return float(arr[0]) if arr.size else 0.0
