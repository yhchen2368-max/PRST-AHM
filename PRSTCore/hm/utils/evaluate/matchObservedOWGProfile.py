"""Port of MRST ``matchObservedOWGProfile.m`` (mrst-2026a/hm/utils/evaluate).

The mismatch FAHM itself uses. Unlike :mod:`matchObservedLW`, which works
on liquid rate and water cut, this one scores each phase rate separately
and can additionally score a well's per-perforation flux profile,
near-well water saturation, and tracer concentration::

    obj(step) = dt/(T*nw) * sum( a.ww*o.ww*(b.ww*(qWs - qWs_obs))^2
                               + a.wo*o.wo*(b.wo*(qOs - qOs_obs))^2
                               + a.wg*o.wg*(b.wg*(qGs - qGs_obs))^2
                               + a.wp*o.wp*(b.wp*(bhp - bhp_obs))^2
                               + tracer + flux profile + saturation )

``alpha`` turns each term on, ``omega`` weights each well, and ``beta``
normalises each quantity to comparable magnitude.

Two MATLAB behaviours are reproduced deliberately:

* ``NormalizationFactor`` (``beta``) has no default. The MATLAB reads
  ``beta.ww`` unconditionally, so omitting it is an error there too; this
  port raises rather than inventing a normalisation, since a silently
  chosen one would change every objective value.

* ``omega.wp(bhp_obs <= 1*atm) = 0`` mutates the caller's ``omega`` and is
  **not** reset between report steps. A well whose observed bhp drops to
  atmospheric at one step therefore stays excluded from the pressure term
  for every *later* step as well. That is sticky state in a function that
  reads as pure, but it is what MRST computes.
"""

import numpy as _np

from PRSTCore.hm.utils.controlIndex import control_index
import scipy.sparse as _sp

from PRSTCore.ad_core.utils.getPerforationToWellMapping import \
    getPerforationToWellMapping

from .getPhaseFlux import getPhaseFlux

ATM = 101325.0

_DEFAULT_ALPHA = {'ww': 1, 'wo': 1, 'wg': 1, 'wp': 1, 'wt': 0, 'wf': 0, 'ws': 0}


def matchObservedOWGProfile(model, states, schedule, observed,
                            ObjectiveWeight=None, NormalizationFactor=None,
                            WellsWeight=None, ComputePartials=False,
                            tStep=None, state=None, from_states=True):
    """Return one mismatch entry per requested report step."""
    active = model.getActivePhases()
    phNames = model.getPhaseNames()
    np_ = int(_np.count_nonzero(active))
    nc = int(_get(_get(model.G, 'cells'), 'num'))

    dts = _np.asarray(schedule['step']['val'], dtype=float).ravel()
    totTime = float(dts.sum())

    W = _control(schedule, -1)['W']
    nw = len(W)
    p2w = getPerforationToWellMapping(W)
    nwc = p2w.size
    cmap = _sp.csr_matrix((p2w.astype(float), (p2w, _np.arange(nwc))),
                          shape=(nw, nwc))
    wCells = _np.concatenate([_np.atleast_1d(_np.asarray(_get(w, 'cells')))
                              .ravel() for w in W]) if nw else _np.zeros(0, int)

    if tStep is None:
        tSteps = _np.arange(dts.size)
        numSteps = dts.size
    else:
        tSteps = _np.atleast_1d(_np.asarray(tStep, dtype=int)).ravel()
        numSteps = 1
        dts = dts[tSteps]

    alpha = dict(_DEFAULT_ALPHA) if ObjectiveWeight is None \
        else _as_dict(ObjectiveWeight)
    if WellsWeight is None:
        w = _np.ones(nw)
        omega = {k: w.copy() for k in ('ww', 'wo', 'wg', 'wp', 'wt', 'wf', 'ws')}
    else:
        omega = _as_dict(WellsWeight)
    if NormalizationFactor is None:
        raise ValueError(
            'matchObservedOWGProfile requires NormalizationFactor: the '
            'MATLAB reads beta.ww unconditionally and has no default.')
    beta = _as_dict(NormalizationFactor)

    obj = []
    for step in range(numSteps):
        sol_obs = observed[int(tSteps[step])]['wellSol']
        qWs_obs = _vertcatIfPresent(sol_obs, 'qWs', nw)
        qOs_obs = _vertcatIfPresent(sol_obs, 'qOs', nw)
        qGs_obs = _vertcatIfPresent(sol_obs, 'qGs', nw)
        bhp_obs = _vertcatIfPresent(sol_obs, 'bhp', nw)
        status_obs = _np.asarray([bool(_get(w, 'status')) for w in sol_obs])

        if ComputePartials:
            st = model.getStateAD(states[int(tSteps[step])], True) \
                if from_states else state
            qWs = _getPropIfPresent(model, st, 'qWs')
            qOs = _getPropIfPresent(model, st, 'qOs')
            qGs = _getPropIfPresent(model, st, 'qGs')
            bhp = _getPropIfPresent(model, st, 'bhp')
        else:
            st = states[int(tSteps[step])]
            qWs = _vertcatIfPresent(st['wellSol'], 'qWs', nw)
            qOs = _vertcatIfPresent(st['wellSol'], 'qOs', nw)
            qGs = _vertcatIfPresent(st['wellSol'], 'qGs', nw)
            bhp = _vertcatIfPresent(st['wellSol'], 'bhp', nw)
        status = _np.asarray([bool(_get(w, 'status')) for w in st['wellSol']])

        if not status.all() or not status_obs.all():
            qWs, qWs_obs = _expandToFull(qWs, qWs_obs, status, status_obs, False)
            qOs, qOs_obs = _expandToFull(qOs, qOs_obs, status, status_obs, False)
            qGs, qGs_obs = _expandToFull(qGs, qGs_obs, status, status_obs, False)
            bhp, bhp_obs = _expandToFull(bhp, bhp_obs, status, status_obs, True)

        # ------------- profile, saturation and tracer ---------------------
        if alpha['wf'] > 0 and _has(sol_obs, 'cqs'):
            # ``schedule.control(schedule.step.control(step))`` -- MRST
            # indexes straight through, with no adjustment at all.
            controls = schedule['control']
            ctrl = _control(schedule, control_index(
                schedule['step'], step,
                1 if isinstance(controls, dict) else len(controls)))
            fstruct = model.getDrivingForces(ctrl)
            model = model.validateModel(fstruct)
            cqs_obs = _stack(sol_obs, 'cqs', nwc, np_)
            cqs = getPhaseFlux(model.FacilityModel, st)
        else:
            cqs_obs = _np.zeros((nwc, np_))
            cqs = _np.zeros((nwc, np_))

        if alpha['ws'] > 0 and _has(sol_obs, 'sw'):
            sWs_obs = _stack(sol_obs, 'sw', nwc, 1)
            sWs = _np.atleast_2d(_np.asarray(
                model.getProp(st, 'sW'), dtype=float).ravel()[wCells]).T
        else:
            sWs_obs = _np.zeros((nwc, 1))
            sWs = _np.zeros((nwc, 1))

        if alpha['wt'] > 0 and _isTracerModel(model):
            cTs_obs = _stack(sol_obs, 'tracer', nw, 1)
            cTs = model.getProp(st, 'tracer')
            cTs = list(cTs) if isinstance(cTs, (list, tuple)) else [cTs]
        else:
            cTs_obs = _np.zeros((nw, 1))
            cTs = [_np.zeros(nc)]
        cTs = [cmap @ _np.asarray(c, dtype=float).ravel()[wCells] for c in cTs]

        # ------------------------------ objective -------------------------
        # Sticky by design -- see the module docstring.
        omega['wp'] = _np.where(_np.asarray(bhp_obs).ravel() <= ATM, 0.0,
                                _np.asarray(omega['wp'], dtype=float).ravel())

        W_misfit = alpha['ww'] * omega['ww'] * (beta['ww'] * (qWs - qWs_obs)) ** 2
        O_misfit = alpha['wo'] * omega['wo'] * (beta['wo'] * (qOs - qOs_obs)) ** 2
        G_misfit = alpha['wg'] * omega['wg'] * (beta['wg'] * (qGs - qGs_obs)) ** 2
        P_misfit = alpha['wp'] * omega['wp'] * (beta['wp'] * (bhp - bhp_obs)) ** 2

        T_misfit = 0
        if _np.any(cTs_obs.sum(axis=1) != 0):
            for k in range(len(cTs)):
                T_misfit = T_misfit + alpha['wt'] * omega['wt'] * (
                    beta['wt'] * (cTs[k] - cTs_obs[:, k])) ** 2

        F_misfit = 0
        if _np.any(cqs_obs.sum(axis=1) != 0):
            omega_wf = _np.asarray(omega['wf'], dtype=float).ravel()[p2w]
            for k in range(np_):
                beta_wf = _getCellRateWeights(beta, p2w, phNames[k])
                F_misfit = F_misfit + alpha['wf'] * omega_wf * (
                    beta_wf * (_col(cqs, k) - cqs_obs[:, k])) ** 2

        S_misfit = 0
        ix = sWs_obs.sum(axis=1) != 0
        if _np.any(ix):
            omega_ws = _np.asarray(omega['ws'], dtype=float).ravel()[p2w]
            for k in range(sWs.shape[1]):
                # MATLAB weights this term by ``ix`` itself, not by beta.
                S_misfit = S_misfit + alpha['ws'] * omega_ws * (
                    ix * (sWs[:, k] - sWs_obs[:, k])) ** 2

        dt = float(dts[step])
        obj.append(dt / totTime / max(nw, 1) * (
            _sum(W_misfit) + _sum(O_misfit) + _sum(G_misfit) + _sum(P_misfit)
            + _sum(T_misfit) + _sum(F_misfit) + _sum(S_misfit)))

    return obj


def _vertcatIfPresent(sol, field, nw):
    """Port of ``vertcatIfPresent``: open wells only, zeros when absent."""
    status = _np.asarray([bool(_get(w, 'status')) for w in sol])
    if not sol or not _has(sol, field):
        return _np.zeros(int(_np.count_nonzero(status)))
    values = _np.asarray([_scalar(_get(w, field)) for w in sol], dtype=float)
    if values.size == 0:
        return _np.zeros(int(_np.count_nonzero(status)))
    assert values.shape[0] == nw
    return values[status]


def _getPropIfPresent(model, state, field):
    """Port of ``getPropIfPresent``: zeros when the model has no such prop."""
    try:
        return model.FacilityModel.getProp(state, field)
    except Exception:
        status = _np.asarray([bool(_get(w, 'status'))
                              for w in state['wellSol']])
        return _np.zeros(int(_np.count_nonzero(status)))


def _expandToFull(v, v_obs, status, status_obs, setToZero):
    """Port of ``expandToFull``: scatter both sides back to every well."""
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


def _getCellRateWeights(beta, p2w, phase):
    """Port of ``getCellRateWeights``: expand a per-well weight to
    perforations."""
    wf = beta['w' + str(phase).lower()]
    wf = _np.atleast_1d(_np.asarray(wf, dtype=float)).ravel()
    if wf.size > 1:
        assert wf.size == int(p2w.max()) + 1
        wf = wf[p2w]
    return wf


def _control(schedule, index):
    controls = schedule['control']
    return controls[index] if not isinstance(controls, dict) else controls


def _stack(sol, field, nrows, ncols):
    rows = [_np.atleast_1d(_np.asarray(_get(w, field), dtype=float)).ravel()
            for w in sol if _get(w, field) is not None]
    if not rows:
        return _np.zeros((nrows, ncols))
    out = _np.vstack([r.reshape(1, -1) if r.ndim == 1 else r for r in rows])
    return out.reshape(-1, out.shape[-1])


def _col(x, k):
    if isinstance(x, (list, tuple)):
        return x[k]
    x = _np.atleast_2d(x)
    return x[:, k]


def _has(sol, field):
    return bool(sol) and (field in sol[0] if isinstance(sol[0], dict)
                          else hasattr(sol[0], field))


def _isTracerModel(model):
    return type(model).__name__.endswith('TracerModel')


def _sum(x):
    return x.sum() if hasattr(x, 'sum') else _np.sum(x)


def _scalar(value):
    if value is None:
        return 0.0
    arr = _np.atleast_1d(_np.asarray(value, dtype=float)).ravel()
    return float(arr[0]) if arr.size else 0.0


def _as_dict(obj):
    return obj if isinstance(obj, dict) else \
        {k: getattr(obj, k) for k in dir(obj) if not k.startswith('_')}


def _get(obj, key):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
