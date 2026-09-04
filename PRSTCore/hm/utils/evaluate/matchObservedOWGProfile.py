"""FAHM's seven-term history-matching objective.

This is the sole PRST implementation of ``matchObservedOWGProfile.m``.
The numerical value and ``ComputePartials`` path share exactly the same
expression, so an AD result's one-row Jacobian is ``dg/dx`` for the value
being minimized rather than for a parallel approximation.

Confirmed FAHM source defects are corrected under the frozen migration
policy: the tracer cell-to-well matrix contains ones rather than one-based
well numbers, BHP masking is local to each report step, and a partial for
report step ``tStep`` resolves that report step's control. Inputs with a
wrong width fail explicitly; this objective never pads or trims vectors.
"""

from __future__ import annotations

import numpy as _np
import scipy.sparse as _sp

from PRSTCore.ad_core.adi import SparseADI as _SparseADI
from PRSTCore.ad_core.adi import is_ad as _is_ad
from PRSTCore.ad_core.utils.getPerforationToWellMapping import \
    getPerforationToWellMapping

from .getPhaseFlux import getPhaseFlux

ATM = 101325.0

TERM_KEYS = ('ww', 'wo', 'wg', 'wp', 'wt', 'wf', 'ws')
TERM_NAMES = {
    'ww': 'Water', 'wo': 'Oil', 'wg': 'Gas', 'wp': 'BHP',
    'wt': 'Tracer', 'wf': 'Profile', 'ws': 'Saturation',
}
_DEFAULT_ALPHA = {
    'ww': 1.0, 'wo': 1.0, 'wg': 1.0, 'wp': 1.0,
    'wt': 0.0, 'wf': 0.0, 'ws': 0.0,
}


def matchObservedOWGProfile(model, states, schedule, observed,
                            ObjectiveWeight=None, NormalizationFactor=None,
                            WellsWeight=None, ComputePartials=False,
                            tStep=None, state=None, from_states=True,
                            return_breakdown=False):
    """Return one scalar mismatch per selected report step.

    ``tStep`` uses PRST's zero-based report-step convention and must be a
    scalar, matching FAHM's adjoint callback usage. With
    ``return_breakdown=True`` the return value is ``(objective, breakdown)``;
    each breakdown entry contains seven ``dt/(T*nw)``-scaled per-well
    summands whose sum is the corresponding objective scalar.
    """
    phase_names = _phase_names(model)
    nphase = len(phase_names)
    nc = int(_get(_get(model.G, 'cells'), 'num'))

    step_data = schedule.get('step') or {}
    dts = _np.asarray(step_data.get('val', []), dtype=float).ravel()
    if dts.size == 0:
        raise ValueError('schedule.step.val must contain at least one step')
    if not _np.all(_np.isfinite(dts)) or _np.any(dts <= 0.0):
        raise ValueError('schedule.step.val must contain finite positive steps')
    total_time = float(dts.sum())

    controls = schedule.get('control') or []
    if not isinstance(controls, (list, tuple)) or not controls:
        raise ValueError('schedule.control must contain at least one control')
    W = _get(controls[-1], 'W') or []
    nw = len(W)
    if nw == 0:
        raise ValueError('matchObservedOWGProfile requires at least one well')

    p2w = getPerforationToWellMapping(W)
    nwc = int(p2w.size)
    # FIX-006: source data=p2w multiplies the second well by two, etc.
    cmap = _sp.csr_matrix(
        (_np.ones(nwc), (p2w, _np.arange(nwc, dtype=int))),
        shape=(nw, nwc),
    )
    w_cells = _well_cells(W, nwc)

    report_steps = _report_steps(tStep, dts.size)
    if not ComputePartials and len(states) < dts.size:
        raise ValueError('states must contain one entry per schedule step')
    if len(observed) < dts.size:
        raise ValueError('observed must contain one entry per schedule step')

    alpha = _alpha(ObjectiveWeight)
    beta = _beta(NormalizationFactor)
    omega = _omega(WellsWeight, nw)
    omega_wp = omega['wp'].copy()

    objective = []
    breakdown = []
    for report_step in report_steps:
        obs_entry = observed[report_step]
        sol_obs = _well_solutions(obs_entry, nw, 'observed', report_step)
        status_obs = _statuses(sol_obs, nw, 'observed', report_step)
        qWs_obs = _vertcat_if_present(sol_obs, 'qWs', nw)
        qOs_obs = _vertcat_if_present(sol_obs, 'qOs', nw)
        qGs_obs = _vertcat_if_present(sol_obs, 'qGs', nw)
        bhp_obs = _vertcat_if_present(sol_obs, 'bhp', nw)

        if ComputePartials:
            if from_states:
                if len(states) <= report_step:
                    raise ValueError(
                        'states does not contain requested tStep %d'
                        % report_step)
                st = model.getStateAD(states[report_step], True)
            else:
                if state is None:
                    raise ValueError(
                        'ComputePartials with from_states=False requires state')
                st = state
            qWs = _get_prop_if_present(model, st, 'qWs')
            qOs = _get_prop_if_present(model, st, 'qOs')
            qGs = _get_prop_if_present(model, st, 'qGs')
            bhp = _get_prop_if_present(model, st, 'bhp')
        else:
            st = states[report_step]
            sol = _well_solutions(st, nw, 'state', report_step)
            qWs = _vertcat_if_present(sol, 'qWs', nw)
            qOs = _vertcat_if_present(sol, 'qOs', nw)
            qGs = _vertcat_if_present(sol, 'qGs', nw)
            bhp = _vertcat_if_present(sol, 'bhp', nw)

        sol = _well_solutions(st, nw, 'state', report_step)
        status = _statuses(sol, nw, 'state', report_step)
        expected_open = int(_np.count_nonzero(status))
        qWs = _require_width(qWs, expected_open, 'qWs')
        qOs = _require_width(qOs, expected_open, 'qOs')
        qGs = _require_width(qGs, expected_open, 'qGs')
        bhp = _require_width(bhp, expected_open, 'bhp')

        if not status.all() or not status_obs.all():
            qWs, qWs_obs = _expand_to_full(
                qWs, qWs_obs, status, status_obs, False)
            qOs, qOs_obs = _expand_to_full(
                qOs, qOs_obs, status, status_obs, False)
            qGs, qGs_obs = _expand_to_full(
                qGs, qGs_obs, status, status_obs, False)
            bhp, bhp_obs = _expand_to_full(
                bhp, bhp_obs, status, status_obs, True)

        # FIX-026: resolve the selected report step, not the one-element
        # loop's local index when tStep is nonzero.
        if alpha['wf'] > 0.0 and _all_have(sol_obs, 'cqs'):
            ctrl = _control_for_report(schedule, report_step)
            driving = model.getDrivingForces(ctrl)
            fstruct = driving[-1] if isinstance(driving, tuple) else driving
            validated = model.validateModel(fstruct)
            if validated is not None:
                model = validated
            cqs_obs = _stack_perforation_field(
                sol_obs, W, 'cqs', nwc, nphase)
            cqs = getPhaseFlux(model.FacilityModel, st)
            if len(cqs) != nphase:
                raise ValueError(
                    'PhaseFlux returned %d phases; expected %d'
                    % (len(cqs), nphase))
            for phase, values in zip(phase_names, cqs):
                _require_width(values, nwc, 'PhaseFlux[%s]' % phase)
        else:
            cqs_obs = _np.zeros((nwc, nphase))
            cqs = [_np.zeros(nwc) for _ in range(nphase)]

        if alpha['ws'] > 0.0 and _all_have(sol_obs, 'sw'):
            sWs_obs = _stack_perforation_field(
                sol_obs, W, 'sw', nwc, nphase)
            sW_cells = _take(model.getProp(st, 'sW'), w_cells)
            _require_width(sW_cells, nwc, 'sW(wCells)')
            sWs = [sW_cells]
        else:
            sWs_obs = _np.zeros((nwc, 1))
            sWs = [_np.zeros(nwc)]

        if alpha['wt'] > 0.0 and _is_tracer_model(model):
            cTs_obs = _stack_well_vectors(sol_obs, 'tracer', nw)
            cTs = _tracer_components(model.getProp(st, 'tracer'), nc)
            if cTs_obs.shape[1] != len(cTs):
                raise ValueError(
                    'observed tracer columns (%d) and model tracers (%d) '
                    'must agree' % (cTs_obs.shape[1], len(cTs)))
        else:
            cTs_obs = _np.zeros((nw, 1))
            cTs = [_np.zeros(nc)]
        cTs = [_linear_map(cmap, _take(c, w_cells)) for c in cTs]

        # FIX-005: a step-local BHP mask, without mutating caller omega.
        local_wp = _np.where(bhp_obs <= ATM, 0.0, omega_wp)
        W_misfit = alpha['ww'] * omega['ww'] * (
            _well_scale(beta['ww'], nw, 'beta.ww') *
            (qWs - qWs_obs)) ** 2
        O_misfit = alpha['wo'] * omega['wo'] * (
            _well_scale(beta['wo'], nw, 'beta.wo') *
            (qOs - qOs_obs)) ** 2
        G_misfit = alpha['wg'] * omega['wg'] * (
            _well_scale(beta['wg'], nw, 'beta.wg') *
            (qGs - qGs_obs)) ** 2
        P_misfit = alpha['wp'] * local_wp * (
            _well_scale(beta['wp'], nw, 'beta.wp') *
            (bhp - bhp_obs)) ** 2

        T_misfit = _np.zeros(nw)
        if _np.any(cTs_obs.sum(axis=1) != 0.0):
            for k, predicted in enumerate(cTs):
                T_misfit = T_misfit + alpha['wt'] * omega['wt'] * (
                    _well_scale(beta['wt'], nw, 'beta.wt') *
                    (predicted - cTs_obs[:, k])) ** 2

        F_misfit = _np.zeros(nwc)
        if _np.any(cqs_obs.sum(axis=1) != 0.0):
            omega_wf = omega['wf'][p2w]
            for k, phase in enumerate(phase_names):
                beta_wf = _cell_rate_weights(beta, p2w, phase, nw)
                F_misfit = F_misfit + alpha['wf'] * omega_wf * (
                    beta_wf * (cqs[k] - cqs_obs[:, k])) ** 2

        S_misfit = _np.zeros(nwc)
        saturation_rows = sWs_obs.sum(axis=1) != 0.0
        if _np.any(saturation_rows):
            omega_ws = omega['ws'][p2w]
            for k, predicted in enumerate(sWs):
                S_misfit = S_misfit + alpha['ws'] * omega_ws * (
                    saturation_rows *
                    (predicted - sWs_obs[:, k])) ** 2

        factor = float(dts[report_step]) / total_time / nw
        per_well = {
            'ww': W_misfit * factor,
            'wo': O_misfit * factor,
            'wg': G_misfit * factor,
            'wp': P_misfit * factor,
            'wt': T_misfit * factor,
            'wf': _linear_map(cmap, F_misfit) * factor,
            'ws': _linear_map(cmap, S_misfit) * factor,
        }
        objective.append(_sum_terms(per_well.values()))
        breakdown.append(per_well)

    return (objective, breakdown) if return_breakdown else objective


def _report_steps(t_step, nsteps):
    if t_step is None:
        return list(range(nsteps))
    values = _np.atleast_1d(_np.asarray(t_step, dtype=int)).ravel()
    if values.size != 1:
        raise ValueError('tStep must select exactly one report step')
    step = int(values[0])
    if step < 0 or step >= nsteps:
        raise IndexError('tStep %d is outside %d report steps' % (step, nsteps))
    return [step]


def _phase_names(model):
    getter = getattr(model, 'getPhaseNames', None)
    if callable(getter):
        names = list(getter())
    else:
        names = [name for name, attr in
                 (('W', 'water'), ('O', 'oil'), ('G', 'gas'))
                 if bool(getattr(model, attr, False))]
    names = [str(name).upper() for name in names]
    if not names or any(name not in ('W', 'O', 'G') for name in names):
        raise ValueError('model must define an active W/O/G phase ordering')
    active_getter = getattr(model, 'getActivePhases', None)
    if callable(active_getter):
        active_count = int(_np.count_nonzero(active_getter()))
        if active_count != len(names):
            raise ValueError('active phase mask and phase names disagree')
    return names


def _alpha(value):
    out = dict(_DEFAULT_ALPHA) if value is None else dict(_as_dict(value))
    missing = [key for key in TERM_KEYS if key not in out]
    if missing:
        raise ValueError('ObjectiveWeight is missing: %s' % ', '.join(missing))
    for key in TERM_KEYS:
        out[key] = float(out[key])
        if not _np.isfinite(out[key]):
            raise ValueError('ObjectiveWeight.%s must be finite' % key)
    return out


def _beta(value):
    if value is None:
        raise ValueError(
            'matchObservedOWGProfile requires NormalizationFactor: FAHM '
            'reads beta.ww unconditionally and defines no default')
    out = dict(_as_dict(value))
    missing = [key for key in ('ww', 'wo', 'wg', 'wp', 'wt')
               if key not in out]
    if missing:
        raise ValueError(
            'NormalizationFactor is missing: %s' % ', '.join(missing))
    return out


def _omega(value, nw):
    source = ({key: _np.ones(nw) for key in TERM_KEYS}
              if value is None else dict(_as_dict(value)))
    missing = [key for key in TERM_KEYS if key not in source]
    if missing:
        raise ValueError('WellsWeight is missing: %s' % ', '.join(missing))
    return {key: _well_scale(source[key], nw, 'omega.' + key).copy()
            for key in TERM_KEYS}


def _well_scale(value, nw, name):
    out = _np.asarray(value, dtype=float).ravel()
    if out.size == 1:
        out = _np.full(nw, float(out[0]))
    if out.size != nw:
        raise ValueError('%s has width %d; expected 1 or %d'
                         % (name, out.size, nw))
    if not _np.all(_np.isfinite(out)):
        raise ValueError('%s must contain only finite values' % name)
    return out


def _well_cells(wells, expected):
    arrays = [_np.asarray(_get(well, 'cells'), dtype=int).ravel()
              for well in wells]
    out = _np.concatenate(arrays) if arrays else _np.zeros(0, dtype=int)
    if out.size != expected:
        raise ValueError('well completion count and perforation map disagree')
    return out


def _well_solutions(container, nw, kind, step):
    sol = _get(container, 'wellSol')
    if sol is None:
        raise ValueError('%s step %d has no wellSol' % (kind, step))
    if len(sol) != nw:
        raise ValueError('%s step %d has %d wells; expected %d'
                         % (kind, step, len(sol), nw))
    return sol


def _statuses(sol, nw, kind, step):
    if any(_get(well, 'status') is None for well in sol):
        raise ValueError('%s step %d has a missing well status' % (kind, step))
    out = _np.asarray([bool(_get(well, 'status')) for well in sol])
    if out.size != nw:
        raise ValueError('%s status width does not match wells' % kind)
    return out


def _vertcat_if_present(sol, field, nw):
    status = _np.asarray([bool(_get(w, 'status')) for w in sol])
    present = [_get(w, field) is not None for w in sol]
    if not any(present):
        return _np.zeros(int(_np.count_nonzero(status)))
    if not all(present):
        raise ValueError('%s must be present for every well or none' % field)
    raw = [_np.asarray(_get(well, field), dtype=float).ravel()
           for well in sol]
    if all(value.size == 0 for value in raw):
        return _np.zeros(int(_np.count_nonzero(status)))
    values = []
    for value in raw:
        if value.size != 1:
            raise ValueError(
                '%s must contain one scalar per well; got %d values'
                % (field, value.size))
        values.append(float(value[0]))
    out = _np.asarray(values)
    if out.size != nw:
        raise ValueError('%s has width %d; expected %d'
                         % (field, out.size, nw))
    return out[status]


def _get_prop_if_present(model, state, field):
    try:
        return model.FacilityModel.getProp(state, field)
    except Exception:
        status = _np.asarray([bool(_get(w, 'status'))
                              for w in state['wellSol']])
        return _np.zeros(int(_np.count_nonzero(status)))


def _require_width(value, expected, name):
    size = value.val.size if _is_ad(value) else _np.asarray(value).size
    if size != expected:
        raise ValueError('%s has width %d; expected %d; padding/trimming is '
                         'forbidden' % (name, size, expected))
    return value if _is_ad(value) else _np.asarray(value, dtype=float).ravel()


def _expand_to_full(value, observed, status, status_obs, zero_mismatch):
    if _is_ad(value):
        value = _SparseADI.scatter(_np.flatnonzero(status), value, status.size)
    else:
        source = _np.asarray(value, dtype=float).ravel()
        if source.size != int(_np.count_nonzero(status)):
            raise ValueError('state open-well vector has the wrong width')
        full = _np.zeros(status.size)
        full[status] = source
        value = full

    observed = _np.asarray(observed, dtype=float).ravel()
    if observed.size != int(_np.count_nonzero(status_obs)):
        raise ValueError('observed open-well vector has the wrong width')
    full_observed = _np.zeros(status_obs.size)
    full_observed[status_obs] = observed
    observed = full_observed

    if zero_mismatch:
        different = status != status_obs
        if _np.any(different):
            if _is_ad(value):
                from PRSTCore.ad_core.adi import ad_select
                value = ad_select(different, _np.zeros(status.size), value)
            else:
                value[different] = 0.0
            observed[different] = 0.0
    return value, observed


def _all_have(sol, field):
    return bool(sol) and all(_get(well, field) is not None for well in sol)


def _stack_perforation_field(sol, wells, field, nrows, ncols):
    blocks = []
    for index, (well_sol, well) in enumerate(zip(sol, wells)):
        nperf = _np.asarray(_get(well, 'cells')).size
        values = _np.asarray(_get(well_sol, field), dtype=float)
        if values.ndim == 1 and ncols == 1:
            values = values.reshape(-1, 1)
        if values.shape != (nperf, ncols):
            raise ValueError(
                '%s for well %d has shape %s; expected (%d, %d)'
                % (field, index, values.shape, nperf, ncols))
        blocks.append(values)
    out = _np.vstack(blocks) if blocks else _np.zeros((0, ncols))
    if out.shape != (nrows, ncols):
        raise ValueError('%s stacked shape %s; expected (%d, %d)'
                         % (field, out.shape, nrows, ncols))
    return out


def _stack_well_vectors(sol, field, nw):
    rows = []
    width = None
    for index, well in enumerate(sol):
        values = _np.asarray(_get(well, field), dtype=float).ravel()
        if width is None:
            width = values.size
        if values.size != width:
            raise ValueError('%s width differs at well %d' % (field, index))
        rows.append(values)
    if width is None or width == 0:
        return _np.zeros((nw, 0))
    out = _np.vstack(rows)
    if out.shape[0] != nw:
        raise ValueError('%s row count does not match wells' % field)
    return out


def _tracer_components(value, nc):
    if isinstance(value, (list, tuple)):
        out = list(value)
    elif _is_ad(value):
        out = [value]
    else:
        array = _np.asarray(value, dtype=float)
        if array.ndim <= 1:
            out = [array.ravel()]
        elif array.shape[0] == nc:
            out = [array[:, column] for column in range(array.shape[1])]
        else:
            raise ValueError('tracer matrix must have one row per cell')
    for index, component in enumerate(out):
        _require_width(component, nc, 'tracer[%d]' % index)
    return out


def _take(value, indices):
    return value[indices] if _is_ad(value) else \
        _np.asarray(value, dtype=float).ravel()[indices]


def _linear_map(matrix, value):
    if _is_ad(value):
        return value.linear_map(matrix)
    return _np.asarray(matrix @ _np.asarray(value, dtype=float).ravel()).ravel()


def _cell_rate_weights(beta, p2w, phase, nw):
    field = 'w' + str(phase).lower()
    if field not in beta:
        raise ValueError('NormalizationFactor is missing: %s' % field)
    values = _np.asarray(beta[field], dtype=float).ravel()
    if values.size == 1:
        return _np.full(p2w.size, float(values[0]))
    if values.size != nw:
        raise ValueError('beta.%s has width %d; expected 1 or %d'
                         % (field, values.size, nw))
    return values[p2w]


def _control_for_report(schedule, report_step):
    controls = schedule['control']
    mapping = _np.asarray(schedule['step'].get('control', []), dtype=int).ravel()
    if mapping.size != _np.asarray(schedule['step']['val']).size:
        raise ValueError('schedule.step.control must map every report step')
    index = int(mapping[report_step])
    if index < 0 or index >= len(controls):
        raise IndexError('schedule.step.control[%d]=%d is invalid'
                         % (report_step, index))
    return controls[index]


def _is_tracer_model(model):
    return any(cls.__name__.endswith('TracerModel')
               for cls in type(model).__mro__)


def _sum_terms(values):
    out = 0.0
    for value in values:
        out = out + (value.sum() if hasattr(value, 'sum')
                     else _np.sum(value))
    return out


def _as_dict(value):
    if isinstance(value, dict):
        return value
    return {key: getattr(value, key) for key in dir(value)
            if not key.startswith('_')}


def _get(obj, key):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
