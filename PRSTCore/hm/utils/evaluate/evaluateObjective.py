"""Port of MRST ``evaluateObjective.m`` (mrst-2026a/hm/utils/evaluate).

Evaluates a history-matching objective at a scaled parameter vector, and
optionally its gradient.

The parameter vector lives in the unit box: each parameter's ``unscale``
maps its slice back to physical units before the simulation runs, and
``scaleGradient`` maps the gradient back into the scaled space afterwards,
so the optimiser only ever sees ``[0, 1]``.

Two gradient methods, as in the MATLAB:

``'AdjointAD'``          one adjoint sweep (the default);
``'PerturbationADNUM'``  one extra forward simulation per parameter,
                         forward-differenced;
``'none'``               objective only.
"""

import numpy as _np


def evaluateObjective(pvec, obj, setup, parameters, Verbose=False,
                      Gradient='AdjointAD', NonlinearSolver=None,
                      AdjointLinearSolver=None, PerturbationSize=1e-7,
                      objScaling=1.0, enforceBounds=True,
                      return_gradient=False, return_states=False,
                      return_setup=False, **extra):
    """Return the objective value, plus whatever else was requested.

    MATLAB signals the extras through ``nargout``; here they are explicit
    flags, and the return is a tuple in the same order:
    ``(objVal[, gradient][, wellSols, states][, setupNew])``.
    """
    from PRSTCore.ad_core.simulators.simulate_schedule_ad import simulate_schedule_ad

    nparam = [int(_get(p, 'nParam')) for p in parameters]
    p_org = _np.asarray(pvec, dtype=float).ravel()
    pvec = _np.clip(p_org, 0.0, 1.0) if enforceBounds else p_org.copy()
    slices = _slices(nparam)

    setupNew = _shallow_setup(setup)
    pval = []
    for k, p in enumerate(parameters):
        value = _get(p, 'unscale')(pvec[slices[k]])
        pval.append(value)
        setupNew = _get(p, 'setParameter')(setupNew, value)

    wellSols, states = simulate_schedule_ad(
        setupNew['state0'], setupNew['model'], setupNew['schedule'],
        nonlinear_solver=NonlinearSolver, **extra)

    objVals = obj(setupNew['model'], states, setupNew['schedule'],
                  False, None, None, True)
    objVal = float(_np.sum(_np.concatenate(
        [_np.atleast_1d(_np.asarray(v, dtype=float)).ravel()
         for v in objVals]))) / objScaling

    out = [objVal]

    if return_gradient:
        out.append(_gradient(Gradient, obj, setup, setupNew, parameters, pval,
                             p_org, objVal, objScaling, PerturbationSize,
                             states, NonlinearSolver, AdjointLinearSolver,
                             slices, nparam))
    if return_states:
        out.extend([wellSols, states])
    if return_setup:
        out.append(setupNew)

    return out[0] if len(out) == 1 else tuple(out)


def _gradient(method, obj, setup, setupNew, parameters, pval, p_org, objVal,
              objScaling, eps_pert, states, nls, adjoint_solver, slices,
              nparam):
    if method == 'none':
        return None

    if method == 'AdjointAD':
        # MRST keeps this in ad-core, not hm; PRSTCore mirrors that path.
        # It takes no LinearSolver argument, so opt.AdjointLinearSolver has
        # nowhere to go -- the MATLAB passes it straight through.
        from PRSTCore.ad_core.simulators import compute_sensitivities_adjoint_ad

        def objh(tstep, model, state):
            return obj(setupNew['model'], states, setupNew['schedule'],
                       True, tstep, state, False)

        gradient = compute_sensitivities_adjoint_ad(
            setupNew, states, parameters, objh)
        scaled = []
        for k, p in enumerate(parameters):
            name = _get(p, 'name')
            scaled.append(_np.atleast_1d(_np.asarray(
                _get(p, 'scaleGradient')(gradient[name], pval[k]),
                dtype=float)).ravel())
        return _np.concatenate(scaled) / objScaling

    if method == 'PerturbationADNUM':
        # One extra forward simulation per parameter, forward-differenced.
        val = _np.empty(p_org.size)
        for i in range(p_org.size):
            val[i] = evaluateObjective(
                _perturb(p_org, i, eps_pert), obj, setup, parameters,
                Gradient='none', NonlinearSolver=nls, objScaling=objScaling,
                enforceBounds=False)
        gradient = (val - objVal) * objScaling / eps_pert
        return gradient / objScaling

    raise ValueError('Gradient method %s is not implemented' % method)


def _perturb(p_org, i, eps_pert):
    """Port of the local ``perturb``."""
    out = _np.array(p_org, dtype=float, copy=True)
    out[i] += eps_pert
    return out


def _slices(nparam):
    bounds = _np.concatenate([[0], _np.cumsum(nparam)]).astype(int)
    return [slice(bounds[i], bounds[i + 1]) for i in range(len(nparam))]


def _shallow_setup(setup):
    """Copy the setup and clear the cached state-function containers.

    MATLAB blanks FlowDiscretization/FlowPropertyFunctions/
    PVTPropertyFunctions so the model rebuilds them against the new
    parameter values instead of reusing the previous evaluation's.
    """
    out = dict(setup)
    model = out['model']
    for field in ('FlowDiscretization', 'FlowPropertyFunctions',
                  'PVTPropertyFunctions'):
        if hasattr(model, field):
            setattr(model, field, None)
    return out


def _get(obj, key):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)
