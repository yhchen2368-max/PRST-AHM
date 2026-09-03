"""Port of MRST ``evaluateMatchSummandsMulti.m``
(mrst-2026a/hm/utils/evaluate).

The least-squares form of :mod:`evaluateObjective`, over several cases at
once: instead of one scalar objective it returns the vector of *residuals*
(square roots of the per-step misfits) and their Jacobian, which is what a
Levenberg-Marquardt optimiser consumes.

All cases share one parameter vector; each contributes a column of
residuals, so the returned ``misfitVals`` is ``(nresidual, ncase)``.

The chain rule for the square root gives the Jacobian scaling

    d(sqrt(f))/dp = (1 / (2*sqrt(f))) * df/dp

which is applied only where the residual is nonzero relative to the
column's norm -- at an exact match the derivative is unbounded, and the
row is left as computed rather than divided by zero.
"""

import numpy as _np
import scipy.sparse as _sp


def evaluateMatchSummandsMulti(pvec, obj, setup, parameters, states_ref,
                               Verbose=False, NonlinearSolver=None,
                               objScaling=1.0, enforceBounds=True,
                               accumulateResiduals=None,
                               return_jacobian=False, return_states=False,
                               return_setup=False, **extra):
    """Return ``misfitVals``, plus whatever else was requested."""
    from PRSTCore.ad_core.simulators.simulate_schedule_ad import simulate_schedule_ad
    from PRSTCore.ad_core.simulators import compute_sensitivities_adjoint_ad

    accum = accumulateResiduals or {'wells': None, 'types': None, 'steps': None}
    nparam = [int(_get(p, 'nParam')) for p in parameters]
    p = _np.asarray(pvec, dtype=float).ravel()
    if enforceBounds:
        p = _np.clip(p, 0.0, 1.0)
    slices = _slices(nparam)

    nc = len(setup)
    setupNew, wellSols, states, misfitVals = [], [], [], []
    pval = [None] * len(parameters)

    for c in range(nc):
        case = dict(setup[c])
        model = case['model']
        for field in ('FlowDiscretization', 'FlowPropertyFunctions'):
            if hasattr(model, field):
                setattr(model, field, None)
        for k, param in enumerate(parameters):
            pval[k] = _get(param, 'unscale')(p[slices[k]])
            case = _get(param, 'setParameter')(case, pval[k])
        setupNew.append(case)

        ws, st = simulate_schedule_ad(case['state0'], case['model'],
                                      case['schedule'],
                                      nonlinear_solver=NonlinearSolver, **extra)
        wellSols.append(ws)
        states.append(st)

        values = obj[c](case['model'], st, case['schedule'], states_ref[c],
                        False, None, None)
        values = _accumulate_steps(values, accum.get('steps'))
        # The residual is the square root of the misfit.
        misfitVals.append(_np.sqrt(_np.concatenate(
            [_np.atleast_1d(_np.asarray(v, dtype=float)).ravel()
             for v in values])))

    misfitVals = _np.column_stack(misfitVals)

    out = [misfitVals]

    if return_jacobian:
        names = [_get(param, 'name') for param in parameters]
        J = []
        for c in range(nc):
            def objh(tstep, model, state, _c=c):
                return obj[_c](setupNew[_c]['model'], states[_c],
                               setupNew[_c]['schedule'], states_ref[_c],
                               True, tstep, state)

            gradient = compute_sensitivities_adjoint_ad(
                setupNew[c], states[c], parameters, objh,
                accumulate_residuals=accum, is_scalar=False)
            scaled = [_np.atleast_2d(_np.asarray(
                _get(param, 'scaleGradient')(gradient[names[k]], pval[k]),
                dtype=float)) for k, param in enumerate(parameters)]
            Jc = _np.vstack(scaled).T / objScaling

            column = misfitVals[:, c]
            norm = _np.linalg.norm(column)
            nz = _np.abs(column) > _np.finfo(float).eps * norm
            if _np.any(nz):
                # d(sqrt(f))/dp = df/dp / (2*sqrt(f))
                Jc[nz, :] = (_sp.diags(1.0 / (2.0 * column[nz]))
                             @ Jc[nz, :])
            J.append(Jc)
        out.append(J)

    if return_states:
        out.extend([wellSols, states])
    if return_setup:
        out.append(setupNew)

    return out[0] if len(out) == 1 else tuple(out)


def _accumulate_steps(values, steps):
    """Merge the per-step misfits into the requested residual groups."""
    if steps is None:
        return values
    steps = _np.atleast_1d(_np.asarray(steps, dtype=int)).ravel()
    out = [0] * int(steps.max())
    for k, value in enumerate(values):
        if k < steps.size and steps[k] > 0:
            out[steps[k] - 1] = out[steps[k] - 1] + value
    return out


def _slices(nparam):
    bounds = _np.concatenate([[0], _np.cumsum(nparam)]).astype(int)
    return [slice(bounds[i], bounds[i + 1]) for i in range(len(nparam))]


def _get(obj, key):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)
