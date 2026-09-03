"""Parameter sensitivities by adjoint -- port of
``computeSensitivitiesAdjointAD.m`` (autodiff/ad-core/simulators).

The backward loop::

    for step = nstep:-1:1
        [lambda, lambdaVec] = model.solveAdjoint(...)
        [eqdth, modelParam] = partialWRTparam(modelParam, ...)
        result = sum_k lambda{k}' * eqdth{k}
        result = result.jac
        sens.(name) += result{k}'

The trick worth naming is in the third line. ``eqdth`` is the residual
assembled on a model whose *parameters* are AD variables and whose
*states* are not -- ``resOnly=true`` sees to the second half. So
``lambda' * eqdth`` is a scalar carrying ``lambda^T dR/dtheta`` in its
Jacobian, and the contribution falls out without ever forming
``dR/dtheta`` as a matrix. One assembly per step, whatever the number of
parameters; that is the whole reason an adjoint is worth having.

The forward states must come from a tightly converged solve. The adjoint
differentiates the *root* of the residual, so a state that merely
satisfies a CNV/MB criterion gives the gradient of a slightly different
problem than the one that was solved.
"""

import numpy as _np

from PRSTCore.ad_core.simulators.solve_adjoint import solve_adjoint


def compute_sensitivities_adjoint(setup, states, params, get_objective,
                                  linear_solver=None, verbose=False):
    """Return ``{name: dJ/dp}``, one entry per parameter.

    ``setup`` carries ``model``, ``schedule`` and ``state0``; ``params``
    is a list of :class:`ModelParameter`.
    """
    from PRSTCore.ad_core.solvers.linear_solver_ad import LinearSolverAD

    solver = linear_solver or LinearSolverAD()
    model = setup['model']
    schedule = setup['schedule']
    state0 = setup.get('state0')

    sens = {p.name: 0.0 for p in params}
    model_param, schedule_param = init_model_parameters_adi(setup, params)

    dt = _np.asarray(schedule['step']['val'], dtype=float).ravel()
    nstep = dt.size

    def get_state(i):
        """``getStateFromInput``: step -1 is state0, past the end is None."""
        if i < 0:
            return state0
        return states[i] if i < len(states) else None

    lambda_vec = None
    for step in range(nstep - 1, -1, -1):
        if verbose:
            print('Solving reverse mode step %d of %d'
                  % (nstep - step, nstep))
        lam, lambda_vec, _report = solve_adjoint(
            model, solver, get_state, get_objective, schedule, lambda_vec,
            step)

        residual = partial_wrt_param(model_param, get_state, schedule_param,
                                     step, params)
        if residual is None:
            continue

        # lambda^T R(theta): a scalar whose Jacobian is lambda^T dR/dtheta.
        result = residual * _np.asarray(lam, dtype=float).ravel()

        # MRST-0 guards this: with no parameter carrying a derivative the
        # sum is a plain number and ``.jac`` does not exist. Stock MRST
        # reaches for it unconditionally.
        total = result.sum()
        if not hasattr(total, 'jac'):
            continue

        jac = total.jac
        contribution = _np.asarray(jac.todense()) if hasattr(jac, 'todense') \
            else _np.asarray(jac)
        # MRST-0 slices the result by *column* (``result(:, i1:i2)``)
        # where stock MRST uses a linear index. The two agree while the
        # objective is scalar and the result is one row; they part as
        # soon as it is not -- a multi-objective match assembles a matrix
        # here, and a linear index would then cut across rows instead of
        # separating the parameters.
        contribution = _np.atleast_2d(contribution)

        offset = 0
        for p in params:
            width = int(p.n_param)
            block = contribution[:, offset:offset + width]
            sens[p.name] = sens[p.name] + (block[0] if block.shape[0] == 1
                                           else block.T)
            offset += width

    return sens


def init_model_parameters_adi(setup, params):
    """Port of ``initModelParametersADI``.

    Every parameter is seeded as an AD variable *together*, so parameter
    ``k`` owns its own block of columns, and then written back into a
    copy of the setup through ``ModelParameter.set_parameter``. The
    indirection is what makes the seeding general: whatever a parameter
    feeds -- an operator, a rock field, a relperm scaler -- the assembly
    carries the derivative through it without knowing that it did.

    Well-control parameters are excluded, as in MRST: the control
    equations carry logic that is not differentiable, and
    ``partialWRTparam`` handles them separately.
    """
    from PRSTCore.ad_core.adi import SparseADI
    from PRSTCore.optimization.utils.parameters import WELL_CONTROL_TYPES

    from PRSTCore.optimization.utils.parameters import _copy_model

    values = [_np.asarray(p.get_parameter(setup), dtype=float).ravel()
              for p in params]
    widths = [v.size for v in values]
    nvar = int(sum(widths))

    # MATLAB copies a struct by value, so MRST's ``setupNew`` is
    # independent of ``setup``. A shallow dict copy is not: seeding a
    # parameter would write the AD object into the *same* model
    # ``solve_adjoint`` is about to assemble a forward Jacobian from,
    # and the two would silently disagree about how wide the system is.
    new = dict(setup)
    new['model'] = _copy_model(setup['model'])
    offset = 0
    for p, value, width in zip(params, values, widths):
        if p.control_type not in WELL_CONTROL_TYPES:
            p.set_parameter(new, SparseADI.variable(value, nvar, offset))
        offset += width
    return new['model'], new.get('schedule')


def partial_wrt_param(model, get_state, schedule, step, params):
    """Port of ``partialWRTparam``: ``dR_n/dtheta`` as an AD residual.

    ``resOnly=True`` is the point -- it stops the assembly seeding the
    states, so the only derivatives in the result are the parameters'.
    The previous state is passed through ``getStateAD(before, False)``
    for the same reason: to clear any cached AD it might carry.
    """
    current = get_state(step)
    before = get_state(step - 1)
    if current is None or before is None:
        return None

    dt_steps = _np.asarray(schedule['step']['val'], dtype=float).ravel()
    dt = float(dt_steps[step])

    from PRSTCore.ad_core.simulators.solve_adjoint import (_control_of,
                                                           _forces_for,
                                                           _validate_facility)
    forces = _forces_for(model, schedule, step)

    # MRST re-validates only when the controls are about to change, since
    # doing it every step is not free.
    revalidate = (step == dt_steps.size - 1
                  or _control_of(schedule, step)
                  != _control_of(schedule, step + 1))
    if revalidate:
        model = _validate_facility(model, forces)

    before = model.getStateAD(before, False)
    problem, _ = model.get_equations(before, current, dt, forces,
                                     ResOnly=True)
    return problem.get('_assembled')
