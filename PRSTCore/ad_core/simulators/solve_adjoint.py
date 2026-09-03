"""One backward step of the adjoint -- port of
``PhysicalModel.solveAdjoint`` (autodiff/ad-core/models).

For an objective summed over report steps, ``J = sum_n g_n(x_n)``, and a
residual ``R_n(x_n, x_{n-1}, p) = 0``, the Lagrange multipliers satisfy

    J_n^T lambda_n = -(dg_n/dx_n)^T - B_{n+1}^T lambda_{n+1}

with ``J_n = dR_n/dx_n`` and ``B_{n+1} = dR_{n+1}/dx_n``. This assembles
both and hands them to the linear solver.

The two Jacobians come from *one* function, ``getAdjointEquations``,
called twice: once forwards for ``J_n``, once with ``reverseMode`` for
``B_{n+1}``. In reverse mode the current state carries no derivative and
the previous one -- seeded here by ``getReverseStateAD`` -- does, so the
Jacobian that falls out is the coupling term. There is no second
expression of the equations to keep in step with the first.

Three details are MRST's and each of them changes the answer:

* **The controls are looked up per step.** A history-matching deck
  restates WCONHIST at every report step, so the well targets, and which
  wells are open at all, differ from step to step. Differentiating all of
  them against the first step's controls gives a gradient for a schedule
  that was never run.
* **The facility model is re-validated with those forces.** That is what
  keeps the well bookkeeping -- which wells exist, in what order, which
  are active -- consistent with the step being differentiated.
* **The forces are only rebuilt when the control index changes.** MRST
  checks ``diff(schedule.step.control([n; n+1])) == 0`` first, which
  matters because re-validating the facility model is not free.
"""

import numpy as _np


def solve_adjoint(model, solver, get_state, get_objective, schedule,
                  lambda_next, step, linear_solver=None):
    """Return ``(lambda, lambda_vec, report)`` for one backward step.

    ``step`` is 0-based here where MRST's is 1-based; everything else
    follows the MATLAB.
    """
    dt_steps = _np.asarray(schedule['step']['val'], dtype=float).ravel()
    nsteps = dt_steps.size

    current = get_state(step)
    before = get_state(step - 1)
    dt = float(dt_steps[step])

    forces = _forces_for(model, schedule, step)
    model = _validate_facility(model, forces)

    problem, _ = model.getAdjointEquations(before, current, dt, forces,
                                           reverseMode=False)

    if step < nsteps - 1:
        after = get_state(step + 1)
        dt_next = float(dt_steps[step + 1])

        # Seed the current state to the width the *next* step's forward
        # assembly uses -- that is the system the coupling term belongs
        # to, and it need not have the same well count as this step's.
        forces_p = forces
        if _control_of(schedule, step) != _control_of(schedule, step + 1):
            forces_p = _forces_for(model, schedule, step + 1)
            model = _validate_facility(model, forces_p)

        nvar = _forward_width(model, current, after, dt_next, forces_p)
        nc = int(model.G['cells']['num'])
        seeded = model.getReverseStateAD(current, True, forces_p, nvar=nvar,
                                         nw=max(nvar - 3 * nc, 0) // 4)
        problem_p, _ = model.getAdjointEquations(seeded, after, dt_next,
                                                 forces_p, reverseMode=True)
    else:
        problem_p = None

    objective = get_objective(step, model, problem.get('State', current))
    lam, lam_vec, report = solver.solveAdjointProblem(
        problem_p, problem, lambda_next, objective, model)
    return lam, lam_vec, {'Types': problem.get('types'),
                          'LinearSolverReport': report}


def _control_of(schedule, step):
    """``schedule.step.control(step)``, 0-based."""
    control = _np.asarray(schedule['step'].get(
        'control', _np.zeros(len(schedule['step']['val']), dtype=int)),
        dtype=int).ravel()
    return int(control[step]) if step < control.size else 0


def _forces_for(model, schedule, step):
    """``model.getDrivingForces(lookupCtrl(step))``."""
    controls = schedule.get('control') or [{'W': []}]
    which = min(_control_of(schedule, step), len(controls) - 1)
    return model.getDrivingForces(controls[which])


def _validate_facility(model, forces):
    """``model.FacilityModel = model.FacilityModel.validateModel(forces)``.

    A model whose facility does not offer it is left alone, as MRST
    falls back to ``model.validateModel(forces)`` for the same case.
    """
    facility = getattr(model, 'FacilityModel', None)
    if facility is not None and hasattr(facility, 'validateModel'):
        model.FacilityModel = facility.validateModel(forces)
    elif hasattr(model, 'validateModel'):
        try:
            model.validateModel(forces)
        except TypeError:
            pass                       # a validateModel that takes no forces
    return model


def _forward_width(model, state0, state, dt, forces):
    """The unknown count the assembly uses for this pair of states.

    Read from the assembly rather than recomputed, because the facility
    model resolves its own well list and a separate count can disagree
    with it -- silently, since every size still checks out and only the
    seeded columns land in the wrong place.
    """
    problem, _ = model.get_equations(state0, state, dt, forces,
                                     ResOnly=True)
    return int(problem['Jacobian'].shape[1])
