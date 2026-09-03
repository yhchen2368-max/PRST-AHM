"""The adjoint backward sweep.

For an objective summed over report steps, ``J = sum_n g_n(x_n)``, and a
residual ``R_n(x_n, x_{n-1}, p) = 0`` at each step, the sensitivity is

    dJ/dp = sum_n  lambda_n^T dR_n/dp

with the multipliers found by sweeping backwards::

    J_N^T lambda_N = -(dg_N/dx_N)^T
    J_n^T lambda_n = -(dg_n/dx_n)^T - B_{n+1}^T lambda_{n+1}

where ``J_n = dR_n/dx_n`` and ``B_n = dR_n/dx_{n-1}``. One linear solve
per step, against the transpose of a Jacobian the forward run already
built -- which is the whole point: the cost does not grow with the
number of parameters.

Every derivative here is checked against finite differences in
:mod:`adjoint_verification` before being used, and the assembled
gradient is checked end to end against a finite-difference gradient of
the same objective. An adjoint that is merely plausible is worthless:
a wrong gradient descends just as smoothly as a right one, to the wrong
answer.
"""

import numpy as _np
import scipy.sparse as _sp
import scipy.sparse.linalg as _spla


def adjoint_gradient(model, state0, states, dts, forces, parameters,
                     objective_partials, linear_solver=None):
    """Return ``{name: dJ/dp}`` for each requested parameter.

    Parameters
    ----------
    model, state0, forces
        As the forward run used them. ``forces`` may be a single set, or
        one per step when the schedule changes control -- a
        history-matching deck usually resets WCONHIST at every report
        step, and differentiating all of them against the first
        step's controls gives a gradient for a schedule that was never
        run.
    states : sequence
        The converged state after each report step, ``states[n]`` for
        step ``n``.
    dts : sequence of float
        Step lengths, matching ``states``.
    parameters : sequence of str
        Any of ``compute_sensitivities_adjoint_ad.SUPPORTED``:
        ``'transmissibility'``, ``'porevolume'``, or one of the eleven
        saturation-function endpoints.
    objective_partials : callable(n, state) -> ndarray
        ``dg_n/dx_n`` as a dense vector over the same unknowns the
        residual uses. Zero for a step the objective does not see.
    linear_solver : callable(A, b) -> x, optional
        Defaults to a sparse direct solve, which is what MRST's
        BackslashSolverAD does.
    """
    from .adjoint_verification import (jacobian_wrt_parameter,
                                       jacobian_wrt_state,
                                       jacobian_wrt_state0)

    solve = linear_solver or _direct_solve
    nsteps = len(states)
    grads = {name: None for name in parameters}

    lam_next = None
    coupling_next = None

    def forces_at(step):
        if isinstance(forces, (list, tuple)) and forces \
                and isinstance(forces[0], (dict, type(forces[0]))) \
                and len(forces) == nsteps:
            return forces[step]
        return forces

    for n in range(nsteps - 1, -1, -1):
        previous = state0 if n == 0 else states[n - 1]
        state = states[n]
        dt = float(dts[n])
        forces_n = forces_at(n)

        J = jacobian_wrt_state(model, previous, state, dt, forces_n)

        rhs = -_np.asarray(objective_partials(n, state), dtype=float).ravel()
        if lam_next is not None:
            # The next step's residual depends on this step's state, so
            # its multiplier feeds back through that coupling.
            rhs = rhs - coupling_next.T @ lam_next

        lam = solve(J.T.tocsc() if hasattr(J, 'tocsc') else J.T, rhs)

        for name in parameters:
            # ``forces_n``, not ``forces``: the latter may be the whole
            # per-step list, and a list reaching the assembly reads as
            # "no wells at all" -- which then writes empty facility
            # primaries and an empty wellSol into the caller's state and
            # strands every step after it.
            dRdp = jacobian_wrt_parameter(model, previous, state, dt,
                                          forces_n, name, forward=J)
            contribution = dRdp.T @ lam
            grads[name] = contribution if grads[name] is None \
                else grads[name] + contribution

        # Carry this step's coupling back one step, for the next
        # iteration of the sweep.
        coupling_next = jacobian_wrt_state0(model, previous, state, dt,
                                            forces_n, forward=J)
        lam_next = lam

    return grads


def _direct_solve(A, b):
    """A sparse direct solve, MRST's BackslashSolverAD by another name."""
    A = A.tocsc() if hasattr(A, 'tocsc') else _sp.csc_matrix(A)
    return _spla.spsolve(A, _np.asarray(b, dtype=float).ravel())


# ------------------------------------------------- finite-difference check --

def finite_difference_gradient(run_objective, base, entries, h=None):
    """dJ/dp by central differences -- the reference the adjoint is
    checked against.

    One forward simulation per entry per side, which is exactly the cost
    the adjoint exists to avoid, so this is a verification tool and not
    a fallback.
    """
    base = _np.asarray(base, dtype=float).ravel()
    if h is None:
        h = 1e-6 * float(_np.max(_np.abs(base)))

    out = _np.zeros(len(entries))
    for j, entry in enumerate(entries):
        up, down = base.copy(), base.copy()
        up[entry] += h
        down[entry] -= h
        out[j] = (run_objective(up) - run_objective(down)) / (2 * h)
    return out
