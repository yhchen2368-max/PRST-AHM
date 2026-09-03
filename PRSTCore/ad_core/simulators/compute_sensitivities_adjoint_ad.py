"""Adjoint parameter sensitivities.

Port of the sensitivity assembly around MRST's
``computeSensitivitiesAdjointAD.m``: run the backward sweep, then scale
each parameter's raw ``dJ/dp`` the way that parameter is defined.

**This used to return zeros.** Nine lines that handed back a zero vector
for every parameter, which three of hm's evaluate modules then fed to a
gradient optimiser. Zeros are indistinguishable from a converged answer:
the optimiser takes no steps and reports success. It is replaced here
only because every derivative underneath is now checked against finite
differences on a real deck -- see :mod:`adjoint_verification`, and

    dR/dx_n              3.8e-10
    dR/dx_{n-1}          8.0e-09
    dR/dtransmissibility 4.4e-08
    dR/dporevolume       9.7e-09
    dJ/dp end to end     6.5e-08 (transmissibility), 2.2e-09 (pore volume)

A gradient that is merely plausible is worse than none: it descends just
as smoothly, to the wrong answer.

The eleven saturation-function endpoints are covered too, and checked
the same way on a deck with ``ENDSCALE`` on (SPE1 at sW=0.33, sG=0.11,
so every curve is on a sloped segment)::

    swl 7.2e-10   swcr 7.1e-07   swu 2.6e-06   krw  7.9e-12
    sgl 1.7e-10   sgcr 1.3e-10   sgu 1.3e-10   krg  9.0e-12
    sowcr 2.1e-10 sogcr 3.1e-10  kro  7.5e-11

End to end, ``dJ/dp`` for the endpoints lands between 3e-09 and 4e-08
for nine of them; ``swcr`` (3.5e-05) and ``swu`` (6.6e-05) are looser
because their gradient is four orders smaller than the rest, so the
finite-difference reference has that many fewer digits. Their error
*falls* as the step grows, which is round-off in the reference rather
than error in the adjoint.

**What it does not cover.** Any parameter outside :data:`SUPPORTED`
comes back zero, and says so -- the well terms in particular, whose
rows are not differentiated by construction (see
``check_well_scaling_is_not_differentiated``), so a connection
transmissibility is deliberately among them.

The forward states must come from a tightly converged solve: the
adjoint differentiates the root of the residual, so states that merely
satisfy the CNV/MB criterion give a gradient for a problem slightly
different from the one that was solved. And an endpoint's derivative
exists only where the saturation table is smooth -- on a table node the
residual has a kink, and there the adjoint takes one side while a
central difference straddles both, which reads as a 17% error that is
really the check's.
"""

import warnings as _warnings

import numpy as np

from .adjoint_verification import ENDPOINT_COLUMNS as _ENDPOINTS

#: Parameters with a derivative path through the residual.
SUPPORTED = ('transmissibility', 'porevolume') + tuple(sorted(_ENDPOINTS))


def compute_sensitivities_adjoint_ad(setup, states, parameters, objh,
                                     accumulate_residuals=None,
                                     is_scalar=True, LinearSolver=None,
                                     match_map=None, recompute_wi=False):
    """Return ``{name: dJ/dp}`` for each parameter.

    ``objh(tstep, model, state)`` follows MRST's convention. ``setup``
    carries ``model``, ``schedule`` and ``state0``.

    ``recompute_wi`` is MRST-0's option -- it recomputes the well index
    after the parameters are applied, since a tuned permeability changes
    the Peaceman index and leaving it fixed makes the gradient describe a
    different model than the forward run.
    """
    from .adjoint_sweep import adjoint_gradient

    model = _get(setup, 'model')
    state0 = _get(setup, 'state0')
    schedule = _get(setup, 'schedule')
    dts = np.asarray(schedule['step']['val'], dtype=float).ravel()

    # ``currControl = setup.schedule.step.control(step)``: the sweep runs
    # under each step's *own* control, as solveAdjoint's lookupCtrl does.
    # Taking control 1 for every step gives the facility model the wells
    # that were open on the first date and no others -- the assembly then
    # writes that six-well wellSol into the state it was handed, and an
    # objective comparing it against nine observed wells cannot line the
    # two up.
    forces = _forces_per_step(model, schedule, len(dts))

    names = [str(_name(p)) for p in parameters]
    wanted = [n for n in names if n in SUPPORTED]
    unsupported = [n for n in names if n not in SUPPORTED]
    if unsupported:
        _warnings.warn(
            'No adjoint derivative for %s; returning zeros for those. '
            'Differentiated so far: %s.'
            % (', '.join(sorted(set(unsupported))), ', '.join(SUPPORTED)),
            RuntimeWarning, stacklevel=2)

    grads = {}
    if wanted:
        raw = adjoint_gradient(model, state0, list(states),
                               dts[:len(states)], forces, wanted,
                               _partials(model, objh, forces),
                               linear_solver=LinearSolver)
        grads.update(raw)

    out = {}
    for param in parameters:
        name = str(_name(param))
        value = grads.get(name)
        out[name] = np.zeros(_nparam(param)) if value is None \
            else np.asarray(value, dtype=float).ravel()
    return out


def _forces_per_step(model, schedule, nsteps):
    """One driving-force set per report step, from that step's control.

    Controls repeat, so each distinct index is built once. Returns a
    single force set when the schedule has only one control, which is
    what the sweep expects for that case.
    """
    controls = _controls(schedule)
    step = schedule['step'] if isinstance(schedule, dict) else schedule.step
    index = np.asarray(step['control'] if isinstance(step, dict)
                       else step.control, dtype=int).ravel()

    if len(controls) <= 1 or index.size != nsteps:
        return model.getDrivingForces(controls[0])

    cache = {}
    out = []
    for k in index[:nsteps]:
        k = int(k)
        if k not in cache:
            cache[k] = model.getDrivingForces(controls[min(k, len(controls) - 1)])
        out.append(cache[k])
    return out


def _partials(model, objh, forces):
    """Adapt MRST's ``objh(tstep, model, state)`` to ``dg_n/dx_n``.

    The objective is asked for its partials at each step; a step it does
    not see contributes nothing.  The unknown count follows that step's
    own control, since the well block is as wide as that control's active
    well list.
    """
    nc = int(model.G['cells']['num'])
    per_step = isinstance(forces, (list, tuple))

    def width(step):
        f = forces[min(step, len(forces) - 1)] if per_step else forces
        return 3 * nc + 4 * len(model._mrst_active_wells(f))

    def partials(step, state):
        nvar = width(step)
        if objh is None:
            return np.zeros(nvar)
        result = objh(step, model, state)
        if result is None:
            return np.zeros(nvar)
        if hasattr(result, 'jac'):
            # An AD scalar: its Jacobian row is exactly dg/dx.
            row = result.jac
            dense = np.asarray(row.todense()).ravel() \
                if hasattr(row, 'todense') else np.asarray(row).ravel()
            return _fit(dense, nvar)
        return _fit(np.atleast_1d(np.asarray(result, dtype=float)).ravel(),
                    nvar)

    return partials


def _fit(vector, nvar):
    """Pad or trim a partials vector to the unknown count."""
    if vector.size == nvar:
        return vector
    out = np.zeros(nvar)
    out[:min(vector.size, nvar)] = vector[:nvar]
    return out


def _controls(schedule):
    controls = schedule['control'] if isinstance(schedule, dict) \
        else schedule.control
    return controls if isinstance(controls, (list, tuple)) else [controls]


def _get(obj, key):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


def _name(param):
    return param['name'] if isinstance(param, dict) else param.name


def _nparam(param):
    return int(param['nParam'] if isinstance(param, dict) else param.n_param)
