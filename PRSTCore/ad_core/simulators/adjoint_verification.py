"""A bench for checking derivatives against finite differences.

The adjoint's whole value is that it is cheap; its whole danger is that a
wrong gradient looks exactly like a right one. Nothing about a smooth
descent curve tells you the gradient was correct -- an optimiser will
happily converge on a wrong one, to the wrong answer.

So every derivative the adjoint rests on gets checked here against a
finite difference of the same quantity, on a real deck-driven model.
Not a synthetic fixture: the AD property stack goes through the deck's
PVT tables, so a model without them exercises none of the code that
matters.

Used from :mod:`tests.test_adjoint_verification`, and runnable directly::

    python -m PRSTCore.ad_core.simulators.adjoint_verification path/to.DATA
"""

import numpy as _np
import scipy.sparse as _sp


def build_case(deck_path, ncells=None):
    """A deck-driven model, its initial state and its first control.

    Returns ``(model, state0, forces, dt)``.
    """
    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import \
        init_eclipse_problem_ad

    state0, model, schedule, _ = init_eclipse_problem_ad(deck_path)
    forces = model.getDrivingForces(schedule['control'][0])
    dt = float(schedule['step']['val'][0])
    return model, state0, forces, dt


def residual(model, state0, state, dt, forces):
    """The residual vector alone, with no derivatives."""
    problem, _ = model.get_equations(state0, state, dt, forces)
    return _np.asarray(problem['Residuals'], dtype=float).ravel()


def jacobian_wrt_state(model, state0, state, dt, forces):
    """dR/dx_n -- what the forward Newton solve already builds."""
    problem, _ = model.get_equations(state0, state, dt, forces)
    return problem['Jacobian']


def jacobian_wrt_state0(model, state0, state, dt, forces, forward=None):
    """dR/dx_{n-1} -- the coupling term the adjoint sweep needs.

    The assembly always seeds the *current* state at columns 0..3nc, and
    :meth:`getStateAD` seeds the previous state at the same offsets. AD
    accumulates linearly, so seeding both gives their sum::

        J(both seeded) = dR/dx_n + dR/dx_{n-1}

    and subtracting the ordinary forward Jacobian leaves the coupling
    term on its own. That avoids having to teach the assembly a second
    column block, and it is exact rather than approximate -- the two
    evaluations differ in nothing but whether state0 carries a seed.
    """
    if forward is None:
        forward = model.get_equations(state0, state, dt, forces)[0]['Jacobian']
    # Seed state0 to the width and well count the assembly actually
    # used. Letting getStateAD work the count out for itself lets the two
    # disagree -- the assembly resolves its wells through the facility
    # state, which need not match a plain read of the driving forces --
    # and the mismatch surfaces as "ADI variables use different primary
    # vectors" deep inside the residual.
    nvar = forward.shape[1]
    nc = int(model.G['cells']['num'])
    seeded = model.getStateAD(state0, True, forces, nvar=nvar,
                              nw=max(nvar - 3 * nc, 0) // 4)
    both = model.get_equations(seeded, state, dt, forces)[0]['Jacobian']
    return both - forward


def finite_difference_columns(model, state0, state, dt, forces, field,
                              which='state', h=None, cells=None,
                              central=True):
    """dR/d(field) by finite differences, one column per cell.

    Central differences by default: their error is O(h^2) against a
    one-sided O(h), which matters because the thing being checked is
    supposed to be exact. With a pressure step of a few Pa on a field at
    1e7 Pa, one-sided differencing alone leaves a 1e-4 relative error
    and there is then no way to tell a small truncation error from a
    small mistake in the derivative.

    ``which`` selects whether the perturbed state is the current one or
    the previous one. ``cells`` limits the columns computed -- a full
    Jacobian would cost two simulations per unknown, and a handful of
    columns is enough to catch a wrong derivative.
    """
    target = state if which == 'state' else state0
    base = _np.asarray(target[field], dtype=float).ravel()
    if cells is None:
        cells = range(base.size)
    cells = list(cells)

    if h is None:
        # Scale the step to the field -- a pressure step of 1e-6 Pa is
        # noise, and a saturation step of 1e-6 is not -- but keep it
        # small enough to stay inside one PVT table interval.
        #
        # This is not a tolerance to be tuned until the test passes. The
        # deck's PVT tables are piecewise linear, so the residual is only
        # piecewise smooth in pressure: a step that crosses a table node
        # differences across a kink and the result is wrong however exact
        # the derivative is. On SPE1 the effect is stark -- a 33 Pa step
        # gives 1.7e-4 relative error and an 8 Pa step gives 3e-10, with
        # no smooth transition between them, because 8 Pa happens to stay
        # within one interval and 33 Pa does not.
        h = max(1e-7 * float(_np.max(_np.abs(base))), 1e-10)

    def at(values):
        perturbed = dict(target)
        perturbed[field] = values
        if which == 'state':
            return residual(model, state0, perturbed, dt, forces)
        return residual(model, perturbed, state, dt, forces)

    r0 = None if central else residual(model, state0, state, dt, forces)
    out = _np.zeros(((r0.size if r0 is not None else at(base).size),
                     len(cells)))
    for j, cell in enumerate(cells):
        up = base.copy()
        up[cell] += h
        if central:
            down = base.copy()
            down[cell] -= h
            out[:, j] = (at(up) - at(down)) / (2 * h)
        else:
            out[:, j] = (at(up) - r0) / h
    return out


def compare(analytic, numeric, name='', rtol=1e-4, atol=None):
    """Report how far apart two derivative blocks are.

    Scaled by the largest entry rather than entry by entry: a Jacobian
    column is mostly zeros, and a relative test on a zero entry says
    nothing useful.
    """
    analytic = _np.asarray(analytic, dtype=float)
    numeric = _np.asarray(numeric, dtype=float)
    reference = float(_np.max(_np.abs(numeric)))

    # An all-zero reference makes any comparison succeed while testing
    # nothing. It happened here for real: the first 540 of SPE1's 740
    # faces are horizontal, the state is horizontally uniform, and
    # dR/dT across a face carrying no flow is identically zero -- so the
    # first version of this check "passed" against a column of zeros.
    if reference <= 0.0:
        return {'name': name, 'max_abs_diff': 0.0, 'scale': 0.0,
                'max_rel_diff': 0.0, 'passed': False,
                'reason': 'the finite-difference reference is identically '
                          'zero, so this compares nothing -- pick entries '
                          'the residual actually depends on',
                'worst_index': None}

    scale = reference
    if atol is None:
        atol = rtol * scale
    diff = _np.abs(analytic - numeric)
    worst = int(_np.argmax(diff))
    return {
        'name': name,
        'max_abs_diff': float(_np.max(diff)),
        'scale': scale,
        'max_rel_diff': float(_np.max(diff) / scale),
        'passed': bool(_np.max(diff) <= atol),
        'worst_index': _np.unravel_index(worst, diff.shape),
    }


def reservoir_rows(model):
    """The conservation equations, excluding the well equations.

    The well equations are scaled by the mean of the *previous* state's
    phase densities, and that normalisation is deliberately not
    differentiated -- MRST takes ``value()`` there, so the model does
    too. A finite difference does not know about the convention and
    duly measures the derivative the model has chosen to drop, so
    checking dR/dx_{n-1} on the well rows compares against a number the
    model never claimed. The reservoir rows are the ones the adjoint's
    parameter sensitivities are built from, and they are checked
    strictly.
    """
    return 3 * int(model.G['cells']['num'])


def check_state_jacobian(model, state0, state, dt, forces, cells=(0, 1, 2),
                         rtol=1e-6):
    """dR/dx_n against finite differences, for a few pressure columns."""
    J = jacobian_wrt_state(model, state0, state, dt, forces)
    analytic = _np.asarray(J.todense())[:, list(cells)]
    numeric = finite_difference_columns(model, state0, state, dt, forces,
                                        'pressure', 'state', cells=cells)
    return compare(analytic, numeric, 'dR/dp_n', rtol)


def check_state0_jacobian(model, state0, state, dt, forces, cells=(0, 1, 2),
                          rtol=1e-6, reservoir_only=True):
    """dR/dx_{n-1} against finite differences -- the new path."""
    J = jacobian_wrt_state0(model, state0, state, dt, forces)
    analytic = _np.asarray(J.todense())[:, list(cells)]
    numeric = finite_difference_columns(model, state0, state, dt, forces,
                                        'pressure', 'state0', cells=cells)
    if reservoir_only:
        n = reservoir_rows(model)
        analytic, numeric = analytic[:n], numeric[:n]
    return compare(analytic, numeric, 'dR/dp_{n-1}', rtol)


def check_well_scaling_is_not_differentiated(model, state0, state, dt,
                                             forces, cells=(0, 1, 2)):
    """State the well-row convention rather than leaving it implicit.

    The analytic derivative of the well rows w.r.t. the previous state is
    exactly zero by construction. Reporting it keeps the choice visible
    instead of hiding behind a loosened tolerance.
    """
    J = jacobian_wrt_state0(model, state0, state, dt, forces)
    n = reservoir_rows(model)
    analytic = _np.asarray(J.todense())[n:, list(cells)]
    numeric = finite_difference_columns(model, state0, state, dt, forces,
                                        'pressure', 'state0', cells=cells)[n:]
    return {'name': 'well rows dR/dp_{n-1}',
            'analytic_all_zero': bool(_np.all(analytic == 0.0)),
            'numeric_max': float(_np.max(_np.abs(numeric)))}



# --------------------------------------------------- parameter Jacobians --


def live_faces(model, state, count=3, tol=1.0):
    """Faces with a pressure difference across them, in face numbering.

    A transmissibility derivative is ``potential * mobility``, so across
    a face carrying no flow it is exactly zero and checking it there
    compares nothing. On an equilibrated deck the horizontal faces are
    all like that -- SPE1 has 540 of them before the first live one.
    """
    c1, c2, _ = model._internal_connections()
    p = _np.asarray(state['pressure'], dtype=float).ravel()
    live = _np.flatnonzero(_np.abs(p[c2] - p[c1]) > tol)
    return live[:count].tolist()

#: Which relperm curve and column each saturation-function parameter is.
#: The four columns are [connate, critical, max-saturation, max-relperm],
#: the same layout ``fluid.krPts`` uses. ``kro`` names two of them: the
#: oil maximum is shared between the water-oil and gas-oil curves, and
#: ECLIPSE's KRO sets both.
ENDPOINT_COLUMNS = {
    'swl': (('w', 0),), 'swcr': (('w', 1),), 'swu': (('w', 2),),
    'krw': (('w', 3),),
    'sgl': (('g', 0),), 'sgcr': (('g', 1),), 'sgu': (('g', 2),),
    'krg': (('g', 3),),
    'sowcr': (('ow', 1),), 'sogcr': (('og', 1),),
    'kro': (('ow', 3), ('og', 3)),
}


def endpoint_base(model, name):
    """The current per-cell value of an endpoint parameter, or None when
    the model has no end-point scaling and the parameter therefore does
    not enter its residual at all."""
    nc = int(model.G['cells']['num'])
    scale = model._get_relperm_scaling(nc, model._get_relperm_tables())
    if scale is None:
        return None
    phase, column = ENDPOINT_COLUMNS[name][0]
    return _np.asarray(scale['target'][phase][:, column], dtype=float).ravel()


def jacobian_wrt_parameter(model, state0, state, dt, forces, name,
                           forward=None):
    """dR/dp for a tuned parameter.

    Seed the parameter into the low columns, subtract the ordinary
    forward Jacobian, and what is left is dR/dp on its own.

    The parameter has fewer entries than there are unknowns (740 faces
    against 908 columns on SPE1), so it occupies the first ``n`` columns
    and the rest are untouched.

    Transmissibility and pore volume enter the residual linearly, so
    for those this is exact. The saturation endpoints do not: they move
    the affine map that rescales saturation before the relperm table is
    interpolated, so their derivative is exact only where the table is
    smooth. At a table node, or where a cell's scaled saturation sits on
    one of the branch boundaries, the residual has a kink and no
    derivative -- the same caveat that already applies to any relperm
    sensitivity, and the reason the finite-difference check below uses a
    step small enough to stay inside one segment.
    """
    from PRSTCore.ad_core.adi import SparseADI

    # The unseeded Jacobian is the same matrix for every parameter and
    # for the state0 coupling, so the sweep assembles it once per step
    # and passes it in. Recomputing it here cost one full assembly per
    # parameter -- eight of nineteen per step on QIEDIE, all identical.
    if forward is None:
        forward = model.get_equations(state0, state, dt, forces)[0]['Jacobian']
    nvar = forward.shape[1]

    ops = model.operators
    if name == 'transmissibility':
        base = _np.asarray(ops['T'], dtype=float).ravel()
        seeded = SparseADI.variable(base, nvar, 0)
        original, ops['T'] = ops['T'], seeded
        try:
            both = model.get_equations(state0, state, dt,
                                       forces)[0]['Jacobian']
        finally:
            ops['T'] = original
    elif name == 'porevolume':
        # ``operators['pv']`` is where MRST keeps it and where
        # ``ModelParameter``'s location points, so it is the one place
        # the accumulation term reads. Seeding ``model.porevolume``
        # instead lands where nothing looks and gives a zero gradient.
        base = _np.asarray(model._porevolume_vector(), dtype=float).ravel()
        seeded = SparseADI.variable(base, nvar, 0)
        ops = model.operators
        original = ops.get('pv')
        ops['pv'] = seeded
        try:
            both = model.get_equations(state0, state, dt,
                                       forces)[0]['Jacobian']
        finally:
            if original is None:
                ops.pop('pv', None)
            else:
                ops['pv'] = original
    elif name in ENDPOINT_COLUMNS:
        base = endpoint_base(model, name)
        if base is None:
            # No ENDSCALE: the endpoint is not part of this model's
            # residual, so its derivative is zero rather than missing.
            return _sp.csr_matrix((forward.shape[0],
                                   int(model.G['cells']['num'])))
        seeded = SparseADI.variable(base, nvar, 0)
        original = getattr(model, '_relperm_endpoint_seed', None)
        model._relperm_endpoint_seed = {
            key: seeded for key in ENDPOINT_COLUMNS[name]}
        try:
            both = model.get_equations(state0, state, dt,
                                       forces)[0]['Jacobian']
        finally:
            model._relperm_endpoint_seed = original
    else:
        raise ValueError('Unknown parameter %r' % name)

    return (both - forward)[:, :base.size]


def finite_difference_parameter(model, state0, state, dt, forces, name,
                                entries=(0, 1, 2), h=None):
    """dR/dp by central differences, one column per entry."""
    ops = model.operators
    if name == 'transmissibility':
        base = _np.asarray(ops['T'], dtype=float).ravel()
        original = ops['T']

        def put(values):
            ops['T'] = values
    elif name == 'porevolume':
        base = _np.asarray(model._porevolume_vector(), dtype=float).ravel()
        original = model.operators.get('pv')

        def put(values):
            model.operators['pv'] = values
    elif name in ENDPOINT_COLUMNS:
        base = endpoint_base(model, name)
        if base is None:
            raise ValueError('%r does not enter this model: it has no '
                             'end-point scaling.' % name)
        scale = model._get_relperm_scaling(int(model.G['cells']['num']),
                                           model._get_relperm_tables())
        original = {phase: table.copy()
                    for phase, table in scale['target'].items()}

        def put(values):
            # Write straight into the cached target table, which is what
            # the scalers are built from -- the same place the tuned
            # value would land.
            for phase, column in ENDPOINT_COLUMNS[name]:
                scale['target'][phase][:, column] = values
    else:
        raise ValueError('Unknown parameter %r' % name)

    if h is None:
        h = 1e-7 * float(_np.max(_np.abs(base)))

    def restore():
        if name in ENDPOINT_COLUMNS:
            for phase, table in original.items():
                scale['target'][phase][...] = table
        else:
            put(original if original is not None else base)

    out = None
    try:
        for j, entry in enumerate(entries):
            up, down = base.copy(), base.copy()
            up[entry] += h
            down[entry] -= h
            put(up)
            r_up = residual(model, state0, state, dt, forces)
            put(down)
            r_down = residual(model, state0, state, dt, forces)
            if out is None:
                out = _np.zeros((r_up.size, len(entries)))
            out[:, j] = (r_up - r_down) / (2 * h)
    finally:
        restore()
    return out


def check_parameter_jacobian(model, state0, state, dt, forces, name,
                             entries=(0, 1, 2), rtol=1e-6,
                             reservoir_only=True):
    """dR/dp against finite differences."""
    analytic = _np.asarray(jacobian_wrt_parameter(
        model, state0, state, dt, forces, name).todense())[:, list(entries)]
    numeric = finite_difference_parameter(model, state0, state, dt, forces,
                                          name, entries)
    if reservoir_only:
        n = reservoir_rows(model)
        analytic, numeric = analytic[:n], numeric[:n]
    return compare(analytic, numeric, 'dR/d' + name, rtol)

def _perturbed_state(state0, dp=1e5, ds=0.01):
    """A state a little away from state0, so the residual is not zero.

    Checking a derivative at the point where the residual vanishes hides
    any error in the terms that vanish with it.
    """
    state = {k: (v.copy() if isinstance(v, _np.ndarray) else v)
             for k, v in state0.items()}
    state['pressure'] = _np.asarray(state0['pressure'], dtype=float) + dp
    sw = _np.asarray(state0['sW'], dtype=float) + ds
    state['sW'] = _np.clip(sw, 0.0, 1.0)
    state['time'] = float(state0.get('time', 0.0)) + 1.0
    return state



# ------------------------------------------------------- end to end --

#: Convergence tight enough that the forward solve lands on the root
#: rather than merely inside the CNV/MB criterion.
#:
#: This is not fussiness. The adjoint differentiates the *root* of the
#: residual; a finite-difference gradient differentiates whatever the
#: solver returned. At the default tolerances SPE1 stops with |R| = 9.6
#: and the two disagree by a factor of 2.8 -- consistently, and without
#: varying when the step size changes, so it reads exactly like a wrong
#: derivative rather than a loose reference. Tightened, they agree to
#: eight digits.
TIGHT = {'toleranceCNV': 1e-12, 'toleranceMB': 1e-16,
         'toleranceWellRate': 1e-12, 'toleranceWellBHP': 1e-6}


def tighten(model):
    """Apply :data:`TIGHT` and return what was there before."""
    previous = {k: getattr(model, k, None) for k in TIGHT}
    for key, value in TIGHT.items():
        setattr(model, key, value)
    return previous


def run_forward(model, state0, dt, forces, nsteps, solver=None,
                parameter=None, values=None):
    """Simulate ``nsteps`` steps, optionally with one operator replaced.

    Restores the operator afterwards, so a caller perturbing a parameter
    for a finite difference cannot leave the model altered.
    """
    from PRSTCore.ad_core.solvers import NonLinearSolver

    solver = solver or NonLinearSolver(verbose=False, maxIterations=200)
    ops = model.operators
    original = None
    scale = None
    if parameter == 'transmissibility':
        original, ops['T'] = ops['T'], values
    elif parameter == 'porevolume':
        original = model.operators.get('pv')
        model.operators['pv'] = values
    elif parameter in ENDPOINT_COLUMNS:
        # The endpoints live in the cached scaling table the scalers are
        # built from, which is where a tuned value would land too.
        scale = model._get_relperm_scaling(int(model.G['cells']['num']),
                                           model._get_relperm_tables())
        original = {phase: table.copy()
                    for phase, table in scale['target'].items()}
        for phase, column in ENDPOINT_COLUMNS[parameter]:
            scale['target'][phase][:, column] = values

    try:
        state, states = state0, []
        for _ in range(nsteps):
            state, _, _ = solver.solveTimestep(state, dt, model,
                                               drivingForces=forces)
            states.append(state)
    finally:
        if parameter == 'transmissibility':
            ops['T'] = original
        elif parameter == 'porevolume':
            model.operators['pv'] = original
        elif scale is not None:
            for phase, table in original.items():
                scale['target'][phase][...] = table
    return states


def pressure_sum_objective(model, forces):
    """``sum over steps of sum over cells of pressure``, and its partials.

    Summing over *every* step rather than the last one is deliberate: an
    objective that sees only the final state leaves the coupling term
    B_n barely exercised, and the coupling term is the entire reason the
    sweep runs backwards.
    """
    nc = int(model.G['cells']['num'])
    nvar = 3 * nc + 4 * len(model._mrst_active_wells(forces))

    def value(states):
        return float(sum(_np.sum(s['pressure']) for s in states))

    def partials(step, state):
        g = _np.zeros(nvar)
        g[:nc] = 1.0
        return g

    return value, partials


def check_gradient(model, state0, forces, dt, nsteps, parameter, entries,
                   rtol=1e-6, relative_step=1e-3):
    """The whole chain against a finite-difference gradient.

    Forward run, backward sweep, sensitivity accumulation -- compared
    against differencing the same objective through complete re-runs.
    """
    from .adjoint_sweep import adjoint_gradient

    value, partials = pressure_sum_objective(model, forces)
    states = run_forward(model, state0, dt, forces, nsteps)

    analytic = adjoint_gradient(model, state0, states, [dt] * nsteps,
                                forces, [parameter], partials)[parameter]

    if parameter == 'transmissibility':
        base = _np.asarray(model.operators['T'], dtype=float).ravel()
    elif parameter in ENDPOINT_COLUMNS:
        base = endpoint_base(model, parameter)
        if base is None:
            raise ValueError('%r does not enter this model: it has no '
                             'end-point scaling.' % parameter)
    else:
        base = _np.asarray(model._porevolume_vector(), dtype=float).ravel()

    numeric = _np.zeros(len(entries))
    for j, entry in enumerate(entries):
        # A relative step is meaningless for a parameter that is zero,
        # and connate gas usually is: it gives h = 0 and a nan gradient
        # that looks like a failure of the adjoint. Fall back to the
        # parameter's own scale.
        h = relative_step * base[entry]
        if h == 0.0:
            scale = float(_np.max(_np.abs(base)))
            h = relative_step * (scale if scale > 0.0 else 1.0)
        up, down = base.copy(), base.copy()
        up[entry] += h
        down[entry] -= h
        numeric[j] = (value(run_forward(model, state0, dt, forces, nsteps,
                                        parameter=parameter, values=up))
                      - value(run_forward(model, state0, dt, forces, nsteps,
                                          parameter=parameter,
                                          values=down))) / (2 * h)

    return compare(_np.array([analytic[e] for e in entries]), numeric,
                   'dJ/d' + parameter, rtol)

def run(deck_path, cells=(0, 1, 2)):
    """Check every derivative the adjoint rests on. Returns the reports."""
    model, state0, forces, dt = build_case(deck_path)
    state = _perturbed_state(state0)
    return [
        check_state_jacobian(model, state0, state, dt, forces, cells),
        check_state0_jacobian(model, state0, state, dt, forces, cells),
        check_parameter_jacobian(model, state0, state, dt, forces,
                                 'transmissibility',
                                 live_faces(model, state)),
        check_parameter_jacobian(model, state0, state, dt, forces,
                                 'porevolume', cells),
    ], check_well_scaling_is_not_differentiated(model, state0, state, dt,
                                                forces, cells)


if __name__ == '__main__':
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 \
        else 'examples/SpE1/SPE1CASE2.DATA'
    reports, wells = run(path)
    for report in reports:
        print('%-16s max|d| = %.3e   scale = %.3e   rel = %.3e   %s'
              % (report['name'], report['max_abs_diff'], report['scale'],
                 report['max_rel_diff'],
                 'PASS' if report['passed'] else 'FAIL'))
    print('%-16s analytic all zero = %s (by design); FD sees %.3e'
          % (wells['name'], wells['analytic_all_zero'], wells['numeric_max']))
