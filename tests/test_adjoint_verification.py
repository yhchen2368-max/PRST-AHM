"""Derivative checks for the adjoint groundwork, on a real deck.

A synthetic fixture is no good here: the AD property stack runs through
the deck's PVT tables, so a model without them exercises none of the
code the adjoint depends on. These run against SPE1 (10x10x3) and skip
if it is not present.
"""

import os

import numpy as np
import pytest

from PRSTCore.ad_core.simulators import adjoint_verification as V

DECK = 'examples/SpE1/SPE1CASE2.DATA'
CELLS = (0, 1, 2)


@pytest.fixture(scope='module')
def case():
    if not os.path.exists(DECK):
        pytest.skip('SPE1 deck not present')
    model, state0, forces, dt = V.build_case(DECK)
    return model, state0, V._perturbed_state(state0), dt, forces


# ------------------------------------------------------------- the bench --

def test_the_case_is_deck_driven_and_uses_the_generic_assembly(case):
    """If either were false the checks below would exercise the wrong
    code: the legacy hand-assembled path, or a fixture with no PVT."""
    model = case[0]
    assert model._use_mrst_generic_assembly is True
    assert model.G['cells']['num'] == 300


def test_the_comparison_state_is_away_from_the_initial_one(case):
    """Checking a derivative where the residual vanishes hides any error
    in the terms that vanish with it."""
    _, state0, state, _, _ = case
    assert not np.allclose(state['pressure'], state0['pressure'])
    assert not np.allclose(state['sW'], state0['sW'])


# --------------------------------------------------------- the Jacobians --

def test_the_forward_jacobian_matches_finite_differences(case):
    model, state0, state, dt, forces = case
    report = V.check_state_jacobian(model, state0, state, dt, forces, CELLS)
    assert report['passed'], report


def test_the_forward_jacobian_agrees_to_nine_digits(case):
    """Stated separately from the pass/fail so a later regression that
    merely loosens agreement is still visible."""
    model, state0, state, dt, forces = case
    report = V.check_state_jacobian(model, state0, state, dt, forces, CELLS)
    assert report['max_rel_diff'] < 1e-8, report


def test_the_state0_jacobian_matches_finite_differences(case):
    """dR/dx_{n-1}: the coupling term the adjoint sweep needs, which the
    equations could not produce at all before state0 was made AD."""
    model, state0, state, dt, forces = case
    report = V.check_state0_jacobian(model, state0, state, dt, forces, CELLS)
    assert report['passed'], report


def test_the_state0_jacobian_is_not_merely_zero(case):
    """A derivative of zero would pass any comparison against a small
    number; the coupling term must actually be there."""
    model, state0, state, dt, forces = case
    J = V.jacobian_wrt_state0(model, state0, state, dt, forces)
    assert np.max(np.abs(np.asarray(J.todense()))) > 1e-12


def test_the_state0_jacobian_is_far_smaller_than_the_forward_one(case):
    """Backward Euler: the previous state enters only through the
    accumulation term, so its Jacobian is the smaller of the two by
    roughly the ratio of accumulation to flux."""
    model, state0, state, dt, forces = case
    forward = np.abs(np.asarray(V.jacobian_wrt_state(
        model, state0, state, dt, forces).todense())).max()
    coupling = np.abs(np.asarray(V.jacobian_wrt_state0(
        model, state0, state, dt, forces).todense())).max()
    assert coupling < forward


# ------------------------------------------------------- the conventions --

def test_the_well_scaling_carries_no_state0_derivative(case):
    """The well equations are normalised by the mean of the previous
    state's densities, and MRST takes value() there. The analytic
    derivative is zero by construction; a finite difference measures the
    small number the model has chosen to drop. Recorded rather than
    hidden behind a loose tolerance."""
    model, state0, state, dt, forces = case
    report = V.check_well_scaling_is_not_differentiated(
        model, state0, state, dt, forces, CELLS)
    assert report['analytic_all_zero']
    assert report['numeric_max'] < 1e-9      # small, as the choice assumes


def test_a_step_across_a_pvt_table_node_spoils_the_comparison(case):
    """Not a tolerance to tune. The deck's PVT tables are piecewise
    linear, so differencing across a table node compares a secant over a
    kink against a one-sided derivative. Pinned so nobody later 'fixes'
    the step size back to something that crosses one."""
    model, state0, state, dt, forces = case
    J = np.asarray(V.jacobian_wrt_state(
        model, state0, state, dt, forces).todense())[:, list(CELLS)]

    fine = V.finite_difference_columns(model, state0, state, dt, forces,
                                       'pressure', 'state', cells=CELLS,
                                       h=8.0)
    coarse = V.finite_difference_columns(model, state0, state, dt, forces,
                                         'pressure', 'state', cells=CELLS,
                                         h=64.0)
    assert np.max(np.abs(J - fine)) < np.max(np.abs(J - coarse)) / 100


# ------------------------------------------------------------ getStateAD --

def test_get_state_ad_seeds_every_cell_primary_variable(case):
    model, state0, _, _, forces = case
    ad = model.getStateAD(state0, True, forces)
    for field in ('pressure', 'sW', 'x'):
        assert hasattr(ad[field], 'val'), field


def test_get_state_ad_reconstructs_the_fields_the_assembly_reads(case):
    """The assembly reads sG and rs, not x. Leaving them plain would
    strip the derivative one level below where it was seeded."""
    model, state0, _, _, forces = case
    ad = model.getStateAD(state0, True, forces)
    assert hasattr(ad['sG'], 'val')
    if model.disgas:
        assert hasattr(ad['rs'], 'val')


def test_get_state_ad_uses_the_forward_column_layout(case):
    """p at 0, sW at nc, x at 2nc -- the same offsets the forward
    assembly uses, so the two Jacobians line up."""
    model, state0, _, _, forces = case
    nc = model.G['cells']['num']
    ad = model.getStateAD(state0, True, forces)
    assert ad['pressure'].jac[:, 0].nnz > 0
    assert ad['sW'].jac[:, nc].nnz > 0
    assert ad['x'].jac[:, 2 * nc].nnz > 0


def test_get_state_ad_can_be_offset(case):
    model, state0, _, _, forces = case
    nc = model.G['cells']['num']
    nvar = 2 * (3 * nc + 8)
    ad = model.getStateAD(state0, True, forces, offset=3 * nc, nvar=nvar)
    assert ad['pressure'].jac.shape[1] == nvar
    assert ad['pressure'].jac[:, 3 * nc].nnz > 0


def test_get_state_ad_leaves_the_state_alone_when_not_initialising(case):
    model, state0, _, _, _ = case
    assert model.getStateAD(state0, False) is state0


# ------------------------------------------------------ parameter dR/dp --

def test_the_transmissibility_jacobian_matches_finite_differences(case):
    """dR/dT: what a transmissibility parameter contributes to the
    gradient."""
    model, state0, state, dt, forces = case
    faces = V.live_faces(model, state)
    report = V.check_parameter_jacobian(model, state0, state, dt, forces,
                                        'transmissibility', faces)
    assert report['passed'], report


def test_the_porevolume_jacobian_matches_finite_differences(case):
    model, state0, state, dt, forces = case
    report = V.check_parameter_jacobian(model, state0, state, dt, forces,
                                        'porevolume', (0, 1, 2))
    assert report['passed'], report


def test_a_dead_face_has_no_transmissibility_derivative(case):
    """Across a face carrying no flow, dR/dT is exactly zero -- the
    derivative is potential times mobility. Checking there compares
    nothing, which is why live_faces exists."""
    model, state0, state, dt, forces = case
    J = V.jacobian_wrt_parameter(model, state0, state, dt, forces,
                                 'transmissibility')
    dead = np.asarray(J.todense())[:, 0]        # face 0 is horizontal
    assert np.allclose(dead, 0.0)


def test_live_faces_are_the_ones_carrying_a_pressure_difference(case):
    model, _, state, _, _ = case
    faces = V.live_faces(model, state, count=5)
    c1, c2, _ = model._internal_connections()
    p = np.asarray(state['pressure'], dtype=float)
    assert faces, 'no face carries flow -- the check would be vacuous'
    assert all(abs(p[c2[f]] - p[c1[f]]) > 1.0 for f in faces)


def test_an_all_zero_reference_is_reported_as_a_failure(case):
    """The guard that would have caught the first version of the
    transmissibility check, which 'passed' against a column of zeros."""
    report = V.compare(np.zeros((4, 2)), np.zeros((4, 2)), 'nothing')
    assert not report['passed']
    assert 'identically' in report['reason']


def test_the_transmissibility_derivative_is_not_merely_zero(case):
    model, state0, state, dt, forces = case
    faces = V.live_faces(model, state)
    J = V.jacobian_wrt_parameter(model, state0, state, dt, forces,
                                 'transmissibility')
    block = np.asarray(J.todense())[:, faces]
    assert np.max(np.abs(block)) > 0.0


def test_seeding_a_parameter_leaves_the_model_as_it_found_it(case):
    """The bench swaps an operator for an AD one and must put it back,
    or every later evaluation carries a stray derivative."""
    model, state0, state, dt, forces = case
    before_T = np.asarray(model.operators['T'], dtype=float).copy()
    before_pv = model.porevolume
    V.jacobian_wrt_parameter(model, state0, state, dt, forces,
                             'transmissibility')
    V.jacobian_wrt_parameter(model, state0, state, dt, forces, 'porevolume')
    assert np.allclose(np.asarray(model.operators['T'], dtype=float),
                       before_T)
    assert model.porevolume is before_pv


def test_an_unknown_parameter_is_rejected(case):
    model, state0, state, dt, forces = case
    with pytest.raises(ValueError, match='Unknown parameter'):
        V.jacobian_wrt_parameter(model, state0, state, dt, forces, 'nosuch')


# ------------------------------------------------ the gradient, end to end --

NSTEPS = 3


@pytest.fixture(scope='module')
def tight(case):
    """A tightly converged case: the adjoint differentiates the root of
    the residual, so a finite-difference reference taken from a loosely
    converged solve measures something else. See V.TIGHT."""
    model, state0, _, dt, forces = case
    V.tighten(model)
    return model, state0, forces, dt / 20.0


def test_the_forward_run_actually_reaches_the_root(tight):
    """If it does not, the finite-difference reference below is the
    derivative of the solver's output rather than of the solution, and
    the comparison is meaningless. At default tolerances SPE1 stops with
    |R| near 10."""
    model, state0, forces, dt = tight
    states = V.run_forward(model, state0, dt, forces, NSTEPS)
    previous = state0
    for state in states:
        residual = V.residual(model, previous, state, dt, forces)
        assert np.max(np.abs(residual)) < 1e-9
        previous = state


def test_the_transmissibility_gradient_matches_finite_differences(tight):
    """Forward run, backward sweep, sensitivity accumulation -- against
    differencing the same objective through complete re-runs."""
    model, state0, forces, dt = tight
    states = V.run_forward(model, state0, dt, forces, NSTEPS)
    faces = V.live_faces(model, states[-1])
    report = V.check_gradient(model, state0, forces, dt, NSTEPS,
                              'transmissibility', faces)
    assert report['passed'], report


def test_the_porevolume_gradient_matches_finite_differences(tight):
    model, state0, forces, dt = tight
    report = V.check_gradient(model, state0, forces, dt, NSTEPS,
                              'porevolume', (0, 1, 150), relative_step=1e-4)
    assert report['passed'], report


def test_the_gradient_is_not_merely_zero(tight):
    """A zero gradient would pass any comparison against a small number.
    compute_sensitivities_adjoint_ad returns exactly that, which is the
    failure this whole bench exists to make impossible."""
    from PRSTCore.ad_core.simulators.adjoint_sweep import adjoint_gradient

    model, state0, forces, dt = tight
    _, partials = V.pressure_sum_objective(model, forces)
    states = V.run_forward(model, state0, dt, forces, NSTEPS)
    grad = adjoint_gradient(model, state0, states, [dt] * NSTEPS, forces,
                            ['porevolume'], partials)['porevolume']
    assert np.max(np.abs(grad)) > 1.0


def test_the_objective_sees_every_step(tight):
    """An objective reading only the final state barely exercises the
    coupling term, which is the entire reason the sweep runs backwards."""
    model, state0, forces, dt = tight
    value, _ = V.pressure_sum_objective(model, forces)
    states = V.run_forward(model, state0, dt, forces, NSTEPS)
    assert value(states) > value(states[-1:])


def test_a_finite_difference_run_leaves_the_model_unchanged(tight):
    """Each perturbed re-run swaps an operator; failing to put it back
    would corrupt every later step of the same check."""
    model, state0, forces, dt = tight
    before = np.asarray(model.operators['T'], dtype=float).copy()
    perturbed = before * 1.5
    V.run_forward(model, state0, dt, forces, 1,
                  parameter='transmissibility', values=perturbed)
    assert np.allclose(np.asarray(model.operators['T'], dtype=float), before)


def test_tighten_reports_what_it_replaced(tight):
    """So a caller can restore the model's own tolerances."""
    model = tight[0]
    previous = V.tighten(model)
    assert set(previous) == set(V.TIGHT)
    assert model.toleranceCNV == V.TIGHT['toleranceCNV']


# ------------------------------------- the MRST-style entry point --

def _params(model):
    class _P:
        def __init__(self, name, n):
            self.name, self.n_param = name, n
    nc = model.G['cells']['num']
    nf = np.size(model.operators['T'])
    # 'conntrans' is the well connection transmissibility, which is
    # deliberately not differentiated -- see
    # check_well_scaling_is_not_differentiated. It stands here for any
    # parameter with no derivative path.
    return [_P('porevolume', nc), _P('transmissibility', nf),
            _P('conntrans', nc)]


def _setup(model, state0, forces, dt, nsteps):
    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import \
        init_eclipse_problem_ad
    _, _, schedule, _ = init_eclipse_problem_ad(DECK)
    return {'model': model, 'state0': state0,
            'schedule': {'step': {'val': [dt] * nsteps,
                                  'control': [0] * nsteps},
                         'control': schedule['control']}}


def test_the_entry_point_returns_a_real_gradient(tight):
    """It returned zeros for every parameter until the derivatives under
    it were verified. Zeros are indistinguishable from convergence: the
    optimiser takes no steps and reports success."""
    from PRSTCore.ad_core.simulators.compute_sensitivities_adjoint_ad import \
        compute_sensitivities_adjoint_ad

    model, state0, forces, dt = tight
    states = V.run_forward(model, state0, dt, forces, NSTEPS)
    _, partials = V.pressure_sum_objective(model, forces)

    grads = compute_sensitivities_adjoint_ad(
        _setup(model, state0, forces, dt, NSTEPS), states, _params(model),
        lambda step, m, state: partials(step, state))

    assert np.max(np.abs(grads['porevolume'])) > 1.0
    assert np.max(np.abs(grads['transmissibility'])) > 1.0


def test_the_entry_point_agrees_with_the_verified_sweep(tight):
    """The same numbers the end-to-end check validated, reached through
    MRST's calling convention."""
    from PRSTCore.ad_core.simulators.adjoint_sweep import adjoint_gradient
    from PRSTCore.ad_core.simulators.compute_sensitivities_adjoint_ad import \
        compute_sensitivities_adjoint_ad

    model, state0, forces, dt = tight
    states = V.run_forward(model, state0, dt, forces, NSTEPS)
    _, partials = V.pressure_sum_objective(model, forces)

    direct = adjoint_gradient(model, state0, states, [dt] * NSTEPS, forces,
                              ['porevolume'], partials)['porevolume']
    through = compute_sensitivities_adjoint_ad(
        _setup(model, state0, forces, dt, NSTEPS), states, _params(model),
        lambda step, m, state: partials(step, state))['porevolume']
    assert np.allclose(direct, through, rtol=1e-12)


def test_an_unsupported_parameter_gets_zeros_and_says_so(tight):
    """Silently returning zeros for a parameter with no derivative path
    is the failure this replaced; returning them loudly is honest."""
    from PRSTCore.ad_core.simulators.compute_sensitivities_adjoint_ad import \
        compute_sensitivities_adjoint_ad

    model, state0, forces, dt = tight
    states = V.run_forward(model, state0, dt, forces, 1)
    _, partials = V.pressure_sum_objective(model, forces)

    with pytest.warns(RuntimeWarning,
                      match='No adjoint derivative for conntrans'):
        grads = compute_sensitivities_adjoint_ad(
            _setup(model, state0, forces, dt, 1), states, _params(model),
            lambda step, m, state: partials(step, state))
    assert np.allclose(grads['conntrans'], 0.0)


def test_every_requested_parameter_comes_back(tight):
    from PRSTCore.ad_core.simulators.compute_sensitivities_adjoint_ad import \
        compute_sensitivities_adjoint_ad

    model, state0, forces, dt = tight
    states = V.run_forward(model, state0, dt, forces, 1)
    _, partials = V.pressure_sum_objective(model, forces)
    with pytest.warns(RuntimeWarning):
        grads = compute_sensitivities_adjoint_ad(
            _setup(model, state0, forces, dt, 1), states, _params(model),
            lambda step, m, state: partials(step, state))
    assert set(grads) == {'porevolume', 'transmissibility', 'conntrans'}
    assert grads['porevolume'].size == model.G['cells']['num']


# ------------------------------- the two entry points must agree --

def _model_parameters(model, setup):
    """Real ``ModelParameter`` objects, which the MRST-faithful entry
    point needs -- it seeds them through ``set_parameter``, so the stub
    used above is not enough."""
    from PRSTCore.optimization.utils.parameters import add_parameter

    params = add_parameter([], setup, name='porevolume',
                           relative_limits=[0.5, 2.0])
    params = add_parameter(params, setup, name='transmissibility',
                           relative_limits=[0.5, 2.0])
    return params


def test_the_faithful_entry_point_agrees_with_the_verified_one(tight):
    """``compute_sensitivities_adjoint`` is the port of MRST's own loop:
    ``solveAdjoint`` plus ``partialWRTparam``, which seeds *every*
    parameter into one assembly per step. ``compute_sensitivities_adjoint_ad``
    reaches the same numbers the slow way -- a separate seeded assembly
    per parameter, plus a repeated forward one -- and is the version the
    finite-difference checks above validate.

    They must agree, because the fast one is what the history match
    should be calling: on QIEDIE the slow path costs nineteen assemblies
    per step against MRST's three.
    """
    from PRSTCore.ad_core.simulators.compute_sensitivities_adjoint_ad import \
        compute_sensitivities_adjoint_ad
    from PRSTCore.ad_core.simulators.sensitivities_adjoint import \
        compute_sensitivities_adjoint

    model, state0, forces, dt = tight
    states = V.run_forward(model, state0, dt, forces, NSTEPS)
    _, partials = V.pressure_sum_objective(model, forces)
    setup = _setup(model, state0, forces, dt, NSTEPS)
    params = _model_parameters(model, setup)

    def objective(step, _model, state):
        return partials(step, state)

    slow = compute_sensitivities_adjoint_ad(setup, states, params, objective)
    fast = compute_sensitivities_adjoint(setup, states, params, objective)

    for name in ('porevolume', 'transmissibility'):
        a = np.asarray(slow[name], dtype=float).ravel()
        b = np.asarray(fast[name], dtype=float).ravel()
        assert a.size == b.size, name
        scale = max(np.max(np.abs(a)), 1e-300)
        assert np.max(np.abs(a - b)) / scale < 1e-8, name
