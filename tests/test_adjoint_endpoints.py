"""Adjoint derivatives for the relative-permeability endpoints.

These are the eleven quantities FAHM's Parameter tab tunes alongside pore
volume and permeability: the connate, critical and maximum saturations of
water and gas, the two critical oil saturations, and the three maximum
relative permeabilities. Before this they had no derivative path, so an
adjoint gradient over them came back zero -- indistinguishable from a
converged answer.

**The deck needs ``ENDSCALE``.** Without it a model has no end-point
scaling: the endpoints do not enter its residual and their derivative is
genuinely zero. SPE1 does not ask for it, so these build a copy that
does.

**The state has to be three-phase.** At SPE1's own initial state
sG = 0 everywhere and the oil saturation sits on a flat segment of the
krow table, so seven of the eleven have a zero derivative -- correctly,
but a check against zero proves nothing. sW = 0.33 and sG = 0.11 puts
every curve on a sloped segment, and off a table node, where the
derivative exists.
"""

import io
import os

import numpy as np
import pytest

from PRSTCore.ad_core.adi import SparseADI, ad_select
from PRSTCore.ad_core.simulators import adjoint_verification as V
from PRSTCore.ad_core.simulators.compute_sensitivities_adjoint_ad import \
    SUPPORTED

DECK = 'examples/SpE1/SPE1CASE2.DATA'

#: Cells to difference. Spread across the grid so a mistake that happens
#: to vanish in one place still shows.
CELLS = (0, 100, 150, 299)

ENDPOINTS = ('swl', 'swcr', 'swu', 'krw', 'sgl', 'sgcr', 'sgu', 'krg',
             'sowcr', 'sogcr', 'kro')


@pytest.fixture(scope='module')
def endscale_deck(tmp_path_factory):
    """SPE1 with end-point scaling switched on."""
    if not os.path.exists(DECK):
        pytest.skip('SPE1 deck not present')
    text = io.open(DECK, encoding='utf-8', errors='replace').read()
    text = text.replace('RUNSPEC', 'RUNSPEC\n\nENDSCALE\n/\n', 1)
    path = tmp_path_factory.mktemp('endscale') / 'SPE1_ENDSCALE.DATA'
    io.open(str(path), 'w', encoding='utf-8').write(text)
    return str(path)


@pytest.fixture(scope='module')
def case(endscale_deck):
    model, state0, forces, dt = V.build_case(endscale_deck)
    V.tighten(model)
    nc = int(model.G['cells']['num'])

    state = {k: (v.copy() if isinstance(v, np.ndarray) else v)
             for k, v in state0.items()}
    state['sW'] = np.full(nc, 0.33)
    state['sG'] = np.full(nc, 0.11)
    if 's' in state:
        state['s'] = np.column_stack([state['sW'],
                                      1.0 - state['sW'] - state['sG'],
                                      state['sG']])
    state['time'] = float(state0.get('time', 0.0)) + 1.0
    return model, state0, state, dt, forces


# --------------------------------------------------------- the groundwork --

def test_end_point_scaling_is_actually_active(case):
    """Everything below is vacuous if the deck's ENDSCALE was ignored:
    the endpoints would not enter the residual and every derivative
    would be zero on both sides."""
    model = case[0]
    nc = int(model.G['cells']['num'])
    scale = model._get_relperm_scaling(nc, model._get_relperm_tables())
    assert scale is not None
    assert set(scale['target']) == {'w', 'ow', 'og', 'g'}


def test_two_point_scaling_assembles_at_all(case):
    """Regression, and the reason this fixture uses two-point scaling.

    ``ad_select`` accepted a constant on the false branch but not the
    true one. Two-point scaling passes a constant zero as ``when_true``
    and so raised ``ad_select requires at least one SparseADI value``
    for every deck that used it -- QIEDIE among them, which is the deck
    being history matched. Three-point scaling passes AD on that side
    and was unaffected, which is why Norne (``SCALECRS YES``) never
    showed it and nothing in the suite caught it.
    """
    model, state0, state, dt, forces = case
    nc = int(model.G['cells']['num'])
    scale = model._get_relperm_scaling(nc, model._get_relperm_tables())
    assert scale['points'] == 2, 'the regression is in the two-point branch'

    problem, _ = model.get_equations(state0, state, dt, forces)
    assert problem['Jacobian'].shape == (908, 908)
    assert np.all(np.isfinite(problem['Residuals']))


def test_three_point_scaling_assembles_too(case):
    """The branch that always worked, kept honest alongside the one that
    did not. SCALECRS chooses between them."""
    model, state0, state, dt, forces = case
    nc = int(model.G['cells']['num'])
    scale = model._get_relperm_scaling(nc, model._get_relperm_tables())
    original = scale['points']
    scale['points'] = 3
    try:
        problem, _ = model.get_equations(state0, state, dt, forces)
        assert np.all(np.isfinite(problem['Residuals']))
    finally:
        scale['points'] = original


def test_ad_select_takes_a_constant_on_either_branch():
    """The narrow form of the same regression."""
    x = SparseADI.variable(np.array([1.0, 2.0, 3.0]), 3, 0)
    mask = np.array([True, False, True])

    from_false = ad_select(mask, np.zeros(3), x)
    from_true = ad_select(~mask, x, np.zeros(3))
    assert np.allclose(from_false.val, from_true.val)
    assert np.allclose(from_false.jac.toarray(), from_true.jac.toarray())
    # The constant branch contributes value but no derivative.
    assert from_false.val.tolist() == [0.0, 2.0, 0.0]


def test_two_constants_are_still_refused():
    """There is no derivative to select; the caller wants numpy.where."""
    with pytest.raises(TypeError):
        ad_select(np.array([True, False]), np.zeros(2), np.ones(2))


def test_the_state_is_three_phase_and_off_a_table_node(case):
    """Both matter. With no free gas, four of the endpoints have no
    effect; on a table node the residual has a kink and no derivative,
    which reads as a 17% adjoint error that is really the check's."""
    model, _state0, state, _dt, _forces = case
    assert np.all(state['sG'] > 0.02)          # above critical gas
    swof = model._get_relperm_tables()['swof']
    assert not np.any(np.isclose(swof[:, 0], state['sW'][0]))


def test_every_endpoint_is_declared_supported():
    """A parameter missing from SUPPORTED silently returns zeros."""
    for name in ENDPOINTS:
        assert name in SUPPORTED


# ------------------------------------------------------------ dR/dendpoint --

@pytest.mark.parametrize('name', ENDPOINTS)
def test_the_endpoint_jacobian_matches_finite_differences(case, name):
    """The step is deliberately large. These enter the residual through
    an affine rescaling of saturation, so a central difference loses
    digits to cancellation long before it gains anything from a smaller
    step -- the error *falls* as the step grows, which is round-off, not
    truncation."""
    model, state0, state, dt, forces = case
    analytic = V.jacobian_wrt_parameter(model, state0, state, dt, forces,
                                        name)
    ad = np.asarray(analytic[:, list(CELLS)].todense())
    fd = V.finite_difference_parameter(model, state0, state, dt, forces,
                                       name, entries=CELLS, h=1e-5)

    scale = float(np.max(np.abs(fd)))
    assert scale > 1e-6, 'reference is zero: %s does not bite here' % name
    assert np.max(np.abs(ad - fd)) / scale < 1e-5, name


def test_an_endpoint_without_endscale_has_no_derivative():
    """Not an omission: with no end-point scaling the parameter is not
    part of the residual, so zero is the right answer."""
    if not os.path.exists(DECK):
        pytest.skip('SPE1 deck not present')
    model, state0, forces, dt = V.build_case(DECK)
    state = V._perturbed_state(state0)
    assert V.endpoint_base(model, 'swcr') is None
    analytic = V.jacobian_wrt_parameter(model, state0, state, dt, forces,
                                        'swcr')
    assert analytic.nnz == 0


def test_seeding_leaves_the_model_as_it_found_it(case):
    """The seed is an AD object living where a float array normally is;
    leaking one would corrupt every later assembly."""
    model, state0, state, dt, forces = case
    before = V.residual(model, state0, state, dt, forces)
    V.jacobian_wrt_parameter(model, state0, state, dt, forces, 'swcr')
    V.finite_difference_parameter(model, state0, state, dt, forces, 'swcr',
                                  entries=(0,), h=1e-5)
    after = V.residual(model, state0, state, dt, forces)
    assert np.array_equal(before, after)
    assert not getattr(model, '_relperm_endpoint_seed', None)


def test_kro_moves_both_oil_curves(case):
    """ECLIPSE's KRO is the maximum oil relperm on the water-oil *and*
    the gas-oil curve; tuning only one of them would leave half the
    derivative behind."""
    from PRSTCore.ad_core.simulators.adjoint_verification import \
        ENDPOINT_COLUMNS
    assert ENDPOINT_COLUMNS['kro'] == (('ow', 3), ('og', 3))
    assert all(len(v) == 1 for k, v in ENDPOINT_COLUMNS.items()
               if k != 'kro')


# ------------------------------------------------------------- end to end --

@pytest.mark.parametrize('name', ['krw', 'krg', 'kro', 'sgcr', 'sowcr'])
def test_the_whole_chain_matches_a_finite_difference_gradient(case, name):
    """Forward run, backward sweep, sensitivity accumulation -- against
    differencing the same objective through complete re-runs.

    The best of a step sweep is taken because the finite-difference
    error is U-shaped in the step and each parameter's minimum sits in a
    different place. The sweep has to be wide enough to bracket all of
    them: ``krg`` bottoms out at 7e-09 with a step of 1e-3 and climbs to
    7e-05 at 1e-1, while ``krw`` runs the other way -- 1.4e-04 at 1e-3
    down to 1.8e-06 at 5e-2. A sweep that covers only one end reports
    whichever side of the curve it happened to land on, which is a
    statement about the reference and not about the adjoint.
    """
    model, state0, _state, dt, forces = case
    nc = int(model.G['cells']['num'])
    start = {k: (v.copy() if isinstance(v, np.ndarray) else v)
             for k, v in state0.items()}
    start['sW'] = np.full(nc, 0.33)
    start['sG'] = np.full(nc, 0.11)
    if 's' in start:
        start['s'] = np.column_stack([start['sW'],
                                      1.0 - start['sW'] - start['sG'],
                                      start['sG']])

    errors = [V.check_gradient(model, start, forces, dt / 50.0, 2, name,
                               (0, 150, 299), rtol=1e-4,
                               relative_step=step)['max_rel_diff']
              for step in (5e-2, 1e-2, 1e-3, 1e-4)]
    assert min(errors) < 1e-5, (name, errors)


def test_a_zero_endpoint_still_gets_a_usable_step(case):
    """Connate gas is zero, and a *relative* step of zero gives a nan
    gradient that reads as a failure of the adjoint rather than of the
    check."""
    model, state0, _state, dt, forces = case
    nc = int(model.G['cells']['num'])
    start = {k: (v.copy() if isinstance(v, np.ndarray) else v)
             for k, v in state0.items()}
    start['sW'] = np.full(nc, 0.33)
    start['sG'] = np.full(nc, 0.11)
    if 's' in start:
        start['s'] = np.column_stack([start['sW'],
                                      1.0 - start['sW'] - start['sG'],
                                      start['sG']])
    assert np.all(V.endpoint_base(model, 'sgl') == 0.0)
    report = V.check_gradient(model, start, forces, dt / 50.0, 2, 'sgl',
                              (0, 150, 299), rtol=1e-4, relative_step=1e-4)
    assert np.isfinite(report['max_rel_diff'])
    assert report['max_rel_diff'] < 1e-5
