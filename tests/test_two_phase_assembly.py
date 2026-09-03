"""The two-phase branches of the generic assembly, and the ResOnly one.

Three code paths in ``GenericBlackOilModel`` that nothing drove:

``_mrst_generic_adi_residual_ow``  water/oil, when a model has no gas
``_mrst_generic_adi_residual_og``  oil/gas, when it has no water
``get_equations(..., ResOnly=True)`` the residual without its Jacobian

All three raised on their first call. The two-phase pair referred to a
``_pvt0fn`` that only ever existed as a local of the three-phase method
-- they were left behind when the adjoint's state0 seeding introduced
``_state0_fns`` -- and ``ResOnly`` went to a second, numeric expression
of the same equations that had drifted into referring to ``rhoG`` and
``rhoG0``, neither of which existed.

None of it was caught because no test reached any of them: the eor and
tracer modules that mention ``_mrst_generic_adi_residual_ow`` in their
own docstrings carry their own equations and never call it, and no
black-oil model sets ``stepFunctionIsLinear``, which is ``ResOnly``'s
only route.

These drive them on a real deck. SPE1 is three-phase, so the two-phase
branches are reached by switching a phase off -- which is exactly the
condition the dispatch tests, and it exercises the assembly against real
PVT and saturation tables rather than a fixture with neither.
"""

import os

import numpy as np
import pytest

from PRSTCore.ad_core.simulators import adjoint_verification as V

DECK = 'examples/SpE1/SPE1CASE2.DATA'
NC = 300


@pytest.fixture
def case():
    """A fresh model per test: these switch phases off, and a shared one
    would carry that into the next test."""
    if not os.path.exists(DECK):
        pytest.skip('SPE1 deck not present')
    return V.build_case(DECK)


def _moved(state0, key, by):
    state = {k: (v.copy() if isinstance(v, np.ndarray) else v)
             for k, v in state0.items()}
    state[key] = np.clip(np.asarray(state0[key], dtype=float) + by, 0.0, 1.0)
    state['time'] = float(state0.get('time', 0.0)) + 1.0
    return state


def test_the_water_oil_branch_assembles(case):
    model, state0, forces, dt = case
    model.gas = False
    problem, _ = model.get_equations(state0, _moved(state0, 'sW', 0.02), dt,
                                     forces)
    residual = np.asarray(problem['Residuals'], dtype=float)
    # Two conservation equations over 300 cells, plus three well rows for
    # each of two wells.
    assert residual.size == 2 * NC + 3 * 2
    assert np.all(np.isfinite(residual))
    assert problem['Jacobian'].nnz > 0


def test_the_oil_gas_branch_assembles(case):
    model, state0, forces, dt = case
    model.water = False
    problem, _ = model.get_equations(state0, _moved(state0, 'sG', 0.02), dt,
                                     forces)
    residual = np.asarray(problem['Residuals'], dtype=float)
    assert residual.size == 2 * NC + 3 * 2
    assert np.all(np.isfinite(residual))


def test_the_two_phase_branches_name_their_own_equations(case):
    """A branch that quietly fell through to the three-phase assembly
    would still produce finite numbers."""
    model, state0, forces, dt = case
    model.gas = False
    problem, _ = model.get_equations(state0, _moved(state0, 'sW', 0.02), dt,
                                     forces)
    assert set(problem['equationNames']) == {'water', 'oil', 'waterWells',
                                             'oilWells', 'closureWells'}
    assert 'gas' not in problem['equationNames']


def test_state0_goes_through_the_same_property_stack_everywhere(case):
    """``_state0_fns`` dispatches on whether state0 is a constant or an
    AD variable. The three-phase assembly has used it since the adjoint
    was added; the two-phase pair called the value stack directly and
    referred to a name that did not exist there."""
    import inspect

    from PRSTCore.ad_core.models.generic_black_oil_model import \
        GenericBlackOilModel as Model
    for name in ('_mrst_generic_adi_residual',
                 '_mrst_generic_adi_residual_ow',
                 '_mrst_generic_adi_residual_og'):
        source = inspect.getsource(getattr(Model, name))
        assert '_state0_fns' in source, name


# ------------------------------------------------------------- ResOnly --

def test_res_only_returns_the_same_residual(case):
    """It used to raise. Now it is the value half of the one assembly, so
    it cannot disagree with the full one -- which is the point of not
    keeping a second expression of the same equations around."""
    model, state0, forces, dt = case
    state = _moved(state0, 'sW', 0.02)
    full, _ = model.get_equations(state0, state, dt, forces)
    bare, _ = model.get_equations(state0, state, dt, forces, ResOnly=True)
    assert np.array_equal(np.asarray(full['Residuals'], dtype=float),
                          np.asarray(bare['Residuals'], dtype=float))


def test_res_only_drops_the_jacobian(case):
    model, state0, forces, dt = case
    state = _moved(state0, 'sW', 0.02)
    full, _ = model.get_equations(state0, state, dt, forces)
    bare, _ = model.get_equations(state0, state, dt, forces, ResOnly=True)
    assert full['Jacobian'].nnz > 0
    assert bare['Jacobian'].nnz == 0
    assert bare['Jacobian'].shape == full['Jacobian'].shape


def test_no_undefined_names_survive_in_the_model():
    """The three defects above were all of one kind: a name that had
    stopped existing, in a branch nothing ran. pyflakes finds those
    statically, which is cheaper than a deck that reaches every branch.
    """
    pyflakes = pytest.importorskip('pyflakes.api')
    import pyflakes.reporter

    class Collect(pyflakes.reporter.Reporter):
        def __init__(self):
            super().__init__(open(os.devnull, 'w'), open(os.devnull, 'w'))
            self.undefined = []

        def flake(self, message):
            if 'undefined name' in str(message):
                self.undefined.append(str(message))

    path = os.path.join('PRSTCore', 'ad_core', 'models',
                        'generic_black_oil_model.py')
    if not os.path.exists(path):
        pytest.skip('model source not present')
    reporter = Collect()
    pyflakes.api.checkPath(path, reporter)
    assert reporter.undefined == []


# ----------------------------------------------------- the ported pieces --
def _assemble(model, state0, forces, dt, **kwargs):
    return model.get_equations(state0, _moved(state0, 'sW', 0.02), dt,
                               forces, **kwargs)[0]


@pytest.mark.parametrize('phase_off', ['gas', 'water'])
def test_both_flux_paths_agree_on_a_two_phase_model(case, phase_off):
    """The face operators must assemble what the general operators do.

    The two-phase branches named ``SparseADI`` outright and went through
    ``linear_map``, so neither the compiled flux kernels nor the diagonal
    representation reached them -- and a two-phase deck is exactly what the
    large field models are.
    """
    model, state0, forces, dt = case
    setattr(model, phase_off, False)

    model.useFaceOperators = False
    model._face_flux_cache = None
    general = _assemble(model, state0, forces, dt)

    model.useFaceOperators = True
    model._face_flux_cache = None
    fast = _assemble(model, state0, forces, dt)
    assert model._face_flux_cache is not None, 'the fast path was not taken'

    scale = max(float(np.abs(general['Residuals']).max()), 1.0)
    assert np.abs(np.asarray(general['Residuals'])
                  - np.asarray(fast['Residuals'])).max() <= 1e-12 * scale

    difference = (general['Jacobian'].tocsr() - fast['Jacobian'].tocsr()).tocoo()
    if difference.nnz:
        largest = max(float(abs(general['Jacobian']).max()), 1.0)
        assert np.abs(difference.data).max() <= 1e-12 * largest, (
            'the flux paths disagree by %g' % np.abs(difference.data).max())


@pytest.mark.parametrize('phase_off', ['gas', 'water'])
def test_both_backends_agree_on_a_two_phase_model(case, phase_off):
    """The diagonal representation must reach the two-phase branches too."""
    model, state0, forces, dt = case
    setattr(model, phase_off, False)

    model.autodiff_backend = 'sparse'
    sparse = _assemble(model, state0, forces, dt)
    model.autodiff_backend = 'diagonal'
    model._face_flux_cache = None
    diagonal = _assemble(model, state0, forces, dt)

    np.testing.assert_allclose(np.asarray(sparse['Residuals']),
                               np.asarray(diagonal['Residuals']),
                               rtol=1e-12, atol=1e-12)
    difference = (sparse['Jacobian'].tocsr() - diagonal['Jacobian'].tocsr()).tocoo()
    if difference.nnz:
        largest = max(float(abs(sparse['Jacobian']).max()), 1.0)
        assert np.abs(difference.data).max() <= 1e-12 * largest


@pytest.mark.parametrize('phase_off', ['gas', 'water'])
def test_reverse_mode_reaches_the_two_phase_branches(case, phase_off):
    """The adjoint needs ``dR/dx`` with the *previous* state seeded.

    Both two-phase branches ignored ``reverseMode`` entirely, so they
    differentiated with respect to the current state whatever was asked --
    an adjoint built on them would have been quietly wrong rather than
    unavailable.  Seeding state0 is what makes the width follow it.
    """
    model, state0, forces, dt = case
    setattr(model, phase_off, False)
    model._face_flux_cache = None

    problem = _assemble(model, state0, forces, dt, reverseMode=True)
    residual = np.asarray(problem['Residuals'])
    assert np.all(np.isfinite(residual))
    # Nothing in the current state is a variable any more, so the assembly
    # carries no derivatives with respect to it -- which is the point.
    assert problem['Jacobian'].shape[0] == residual.size
    # And the fast flux path stood down, because its column layout no
    # longer describes the seeded variables.
    assert getattr(model, '_face_flux_cache', None) is None
