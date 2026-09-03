import numpy as np

from PRSTCore.ad_core.models.generic_black_oil_model import GenericBlackOilModel
from PRSTCore.ad_core.operators import setup_operators


def _make_two_cell_model(gas=True):
    G = {
        'type': 'tensor',
        'xfaces': np.array([0.0, 1.0, 2.0]),
        'yfaces': np.array([0.0, 1.0]),
        'zfaces': np.array([0.0, 1.0]),
        'cells': {'num': 2, 'volumes': np.array([1.0, 1.0])},
        'cartDims': (2, 1, 1),
    }
    rock = {'perm': np.array([100.0, 100.0]), 'poro': np.array([0.2, 0.2])}
    model = GenericBlackOilModel(G=G, rock=rock, fluid={'viscosity': [1.0, 3.0]}, gas=gas)
    model.operators = setup_operators(G, rock)
    return model


def test_equations_sparse_jacobian_shape():
    model = _make_two_cell_model()
    state0 = {
        'pressure': np.array([100.0, 90.0]),
        'sW': np.array([0.2, 0.2]),
        'sG': np.array([0.0, 0.0]),
        'time': 0.0,
        'wellSol': [],
    }
    state = {
        'pressure': np.array([100.0, 90.0]),
        'sW': np.array([0.25, 0.15]),
        'sG': np.array([0.0, 0.0]),
        'time': 1.0,
        'wellSol': [],
    }
    driving = {'W': []}

    problem, _ = model.get_equations(state0, state, 1.0, driving)
    J = problem['Jacobian']
    r = problem['Residuals']

    assert r.shape == (6,)  # 3 eqns × 2 cells
    assert J.shape == (6, 6)  # (p,sW,sG) × 2
    assert hasattr(J, 'nnz')
    assert J.nnz > 0


def test_rate_producer_at_zero_sw_has_zero_water_rate():
    model = _make_two_cell_model()
    state0 = {
        'pressure': np.array([100.0, 100.0]),
        'sW': np.array([0.0, 0.0]),
        'sG': np.array([0.0, 0.0]),
        'time': 0.0,
        'wellSol': [],
    }
    state = {
        'pressure': np.array([100.0, 100.0]),
        'sW': np.array([0.0, 0.0]),
        'sG': np.array([0.0, 0.0]),
        'time': 1.0,
        'wellSol': [],
    }
    driving = {
        'W': [
            {
                'name': 'P1',
                'type': 'rate',
                'val': 1.0,
                'sign': -1,
                'status': True,
                'i': 1,
                'j': 1,
                'k': [1],
            }
        ]
    }

    problem, _ = model.get_equations(state0, state, 1.0, driving)
    assert len(problem['wellSol']) == 1
    ws = problem['wellSol'][0]
    assert ws['qOs'] > 0.0
    assert np.isclose(ws['qWs'], 0.0)
    assert np.isclose(ws.get('qGs', 0.0), 0.0)


def test_three_phase_jacobian_blocks():
    """Verify 3×3 block structure of Jacobian."""
    model = _make_two_cell_model()
    state0 = {
        'pressure': np.array([100.0, 90.0]),
        'sW': np.array([0.3, 0.3]),
        'sG': np.array([0.1, 0.1]),
        'time': 0.0, 'wellSol': [],
    }
    state = {
        'pressure': np.array([102.0, 88.0]),
        'sW': np.array([0.32, 0.28]),
        'sG': np.array([0.12, 0.08]),
        'time': 1.0, 'wellSol': [],
    }
    problem, state_out = model.get_equations(state0, state, 1.0, {'W': []})
    J = problem['Jacobian']
    assert J.shape == (6, 6)
    assert 'rs' in state_out
    assert 'rv' in state_out


# ------------------------------------- state0 differentiability (adjoint) --

def test_a_plain_state0_field_is_the_same_array_as_before():
    """The helper that lets AD through must be an exact no-op for the
    ordinary forward path."""
    model = _make_two_cell_model()
    values = np.array([1.0, 2.0, 3.0])
    out = model._state0_value(values)
    assert isinstance(out, np.ndarray) and out.dtype == float
    assert list(out) == [1.0, 2.0, 3.0]


def test_a_state0_field_given_as_a_list_is_still_an_array():
    model = _make_two_cell_model()
    assert isinstance(model._state0_value([1, 2]), np.ndarray)


def test_an_ad_state0_field_keeps_its_derivative():
    """Forcing state0 to plain floats -- which every assembly did -- makes
    dR/dx_{n-1} unobtainable, and with it the adjoint."""
    from PRSTCore.ad_core.adi import SparseADI

    model = _make_two_cell_model()
    ad = SparseADI.variable(np.array([1.0, 2.0]), 4, 0)
    out = model._state0_value(ad)
    assert hasattr(out, 'val'), 'the derivative was stripped'
    assert out is ad


def test_state0_selects_the_value_evaluators_when_it_is_plain():
    """A forward evaluation must go on using the value stack."""
    model = _make_two_cell_model()
    pp, pvt, pv = model._state0_fns(np.array([1.0]))
    assert pp == model._phase_pressures
    assert pv == model._mrst_pore_volume


def test_state0_selects_the_ad_evaluators_when_it_is_ad():
    from PRSTCore.ad_core.adi import SparseADI

    model = _make_two_cell_model()
    pp, pvt, pv = model._state0_fns(SparseADI.variable(np.array([1.0]), 1, 0))
    assert pp == model._phase_pressures_adi
    assert pv == model._mrst_pore_volume_adi
