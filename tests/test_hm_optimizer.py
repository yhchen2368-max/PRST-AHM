"""Tests for the ported parts of MRST ``hm/utils/evaluate`` and
``hm/utils/optimizer``."""

import numpy as np
import pytest

from PRSTCore.hm.utils.evaluate.evaluateMatchFromEclipseRun import (
    build_simulator_command, needs_well_index_recompute)
from PRSTCore.hm.utils.evaluate.evaluateMatchFromJutulRun import (
    build_jutul_driver)
from PRSTCore.hm.utils.evaluate.getEclipseSimResults import (_sortWellSol,
                                                             _stable_order)
from PRSTCore.hm.utils.evaluate.wellSensitivitesOW import wellSensitivitesOW
from PRSTCore.hm.utils.optimizer.checkParameterConsistency import (
    _enforceBoxLimits, _intersect_stable, _setdiff_stable)
from PRSTCore.hm.utils.optimizer.unitBoxLMMulti import unitBoxLMMulti
from PRSTCore.hm.utils.optimizer.unitBoxLMMulti2 import unitBoxLMMulti2

DAY = 86400.0


# ------------------------------------------------- wellSensitivitesOW --

def _sol(qWs, qOs, bhp, status=True):
    return {'name': 'W', 'qWs': qWs, 'qOs': qOs, 'bhp': bhp, 'status': status}


def test_sensitivity_sums_the_selected_quantity():
    schedule = {'step': {'val': np.array([DAY]), 'control': np.array([1])}}
    states = [{'wellSol': [_sol(-1.0, -2.0, 5.0), _sol(-3.0, -4.0, 7.0)]}]
    observed = states
    out = wellSensitivitesOW(None, states, schedule, observed,
                             ProductionIndices='qOs')
    assert out[0] == pytest.approx(-6.0)


def test_sensitivity_honours_the_well_subset():
    schedule = {'step': {'val': np.array([DAY]), 'control': np.array([1])}}
    states = [{'wellSol': [_sol(-1.0, -2.0, 5.0), _sol(-3.0, -4.0, 7.0)]}]
    out = wellSensitivitesOW(None, states, schedule, states,
                             ProductionIndices='qWs', WellIndices=[1])
    assert out[0] == pytest.approx(-3.0)


def test_sensitivity_rejects_an_unknown_quantity():
    with pytest.raises(ValueError, match='ProductionIndices'):
        wellSensitivitesOW(None, [], {'step': {'val': np.array([DAY])}}, [],
                           ProductionIndices='nosuch')


# ------------------------------------------------- simulator plumbing --

@pytest.mark.parametrize('simulator, expected', [
    ('eclipse', 'eclrun eclipse case.DATA'),
    ('e300', 'eclrun e300 case.DATA'),
])
def test_eclipse_commands(simulator, expected):
    assert build_simulator_command(simulator, 'case.DATA') == expected


def test_tnavigator_command_carries_its_flags():
    cmd = build_simulator_command('tNavigator', 'case.DATA')
    assert cmd.startswith('tNavigator ') and '--no-gui' in cmd


def test_unsupported_simulator_is_rejected():
    with pytest.raises(ValueError, match='Unsupported simulator'):
        build_simulator_command('nosuch', 'case.DATA')


@pytest.mark.parametrize('name, expected', [
    ('permx', True), ('PERMZ', True), ('porevolume', False),
])
def test_permeability_parameters_trigger_a_well_index_recompute(name, expected):
    assert needs_well_index_recompute([{'name': name}]) is expected


def test_jutul_driver_carries_the_solver_settings():
    text = build_jutul_driver('/tmp/case')
    assert 'using Jutul, JutulDarcy' in text
    assert 'jpth = "/tmp/case"' in text
    assert 'tol_cnv = 1e-2,' in text
    assert 'linear_solver = :gmres,' in text


# ------------------------------------------------ restart realignment --

def test_stable_order_puts_eclipse_wells_back_in_schedule_order():
    assert _stable_order(['A', 'B', 'C'], ['C', 'A', 'B']) == [1, 2, 0]


def test_sort_wellsol_reorders_per_perforation_arrays():
    wellSol = [{'name': 'P1', 'cells': np.array([5, 3, 4]),
                'WI': np.array([0.5, 0.3, 0.4])}]
    out = _sortWellSol(wellSol, [np.array([1, 2, 0])])
    assert list(out[0]['cells']) == [3, 4, 5]
    assert np.allclose(out[0]['WI'], [0.3, 0.4, 0.5])
    assert out[0]['name'] == 'P1'          # scalars untouched


# --------------------------------------------- consistency set helpers --

def test_setdiff_stable_keeps_order_and_positions():
    values, idx = _setdiff_stable([5, 3, 9, 3], [3])
    assert list(values) == [5, 9]
    assert list(idx) == [0, 2]


def test_intersect_stable_returns_both_index_vectors():
    values, ia, ib = _intersect_stable([5, 3, 9], [9, 5])
    assert list(values) == [5, 9]
    assert list(ia) == [0, 2]
    assert list(ib) == [1, 0]


def test_box_limits_tighten_only_where_they_are_too_wide():
    limits = np.array([[0.0, 1.0], [0.0, 0.2]])
    out, done = _enforceBoxLimits(limits, np.array([0.5, 0.5]),
                                  np.array([0, 1]), 'u')
    assert done is True
    assert out[0, 1] == pytest.approx(0.5)   # 1.0 was too wide
    assert out[1, 1] == pytest.approx(0.2)   # 0.2 already tighter


def test_box_limits_report_when_nothing_changed():
    limits = np.array([[0.0, 0.1]])
    _, done = _enforceBoxLimits(limits, np.array([0.5]), np.array([0]), 'u')
    assert done is False


def test_lower_bound_flag_raises_the_floor():
    limits = np.array([[0.0, 1.0]])
    out, done = _enforceBoxLimits(limits, np.array([0.3]), np.array([0]), 'l')
    assert done is True and out[0, 0] == pytest.approx(0.3)


# --------------------------------------------------------- unitBoxLM --

def _quadratic(target):
    """A linear least-squares problem whose minimum is at ``target``."""
    target = np.asarray(target, dtype=float)

    def f(u):
        u = np.asarray(u, dtype=float)
        res = (u - target).reshape(-1, 1)
        J = [np.eye(u.size)]
        return res, J
    return f


def test_lm_finds_an_interior_minimum():
    v, u, _ = unitBoxLMMulti(_quadratic([0.3, 0.7]), np.array([0.5, 0.5]),
                             maxIt=30, verbose=False)
    assert np.allclose(u, [0.3, 0.7], atol=1e-3)
    # The default stopping criterion is resTolAbs = 1e-5, so that is the
    # bar the objective has to clear, not zero.
    assert v < 1e-5


def test_lm_projects_onto_the_unit_box():
    """The minimum sits outside the box; the answer must stay inside."""
    _, u, _ = unitBoxLMMulti(_quadratic([-0.5, 1.5]), np.array([0.5, 0.5]),
                             maxIt=30, verbose=False)
    assert np.all(u >= -1e-12) and np.all(u <= 1 + 1e-12)
    assert u[0] == pytest.approx(0.0, abs=1e-6)
    assert u[1] == pytest.approx(1.0, abs=1e-6)


def test_lm_history_records_each_accepted_iteration():
    _, _, h = unitBoxLMMulti(_quadratic([0.3, 0.7]), np.array([0.5, 0.5]),
                             maxIt=5, verbose=False)
    assert np.isfinite(h['val'][1])
    assert h['u'][1] is not None


def test_lm_respects_the_iteration_cap():
    _, _, h = unitBoxLMMulti(_quadratic([0.3, 0.7]), np.array([0.9, 0.1]),
                             maxIt=3, verbose=False)
    assert np.count_nonzero(np.isfinite(h['val'])) <= 5


def test_lm_trust_region_strategy_also_converges():
    v, u, _ = unitBoxLMMulti(_quadratic([0.3, 0.7]), np.array([0.5, 0.5]),
                             updateStrategy='TR', maxIt=30, verbose=False)
    assert np.allclose(u, [0.3, 0.7], atol=1e-2)


def test_lm_scaled_damping_also_converges():
    _, u, _ = unitBoxLMMulti(_quadratic([0.3, 0.7]), np.array([0.5, 0.5]),
                             scaledDamping=True, maxIt=30, verbose=False)
    assert np.allclose(u, [0.3, 0.7], atol=1e-2)


def test_lm_variants_agree_on_a_single_case():
    """With one case the two gradient forms coincide."""
    _, u1, _ = unitBoxLMMulti(_quadratic([0.3, 0.7]), np.array([0.5, 0.5]),
                              maxIt=30, verbose=False)
    _, u2, _ = unitBoxLMMulti2(_quadratic([0.3, 0.7]), np.array([0.5, 0.5]),
                               maxIt=30, verbose=False)
    assert np.allclose(u1, u2, atol=1e-6)


def test_lm_variants_differ_when_cases_have_different_jacobians():
    """unitBoxLMMulti weights each case by its own sensitivity;
    unitBoxLMMulti2 by the ensemble-mean sensitivity."""
    def f(u):
        u = np.asarray(u, dtype=float)
        res = np.column_stack([u - 0.2, 2.0 * (u - 0.8)])
        return res, [np.eye(u.size), 2.0 * np.eye(u.size)]

    _, u1, _ = unitBoxLMMulti(f, np.array([0.5]), maxIt=30, verbose=False)
    _, u2, _ = unitBoxLMMulti2(f, np.array([0.5]), maxIt=30, verbose=False)
    assert not np.allclose(u1, u2, atol=1e-3)
