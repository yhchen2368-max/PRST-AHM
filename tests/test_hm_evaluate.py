"""Tests for the port of MRST ``hm/utils/evaluate``."""

import numpy as np
import pytest

from PRSTCore.hm.utils.evaluate.evaluateObjective import (_perturb, _slices,
                                                          evaluateObjective)
from PRSTCore.hm.utils.evaluate.matchObservedLW import (_expandToFull,
                                                        _getWeights,
                                                        matchObservedLW)

DAY = 86400.0


def _patch_simulator(monkeypatch):
    """Stub the forward simulation.

    ``simulators/__init__`` rebinds the submodule name to the function it
    exports, so the module has to be reached through sys.modules.
    """
    import importlib
    module = importlib.import_module(
        'PRSTCore.ad_core.simulators.simulate_schedule_ad')
    monkeypatch.setattr(module, 'simulate_schedule_ad',
                        lambda *a, **k: ([], [{}]))


def _sol(qWs, qOs, bhp, sign=-1, status=True):
    return {'name': 'W', 'qWs': qWs, 'qOs': qOs, 'bhp': bhp,
            'sign': sign, 'status': status}


def _case(sim, obs, nstep=1):
    """One-well, ``nstep``-step case with the given sim/obs values."""
    schedule = {'step': {'val': np.full(nstep, DAY),
                         'control': np.ones(nstep, dtype=int)}}
    states = [{'wellSol': [_sol(*sim)]} for _ in range(nstep)]
    observed = [{'wellSol': [_sol(*obs)]} for _ in range(nstep)]
    return None, states, schedule, observed


# ------------------------------------------------------------- weights --

def test_rate_weight_is_the_reciprocal_magnitude():
    wl, wc, wp = _getWeights(np.array([4.0]), np.array([1.0]),
                             np.array([1.0, 3.0]), None, None, None)
    assert np.allclose(wl, 0.25)
    assert wc == 1.0
    assert wp == pytest.approx(0.5)          # 1 / (3 - 1)


def test_rate_weight_is_zero_when_every_rate_is_zero():
    wl, _, _ = _getWeights(np.zeros(2), np.ones(2), np.ones(2),
                           None, None, None)
    assert wl == 0.0


def test_bhp_weight_is_zero_when_pressure_is_flat():
    _, _, wp = _getWeights(np.ones(2), np.ones(2), np.full(2, 5.0),
                           None, None, None)
    assert wp == 0.0


def test_explicit_weights_win():
    wl, wc, wp = _getWeights(np.ones(2), np.ones(2), np.ones(2), 2.0, 3.0, 4.0)
    assert (wl, wc, wp) == (2.0, 3.0, 4.0)


# ------------------------------------------------------------ mismatch --

def test_a_perfect_match_gives_zero():
    model, states, schedule, observed = _case((-1.0, -2.0, 2.0e7),
                                              (-1.0, -2.0, 2.0e7))
    obj = matchObservedLW(model, states, schedule, observed,
                          LiquidRateWeight=1.0, WaterCutWeight=1.0,
                          BHPWeight=1.0, fix_observed_water_cut=True)
    assert obj[0] == pytest.approx(0.0)


def test_mismatch_grows_with_the_discrepancy():
    close = matchObservedLW(*_case((-1.0, -2.0, 2.0e7), (-1.1, -2.0, 2.0e7)),
                            LiquidRateWeight=1.0, WaterCutWeight=0.0,
                            BHPWeight=0.0)[0]
    far = matchObservedLW(*_case((-1.0, -2.0, 2.0e7), (-2.0, -2.0, 2.0e7)),
                          LiquidRateWeight=1.0, WaterCutWeight=0.0,
                          BHPWeight=0.0)[0]
    assert 0 < close < far


def test_steps_are_weighted_by_their_share_of_total_time():
    """Each step contributes dt/T, so N identical steps sum to one step."""
    one = matchObservedLW(*_case((-1.0, -2.0, 2.0e7), (-2.0, -2.0, 2.0e7), 1),
                          LiquidRateWeight=1.0, WaterCutWeight=0.0,
                          BHPWeight=0.0)
    four = matchObservedLW(*_case((-1.0, -2.0, 2.0e7), (-2.0, -2.0, 2.0e7), 4),
                           LiquidRateWeight=1.0, WaterCutWeight=0.0,
                           BHPWeight=0.0)
    assert sum(four) == pytest.approx(one[0])


def test_tstep_selects_a_single_step():
    model, states, schedule, observed = _case((-1.0, -2.0, 2.0e7),
                                              (-2.0, -2.0, 2.0e7), 3)
    obj = matchObservedLW(model, states, schedule, observed, tStep=[1],
                          LiquidRateWeight=1.0, WaterCutWeight=0.0,
                          BHPWeight=0.0)
    assert len(obj) == 1


def test_match_only_producers_skips_injectors():
    schedule = {'step': {'val': np.array([DAY]), 'control': np.array([1])}}
    states = [{'wellSol': [_sol(-1.0, -2.0, 2.0e7, sign=-1),
                           _sol(5.0, 0.0, 3.0e7, sign=1)]}]
    observed = [{'wellSol': [_sol(-1.0, -2.0, 2.0e7, sign=-1),
                             _sol(9.0, 0.0, 9.0e7, sign=1)]}]
    obj = matchObservedLW(None, states, schedule, observed,
                          matchOnlyProducers=True, LiquidRateWeight=1.0,
                          WaterCutWeight=0.0, BHPWeight=1.0)
    # The producer matches exactly; the injector's large error is excluded.
    assert obj[0] == pytest.approx(0.0)


def test_observed_water_cut_defect_is_reproduced():
    """MRST computes wct_obs = qLs_obs./qLs_obs, i.e. identically 1."""
    model, states, schedule, observed = _case((-1.0, -1.0, 0.0),
                                              (-1.0, -1.0, 0.0))
    # Simulated wct = qWs/qLs = -1/-2 = 0.5, observed reads as 1.
    faithful = matchObservedLW(model, states, schedule, observed,
                               LiquidRateWeight=0.0, WaterCutWeight=1.0,
                               BHPWeight=0.0)[0]
    assert faithful == pytest.approx((0.5 - 1.0) ** 2)

    corrected = matchObservedLW(model, states, schedule, observed,
                                LiquidRateWeight=0.0, WaterCutWeight=1.0,
                                BHPWeight=0.0,
                                fix_observed_water_cut=True)[0]
    assert corrected == pytest.approx(0.0)


def test_unsummed_mismatch_returns_the_three_terms():
    model, states, schedule, observed = _case((-1.0, -2.0, 2.0e7),
                                              (-2.0, -2.0, 3.0e7))
    obj = matchObservedLW(model, states, schedule, observed, mismatchSum=False,
                          LiquidRateWeight=1.0, WaterCutWeight=1.0,
                          BHPWeight=1.0)
    assert np.asarray(obj[0]).size == 3      # liquid, water cut, bhp


def test_accumulate_types_merges_the_terms():
    model, states, schedule, observed = _case((-1.0, -2.0, 2.0e7),
                                              (-2.0, -2.0, 3.0e7))
    obj = matchObservedLW(model, states, schedule, observed, mismatchSum=False,
                          accumulateTypes=[1, 1, 2],
                          LiquidRateWeight=1.0, WaterCutWeight=1.0,
                          BHPWeight=1.0)
    assert np.asarray(obj[0]).size == 2      # {liquid+wct}, {bhp}


# ------------------------------------------------------- expandToFull --

def test_expand_scatters_both_sides_over_every_well():
    status = np.array([True, False, True])
    status_obs = np.array([True, False, True])
    v, v_obs = _expandToFull(np.array([1.0, 3.0]), np.array([2.0, 4.0]),
                             status, status_obs, False)
    assert np.allclose(v, [1.0, 0.0, 3.0])
    assert np.allclose(v_obs, [2.0, 0.0, 4.0])


def test_disagreeing_status_is_blanked_on_both_sides():
    """A well shut on one side only must contribute nothing."""
    status = np.array([True, True])
    status_obs = np.array([True, False])
    v, v_obs = _expandToFull(np.array([1.0, 5.0]), np.array([2.0]),
                             status, status_obs, True)
    assert np.allclose(v, [1.0, 0.0])
    assert np.allclose(v_obs, [2.0, 0.0])


# ---------------------------------------------------- evaluateObjective --

def test_parameter_slices_partition_the_vector():
    assert _slices([2, 3]) == [slice(0, 2), slice(2, 5)]


def test_perturb_touches_one_entry():
    out = _perturb(np.zeros(3), 1, 0.5)
    assert np.allclose(out, [0.0, 0.5, 0.0])


def test_objective_unscales_then_rescales_the_gradient(monkeypatch):
    """The optimiser sees the unit box; the model sees physical units."""
    seen = {}

    class _P:
        name = 'p'
        nParam = 2

        @staticmethod
        def unscale(x):
            seen['unscaled'] = np.asarray(x) * 10.0
            return seen['unscaled']

        @staticmethod
        def setParameter(setup, value):
            seen['set'] = value
            return setup

        @staticmethod
        def scaleGradient(g, pval):
            return np.asarray(g) * 10.0

    _patch_simulator(monkeypatch)

    setup = {'model': object(), 'state0': {}, 'schedule': {}}
    value = evaluateObjective(np.array([0.5, 0.5]),
                              lambda *a, **k: [np.array([3.0])],
                              setup, [_P()], Gradient='none')
    assert value == pytest.approx(3.0)
    assert np.allclose(seen['unscaled'], [5.0, 5.0])


def test_bounds_are_enforced_by_default(monkeypatch):
    seen = {}

    class _P:
        name = 'p'
        nParam = 1

        @staticmethod
        def unscale(x):
            seen['x'] = np.asarray(x).copy()
            return x

        @staticmethod
        def setParameter(setup, value):
            return setup

    _patch_simulator(monkeypatch)

    setup = {'model': object(), 'state0': {}, 'schedule': {}}
    evaluateObjective(np.array([5.0]), lambda *a, **k: [np.array([0.0])],
                      setup, [_P()], Gradient='none')
    assert np.allclose(seen['x'], [1.0])     # clipped into the unit box


def test_unknown_gradient_method_is_rejected(monkeypatch):
    class _P:
        name = 'p'
        nParam = 1
        unscale = staticmethod(lambda x: x)
        setParameter = staticmethod(lambda s, v: s)

    _patch_simulator(monkeypatch)

    setup = {'model': object(), 'state0': {}, 'schedule': {}}
    with pytest.raises(ValueError, match='not implemented'):
        evaluateObjective(np.array([0.5]), lambda *a, **k: [np.array([0.0])],
                          setup, [_P()], Gradient='nosuch',
                          return_gradient=True)
