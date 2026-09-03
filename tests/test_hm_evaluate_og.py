"""Tests for matchObservedOG, evaluateMatchSummandsMulti and
updateDeckSchedule in the ``hm`` port."""

import numpy as np
import pytest

from PRSTCore.hm.utils.evaluate.evaluateMatchSummandsMulti import (
    _accumulate_steps, evaluateMatchSummandsMulti)
from PRSTCore.hm.utils.evaluate.matchObservedOG import (_expandWeightsToFull,
                                                        _getWeights,
                                                        matchObservedOG)
from PRSTCore.hm.utils.evaluate.updateDeckSchedule import updateDeckSchedule

DAY = 86400.0


def _sol(qGs, qOs, bhp, sign=-1, status=True, name='W'):
    return {'name': name, 'qGs': qGs, 'qOs': qOs, 'bhp': bhp,
            'sign': sign, 'status': status}


def _case(sim, obs, nstep=1):
    schedule = {'step': {'val': np.full(nstep, DAY),
                         'control': np.ones(nstep, dtype=int)}}
    states = [{'wellSol': [_sol(*sim)]} for _ in range(nstep)]
    observed = [{'wellSol': [_sol(*obs)]} for _ in range(nstep)]
    return None, states, schedule, observed


# -------------------------------------------------------- OG weights --

def test_og_weights_are_per_well_reciprocals():
    wg, wo, wp = _getWeights(np.array([4.0, 2.0]), np.array([5.0, 10.0]),
                             np.array([1.0, 3.0]), None, None, None)
    assert np.allclose(wg, [0.25, 0.5])
    assert np.allclose(wo, [0.2, 0.1])
    assert np.allclose(wp, [0.5, 0.5])       # 1 / (3 - 1)


def test_a_well_without_a_measured_rate_gets_zero_weight():
    """Rather than dividing by zero."""
    wg, wo, wp = _getWeights(np.array([4.0, 0.0]), np.array([5.0, 0.0]),
                             np.array([1.0, 3.0]), None, None, None)
    assert wg[1] == 0.0 and wo[1] == 0.0
    # Neither phase measured, so no pressure weight either.
    assert wp[1] == 0.0
    assert wp[0] > 0.0


def test_scalar_weights_are_broadcast_per_well():
    wg, wo, wp = _getWeights(np.zeros(3), np.zeros(3), np.zeros(3),
                             2.0, 3.0, 4.0)
    assert np.allclose(wg, 2.0) and wg.size == 3
    assert np.allclose(wo, 3.0) and np.allclose(wp, 4.0)


def test_weights_follow_their_wells_when_expanded():
    wg, wo, wp = _expandWeightsToFull(np.array([1.0, 2.0]),
                                      np.array([3.0, 4.0]),
                                      np.array([5.0, 6.0]),
                                      np.array([True, False, True]))
    assert np.allclose(wg, [1.0, 0.0, 2.0])
    assert np.allclose(wp, [5.0, 0.0, 6.0])


# ------------------------------------------------------- OG mismatch --

def test_og_perfect_match_gives_zero():
    obj = matchObservedOG(*_case((-3.0, -2.0, 2.0e7), (-3.0, -2.0, 2.0e7)),
                          GasRateWeight=1.0, OilRateWeight=1.0, BHPWeight=1.0)
    assert obj[0] == pytest.approx(0.0)


def test_og_mismatch_grows_with_the_discrepancy():
    close = matchObservedOG(*_case((-3.0, -2.0, 0.0), (-3.1, -2.0, 0.0)),
                            GasRateWeight=1.0, OilRateWeight=0.0,
                            BHPWeight=0.0)[0]
    far = matchObservedOG(*_case((-3.0, -2.0, 0.0), (-5.0, -2.0, 0.0)),
                          GasRateWeight=1.0, OilRateWeight=0.0,
                          BHPWeight=0.0)[0]
    assert 0 < close < far


def test_match_well_indices_selects_an_explicit_subset():
    schedule = {'step': {'val': np.array([DAY]), 'control': np.array([1])}}
    states = [{'wellSol': [_sol(-3.0, -2.0, 0.0, name='A'),
                           _sol(-1.0, -1.0, 0.0, name='B')]}]
    observed = [{'wellSol': [_sol(-3.0, -2.0, 0.0, name='A'),
                             _sol(-9.0, -9.0, 0.0, name='B')]}]
    obj = matchObservedOG(None, states, schedule, observed,
                          matchWellIndices=[0], GasRateWeight=1.0,
                          OilRateWeight=1.0, BHPWeight=0.0)
    # Only well A is matched, and it matches exactly.
    assert obj[0] == pytest.approx(0.0)


def test_og_steps_are_weighted_by_their_share_of_total_time():
    one = matchObservedOG(*_case((-3.0, -2.0, 0.0), (-5.0, -2.0, 0.0), 1),
                          GasRateWeight=1.0, OilRateWeight=0.0, BHPWeight=0.0)
    four = matchObservedOG(*_case((-3.0, -2.0, 0.0), (-5.0, -2.0, 0.0), 4),
                           GasRateWeight=1.0, OilRateWeight=0.0, BHPWeight=0.0)
    assert sum(four) == pytest.approx(one[0])


def test_og_unsummed_returns_three_terms():
    obj = matchObservedOG(*_case((-3.0, -2.0, 1.0), (-5.0, -4.0, 2.0)),
                          mismatchSum=False, GasRateWeight=1.0,
                          OilRateWeight=1.0, BHPWeight=1.0)
    assert np.asarray(obj[0]).size == 3


# --------------------------------------------- evaluateMatchSummandsMulti --

def test_step_accumulation_merges_the_residual_groups():
    out = _accumulate_steps([1.0, 2.0, 3.0], [1, 1, 2])
    assert out == [3.0, 3.0]


def test_step_accumulation_drops_zero_groups():
    out = _accumulate_steps([1.0, 2.0, 3.0], [1, 0, 1])
    assert out == [4.0]


def test_no_accumulation_passes_the_values_through():
    values = [1.0, 2.0]
    assert _accumulate_steps(values, None) is values


def test_residuals_are_square_roots_of_the_misfits(monkeypatch):
    import importlib
    module = importlib.import_module(
        'PRSTCore.ad_core.simulators.simulate_schedule_ad')
    monkeypatch.setattr(module, 'simulate_schedule_ad',
                        lambda *a, **k: ([], [{}]))

    class _P:
        name = 'p'
        nParam = 1
        unscale = staticmethod(lambda x: x)
        setParameter = staticmethod(lambda s, v: s)

    setup = [{'model': object(), 'state0': {}, 'schedule': {}}]
    obj = [lambda *a, **k: [np.array([4.0]), np.array([9.0])]]
    out = evaluateMatchSummandsMulti(np.array([0.5]), obj, setup, [_P()],
                                     [None])
    assert out.shape == (2, 1)
    assert np.allclose(out[:, 0], [2.0, 3.0])


def test_several_cases_become_separate_columns(monkeypatch):
    import importlib
    module = importlib.import_module(
        'PRSTCore.ad_core.simulators.simulate_schedule_ad')
    monkeypatch.setattr(module, 'simulate_schedule_ad',
                        lambda *a, **k: ([], [{}]))

    class _P:
        name = 'p'
        nParam = 1
        unscale = staticmethod(lambda x: x)
        setParameter = staticmethod(lambda s, v: s)

    setup = [{'model': object(), 'state0': {}, 'schedule': {}}] * 2
    obj = [lambda *a, **k: [np.array([1.0])],
           lambda *a, **k: [np.array([16.0])]]
    out = evaluateMatchSummandsMulti(np.array([0.5]), obj, setup, [_P()],
                                     [None, None])
    assert out.shape == (1, 2)
    assert np.allclose(out[0, :], [1.0, 4.0])


# ------------------------------------------------------ updateDeckSchedule --

def _deck_and_schedule():
    deck = {'SCHEDULE': {'control': [{
        'WELSPECS': [['P1', 'G', 0, 0, 0.0]],
        'COMPDAT': [['P1', 0, 0, 0, 0, 'OPEN', 0, 0.0, 0.0, 0, 0, 0, 'Z']],
        'WCONINJE': [],
        'WCONPROD': [['P1', 'OPEN', 'ORAT', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    }]}}
    G = {'cartDims': [4, 4, 2], 'cells': {'num': 32, 'indexMap': np.arange(32)}}
    schedule = {'control': [{'W': [{
        'name': 'P1', 'cells': np.array([0, 4]), 'type': 'orat', 'val': 5.0,
        'sign': -1, 'r': np.full(2, 0.1), 'WI': np.full(2, 2.0),
        'dir': np.array(['Z', 'Z']), 'status': True,
        'cstatus': np.array([True, False])}]}]}
    return deck, G, schedule


def test_compdat_gets_one_record_per_perforation():
    deck, G, schedule = _deck_and_schedule()
    out = updateDeckSchedule(deck, G, schedule)
    compdat = out['control'][0]['COMPDAT']
    assert len(compdat) == 2
    assert compdat[0][5] == 'OPEN'
    assert compdat[1][5] == 'SHUT'          # cstatus False
    assert compdat[0][7] == pytest.approx(2.0)   # WI
    assert compdat[0][8] == pytest.approx(0.1)   # radius
    assert compdat[0][9] == -1                   # defaulted Kh


def test_production_target_is_negated_back_to_eclipse_sign():
    deck, G, schedule = _deck_and_schedule()
    out = updateDeckSchedule(deck, G, schedule)
    row = out['control'][0]['WCONPROD'][0]
    assert row[2] == 'ORAT'
    assert row[3] == pytest.approx(-5.0)     # val * sign


def test_pressure_control_is_written_unsigned():
    deck, G, schedule = _deck_and_schedule()
    schedule['control'][0]['W'][0].update({'type': 'bhp', 'val': 2.0e7})
    out = updateDeckSchedule(deck, G, schedule)
    row = out['control'][0]['WCONPROD'][0]
    assert row[2] == 'BHP'
    assert row[8] == pytest.approx(2.0e7)


def test_shut_well_is_marked_in_the_control_keyword():
    deck, G, schedule = _deck_and_schedule()
    schedule['control'][0]['W'][0]['status'] = False
    out = updateDeckSchedule(deck, G, schedule)
    assert out['control'][0]['WCONPROD'][0][1] == 'SHUT'


def test_welspecs_takes_the_first_perforations_indices():
    deck, G, schedule = _deck_and_schedule()
    out = updateDeckSchedule(deck, G, schedule)
    row = out['control'][0]['WELSPECS'][0]
    assert (row[2], row[3]) == (1, 1)        # cell 0 -> I=1, J=1


def test_mismatched_control_counts_are_rejected():
    deck, G, schedule = _deck_and_schedule()
    schedule['control'].append({'W': []})
    with pytest.raises(AssertionError):
        updateDeckSchedule(deck, G, schedule)


def test_the_original_deck_is_not_mutated():
    deck, G, schedule = _deck_and_schedule()
    updateDeckSchedule(deck, G, schedule)
    assert len(deck['SCHEDULE']['control'][0]['COMPDAT']) == 1
