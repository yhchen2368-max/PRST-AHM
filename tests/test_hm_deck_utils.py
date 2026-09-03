"""Tests for the deck/wellsol helpers in the MRST ``hm/utils`` port."""

import numpy as np
import pytest

from PRSTCore.hm.utils.addFieldRates import addFieldRates
from PRSTCore.hm.utils.processEclipseDeck import processEclipseDeck
from PRSTCore.hm.utils.reduceEclipseDeckSchedule import reduceEclipseDeckSchedule
from PRSTCore.hm.utils.S_getCurrentDate2021 import S_getCurrentDate2021
from PRSTCore.hm.utils.updateDeckFromModelParameter import updateDeckFromModelParameter


def _w(name, sign, qWs, qOs, qGs, bhp, status=True):
    return {'name': name, 'sign': sign, 'qWs': qWs, 'qOs': qOs,
            'qGs': qGs, 'bhp': bhp, 'status': status}


# ---------------------------------------------------------- addFieldRates --

def test_field_rates_sum_producers_and_injectors_separately():
    ws = [[_w('P1', -1, -1.0, -2.0, -3.0, 200.0),
           _w('P2', -1, -4.0, -5.0, -6.0, 300.0),
           _w('I1', 1, 10.0, 0.0, 0.0, 400.0)]]
    out = addFieldRates(ws)[0]
    prod = next(w for w in out if w['name'] == 'producers')
    inj = next(w for w in out if w['name'] == 'injectors')
    assert prod['qWs'] == pytest.approx(-5.0)
    assert prod['qOs'] == pytest.approx(-7.0)
    assert prod['bhp'] == pytest.approx(250.0)      # mean, not sum
    assert inj['qWs'] == pytest.approx(10.0)
    assert inj['bhp'] == pytest.approx(400.0)


def test_field_rates_skip_shut_wells():
    ws = [[_w('P1', -1, -1.0, -2.0, -3.0, 200.0),
           _w('P2', -1, -9.0, -9.0, -9.0, 999.0, status=False)]]
    prod = next(w for w in addFieldRates(ws)[0] if w['name'] == 'producers')
    assert prod['qWs'] == pytest.approx(-1.0)


def test_field_rates_report_absent_groups_as_shut():
    ws = [[_w('P1', -1, -1.0, -2.0, -3.0, 200.0)]]
    inj = next(w for w in addFieldRates(ws)[0] if w['name'] == 'injectors')
    assert inj['status'] is False
    assert inj['qWs'] == 0.0 and inj['bhp'] == 0.0


def test_field_rates_are_idempotent():
    ws = [[_w('P1', -1, -1.0, -2.0, -3.0, 200.0),
           _w('I1', 1, 10.0, 0.0, 0.0, 400.0)]]
    twice = addFieldRates(addFieldRates(ws))
    names = [w['name'] for w in twice[0]]
    assert names.count('producers') == 1 and names.count('injectors') == 1


# ------------------------------------------------------ processEclipseDeck --

def _deck():
    return {
        'RUNSPEC': {'cartDims': [2, 2, 1]},
        'PROPS': {},
        'SCHEDULE': {
            'control': [
                {'WELSPECS': [], 'COMPDAT': [], 'WCONPROD': [], 'WCONINJE': [],
                 'WCONINJH': [], 'WCONHIST': [], 'WEFAC': []},
                {'WELSPECS': [['P1', 1, 1], ['GHOST', 2, 2]],
                 'COMPDAT': [['P1', 1, 1, 1, 1], ['GHOST', 2, 2, 1, 1]],
                 'WCONPROD': [['P1', 'OPEN', 'ORAT', 100.0, 200.0, 300.0]],
                 'WCONINJE': [], 'WCONINJH': [], 'WCONHIST': [],
                 'WEFAC': [['P1', 0.5]]},
            ],
            'step': {'control': np.array([0, 1, 1]),
                     'val': np.array([1.0, 2.0, 3.0])},
        },
    }


def test_process_deck_forces_output_and_endscale():
    deck = processEclipseDeck(_deck())
    assert deck['RUNSPEC']['UNIFOUT'] is True
    assert deck['RUNSPEC']['FMTOUT'] is False
    assert deck['RUNSPEC']['ENDSCALE'][0] == 'NODIR'
    assert deck['PROPS']['SCALECRS'] == ['NO']


def test_process_deck_drops_the_leading_empty_control():
    deck = processEclipseDeck(_deck())
    assert len(deck['SCHEDULE']['control']) == 1
    assert deck['SCHEDULE']['control'][0]['WELSPECS']


def test_process_deck_preserves_total_time():
    deck = processEclipseDeck(_deck())
    assert deck['SCHEDULE']['step']['val'].sum() == pytest.approx(6.0)


def test_process_deck_removes_wells_without_a_control_record():
    ctrl = processEclipseDeck(_deck())['SCHEDULE']['control'][0]
    assert [r[0] for r in ctrl['WELSPECS']] == ['P1']
    assert [r[0] for r in ctrl['COMPDAT']] == ['P1']


def test_process_deck_applies_wefac_to_the_target_rates():
    row = processEclipseDeck(_deck())['SCHEDULE']['control'][0]['WCONPROD'][0]
    assert row[3] == pytest.approx(50.0)     # 100 * 0.5
    assert row[4] == pytest.approx(100.0)    # 200 * 0.5
    assert row[5] == pytest.approx(150.0)    # 300 * 0.5


def test_process_deck_leaves_wells_without_a_wefac_entry_alone():
    deck = _deck()
    deck['SCHEDULE']['control'][1]['WEFAC'] = [['OTHER', 0.5]]
    out = processEclipseDeck(deck)
    assert out['SCHEDULE']['control'][0]['WCONPROD'][0][3] == pytest.approx(100.0)


# ------------------------------------------------ reduceEclipseDeckSchedule --

def _sched(welspecs_per_control):
    return {'SCHEDULE': {'control': [
        {'WELSPECS': w, 'COMPDAT': [], 'GINJGAS': [], 'GRUPTREE': []}
        for w in welspecs_per_control]}}


def test_reduce_schedule_drops_repeated_records():
    deck = _sched([[['P1', 1, 1]],
                   [['P1', 1, 1]],
                   [['P1', 1, 1], ['P2', 2, 2]]])
    out = reduceEclipseDeckSchedule(deck)['reducedSCHEDULE']['control']
    assert out[0]['WELSPECS'] == [['P1', 1, 1]]
    assert out[1]['WELSPECS'] == []                 # unchanged -> dropped
    assert out[2]['WELSPECS'] == [['P2', 2, 2]]     # only what is new


def test_reduce_schedule_leaves_the_original_intact():
    deck = _sched([[['P1', 1, 1]], [['P1', 1, 1]]])
    reduceEclipseDeckSchedule(deck)
    assert deck['SCHEDULE']['control'][1]['WELSPECS'] == [['P1', 1, 1]]


def test_reduce_schedule_replaces_nan_defaults():
    """A NaN never compares equal to itself, so it becomes -1 first."""
    deck = _sched([[['P1', float('nan'), 1]], [['P1', float('nan'), 1]]])
    out = reduceEclipseDeckSchedule(deck)['reducedSCHEDULE']['control']
    assert out[0]['WELSPECS'] == [['P1', -1, 1]]
    assert out[1]['WELSPECS'] == []


# ------------------------------------------ updateDeckFromModelParameter --

class _Model:
    def __init__(self, nc=4):
        self.G = {'cells': {'num': nc, 'indexMap': np.arange(nc)}}
        self.fluid = {'krPts': {'w': np.array([[0.1, 0.2, 0.9, 0.8]])}}


def _param(name, value, location=None):
    return {'name': name, 'belongsTo': 'model', 'location': location or [],
            'getfun': lambda owner, *loc: value}


def test_parameters_land_in_their_deck_sections():
    deck = {'RUNSPEC': {'cartDims': [2, 2, 1]}}
    updateDeckFromModelParameter(deck, {'model': _Model()}, [
        _param('porevolume', np.full(4, 5.0)),
        _param('permx', np.full(4, 1e-13)),
    ])
    assert np.allclose(deck['EDIT']['PORV'], 5.0)
    assert np.allclose(deck['GRID']['PERMX'], 1e-13)


def test_absent_endpoint_keyword_seeds_from_the_tabulated_points():
    deck = {'RUNSPEC': {'cartDims': [2, 2, 1]}, 'PROPS': {},
            'REGIONS': {'SATNUM': np.ones(4, dtype=int)}}
    updateDeckFromModelParameter(deck, {'model': _Model()}, [
        _param('swl', np.full(4, 0.33), location=['a', 'b', 'c', 'w', 0]),
    ])
    assert np.allclose(deck['PROPS']['SWL'], 0.33)


def test_unknown_parameter_names_are_ignored():
    deck = {'RUNSPEC': {'cartDims': [2, 2, 1]}}
    updateDeckFromModelParameter(deck, {'model': _Model()},
                                 [_param('nosuchthing', np.zeros(4))])
    assert 'EDIT' not in deck and 'GRID' not in deck


# ------------------------------------------------------------------ misc --

def test_current_date_is_iso_formatted():
    value = S_getCurrentDate2021()
    assert len(value) == 10 and value[4] == '-' and value[7] == '-'


# ------------------------------------------- processEclipseDeck trimming --

def _trim_schedule(step_control, step_val, controls):
    return {'RUNSPEC': {}, 'PROPS': {},
            'SCHEDULE': {'step': {'control': list(step_control),
                                  'val': list(step_val)},
                         'control': controls}}


def _welspec(name='W1'):
    return {'WELSPECS': [[name]], 'COMPDAT': [[name]],
            'WCONHIST': [[name, 'OPEN', 'LRAT', 1.0, 2.0, 3.0]]}


def test_a_schedule_that_starts_at_control_zero_is_left_alone():
    """``find(step.control > 0, 1)`` drops the leading steps that have no
    control -- MATLAB writes *no control* as 0 because it numbers them
    from 1. PRSTCore numbers them from 0, so control 0 is a real control
    and testing ``> 0`` threw away the first report step of every deck,
    folding its duration into the next one.
    """
    from PRSTCore.hm.utils.processEclipseDeck import processEclipseDeck

    deck = _trim_schedule([0, 1, 2], [7.0, 7.0, 7.0],
                          [_welspec(), _welspec(), _welspec()])
    out = processEclipseDeck(deck)['SCHEDULE']
    assert list(np.asarray(out['step']['control'])) == [0, 1, 2]
    assert np.allclose(np.asarray(out['step']['val'], dtype=float),
                       [7.0, 7.0, 7.0])
    assert len(out['control']) == 3


def test_leading_steps_with_no_control_are_folded_into_the_first_real_one():
    """A negative index is what "no control" looks like here."""
    from PRSTCore.hm.utils.processEclipseDeck import processEclipseDeck

    deck = _trim_schedule([-1, -1, 0, 1], [3.0, 4.0, 7.0, 7.0],
                          [_welspec(), _welspec()])
    out = processEclipseDeck(deck)['SCHEDULE']
    assert list(np.asarray(out['step']['control'])) == [0, 1]
    # The dropped steps' durations are carried into the first survivor.
    assert np.allclose(np.asarray(out['step']['val'], dtype=float),
                       [3.0 + 4.0 + 7.0, 7.0])


def test_controls_before_the_first_welspecs_are_dropped_and_renumbered():
    """``step.control(ix:end) - step.control(ix-1)`` puts the first
    surviving control at the lowest valid index -- 1 in MATLAB, 0 here."""
    from PRSTCore.hm.utils.processEclipseDeck import processEclipseDeck

    deck = _trim_schedule([0, 1, 2], [5.0, 7.0, 7.0],
                          [{'WELSPECS': [], 'COMPDAT': []},
                           _welspec(), _welspec()])
    out = processEclipseDeck(deck)['SCHEDULE']
    assert len(out['control']) == 2
    assert list(np.asarray(out['step']['control'])) == [0, 1]
    assert np.allclose(np.asarray(out['step']['val'], dtype=float),
                       [5.0 + 7.0, 7.0])


# ------------------------------------------------- the control index --

def test_the_control_index_is_taken_as_written():
    """MATLAB's ``schedule.control(ctrl)`` is 1-based, so a literal
    transcription subtracts one. PRSTCore's ``step['control']`` is already
    0-based, and subtracting again shifts every step back by one: step 0
    reads the *last* control and every later step reads its predecessor's.

    Nothing raises and nothing is out of range -- the values that come
    back are real well controls, just the wrong date's. It had been
    written four times in four different files.
    """
    from PRSTCore.hm.utils.controlIndex import control_index

    step = {'control': [0, 1, 2, 3]}
    assert [control_index(step, i, 4) for i in range(4)] == [0, 1, 2, 3]


def test_repeated_controls_map_each_step_to_its_own():
    from PRSTCore.hm.utils.controlIndex import control_index

    step = {'control': [0, 0, 1, 1, 2]}
    assert [control_index(step, i, 3) for i in range(5)] == [0, 0, 1, 1, 2]


def test_an_index_past_the_end_is_clamped_rather_than_wrapping():
    """Negative indexing is the failure this exists to prevent: ``-1``
    silently selects the last control instead of raising."""
    from PRSTCore.hm.utils.controlIndex import control_index

    assert control_index({'control': [5]}, 0, 3) == 2
    assert control_index({'control': [-1]}, 0, 3) == 0
