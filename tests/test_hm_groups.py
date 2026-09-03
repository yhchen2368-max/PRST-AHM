"""Tests for the group/placement helpers in the MRST ``hm/utils`` port."""

import numpy as np
import pytest

from PRSTCore.hm.utils.convertSummaryToGroupSols import convertSummaryToGroupSols
from PRSTCore.hm.utils.imposeMultipointVerticalRelpermScaling import (
    imposeMultipointVerticalRelpermScaling)
from PRSTCore.hm.utils.optimWellPlacementSimple import optimWellPlacementSimple
from PRSTCore.hm.utils.processGroups import processGroups
from PRSTCore.hm.utils.processJutulStates import processJutulStates


# ------------------------------------------------------------ processGroups --

def test_injection_groups_carry_a_positive_target_and_phase_compi():
    control = {'GCONINJE': [['G1', 'WATER', 'RATE', 500.0, 0.0]]}
    G = processGroups(control)
    assert len(G) == 1
    assert G[0]['name'] == 'G1' and G[0]['sign'] == 1
    assert G[0]['val'] == pytest.approx(500.0)
    assert np.allclose(G[0]['compi'], [1.0, 0.0, 0.0])
    assert G[0]['lims']['rate'] == pytest.approx(500.0)


def test_injection_resv_reads_the_reservoir_column():
    control = {'GCONINJE': [['G1', 'GAS', 'RESV', 500.0, 900.0]]}
    G = processGroups(control)
    assert G[0]['val'] == pytest.approx(900.0)
    assert np.allclose(G[0]['compi'], [0.0, 0.0, 1.0])


def test_unknown_injection_mode_gives_a_zero_target():
    G = processGroups({'GCONINJE': [['G1', 'WATER', 'NOSUCH', 5.0, 6.0]]})
    assert G[0]['val'] == 0


def test_unknown_injection_phase_drops_the_group():
    assert processGroups({'GCONINJE': [['G1', 'ZZZ', 'RATE', 5.0, 6.0]]}) == []


@pytest.mark.parametrize('mode, column, compi', [
    ('ORAT', 2, [0.0, 1.0, 0.0]),
    ('WRAT', 3, [1.0, 0.0, 0.0]),
    ('GRAT', 4, [0.0, 0.0, 1.0]),
    ('LRAT', 5, [1.0, 1.0, 0.0]),
])
def test_production_modes_select_their_column_and_are_negated(mode, column, compi):
    row = ['G1', mode, 10.0, 20.0, 30.0, 40.0]
    G = processGroups({'GCONPROD': [row]})
    assert G[0]['sign'] == -1
    assert G[0]['val'] == pytest.approx(-row[column])
    assert np.allclose(G[0]['compi'], compi)


def test_production_limits_are_all_negated():
    G = processGroups({'GCONPROD': [['G1', 'ORAT', 10.0, 20.0, 30.0, 40.0]]})
    lims = G[0]['lims']
    assert lims['orat'] == pytest.approx(-10.0)
    assert lims['wrat'] == pytest.approx(-20.0)
    assert lims['grat'] == pytest.approx(-30.0)
    assert lims['lrat'] == pytest.approx(-40.0)


def test_injection_groups_come_before_production_groups():
    control = {'GCONPROD': [['P', 'ORAT', 1.0, 2.0, 3.0, 4.0]],
               'GCONINJE': [['I', 'WATER', 'RATE', 5.0, 6.0]]}
    assert [g['name'] for g in processGroups(control)] == ['I', 'P']


# ------------------------------------------------- convertSummaryToGroupSols --

def _smry(rows, ntime=3):
    """Minimal stand-in for read_eclipse_summary's return value."""
    names = [r[0] for r in rows]
    kwrds = [r[1] for r in rows]
    data = np.array([r[2] for r in rows], dtype=float)

    def get(name, keyword):
        for i, (n, k) in enumerate(zip(names, kwrds)):
            if n == name and k == keyword:
                return data[i, :]
        return None

    def get_names(keyword):
        return sorted({n for n, k in zip(names, kwrds) if k == keyword})

    def get_keywords(name):
        return sorted({k for n, k in zip(names, kwrds) if n == name})

    return {'KEYWORDS': kwrds, 'data': data, 'get': get,
            'get_names': get_names, 'get_keywords': get_keywords,
            'intehead_unit': 1}


def test_group_sols_negate_production_rates():
    smry = _smry([[':+:+:+:+', 'TIME', [1.0, 2.0, 3.0]],
                  ['G1', 'GOPR', [10.0, 10.0, 10.0]]])
    sols, time = convertSummaryToGroupSols(smry)
    assert len(sols) == 3 and len(sols[0]) == 1
    assert sols[0][0]['qOs'] < 0
    assert sols[0][0]['sign'] == -1
    assert time.size == 3


def test_group_sols_add_injection_to_production():
    """GWIR is added, so a pure injector reads positive."""
    smry = _smry([[':+:+:+:+', 'TIME', [1.0, 2.0, 3.0]],
                  ['G1', 'GWIR', [5.0, 5.0, 5.0]]])
    sols, _ = convertSummaryToGroupSols(smry)
    assert sols[0][0]['qWs'] > 0
    assert sols[0][0]['sign'] == 1


def test_gas_rate_falls_back_to_the_gas_oil_ratio():
    smry = _smry([[':+:+:+:+', 'TIME', [1.0, 2.0]],
                  ['G1', 'GOPR', [10.0, 10.0]],
                  ['G1', 'GGOR', [2.0, 2.0]]])
    sols, _ = convertSummaryToGroupSols(smry)
    # qGs = qOs * GGOR, and qOs is already negative.
    assert sols[0][0]['qGs'] == pytest.approx(2.0 * sols[0][0]['qOs'])


def test_water_rate_falls_back_to_the_water_cut():
    smry = _smry([[':+:+:+:+', 'TIME', [1.0]],
                  ['G1', 'GOPR', [10.0]],
                  ['G1', 'GWCT', [0.5]]])
    sols, _ = convertSummaryToGroupSols(smry)
    # wcut/(1-wcut) = 1 at wcut = 0.5, so qWs equals qOs.
    assert sols[0][0]['qWs'] == pytest.approx(sols[0][0]['qOs'])


def test_a_shut_group_reports_zero_status():
    smry = _smry([[':+:+:+:+', 'TIME', [1.0]], ['G1', 'GOPR', [0.0]]])
    sols, _ = convertSummaryToGroupSols(smry)
    assert sols[0][0]['status'] is False


def test_groupname_filter_selects_a_subset():
    smry = _smry([[':+:+:+:+', 'TIME', [1.0]],
                  ['G1', 'GOPR', [1.0]], ['G2', 'GOPR', [2.0]]])
    sols, _ = convertSummaryToGroupSols(smry, groupname='G2')
    assert [g['name'] for g in sols[0]] == ['G2']


# --------------------------------------- imposeMultipointVerticalRelpermScaling --

class _MPModel:
    def __init__(self):
        self.G = {'cells': {'num': 4, 'indexMap': np.arange(4)}}
        self.rock = {}
        self.fluid = {'krPts': {'w': np.array([[0.15, 0.2, 0.9, 0.8]])}}
        self.inputdata = None


def test_multipoint_water_curve_starts_after_the_last_immobile_point():
    table = np.array([[0.1, 0.0], [0.2, 0.0], [0.5, 0.3], [0.9, 0.8]])
    model = imposeMultipointVerticalRelpermScaling(_MPModel(), SW_KRW=table)
    got = model.rock['krscale']['multipoint']['w']
    assert np.allclose(got, [[0.5, 0.3], [0.9, 0.8]])


def test_multipoint_oil_curve_stops_at_the_first_immobile_point():
    """SW_KROW keeps the mobile head and converts Sw to So = 1 - Sw."""
    table = np.array([[0.1, 0.9], [0.4, 0.4], [0.8, 0.0], [0.9, 0.0]])
    model = imposeMultipointVerticalRelpermScaling(_MPModel(), SW_KROW=table)
    got = model.rock['krscale']['multipoint']['ow']
    assert np.allclose(got[:, 0], [1 - 0.1, 1 - 0.4])
    assert np.allclose(got[:, 1], [0.9, 0.4])


def test_multipoint_gas_oil_curve_subtracts_connate_water():
    table = np.array([[0.1, 0.9], [0.4, 0.0]])
    model = imposeMultipointVerticalRelpermScaling(_MPModel(), SG_KROG=table)
    got = model.rock['krscale']['multipoint']['og']
    assert np.allclose(got[:, 0], [1 - 0.1 - 0.15])


def test_multipoint_requires_krpts():
    model = _MPModel()
    model.fluid = {}
    with pytest.raises(AssertionError, match='krPts'):
        imposeMultipointVerticalRelpermScaling(model, SW_KRW=np.zeros((2, 2)))


def test_multipoint_rejects_a_bad_point_count():
    with pytest.raises(AssertionError, match='2- or 3-point'):
        imposeMultipointVerticalRelpermScaling(_MPModel(), nPoints=4,
                                               SW_KRW=np.zeros((2, 2)))


def test_multipoint_without_arguments_is_a_no_op():
    model = _MPModel()
    assert imposeMultipointVerticalRelpermScaling(model) is model
    assert 'krscale' not in model.rock


# ---------------------------------------------------- optimWellPlacementSimple --

class _Ctrl:
    nPoints = 1

    def __init__(self, n_param):
        self.parameters = {'nParam': n_param}

    @staticmethod
    def getProjectedUpdate(u, du, flag):
        return du


def test_placement_ascends_a_simple_quadratic():
    """f(u) = -(u - 0.7)^2 peaks at 0.7; the search maximises."""
    target = np.array([0.7, 0.7, 0.7])

    def f(u):
        u = np.asarray(u, dtype=float)
        return float(-np.sum((u - target) ** 2)), -2.0 * (u - target)

    W = [{'posControl': _Ctrl(3)}]
    u = optimWellPlacementSimple(np.zeros(3), f, W, maxSteps=60,
                                 maxRelative=0.05, verbose=False)
    assert np.allclose(u, target, atol=0.02)


def test_placement_keeps_the_controls_in_the_unit_box():
    def f(u):
        u = np.asarray(u, dtype=float)
        return float(np.sum(u)), np.ones_like(u)     # push past 1

    W = [{'posControl': _Ctrl(3)}]
    u = optimWellPlacementSimple(np.full(3, 0.9), f, W, maxSteps=10,
                                 verbose=False)
    assert np.all(u <= 1.0 + 1e-12) and np.all(u >= 0.0)


def test_placement_stops_when_no_step_improves():
    calls = {'n': 0}

    def f(u):
        calls['n'] += 1
        return 1.0, np.ones_like(np.asarray(u, dtype=float))   # never improves

    W = [{'posControl': _Ctrl(2)}]
    u0 = np.array([0.5, 0.5])
    u = optimWellPlacementSimple(u0, f, W, maxSteps=5, maxLineSearchIts=3,
                                 verbose=False)
    assert np.allclose(u, u0)
    # One initial evaluation plus one exhausted line search.
    assert calls['n'] == 1 + 3


# -------------------------------------------------------- processJutulStates --

def test_jutul_states_refresh_the_schedule_wells():
    setup = {'schedule': {
        'step': {'control': [1, 1]},
        'control': [{'W': [{'name': 'P1', 'qWs': 0.0, 'qOs': 0.0, 'qGs': 0.0,
                            'bhp': 0.0, 'status': False}]}],
    }}
    wellSols = [[{'name': 'P1', 'qWs': 1.0, 'qOs': 2.0, 'qGs': 3.0,
                  'bhp': 4.0, 'status': True}],
                [{'name': 'P1', 'qWs': 5.0, 'qOs': 6.0, 'qGs': 7.0,
                  'bhp': 8.0, 'status': True}]]
    states = [{}, {}]
    ws, st = processJutulStates(setup, wellSols, states)
    assert ws[0][0]['qWs'] == pytest.approx(1.0)
    assert ws[1][0]['bhp'] == pytest.approx(8.0)
    assert st[0]['wellSols'][0]['status'] is True
    # The schedule itself must not be mutated.
    assert setup['schedule']['control'][0]['W'][0]['qWs'] == 0.0


def test_jutul_states_ignore_wells_absent_from_the_report():
    setup = {'schedule': {
        'step': {'control': [1]},
        'control': [{'W': [{'name': 'P1', 'qWs': 0.0}, {'name': 'P2', 'qWs': 0.0}]}],
    }}
    ws, _ = processJutulStates(setup, [[{'name': 'P1', 'qWs': 9.0}]], [{}])
    assert ws[0][0]['qWs'] == pytest.approx(9.0)
    assert ws[0][1]['qWs'] == pytest.approx(0.0)


# ---------------------------------------- groups reach the schedule --

def test_group_controls_are_attached_to_the_mrst_schedule():
    """``G = processGroups(ctrl); controlMRST(i).G = G``.

    MRST-0 puts each control's group targets beside its wells; 2026a has
    no group handling at all. Following 2026a left ``processGroups``
    ported with nothing calling it, so a deck's GCONPROD/GCONINJE records
    reached the model nowhere.
    """
    import os

    import pytest
    deck = 'examples/SPE9/SPE9_CP_GROUP.DATA'
    if not os.path.exists(deck):
        pytest.skip('SPE9 group deck not present')
    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import \
        init_eclipse_problem_ad

    _state0, _model, schedule, _ = init_eclipse_problem_ad(deck)
    groups = [c.get('G') for c in schedule['control']]
    assert any(groups), 'no control carried a group target'

    named = [(g['name'], g['type'], g['val'])
             for g in schedule['control'][1]['G']]
    assert ('P', 'orat', -1500.0) in named


def test_a_deck_without_groups_gains_no_group_field():
    """``if isempty(vertcat(controlMRST.G)), controlMRST = rmfield(...)``
    -- the field is dropped entirely rather than left as a row of empties,
    so downstream code can test for it."""
    import os

    import pytest
    deck = 'examples/SPE9/SPE9_CP.DATA'
    if not os.path.exists(deck):
        pytest.skip('SPE9 deck not present')
    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import \
        init_eclipse_problem_ad

    _state0, _model, schedule, _ = init_eclipse_problem_ad(deck)
    assert not any('G' in c for c in schedule['control'])
