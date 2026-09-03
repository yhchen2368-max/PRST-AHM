"""Tests for the restart well-solution reader and processWellStates.

The synthetic cases pin the record layout and the three well-state
passes. The tests against a real ECLIPSE restart run only when one is
present, and cross-check it against the summary file -- two independent
paths through the same run.
"""

import os

import numpy as np
import pytest

from PRSTCore.deckformat.resultinput.convert_restart_to_states import (
    convert_restart_to_states, process_well_states)
from PRSTCore.deckformat.resultinput.get_restart_well_info import (
    getINTEHEAD, getRestartWellInfo)


def _intehead(nwell=1, ncwma=2, niwel=12, nswel=12, nxwel=12, nzwel=3,
              nicon=14, nscon=4, nxcon=52):
    ih = np.zeros(100, dtype=int)
    ih[2] = 1          # metric
    ih[8:11] = (2, 2, 2)
    ih[16] = nwell
    ih[17] = ncwma
    ih[24], ih[25], ih[26], ih[27] = niwel, nswel, nxwel, nzwel
    ih[32], ih[33], ih[34] = nicon, nscon, nxcon
    return ih


# ------------------------------------------------------------ INTEHEAD --

def test_intehead_positions_match_mrst():
    """MRST indexes these 1-based; a slip of one silently reads a
    neighbouring dimension and every stride comes out wrong."""
    ih = getINTEHEAD(_intehead(nwell=7, ncwma=5))
    assert ih['nwell'] == 7 and ih['ncwma'] == 5
    assert (ih['nx'], ih['ny'], ih['nz']) == (2, 2, 2)


def test_intehead_tolerates_a_short_header():
    assert getINTEHEAD(np.zeros(10, dtype=int))['nwell'] == 0


# ------------------------------------------------------- well records --

def test_no_wells_gives_an_empty_list():
    wells, _ = getRestartWellInfo({'INTEHEAD': _intehead(nwell=0)})
    assert wells == []


def test_the_name_is_the_first_of_the_char_slots():
    records = {'INTEHEAD': _intehead(nwell=2, nzwel=3),
               'ZWEL': np.array(['P1', '', '', 'I1', '', ''])}
    wells, _ = getRestartWellInfo(records)
    assert [w['name'] for w in wells] == ['P1', 'I1']


def test_iwel_fields_are_read_from_mrst_positions():
    iwel = np.zeros(12, dtype=int)
    iwel[0:3] = (5, 6, 7)      # i, j, k
    iwel[4] = 3                # ncon
    iwel[6] = 1                # type: producer
    iwel[8] = 2                # control mode
    iwel[10] = 1               # open
    wells, _ = getRestartWellInfo({'INTEHEAD': _intehead(), 'IWEL': iwel})
    w = wells[0]
    assert list(w['ijk']) == [5, 6, 7]
    assert w['ncon'] == 3 and w['type'] == 1 and w['cntr'] == 2
    assert w['stat'] is True


def test_rates_come_back_negated_into_mrst_sign_convention():
    """ECLIPSE writes production positive; MRST wants it negative."""
    xwel = np.zeros(12)
    xwel[0:3] = (10.0, 20.0, 30.0)      # oil, water, gas
    xwel[6] = 250.0                     # bhp, not negated
    wells, _ = getRestartWellInfo({'INTEHEAD': _intehead(), 'XWEL': xwel})
    w = wells[0]
    assert (w['qOs'], w['qWs'], w['qGs']) == (-10.0, -20.0, -30.0)
    assert w['bhp'] == 250.0


def test_connections_are_read_only_up_to_the_wells_own_count():
    """ICON is stored ncwmax per well whether or not that many exist."""
    ih = _intehead(nwell=1, ncwma=4, nicon=14)
    iwel = np.zeros(12, dtype=int)
    iwel[4] = 2                     # only two real connections
    icon = np.zeros(14 * 4, dtype=int)
    for c in range(4):
        icon[c * 14 + 1: c * 14 + 4] = (c + 1, 1, 1)
    wells, _ = getRestartWellInfo({'INTEHEAD': ih, 'IWEL': iwel,
                                   'ICON': icon})
    assert wells[0]['cijk'].shape == (2, 3)


def test_connection_direction_is_decoded():
    ih = _intehead(nwell=1, ncwma=3, nicon=14)
    iwel = np.zeros(12, dtype=int)
    iwel[4] = 3
    icon = np.zeros(14 * 3, dtype=int)
    icon[13] = 1                    # x
    icon[14 + 13] = 2               # y
    icon[28 + 13] = 3               # anything else is z
    wells, _ = getRestartWellInfo({'INTEHEAD': ih, 'IWEL': iwel,
                                   'ICON': icon})
    assert list(wells[0]['cdir']) == ['x', 'y', 'z']


def test_a_second_well_starts_at_its_own_stride():
    ih = _intehead(nwell=2, ncwma=2, niwel=12)
    iwel = np.zeros(24, dtype=int)
    iwel[0:3] = (1, 1, 1)
    iwel[12:15] = (9, 9, 9)
    wells, _ = getRestartWellInfo({'INTEHEAD': ih, 'IWEL': iwel})
    assert list(wells[1]['ijk']) == [9, 9, 9]


def test_a_missing_record_leaves_its_fields_empty():
    """Restart files vary by simulator and version, so an absent record
    must not raise."""
    wells, _ = getRestartWellInfo({'INTEHEAD': _intehead()})
    assert wells[0]['name'] is None and wells[0]['qOs'] is None


# --------------------------------------------------- processWellStates --

def _well(name='P1', sign=-1.0, status=True, resv=-1.0, flux=None):
    flux = np.array([-1.0, -1.0]) if flux is None else np.asarray(flux)
    return {'name': name, 'sign': sign, 'status': status, 'resv': resv,
            'cells': np.array([0, 1]), 'flux': flux,
            'cqs': np.zeros((2, 3)), 'cstatus': np.ones(2, bool)}


def test_crossflow_is_zeroed_not_counted():
    """A connection flowing against the well's sign is crossflow."""
    states = [{'wellSol': [_well(flux=[-2.0, 1.0])]}]
    out = process_well_states(states, remove_crossflow=True,
                              remove_closed_wells=False)
    assert list(out[0]['wellSol'][0]['flux']) == [-2.0, 0.0]


def test_crossflow_is_kept_when_not_asked_for():
    states = [{'wellSol': [_well(flux=[-2.0, 1.0])]}]
    out = process_well_states(states, remove_crossflow=False,
                              remove_closed_wells=False)
    assert list(out[0]['wellSol'][0]['flux']) == [-2.0, 1.0]


def test_a_well_below_the_tolerance_is_treated_as_shut():
    states = [{'wellSol': [_well(resv=-1e-9)]}]
    out = process_well_states(states, set_to_closed_tol=1e-6,
                              remove_closed_wells=False)
    well = out[0]['wellSol'][0]
    assert well['status'] is False
    assert not well['cstatus'].any() and well['resv'] == 0.0
    assert not well['flux'].any()


def test_a_flowing_well_is_left_open():
    states = [{'wellSol': [_well(resv=-5.0)]}]
    out = process_well_states(states, set_to_closed_tol=1e-6,
                              remove_closed_wells=False)
    assert out[0]['wellSol'][0]['status'] is True


def test_a_well_shut_in_every_step_is_dropped():
    states = [{'wellSol': [_well('P1'), _well('P2', status=False)]},
              {'wellSol': [_well('P1'), _well('P2', status=False)]}]
    out = process_well_states(states, remove_closed_wells=True)
    assert [w['name'] for w in out[0]['wellSol']] == ['P1']


def test_a_well_shut_in_only_some_steps_is_kept():
    states = [{'wellSol': [_well('P1', status=False)]},
              {'wellSol': [_well('P1', status=True)]}]
    out = process_well_states(states, remove_closed_wells=True)
    assert len(out[0]['wellSol']) == 1


def test_a_sign_changing_well_is_split_in_two():
    """One curve through an injection/production flip is meaningless, so
    MRST makes it two wells."""
    states = [{'wellSol': [_well('W', sign=1.0, resv=5.0)]},
              {'wellSol': [_well('W', sign=-1.0, resv=-5.0)]}]
    out = process_well_states(states, split_wells_on_sign_change=True,
                              remove_closed_wells=False)
    names = [w['name'] for w in out[0]['wellSol']]
    assert names == ['W (inj)', 'W (prod)']


def test_each_half_of_a_split_well_is_open_only_in_its_own_steps():
    states = [{'wellSol': [_well('W', sign=1.0, resv=5.0)]},
              {'wellSol': [_well('W', sign=-1.0, resv=-5.0)]}]
    out = process_well_states(states, split_wells_on_sign_change=True,
                              remove_closed_wells=False)
    assert [w['status'] for w in out[0]['wellSol']] == [True, False]
    assert [w['status'] for w in out[1]['wellSol']] == [False, True]


def test_a_steady_well_is_not_split():
    states = [{'wellSol': [_well('W')]}, {'wellSol': [_well('W')]}]
    out = process_well_states(states, split_wells_on_sign_change=True,
                              remove_closed_wells=False)
    assert len(out[0]['wellSol']) == 1


# ------------------------------------------------ against a real restart --

_PREFIX = os.environ.get('ECLIPSE_RESTART_PREFIX')


def _case():
    if not _PREFIX or not os.path.exists(_PREFIX + '.UNRST'):
        pytest.skip('set ECLIPSE_RESTART_PREFIX to a run with a restart')
    from PRSTCore.deckformat.resultinput.init_grid_from_eclipse_output import \
        init_grid_from_eclipse_output
    from PRSTCore.deckformat.resultinput.read_eclipse_output_file_unfmt import \
        read_eclipse_output_file_unfmt as rd
    G, _, _, _ = init_grid_from_eclipse_output(rd(_PREFIX + '.INIT'),
                                               rd(_PREFIX + '.EGRID'))
    states, _ = convert_restart_to_states(_PREFIX, G, remove_crossflow=False,
                                          remove_closed_wells=False)
    return G, states


def test_real_restart_gives_the_wells_their_real_names():
    _, states = _case()
    names = [w['name'] for w in states[-1]['wellSol']]
    assert names == ['WELL%d' % i for i in range(1, 10)]


def test_real_restart_rates_agree_with_the_summary_file():
    """The restart and the summary are independent paths through the same
    run; if the record layout were misread they would not agree."""
    from PRSTCore.deckformat.resultinput.read_eclipse_summary import \
        convert_summary_to_well_sols
    _, states = _case()
    summary, _ = convert_summary_to_well_sols(_PREFIX)

    from_restart = {w['name']: w['qOs'] for w in states[-1]['wellSol']}
    from_summary = {w['name']: w['qOs'] for w in summary[-1]}
    for name, value in from_summary.items():
        assert from_restart[name] == pytest.approx(value, rel=1e-4), name


def test_real_restart_maps_connections_onto_grid_cells():
    G, states = _case()
    cells = states[-1]['wellSol'][0]['cells']
    assert cells.size > 0
    assert cells.min() >= 0 and cells.max() < G['cells']['num']


def test_real_restart_connections_stack_in_the_k_direction():
    """A vertical well's cells step by nx*ny between layers."""
    G, states = _case()
    cells = np.sort(states[-1]['wellSol'][0]['cells'])
    nx, ny = int(G['cartDims'][0]), int(G['cartDims'][1])
    assert np.all(np.diff(cells) == nx * ny)


def test_real_restart_producers_have_negative_rates():
    _, states = _case()
    for well in states[-1]['wellSol']:
        if well['sign'] < 0 and well['status']:
            assert well['qOs'] <= 0 and well['qWs'] <= 0
