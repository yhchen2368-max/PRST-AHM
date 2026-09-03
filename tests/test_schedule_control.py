"""Tests for the port of readSCHEDULE's control splitting.

The behaviour under test is MRST's, so each test names the MATLAB it
follows: ``readSCHEDULE.m`` for the control/step structure itself and
``readWellKW.m`` for how one keyword's records merge into the running
control.  MRST-0 is the reference tree -- see
:mod:`PRSTCore.deckformat.deckinput.schedule_control`.
"""

import os

import numpy as np
import pytest

from PRSTCore.deckformat.deckinput.schedule_control import (
    CONTROL_KEYWORDS, default_control, parse_eclipse_date,
    read_schedule_control)


def _schedule(text):
    control, step, missing = read_schedule_control(text, start='01 JAN 2020')
    return control, step, missing


# ------------------------------------------------------- the structure --

_TWO_CONTROLS = """
WELSPECS
 'P1' 'G1' 5 5 1* OIL /
 'I1' 'G1' 1 1 1* WATER /
/

COMPDAT
 'P1' 5 5 1 3 OPEN 1* 1* 0.2 /
 'I1' 1 1 1 3 OPEN 1* 1* 0.2 /
/

WCONPROD
 'P1' OPEN ORAT 100 /
/

WCONINJE
 'I1' WATER OPEN RATE 150 /
/

DATES
 1 FEB 2020 /
/

WCONPROD
 'P1' OPEN ORAT 200 /
/

DATES
 1 MAR 2020 /
/
"""


def test_a_dates_record_closes_the_running_control():
    control, step, _ = _schedule(_TWO_CONTROLS)
    assert len(control) == 2
    assert step['control'] == [0, 1]
    assert np.allclose(step['val'], [31.0, 29.0])   # 2020 is a leap year


def test_wells_and_completions_carry_into_the_next_control():
    """``defaultControl`` accumulates WELSPECS and COMPDAT, so a control
    that declares neither still knows about both."""
    control, _, _ = _schedule(_TWO_CONTROLS)
    assert [row[0] for row in control[1]['WELSPECS']] == ['P1', 'I1']
    assert len(control[1]['COMPDAT']) == 2


def test_rate_targets_carry_forward_until_restated():
    """WCONINJE is in ``defaultControl``'s copy list, so the injector
    keeps its target through a control that only restates the producer."""
    control, _, _ = _schedule(_TWO_CONTROLS)
    assert control[0]['WCONPROD'][0][3] == 100.0
    assert control[1]['WCONPROD'][0][3] == 200.0
    assert control[1]['WCONINJE'][0][4] == 150.0


def test_the_step_vector_indexes_controls_from_zero():
    """MRST numbers controls from 1; PRSTCore addresses them from 0, and
    writeSchedule's ``cstep`` loop relies on that."""
    control, step, _ = _schedule(_TWO_CONTROLS)
    assert min(step['control']) == 0
    assert max(step['control']) == len(control) - 1


def test_a_control_declares_every_field_even_when_unused():
    control, _, _ = _schedule(_TWO_CONTROLS)
    for name in ('WCONHIST', 'WCONINJH', 'GRUPTREE', 'WELTARG', 'WEFAC'):
        assert control[0][name] == []


def test_tstep_records_advance_the_clock_too():
    control, step, _ = _schedule("""
WELSPECS
 'P1' 'G1' 5 5 1* OIL /
/
WCONPROD
 'P1' OPEN ORAT 100 /
/
TSTEP
 10 3*20 /
""")
    assert len(control) == 1
    assert np.allclose(step['val'], [10.0, 20.0, 20.0, 20.0])
    assert step['control'] == [0, 0, 0, 0]


# ---------------------------------------------------- defaulted records --

def test_a_repeat_count_skips_that_many_items():
    """``readDefaultedRecord``: ``3*`` leaves three items at their
    template default rather than consuming three values."""
    control, _, _ = _schedule("""
WELSPECS
 'P1' 'G1' 5 5 1* OIL /
/
COMPDAT
 'P1' 5 5 1 1 OPEN 1* 7.5 0.16 549.3 3* 3.86 /
/
TSTEP
 1 /
""")
    row = control[0]['COMPDAT'][0]
    assert row[7] == 7.5 and row[8] == 0.16 and row[9] == 549.3
    assert row[10] == 0.0 and row[11] == -1.0 and row[12] == 'Z'
    assert row[13] == pytest.approx(3.86)


def test_an_all_defaulted_record_terminates_the_keyword():
    """A record identical to the template is how ``readDefaultedKW``
    recognises the terminating ``/``."""
    control, _, _ = _schedule("""
WELSPECS
 'P1' 'G1' 5 5 1* OIL /
/
WCONPROD
 'P1' OPEN ORAT 100 /
/
TSTEP
 1 /
""")
    assert len(control[0]['WCONPROD']) == 1


def test_a_trailing_comment_does_not_end_the_record():
    control, _, _ = _schedule("""
WELSPECS
 'P1' 'G1' 5 5 1* OIL / -- the producer
/
WCONPROD
 'P1' OPEN ORAT 100 /   -- rate target
/
TSTEP
 1 /
""")
    assert control[0]['WELSPECS'][0][0] == 'P1'
    assert control[0]['WCONPROD'][0][3] == 100.0


def test_a_record_may_span_lines():
    """``readRecordString`` accumulates until the ``/``, which is how
    QIEDIE's COMPDAT records wrap."""
    control, _, _ = _schedule("""
WELSPECS
 'P1' 'G1' 5 5 1* OIL /
/
COMPDAT
 'P1' 5 5 1 1 OPEN 1* 7.5 0.16 549.3 3*
                  3.86 /
/
TSTEP
 1 /
""")
    assert control[0]['COMPDAT'][0][13] == pytest.approx(3.86)


def test_an_empty_quoted_entry_reads_as_defaulted():
    """``replaceEmpty`` -- one of MRST-0's `% edited by zhang` changes."""
    control, _, _ = _schedule("""
WELSPECS
 'P1' '' 5 5 1* OIL /
/
WCONPROD
 'P1' OPEN ORAT 100 /
/
TSTEP
 1 /
""")
    assert control[0]['WELSPECS'][0][1] == 'Default'


# -------------------------------------------------- MRST-0's own edits --

def test_a_well_name_longer_than_eight_characters_is_chopped():
    """``relpaceWellName``: ECLIPSE names are at most eight characters."""
    with pytest.warns(RuntimeWarning, match='great than 8'):
        control, _, _ = _schedule("""
WELSPECS
 'PRODUCER01' 'G1' 5 5 1* OIL /
/
TSTEP
 1 /
""")
    assert control[0]['WELSPECS'][0][0] == 'PRODUCER'


def test_a_multi_letter_completion_direction_keeps_its_first_letter():
    """``readCompDat`` `% edited by zhang`: anything but FX/FY/FZ is
    truncated, so tNavigator's 'ZZ' reads as 'Z'."""
    control, _, _ = _schedule("""
WELSPECS
 'P1' 'G1' 5 5 1* OIL /
/
COMPDAT
 'P1' 5 5 1 1 OPEN 1* 7.5 0.16 549.3 0.0 -1 ZZ 3.86 /
 'P1' 5 5 2 2 OPEN 1* 7.5 0.16 549.3 0.0 -1 FX 3.86 /
/
TSTEP
 1 /
""")
    assert control[0]['COMPDAT'][0][12] == 'Z'
    assert control[0]['COMPDAT'][1][12] == 'FX'


def test_compdat_item_twelve_is_numeric():
    """MRST-0 moved item 12 into the numeric set; 2026a leaves it text."""
    control, _, _ = _schedule("""
WELSPECS
 'P1' 'G1' 5 5 1* OIL /
/
COMPDAT
 'P1' 5 5 1 1 OPEN 1* 7.5 0.16 549.3 0.0 2.5 Z 3.86 /
/
TSTEP
 1 /
""")
    assert control[0]['COMPDAT'][0][11] == pytest.approx(2.5)


def test_wefac_is_a_control_keyword():
    """`% edited by zhang` added WEFAC to readSCHEDULE's case list, so it
    opens a control of its own."""
    assert 'WEFAC' in CONTROL_KEYWORDS
    control, _, _ = _schedule("""
WELSPECS
 'P1' 'G1' 5 5 1* OIL /
/
TSTEP
 1 /
WEFAC
 'P1' 0.8 /
/
TSTEP
 1 /
""")
    assert len(control) == 2
    assert control[1]['WEFAC'][0][1] == pytest.approx(0.8)


# ------------------------------------------------- record bookkeeping --

def test_a_well_moved_to_another_rate_keyword_loses_its_old_record():
    """``assignControlRecords`` removes the well from the other four rate
    keywords -- ``excludeSet``."""
    control, _, _ = _schedule("""
WELSPECS
 'W1' 'G1' 5 5 1* OIL /
/
WCONPROD
 'W1' OPEN ORAT 100 /
/
TSTEP
 1 /
WCONINJE
 'W1' WATER OPEN RATE 150 /
/
TSTEP
 1 /
""")
    assert control[0]['WCONPROD'] and not control[0]['WCONINJE']
    assert control[1]['WCONINJE'] and not control[1]['WCONPROD']


def test_restating_a_well_replaces_its_carried_record():
    """``appendSpec`` drops the copy defaultControl carried forward."""
    control, _, _ = _schedule("""
WELSPECS
 'W1' 'G1' 5 5 1* OIL /
 'W2' 'G1' 6 6 1* OIL /
/
WCONPROD
 'W1' OPEN ORAT 100 /
 'W2' OPEN ORAT 300 /
/
TSTEP
 1 /
WCONPROD
 'W1' OPEN ORAT 200 /
/
TSTEP
 1 /
""")
    rows = {row[0]: row for row in control[1]['WCONPROD']}
    assert len(control[1]['WCONPROD']) == 2
    assert rows['W1'][3] == 200.0       # restated
    assert rows['W2'][3] == 300.0       # carried


def test_a_wildcard_control_expands_over_the_declared_wells():
    control, _, _ = _schedule("""
WELSPECS
 'PA' 'G1' 5 5 1* OIL /
 'PB' 'G1' 6 6 1* OIL /
 'IA' 'G1' 1 1 1* WATER /
/
WCONPROD
 'P*' OPEN ORAT 100 /
/
TSTEP
 1 /
""")
    assert sorted(row[0] for row in control[0]['WCONPROD']) == ['PA', 'PB']


def test_a_restated_well_replaces_its_welspecs_row_in_place():
    """``readWellSpec`` overwrites the matching row rather than appending
    a second one."""
    control, _, _ = _schedule("""
WELSPECS
 'W1' 'G1' 5 5 1* OIL /
/
TSTEP
 1 /
WELSPECS
 'W1' 'G2' 7 7 1* OIL /
 'W2' 'G2' 8 8 1* OIL /
/
TSTEP
 1 /
""")
    rows = control[1]['WELSPECS']
    assert [row[0] for row in rows] == ['W1', 'W2']
    assert rows[0][1] == 'G2' and rows[0][2] == 7.0


def test_new_completions_append_and_restated_ones_replace():
    """``handleOverlapCompdat``: same (well, I, J, K1, K2) is an update."""
    control, _, _ = _schedule("""
WELSPECS
 'W1' 'G1' 5 5 1* OIL /
/
COMPDAT
 'W1' 5 5 1 1 OPEN 1* 7.5 0.16 549.3 /
/
TSTEP
 1 /
COMPDAT
 'W1' 5 5 1 1 OPEN 1* 9.9 0.16 549.3 /
 'W1' 5 5 2 2 OPEN 1* 8.8 0.16 549.3 /
/
TSTEP
 1 /
""")
    rows = control[1]['COMPDAT']
    assert len(rows) == 2
    by_layer = {row[3]: row for row in rows}
    assert by_layer[1.0][7] == pytest.approx(9.9)     # replaced
    assert by_layer[2.0][7] == pytest.approx(8.8)     # appended


def test_welopen_with_defaulted_perforations_shuts_the_well():
    control, _, _ = _schedule("""
WELSPECS
 'W1' 'G1' 5 5 1* OIL /
/
WCONPROD
 'W1' OPEN ORAT 100 /
/
TSTEP
 1 /
WELOPEN
 'W1' SHUT /
/
TSTEP
 1 /
""")
    assert control[0]['WCONPROD'][0][1] == 'OPEN'
    assert control[1]['WCONPROD'][0][1] == 'SHUT'


# ------------------------------------------------------------- dates --

def test_dates_become_step_lengths_measured_from_start():
    _, step, _ = read_schedule_control("""
WELSPECS
 'W1' 'G1' 5 5 1* OIL /
/
DATES
 15 JAN 2020 /
 1 FEB 2020 /
/
""", start='01 JAN 2020')
    assert np.allclose(step['val'], [14.0, 17.0])


def test_jly_is_accepted_as_july():
    assert parse_eclipse_date('1 JLY 2020') == parse_eclipse_date('1 JUL 2020')


def test_a_date_that_does_not_advance_the_clock_is_an_error():
    with pytest.raises(ValueError, match='advance'):
        read_schedule_control("""
WELSPECS
 'W1' 'G1' 5 5 1* OIL /
/
DATES
 1 JAN 2020 /
/
""", start='01 JAN 2020')


# ------------------------------------------------------ the deck path --

def test_default_control_starts_empty_and_inherits_nothing():
    control = default_control()
    assert control['WELSPECS'] == [] and control['COMPDAT'] == []
    assert control['DRSDT'] == [float('inf'), 'ALL']


def test_default_control_accumulates_wells_but_replaces_targets():
    first = default_control()
    first['WELSPECS'] = [['W1']]
    first['WCONPROD'] = [['W1', 'OPEN']]
    second = default_control(first)
    assert second['WELSPECS'] == [['W1']]
    assert second['WCONPROD'] == [['W1', 'OPEN']]
    # Accumulated fields must be copies -- appending to the new control
    # may not reach back into the old one.
    second['WELSPECS'].append(['W2'])
    assert first['WELSPECS'] == [['W1']]


def _qiedie():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, 'examples', 'HM', 'QIEDIE.DATA')


@pytest.mark.skipif(not os.path.exists(_qiedie()), reason='QIEDIE absent')
def test_qiedie_splits_into_one_control_per_reporting_date():
    from PRSTCore.deckformat.deckinput.read_eclipse_deck import \
        read_eclipse_deck

    deck = read_eclipse_deck(_qiedie())
    schedule = deck['SCHEDULE']

    assert len(schedule['control']) == 63
    assert len(schedule['step']['val']) == 63
    assert schedule['step']['control'] == list(range(63))

    first, last = schedule['control'][0], schedule['control'][-1]
    for control in (first, last):
        assert len(control['WELSPECS']) == 9
        assert len(control['COMPDAT']) == 180        # 9 wells x 20 layers
        assert len(control['WCONHIST']) == 5
        assert len(control['WCONINJH']) == 4

    # The rate targets differ between controls -- that is the history
    # being matched, and it is what a single merged keyword list loses.
    assert first['WCONHIST'][0][3] != last['WCONHIST'][0][3]


@pytest.mark.skipif(not os.path.exists(_qiedie()), reason='QIEDIE absent')
def test_qiedie_keeps_its_flat_keyword_lists():
    """The control structure is additive: nothing that read the deck
    before this existed may see a different SCHEDULE."""
    from PRSTCore.deckformat.deckinput.read_eclipse_deck import \
        read_eclipse_deck

    schedule = read_eclipse_deck(_qiedie())['SCHEDULE']
    for keyword in ('WELSPECS', 'COMPDAT', 'WCONHIST', 'WCONINJH', 'DATES'):
        assert keyword in schedule
    assert '_order' in schedule
