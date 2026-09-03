"""Tests for ``write_deck`` / ``write_schedule``.

The substantive test is the round trip: a deck written out and read back
must carry the same arrays. Anything less only checks that the writer
produces text.
"""

import os

import numpy as np
import pytest

from PRSTCore.deckformat.deckinput.read_eclipse_deck import read_eclipse_deck
from PRSTCore.deckformat.deckoutput.write_deck import write_deck
from PRSTCore.deckformat.deckoutput.write_schedule import writeSchedule


def _deck():
    """A deck small enough to check by eye, complete enough to write."""
    nx, ny, nz = 2, 2, 1
    return {
        'RUNSPEC': {'cartDims': [nx, ny, nz], 'DIMENS': [nx, ny, nz],
                    'METRIC': True, 'OIL': True, 'WATER': True,
                    'TITLE': 'ROUNDTRIP'},
        'GRID': {'cartDims': [nx, ny, nz],
                 'PERMX': np.array([100.0, 200.0, 300.0, 400.0]),
                 'PORO': np.array([0.1, 0.2, 0.3, 0.4]),
                 'ACTNUM': np.array([1, 1, 1, 1])},
        'PROPS': {},
        'REGIONS': {},
        'SOLUTION': {},
        'SCHEDULE': {'control': [], 'step': {'val': [], 'control': []}},
    }


def test_write_deck_creates_the_data_file(tmp_path):
    path = write_deck(_deck(), str(tmp_path), filename='CASE', unit='metric')
    assert os.path.exists(path) and path.endswith('CASE.DATA')


def test_bulk_arrays_go_to_their_own_include_files(tmp_path):
    write_deck(_deck(), str(tmp_path), filename='CASE')
    for name in ('PERMX.INC', 'PORO.INC', 'ACTNUM.INC'):
        assert (tmp_path / name).exists(), name


def test_the_data_file_references_every_include(tmp_path):
    path = write_deck(_deck(), str(tmp_path), filename='CASE')
    text = open(path).read()
    for name in ('PERMX.INC', 'PORO.INC'):
        assert "'%s'" % name in text


def test_sections_appear_in_eclipse_order(tmp_path):
    path = write_deck(_deck(), str(tmp_path), filename='CASE')
    text = open(path).read()
    order = [text.index(s) for s in ('RUNSPEC', 'GRID', 'PROPS', 'SOLUTION',
                                     'SUMMARY', 'SCHEDULE')]
    assert order == sorted(order)


def test_phase_flags_are_written(tmp_path):
    path = write_deck(_deck(), str(tmp_path), filename='CASE')
    text = open(path).read()
    assert '\nOIL\n' in text and '\nWATER\n' in text


def test_round_trip_preserves_the_grid_arrays(tmp_path):
    """The real test: read back what was written."""
    write_deck(_deck(), str(tmp_path), filename='CASE')
    back = read_eclipse_deck(str(tmp_path / 'CASE.DATA'))
    assert np.allclose(np.asarray(back['GRID']['PERMX'], dtype=float),
                       [100.0, 200.0, 300.0, 400.0])
    assert np.allclose(np.asarray(back['GRID']['PORO'], dtype=float),
                       [0.1, 0.2, 0.3, 0.4])


def test_round_trip_preserves_the_grid_dimensions(tmp_path):
    write_deck(_deck(), str(tmp_path), filename='CASE')
    back = read_eclipse_deck(str(tmp_path / 'CASE.DATA'))
    assert list(back['RUNSPEC']['cartDims']) == [2, 2, 1]


def test_actnum_is_written_as_integers(tmp_path):
    write_deck(_deck(), str(tmp_path), filename='CASE')
    body = (tmp_path / 'ACTNUM.INC').read_text()
    assert '.' not in body.split('ACTNUM')[1].split('/')[0]


def test_a_pvto_table_keeps_its_key_and_record_structure(tmp_path):
    deck = _deck()
    deck['PROPS'] = {'PVTO': [{'key': np.array([0.5, 1.0]),
                               'pos': np.array([0, 2, 3]),
                               'data': np.array([[50.0, 1.1, 0.9],
                                                 [80.0, 1.0, 1.0],
                                                 [90.0, 1.2, 0.8]])}]}
    write_deck(deck, str(tmp_path), filename='CASE')
    body = (tmp_path / 'PVTO.INC').read_text()
    assert body.startswith('PVTO')
    # Two saturated keys, each closed by '/', then the table's own '/'.
    assert body.count('/') == 3


def test_nan_becomes_zero_in_a_table(tmp_path):
    deck = _deck()
    deck['PROPS'] = {'DENSITY': np.array([[900.0, np.nan, 1.0]])}
    write_deck(deck, str(tmp_path), filename='CASE')
    body = (tmp_path / 'DENSITY.INC').read_text()
    assert 'nan' not in body.lower()


# ------------------------------------------------------ writeSchedule --

def _schedule():
    return {
        'control': [{
            'WELSPECS': [['P1', 'G1', 1, 1, 2000.0, 'OIL', 0, 'STD', 'SHUT',
                          'YES', 0, 'SEG', 0]],
            'COMPDAT': [['P1', 1, 1, 1, 1, 'OPEN', -1, 1e-12, 0.2, 0.0, 0.0,
                         'Default', 'Z', 0.0]],
            'WCONPROD': [['P1', 'OPEN', 'BHP', np.inf, np.inf, np.inf,
                          np.inf, np.inf, 100.0, np.nan, 0, 0]],
        }],
        'step': {'val': [1.0, 2.0], 'control': [0, 0]},
    }


def test_schedule_writes_each_keyword_block(tmp_path):
    out = tmp_path / 'SCH.INC'
    with open(out, 'w') as fh:
        writeSchedule(fh, str(tmp_path), _schedule())
    text = out.read_text()
    for kw in ('WELSPECS', 'COMPDAT', 'WCONPROD', 'TSTEP'):
        assert kw in text, kw


def test_schedule_marks_defaults_with_the_eclipse_star(tmp_path):
    """'Default', NaN and inf all become 1*, which is what ECLIPSE reads
    as 'use the default'."""
    out = tmp_path / 'SCH.INC'
    with open(out, 'w') as fh:
        writeSchedule(fh, str(tmp_path), _schedule())
    text = out.read_text()
    assert '1*' in text
    assert 'inf' not in text.lower() and 'nan' not in text.lower()


def test_schedule_blanks_a_negative_compdat_entry(tmp_path):
    """A negative entry in COMPDAT means 'defaulted', so it is written as
    1* rather than passed through as a number."""
    out = tmp_path / 'SCH.INC'
    with open(out, 'w') as fh:
        writeSchedule(fh, str(tmp_path), _schedule())
    compdat = out.read_text().split('COMPDAT')[1].split('/\n\n')[0]
    assert '-1' not in compdat and '1*' in compdat


def test_schedule_emits_the_steps_of_its_own_control(tmp_path):
    sched = _schedule()
    sched['control'].append(dict(sched['control'][0]))
    sched['step'] = {'val': [1.0, 2.0, 5.0], 'control': [0, 0, 1]}
    out = tmp_path / 'SCH.INC'
    with open(out, 'w') as fh:
        writeSchedule(fh, str(tmp_path), sched)
    blocks = out.read_text().split('TSTEP')
    assert '5.0000' in blocks[2]          # the second control's step
    assert '5.0000' not in blocks[1]


def test_schedule_field_filter_restricts_what_is_written(tmp_path):
    out = tmp_path / 'SCH.INC'
    with open(out, 'w') as fh:
        writeSchedule(fh, str(tmp_path), _schedule(), fields=('COMPDAT',))
    text = out.read_text()
    assert 'COMPDAT' in text and 'WELSPECS' not in text


def test_schedule_include_mode_puts_the_body_in_a_separate_file(tmp_path):
    main = tmp_path / 'MAIN.INC'
    with open(main, 'w') as fh:
        writeSchedule(fh, str(tmp_path), _schedule(), writeInclude=True,
                      includeName='SCHEDULE.INC')
    assert "'SCHEDULE.INC'" in main.read_text()
    assert 'COMPDAT' in (tmp_path / 'SCHEDULE.INC').read_text()
    assert 'COMPDAT' not in main.read_text()


# ------------------------------------------- MRST-0 writeDeck behaviour --

def test_summary_section_is_generated_not_echoed(tmp_path):
    """The deck being written has no SUMMARY at all, yet the output must
    request the full standard set -- otherwise a written deck produces no
    summary vectors to match against."""
    write_deck(_deck(), str(tmp_path), filename='CASE')
    body = (tmp_path / 'SUM.INC').read_text()
    for kw in ('FOPR', 'WOPR', 'WBHP', 'WWCT', 'FGOR'):
        assert '\n%s\n' % kw in '\n' + body, kw


def test_summary_requests_the_history_vectors_too(tmp_path):
    """The *H vectors are the observed data a history match is scored
    against; without them the written deck cannot be matched."""
    write_deck(_deck(), str(tmp_path), filename='CASE')
    body = (tmp_path / 'SUM.INC').read_text()
    for kw in ('WOPRH', 'WWPRH', 'WBHPH', 'FOPRH'):
        assert '\n%s\n' % kw in '\n' + body, kw


def test_well_summary_vectors_carry_a_record_terminator(tmp_path):
    """A well vector needs a well-name record; a field vector does not."""
    write_deck(_deck(), str(tmp_path), filename='CASE')
    body = (tmp_path / 'SUM.INC').read_text()
    assert 'WOPR\n/' in body
    assert 'FOPR\n/' not in body


def test_the_data_file_includes_the_summary(tmp_path):
    path = write_deck(_deck(), str(tmp_path), filename='CASE')
    assert "'SUM.INC'" in open(path).read()


def test_nosim_is_written_when_asked(tmp_path):
    path = write_deck(_deck(), str(tmp_path), filename='CASE', NOSIM=True)
    assert '\nNOSIM\n' in open(path).read()


def test_nosim_is_absent_by_default(tmp_path):
    path = write_deck(_deck(), str(tmp_path), filename='CASE')
    assert '\nNOSIM\n' not in open(path).read()


def test_filename_names_the_data_file(tmp_path):
    path = write_deck(_deck(), str(tmp_path), filename='mycase')
    assert os.path.basename(path) == 'MYCASE.DATA'


def test_without_filename_the_directory_names_it(tmp_path):
    work = tmp_path / 'runcase'
    work.mkdir()
    path = write_deck(_deck(), str(work))
    assert os.path.basename(path) == 'RUNCASE.DATA'


def test_an_empty_edit_section_still_gets_a_porv_include(tmp_path):
    """MRST-0 writes an empty PORV.INC so the deck's structure does not
    depend on whether EDIT happened to be populated."""
    write_deck(_deck(), str(tmp_path), filename='CASE')
    assert (tmp_path / 'PORV.INC').exists()
    assert (tmp_path / 'PORV.INC').read_text() == ''
    assert "'PORV.INC'" in open(tmp_path / 'CASE.DATA').read()


def test_restart_output_is_requested(tmp_path):
    """RPTSOL/RPTRST are rewritten, not copied, so states can be read back."""
    path = write_deck(_deck(), str(tmp_path), filename='CASE')
    text = open(path).read()
    assert 'RPTSOL' in text and 'RESTART=2' in text
    assert 'RPTRST' in text and 'BASIC=2' in text


def test_swatinit_adds_pcow_to_the_restart_request(tmp_path):
    """Capillary scaling cannot be recovered without PCOW in the restart."""
    deck = _deck()
    deck['PROPS'] = {'SWATINIT': np.array([0.2, 0.2, 0.2, 0.2])}
    path = write_deck(deck, str(tmp_path), filename='CASE')
    assert 'PCOW' in open(path).read()


def test_arrays_are_written_six_values_to_a_line(tmp_path):
    deck = _deck()
    deck['GRID']['PORO'] = np.full(13, 0.2)
    write_deck(deck, str(tmp_path), filename='CASE')
    rows = [r for r in (tmp_path / 'PORO.INC').read_text().splitlines()[1:]
            if r.strip() and not r.startswith('/')]
    assert len(rows) == 3          # 6 + 6 + 1


def test_start_date_switches_the_schedule_to_dates(tmp_path):
    """With a start date the schedule carries absolute DATES, not TSTEP."""
    out = tmp_path / 'SCH.INC'
    with open(out, 'w') as fh:
        writeSchedule(fh, str(tmp_path), _schedule(), start=[1, 1, 2020])
    text = out.read_text()
    assert 'DATES' in text and 'TSTEP' not in text
    assert 'JAN 2020' in text


def test_dates_accumulate_from_the_very_first_step(tmp_path):
    """Each DATES record is the elapsed time to the end of that step,
    counted from step one -- not from the current control's start."""
    out = tmp_path / 'SCH.INC'
    with open(out, 'w') as fh:
        writeSchedule(fh, str(tmp_path), _schedule(), start=[1, 1, 2020])
    text = out.read_text()
    assert ' 2 JAN 2020' in text        # after 1 day
    assert ' 4 JAN 2020' in text        # after 1 + 2 days


# ------------------------------ MRST-0 writeSchedule: blanking passes --

def _one(keyword, row):
    return {'control': [{keyword: [row]}],
            'step': {'val': [], 'control': []}}


def test_rate_keywords_blank_infinity(tmp_path):
    """An unlimited target is written 1*, not 'inf' -- ECLIPSE cannot
    read the latter."""
    out = tmp_path / 'S.INC'
    with open(out, 'w') as fh:
        writeSchedule(fh, str(tmp_path),
                      _one('WCONPROD', ['P1', 'OPEN', 'BHP', np.inf, 1.0,
                                        1.0, 1.0, 1.0, 100.0, 1.0, 0, 0]))
    text = out.read_text()
    assert 'inf' not in text.lower() and '1*' in text


def test_rate_keywords_keep_a_negative_value(tmp_path):
    """WCONPROD runs default/nan/inf -- not negative -- so a negative
    number here is a real value and must survive."""
    out = tmp_path / 'S.INC'
    with open(out, 'w') as fh:
        writeSchedule(fh, str(tmp_path),
                      _one('WCONPROD', ['P1', 'OPEN', 'BHP', -5.0, 1.0, 1.0,
                                        1.0, 1.0, 100.0, 1.0, 0, 0]))
    assert '-5.0000' in out.read_text()


def test_welspecs_blanks_a_negative_value(tmp_path):
    """WELSPECS runs default/nan/negative, so the sign is the marker."""
    out = tmp_path / 'S.INC'
    with open(out, 'w') as fh:
        writeSchedule(fh, str(tmp_path),
                      _one('WELSPECS', ['P1', 'G1', 1, 1, -1.0, 'OIL', 0,
                                        'STD', 'SHUT', 'YES', 0, 'SEG', 0]))
    assert '1*' in out.read_text()


def test_welspecs_keeps_infinity_out_of_its_passes(tmp_path):
    """WELSPECS has no inf pass, so an infinity is not blanked there --
    reproducing MRST-0 rather than blanking everything everywhere."""
    out = tmp_path / 'S.INC'
    with open(out, 'w') as fh:
        writeSchedule(fh, str(tmp_path),
                      _one('WELSPECS', ['P1', 'G1', 1, 1, np.inf, 'OIL', 0,
                                        'STD', 'SHUT', 'YES', 0, 'SEG', 0]))
    assert 'inf' in out.read_text().lower()


def test_default_text_becomes_the_eclipse_star(tmp_path):
    out = tmp_path / 'S.INC'
    with open(out, 'w') as fh:
        writeSchedule(fh, str(tmp_path),
                      _one('WELSPECS', ['P1', ' Default ', 1, 1, 2000.0,
                                        'OIL', 0, 'STD', 'SHUT', 'YES', 0,
                                        'SEG', 0]))
    text = out.read_text()
    assert '1*' in text and 'Default' not in text


# ------------------------------------- MRST-0 writeSchedule: keywords --

@pytest.mark.parametrize('keyword, row', [
    ('WPIMULT', ['P1', 2.0, 1, 1, 1, 1, 1]),
    ('WELTARG', ['P1', 'BHP', 250.0]),
    ('WEFAC', ['P1', 0.8, 'YES']),
    ('WTEMP', ['P1', 350.0]),
    ('WPOLYMER', ['P1', 1.0, 0.0, 'G1', 'G2']),
    ('WSOLVENT', ['P1', 1.0]),
    ('WSURFACT', ['P1', 1.0]),
    ('GRUPTREE', ['G1', 'FIELD']),
    ('GINJGAS', ['G1', 'GAS', 'G2', 'STD', 1]),
])
def test_extra_schedule_keywords_are_written(tmp_path, keyword, row):
    """These are in MRST-0's writeSchedule and absent from 2026a's."""
    out = tmp_path / 'S.INC'
    with open(out, 'w') as fh:
        writeSchedule(fh, str(tmp_path), _one(keyword, row),
                      writeWEFAC=True)
    assert keyword in out.read_text()


def test_wefac_is_off_by_default(tmp_path):
    out = tmp_path / 'S.INC'
    with open(out, 'w') as fh:
        writeSchedule(fh, str(tmp_path), _one('WEFAC', ['P1', 0.8, 'YES']))
    assert 'WEFAC' not in out.read_text()


def test_tuning_lines_are_written_verbatim(tmp_path):
    """TUNING is stored as ready-made lines, not as fields."""
    sched = {'control': [{'TUNING': ['1 30 0.1', '0.1 0.001', '12 1 25']}],
             'step': {'val': [], 'control': []}}
    out = tmp_path / 'S.INC'
    with open(out, 'w') as fh:
        writeSchedule(fh, str(tmp_path), sched)
    text = out.read_text()
    assert 'TUNING' in text and '1 30 0.1 /' in text


def test_box_values_are_run_length_encoded(tmp_path):
    """A box edit over many cells is written as repeat counts."""
    sched = {'control': [{'BOX': [{'box': [1, 10, 1, 10, 1, 3],
                                   'name': 'PORO',
                                   'values': np.array([0.2] * 12)}]}],
             'step': {'val': [], 'control': []}}
    out = tmp_path / 'S.INC'
    with open(out, 'w') as fh:
        writeSchedule(fh, str(tmp_path), sched)
    text = out.read_text()
    assert 'BOX' in text and 'ENDBOX' in text
    assert '12*0.2' in text


def test_skiprestart_is_written_when_set(tmp_path):
    sched = {'SKIPRESTART': True, 'control': [],
             'step': {'val': [], 'control': []}}
    out = tmp_path / 'S.INC'
    with open(out, 'w') as fh:
        writeSchedule(fh, str(tmp_path), sched)
    assert 'SKIPRESTART' in out.read_text()
