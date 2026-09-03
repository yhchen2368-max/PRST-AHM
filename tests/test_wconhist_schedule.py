"""``WCONHIST`` / ``WCONINJH``: the well controls a history-matching deck
is written with.

These are the keywords that carry the *observed* rates, and they are what
a deck like QIEDIE states every one of its 63 report steps with.
``init_eclipse_problem_ad`` handled ``WCONPROD`` and ``WCONINJE`` and
skipped these two entirely, which is not an error anywhere: the deck
parses, the model builds, a simulation runs. It simply has no wells --
every one of them left at its WELSPECS default of shut with a zero
target, and 63 control blocks collapsed into one.

That is why the adjoint gradient could not be used against such a deck
and the loop fell back to finite differences: not because the adjoint
was wrong, but because the model it would have differentiated had
nothing in it to match.

Ported from ``processWells.m``'s ``process_wconhist`` and
``process_wconinjh``.
"""

import os

import numpy as np
import pytest

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import \
    _wconhist_target

DECK = 'examples/HM/QIEDIE.DATA'

DAY = 86400.0


# ------------------------------------------------------- the control switch --

def test_orat_targets_the_observed_oil_rate():
    value, compi, kind = _wconhist_target('orat', 100.0, 20.0, 5000.0,
                                          np.nan)
    assert value == -100.0
    assert compi == [0.0, 1.0, 0.0]
    assert kind == 'orat'


def test_wrat_targets_the_observed_water_rate():
    value, compi, _ = _wconhist_target('wrat', 100.0, 20.0, 5000.0, np.nan)
    assert value == -20.0
    assert compi == [1.0, 0.0, 0.0]


def test_grat_targets_the_observed_gas_rate():
    value, compi, _ = _wconhist_target('grat', 100.0, 20.0, 5000.0, np.nan)
    assert value == -5000.0
    assert compi == [0.0, 0.0, 1.0]


def test_lrat_targets_the_observed_liquid_rate():
    """Liquid is oil plus water, and a producer's rates are negative."""
    value, _compi, _ = _wconhist_target('lrat', 100.0, 20.0, 5000.0, np.nan)
    assert value == pytest.approx(-120.0)


def test_lrat_composition_reproduces_mrsts_swapped_slots():
    """**A defect carried over from MRST, deliberately.**

    ``process_wconhist`` builds LRAT's composition as ``[rates, 0]/val``
    from ``rates = -[orat, wrat]``, so the oil fraction lands in the
    water slot and the water fraction in the oil slot -- MRST's compi is
    ordered water, oil, gas while the deck's rates are oil, water, gas.
    The RESV branch immediately below it *does* swap them, with the
    comment "Account for OWG ordering. MRST uses WOG", so this is an
    oversight in MRST rather than a convention.

    It is reproduced so a deck matched here agrees with the same deck
    matched in MRST. This test exists to make that a decision on record
    rather than a surprise, and it fails the day someone corrects it --
    at which point the numbers stop matching MRST, which is the thing
    worth knowing.
    """
    _value, compi, _ = _wconhist_target('lrat', 90.0, 10.0, 0.0, np.nan)
    assert compi == [pytest.approx(0.9), pytest.approx(0.1), 0.0]
    # What it would be with the slots the right way round:
    assert compi != [pytest.approx(0.1), pytest.approx(0.9), 0.0]


def test_resv_does_swap_the_slots():
    """The branch that gets it right, kept beside the one that does not."""
    _value, compi, kind = _wconhist_target('resv', 60.0, 30.0, 10.0, np.nan)
    total = 100.0
    assert compi == [pytest.approx(30.0 / total),   # water
                     pytest.approx(60.0 / total),   # oil
                     pytest.approx(10.0 / total)]
    assert kind == 'resv_history'


def test_a_shut_in_well_gets_an_even_split_rather_than_a_nan():
    """Zero liquid would divide by zero; MRST falls back to half and
    half so the composition stays usable."""
    _value, compi, _ = _wconhist_target('lrat', 0.0, 0.0, 0.0, np.nan)
    assert compi == [0.5, 0.5, 0.0]

    _value, compi, _ = _wconhist_target('resv', 0.0, 0.0, 0.0, np.nan)
    assert compi == [pytest.approx(1 / 3)] * 3


def test_bhp_control_falls_back_to_one_atmosphere():
    value, _compi, _ = _wconhist_target('bhp', 100.0, 20.0, 0.0, np.nan)
    assert value == pytest.approx(101325.0)
    value, _compi, _ = _wconhist_target('bhp', 100.0, 20.0, 0.0, 250e5)
    assert value == pytest.approx(250e5)


def test_an_unsupported_mode_is_ignored_not_guessed():
    value, _compi, kind = _wconhist_target('thp', 100.0, 20.0, 0.0, np.nan)
    assert value == 0.0
    assert kind == 'thp'


# --------------------------------------------------------- against QIEDIE --

@pytest.fixture(scope='module')
def converted():
    if not os.path.exists(DECK):
        pytest.skip('QIEDIE.DATA not present')
    _state0, model, schedule, _ = init()
    return model, schedule


def init():
    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import \
        init_eclipse_problem_ad
    return init_eclipse_problem_ad(DECK)


def test_every_report_step_gets_its_own_control(converted):
    """The deck states WCONHIST 63 times. Collapsing them into one
    control would match every step against the first step's rates."""
    _model, schedule = converted
    assert len(schedule['control']) == 63
    assert np.size(schedule['step']['val']) == 63
    assert np.array_equal(np.asarray(schedule['step']['control']),
                          np.arange(63))


def test_the_wells_are_open_with_real_targets(converted):
    """All nine, not six.

    The deck opens five producers (WCONHIST) and four injectors
    (WCONINJH) in every control. Three of the injectors used to come back
    shut because their WCONINJH records wrap onto a second line and the
    reader took the continuation (``RATE /``) for a new keyword, so only
    WELL6's record survived each block. This asserted six for exactly
    that reason.
    """
    model, schedule = converted
    wells = schedule['control'][0]['W']
    assert len(wells) == 9
    active = model._mrst_active_wells({'W': wells})
    assert len(active) == 9
    assert all(w['val'] != 0.0 for w in active)


def test_producers_and_injectors_take_opposite_signs(converted):
    _model, schedule = converted
    wells = schedule['control'][0]['W']
    producers = [w for w in wells if w['sign'] < 0 and w['status']]
    injectors = [w for w in wells if w['sign'] > 0 and w['status']]
    assert producers and injectors
    assert all(w['val'] < 0 for w in producers)
    assert all(w['val'] > 0 for w in injectors)
    assert all(w['type'] == 'rate' for w in injectors)


def test_the_rates_are_converted_out_of_deck_units(converted):
    """QIEDIE is METRIC, so its rates are sm3/day and the model wants
    m3/s. WELL1's first record is 7384.14 oil and 836.1 water."""
    _model, schedule = converted
    well1 = next(w for w in schedule['control'][0]['W']
                 if w['name'] == 'WELL1')
    assert well1['type'] == 'lrat'
    assert well1['val'] == pytest.approx(-(7384.14 + 836.1) / DAY, rel=1e-9)


def test_the_injector_rate_is_converted_too(converted):
    """WELL6 injects 11012.53 sm3/day of water in the first record."""
    _model, schedule = converted
    well6 = next(w for w in schedule['control'][0]['W']
                 if w['name'] == 'WELL6')
    assert well6['sign'] == 1
    assert well6['phase'].upper().startswith('W')
    assert well6['compi'] == [1.0, 0.0, 0.0]
    assert well6['val'] == pytest.approx(11012.53 / DAY, rel=1e-9)


def test_the_controls_actually_differ_between_steps(converted):
    """63 identical controls would be the old bug wearing a new shape."""
    _model, schedule = converted
    first = [w['val'] for w in schedule['control'][0]['W']]
    last = [w['val'] for w in schedule['control'][-1]['W']]
    assert first != last
