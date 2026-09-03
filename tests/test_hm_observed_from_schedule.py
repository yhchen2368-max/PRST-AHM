"""The observed data a history match is scored against.

Three defects lived here, and none of them raised. Each produced numbers
that were the right shape, the right magnitude, and wrong.

**The rates were never written onto the wells.**
``getObservedFromSchedule`` reads ``W.qWs``/``qOs``/``qGs``/``bhp``, and
nothing set them -- stock MRST's ``processWells.m`` does not, as a search
of the whole tree confirms. It is MRST-0 that does, under a comment
naming it: *"edited by zhang. Major modification: write observed data
(eg., qWs, qOs, qGs, etc.) to schedule"*. Without it the observed
container is full of ``None`` and every normalisation factor is zero.

**The control index was off by one.** MATLAB's ``schedule.control(ctrl)``
is 1-based so the port subtracted one, but PRSTCore's ``step['control']``
is already 0-based. Step 0 then read ``controls[-1]`` -- the *last*
control -- and every step was matched against the wrong date's rates.
The numbers stayed entirely plausible: real production figures, just not
this step's, and the normalisation factors differed by a couple of
percent.

**A well no control had yet mentioned had no rates at all.** MRST-0
opens each well's processing with ``[qWs, qOs, qGs, bhp] = deal(0, 0, 0,
1*atm)``; a shut well observes zero rate at one atmosphere.
"""

import os

import numpy as np
import pytest

from PRSTCore.hm.utils.observed.getObservedFromSchedule import \
    getObservedFromSchedule

DECK = 'examples/HM/QIEDIE.DATA'
DAY = 86400.0

#: The first three WCONHIST records for WELL1, straight from the deck:
#: ``'WELL1' OPEN LRAT <orat> <wrat> <grat> 3* 100 /``
WELL1 = [(7384.14, 836.10, 769497.94),
         (8822.34, 886.90, 916063.51),
         (9020.82, 907.01, 935694.15)]


@pytest.fixture(scope='module')
def case():
    if not os.path.exists(DECK):
        pytest.skip('QIEDIE.DATA not present')
    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import \
        init_eclipse_problem_ad
    _state0, _model, schedule, _ = init_eclipse_problem_ad(DECK)
    return schedule, getObservedFromSchedule(schedule)


def _well(entry, name):
    return next(w for w in entry['wellSol'] if w['name'] == name)


# ------------------------------------------------ the rates are present --

def test_every_well_reports_a_number(case):
    """A ``None`` here is not "unobserved" -- the objective multiplies
    these, so it is a crash at best and a silent skip at worst."""
    _schedule, observed = case
    for step, entry in enumerate(observed):
        for well in entry['wellSol']:
            for field in ('qWs', 'qOs', 'qGs', 'bhp'):
                assert well[field] is not None, (step, well['name'], field)
                assert np.isfinite(float(well[field]))


def test_a_well_no_control_has_mentioned_observes_nothing():
    """MRST-0 opens each well's processing with ``[qWs, qOs, qGs, bhp] =
    deal(0, 0, 0, 1*atm)``: a well no control record has reached observes
    zero rate at one atmosphere, not ``None``.

    This used to read the case off QIEDIE, on the belief that WELL7-9
    were absent from its first control. They were not -- the reader was
    splitting their two-line WCONINJH records and losing three of the
    four injectors, so they came back shut. All nine wells are open in
    every QIEDIE control, so the defaults are exercised on a well that
    genuinely has no record instead.
    """
    schedule = {'step': {'val': [1.0], 'control': [0]},
                'control': [{'W': [
                    {'name': 'W1', 'status': True, 'sign': -1,
                     'qWs': -1.0, 'qOs': -2.0, 'qGs': -3.0, 'bhp': 2.5e7},
                    # No WCON* record ever named this one.
                    {'name': 'W2', 'status': False, 'sign': -1,
                     'qWs': 0.0, 'qOs': 0.0, 'qGs': 0.0, 'bhp': 101325.0},
                ]}]}
    observed = getObservedFromSchedule(schedule)
    quiet = _well(observed[0], 'W2')
    assert quiet['qWs'] == 0.0 and quiet['qOs'] == 0.0 and quiet['qGs'] == 0.0
    assert quiet['bhp'] == pytest.approx(101325.0)
    assert quiet['status'] is False


def test_every_qiedie_well_is_open_in_the_first_control(case):
    """Nine wells, all open: five producers under WCONHIST and four
    injectors under WCONINJH. Three of the injectors used to come back
    shut because their records wrap onto a second line."""
    _schedule, observed = case
    first = observed[0]['wellSol']
    assert len(first) == 9
    assert all(w['status'] for w in first)


# ------------------------------------------------------- the right step --

@pytest.mark.parametrize('step', [0, 1, 2])
def test_each_step_reports_its_own_records_rates(case, step):
    """The off-by-one that made this read the last control. Checked
    against the deck's own numbers, not against whatever came out."""
    _schedule, observed = case
    orat, wrat, grat = WELL1[step]
    well = _well(observed[step], 'WELL1')
    assert well['qOs'] * DAY == pytest.approx(-orat, rel=1e-9)
    assert well['qWs'] * DAY == pytest.approx(-wrat, rel=1e-9)
    assert well['qGs'] * DAY == pytest.approx(-grat, rel=1e-9)


def test_the_steps_are_not_all_the_same(case):
    """63 identical entries would be the old bug in a new shape."""
    _schedule, observed = case
    first = [w['qOs'] for w in observed[0]['wellSol']]
    last = [w['qOs'] for w in observed[-1]['wellSol']]
    assert first != last


def test_there_is_one_entry_per_report_step(case):
    schedule, observed = case
    assert len(observed) == np.size(schedule['step']['val']) == 63


# ------------------------------------------- rates reach the well struct --

def test_the_observed_rates_are_independent_of_the_control_mode(case):
    """WELL1 is on LRAT, so its *target* is the liquid total -- but oil
    and water are observed separately and a match needs both. MRST-0
    records all three regardless of which one controls."""
    schedule, _observed = case
    well = next(w for w in schedule['control'][0]['W']
                if w['name'] == 'WELL1')
    orat, wrat, _grat = WELL1[0]
    assert well['type'] == 'lrat'
    assert well['val'] * DAY == pytest.approx(-(orat + wrat), rel=1e-9)
    assert well['qOs'] * DAY == pytest.approx(-orat, rel=1e-9)
    assert well['qWs'] * DAY == pytest.approx(-wrat, rel=1e-9)


def test_an_injector_records_its_phase_only(case):
    """WELL6 injects water: qWs carries the rate, qOs and qGs are zero."""
    schedule, _observed = case
    well = next(w for w in schedule['control'][0]['W']
                if w['name'] == 'WELL6')
    assert well['sign'] == 1
    assert well['qWs'] > 0 and well['qOs'] == 0.0 and well['qGs'] == 0.0


# ------------------------------------------------- normalisation factors --

def test_the_weights_are_not_all_zero(case):
    """``getNormalizationFactors`` divides by the observed totals, so an
    all-None container gives every weight as zero -- an objective that
    is identically zero and an optimiser that stops at once."""
    from PRSTCore.hm.utils.observed.getNormalizationFactors import \
        getNormalizationFactors
    _schedule, observed = case
    beta = dict(getNormalizationFactors(observed))
    assert float(np.ravel(beta['ww'])[0]) > 0
    assert float(np.ravel(beta['wo'])[0]) > 0
    assert float(np.ravel(beta['wg'])[0]) > 0
