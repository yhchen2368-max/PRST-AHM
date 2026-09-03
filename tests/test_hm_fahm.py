"""Tests for the FAHM pipeline (``PRSTCore.hm.APP.fahm``) and for
``matchObservedOWGProfile``.

The pipeline tests run against a summary produced by a real ECLIPSE run
of ``examples/HM/QIEDIE.DATA``; they skip when that run has not been made,
so the suite stays runnable without a licensed simulator.
"""

import os

import numpy as np
import pytest

from PRSTCore.ad_core.utils.getPerforationToWellMapping import \
    getPerforationToWellMapping
from PRSTCore.hm.APP.fahm import (DEFAULT_PARAMETER_LIMITS, FahmConfig,
                                  mismatch_by_type, well_series,
                                  mismatch, observed_from_history,
                                  simulated_from_summary)
from PRSTCore.hm.utils.evaluate.matchObservedOWGProfile import \
    matchObservedOWGProfile

DAY = 86400.0


# --------------------------------------------- getPerforationToWellMapping --

def test_perf2well_repeats_each_well_by_its_perforation_count():
    W = [{'cells': [1, 2, 3]}, {'cells': [7]}]
    assert list(getPerforationToWellMapping(W)) == [0, 0, 0, 1]


def test_perf2well_scatter_matrix_places_one_entry_per_perforation():
    W = [{'cells': [1, 2]}, {'cells': [5, 6]}]
    p2w, Rw = getPerforationToWellMapping(W, with_Rw=True)
    assert Rw.shape == (4, 2)
    assert np.allclose(Rw.toarray().sum(axis=0), [2, 2])


def test_perf2well_returns_scalar_one_when_every_well_has_one_perforation():
    _, Rw = getPerforationToWellMapping([{'cells': [1]}, {'cells': [2]}],
                                        with_Rw=True)
    assert Rw == 1


def test_perf2well_handles_no_wells():
    assert getPerforationToWellMapping([]).size == 0


# ------------------------------------------------ matchObservedOWGProfile --

class _Model:
    """The smallest model the objective actually reads."""
    G = {'cells': {'num': 4}}

    def getActivePhases(self):
        return np.array([True, True, False])

    def getPhaseNames(self):
        return ['W', 'O']


def _sol(qWs, qOs, bhp, status=True):
    return {'name': 'W', 'qWs': qWs, 'qOs': qOs, 'qGs': 0.0, 'bhp': bhp,
            'status': status}


def _setup(qW_sim, qO_sim, qW_obs, qO_obs, bhp=200e5):
    schedule = {'step': {'val': np.array([DAY]), 'control': np.array([1])},
                'control': [{'W': [{'cells': [0]}]}]}
    states = [{'wellSol': [_sol(qW_sim, qO_sim, bhp)]}]
    observed = [{'wellSol': [_sol(qW_obs, qO_obs, bhp)]}]
    return schedule, states, observed


_BETA = {'ww': 1.0, 'wo': 1.0, 'wg': 1.0, 'wp': 1.0, 'wt': 1.0, 'wf': 1.0,
         'ws': 1.0}


def test_owg_profile_scores_each_phase_separately():
    schedule, states, observed = _setup(-3.0, -4.0, -1.0, -1.0)
    obj = matchObservedOWGProfile(_Model(), states, schedule, observed,
                                  NormalizationFactor=_BETA)
    # dt/T/nw == 1, bhp matches, so only the two rate terms contribute.
    assert obj[0] == pytest.approx((-3.0 + 1.0) ** 2 + (-4.0 + 1.0) ** 2)


def test_owg_profile_is_zero_for_a_perfect_match():
    schedule, states, observed = _setup(-3.0, -4.0, -3.0, -4.0)
    obj = matchObservedOWGProfile(_Model(), states, schedule, observed,
                                  NormalizationFactor=_BETA)
    assert obj[0] == pytest.approx(0.0)


def test_owg_profile_requires_a_normalization_factor():
    """The MATLAB reads beta.ww unconditionally; there is no default."""
    schedule, states, observed = _setup(-3.0, -4.0, -1.0, -1.0)
    with pytest.raises(ValueError, match='NormalizationFactor'):
        matchObservedOWGProfile(_Model(), states, schedule, observed)


def test_owg_profile_alpha_switches_a_term_off():
    schedule, states, observed = _setup(-3.0, -4.0, -1.0, -1.0)
    alpha = {'ww': 0, 'wo': 1, 'wg': 1, 'wp': 1, 'wt': 0, 'wf': 0, 'ws': 0}
    obj = matchObservedOWGProfile(_Model(), states, schedule, observed,
                                  ObjectiveWeight=alpha,
                                  NormalizationFactor=_BETA)
    assert obj[0] == pytest.approx((-4.0 + 1.0) ** 2)   # water term dropped


def test_owg_profile_drops_the_pressure_term_below_one_atmosphere():
    """omega.wp is zeroed where the observed bhp is atmospheric or less."""
    schedule, states, observed = _setup(-1.0, -1.0, -1.0, -1.0)
    states[0]['wellSol'][0]['bhp'] = 300e5
    observed[0]['wellSol'][0]['bhp'] = 1.0e5          # below 1 atm
    obj = matchObservedOWGProfile(_Model(), states, schedule, observed,
                                  NormalizationFactor=_BETA)
    assert obj[0] == pytest.approx(0.0)


def test_owg_profile_weights_each_step_by_its_share_of_time():
    schedule = {'step': {'val': np.array([DAY, 3 * DAY]),
                         'control': np.array([1, 1])},
                'control': [{'W': [{'cells': [0]}]}]}
    states = [{'wellSol': [_sol(-3.0, 0.0, 200e5)]}] * 2
    observed = [{'wellSol': [_sol(-1.0, 0.0, 200e5)]}] * 2
    obj = matchObservedOWGProfile(_Model(), states, schedule, observed,
                                  NormalizationFactor=_BETA)
    assert obj[1] == pytest.approx(3.0 * obj[0])


def test_owg_profile_blanks_a_well_shut_on_one_side_only():
    """A well shut in the simulation but open in the data must not be
    scored on pressure."""
    schedule = {'step': {'val': np.array([DAY]), 'control': np.array([1])},
                'control': [{'W': [{'cells': [0]}, {'cells': [1]}]}]}
    states = [{'wellSol': [_sol(-1.0, 0.0, 200e5),
                           _sol(0.0, 0.0, 0.0, status=False)]}]
    observed = [{'wellSol': [_sol(-1.0, 0.0, 200e5),
                             _sol(-5.0, 0.0, 300e5)]}]
    obj = matchObservedOWGProfile(_Model(), states, schedule, observed,
                                  NormalizationFactor=_BETA)
    # The shut well contributes its rate difference but no pressure term.
    assert obj[0] == pytest.approx((0.0 + 5.0) ** 2 / 2.0)


# ---------------------------------------------------------- FAHM pipeline --

_PREFIX = os.environ.get('FAHM_TEST_PREFIX')

_DECK = 'examples/HM/QIEDIE.DATA'


def _cached_run():
    """Run QIEDIE through ECLIPSE once, and keep the result.

    These four tests are the only ones that read a real summary, and
    they are what proves the reader tells the *H history vectors apart
    from the simulated ones -- a mistake that would make every misfit
    zero and every history match succeed instantly. Leaving them behind
    an environment variable meant they never ran.

    The run costs a few minutes, so its output is cached outside the
    repository and keyed on the deck's modification time: the first
    session pays for it and later ones do not. Set
    ``PRSTCORE_NO_SIMULATOR`` to opt out, or ``FAHM_TEST_PREFIX`` to
    point at a run you already have.
    """
    import tempfile

    from PRSTCore.hm.APP.fahm import (DEFAULT_ECLIPSE, FahmConfig,
                                      prepare_run_dir, run_eclipse)

    if os.environ.get('PRSTCORE_NO_SIMULATOR'):
        pytest.skip('PRSTCORE_NO_SIMULATOR is set')
    if not os.path.exists(_DECK):
        pytest.skip('QIEDIE.DATA not present')
    simulator = os.environ.get('FAHM_ECLIPSE', DEFAULT_ECLIPSE)
    if not os.path.exists(simulator):
        pytest.skip('ECLIPSE not installed at %s' % simulator)

    stamp = int(os.path.getmtime(_DECK))
    cache = os.path.join(tempfile.gettempdir(),
                         'prstcore-fahm-%d' % stamp)
    prefix = os.path.join(cache, 'QIEDIE')
    if os.path.exists(prefix + '.UNSMRY'):
        return prefix

    config = FahmConfig(deck_path=_DECK, work_dir=cache,
                        simulator=simulator)
    prepare_run_dir(config, cache)
    return run_eclipse(config, cache, case_name='QIEDIE')


def _summary():
    if _PREFIX and os.path.exists(_PREFIX + '.UNSMRY'):
        return _PREFIX
    return _cached_run()


def test_config_falls_back_to_a_wide_box_for_an_unknown_parameter():
    config = FahmConfig(deck_path='x.DATA', work_dir='.')
    assert config.limits_for('permx') == DEFAULT_PARAMETER_LIMITS['permx']
    assert config.limits_for('nosuch') == (0.1, 10.0)


def test_config_matches_parameter_names_case_insensitively():
    config = FahmConfig(deck_path='x.DATA', work_dir='.')
    assert config.limits_for('PERMX') == config.limits_for('permx')


def test_observed_and_simulated_cover_the_same_wells_and_steps():
    prefix = _summary()
    observed, t_obs = observed_from_history(prefix)
    simulated, t_sim = simulated_from_summary(prefix)
    assert len(observed) == len(simulated)
    assert np.allclose(t_obs, t_sim)
    assert ([w['name'] for w in observed[0]['wellSol']]
            == [w['name'] for w in simulated[0]['wellSol']])


def test_observed_history_differs_from_the_simulation():
    """If these agreed there would be nothing to history match -- and it
    would mean the *H vectors were being read as the simulated ones."""
    prefix = _summary()
    observed, _ = observed_from_history(prefix)
    simulated, _ = simulated_from_summary(prefix)
    o = np.array([w['qOs'] for w in observed[-1]['wellSol']])
    s = np.array([w['qOs'] for w in simulated[-1]['wellSol']])
    assert not np.allclose(o, s)


def test_a_perfect_match_scores_exactly_zero():
    prefix = _summary()
    simulated, time = simulated_from_summary(prefix)
    simulated, time = simulated[1:], time[1:]
    dt = np.diff(np.concatenate([[0.0], time]))
    schedule = {'step': {'val': dt, 'control': np.ones(dt.size, dtype=int)},
                'control': [{'W': []}]}
    weights = {'oil': 1.0, 'water': 1.0, 'gas': 0.0, 'bhp': 0.0}
    assert mismatch(simulated, simulated, schedule, weights) == 0.0


def test_the_mismatch_is_finite_and_positive():
    prefix = _summary()
    observed, time = observed_from_history(prefix)
    simulated, _ = simulated_from_summary(prefix)
    observed, simulated, time = observed[1:], simulated[1:], time[1:]
    dt = np.diff(np.concatenate([[0.0], time]))
    schedule = {'step': {'val': dt, 'control': np.ones(dt.size, dtype=int)},
                'control': [{'W': []}]}
    value = mismatch(observed, simulated, schedule,
                     {'oil': 1.0, 'water': 1.0, 'gas': 0.0, 'bhp': 0.0})
    assert np.isfinite(value) and value > 0


# ------------------------------------------------- the score breakdown --

# Two wells over two 30-day steps, with the simulation missing the
# history in every quantity so no term is trivially zero.
def _pair():
    def sols(qw, qo, bhp):
        return {'wellSol': [{'name': 'W%d' % i, 'qWs': float(a),
                             'qOs': float(b), 'bhp': float(c),
                             'status': True, 'sign': -1}
                            for i, (a, b, c) in enumerate(zip(qw, qo, bhp))]}

    observed = [sols([1.0, 2.0], [10.0, 20.0], [200e5, 210e5]),
                sols([1.5, 2.5], [11.0, 21.0], [201e5, 211e5])]
    simulated = [sols([1.2, 2.4], [10.5, 19.0], [202e5, 209e5]),
                 sols([1.4, 2.9], [12.0, 20.0], [203e5, 212e5])]
    dt = np.array([30.0, 30.0]) * DAY
    schedule = {'step': {'val': dt, 'control': np.ones(2, dtype=int)},
                'control': [{'W': []}]}
    return observed, simulated, schedule


WEIGHTS = {'oil': 1.0, 'water': 1.0, 'gas': 0.0, 'bhp': 1e-5}


def test_the_breakdown_sums_to_the_mismatch():
    """The Mismatch Scores panel and the objective must be two readings
    of one computation, not two computations that could drift apart."""
    observed, simulated, schedule = _pair()
    total = mismatch(observed, simulated, schedule, WEIGHTS,
                     match_only_producers=False)
    parts = mismatch_by_type(observed, simulated, schedule, WEIGHTS,
                             match_only_producers=False)
    assert sum(parts.values()) == pytest.approx(total, rel=1e-12)


def test_the_breakdown_names_what_the_objective_covers():
    """Water-cut formulation: liquid rate, water cut, bhp. Gas is not
    among them and must be absent rather than reported as zero."""
    observed, simulated, schedule = _pair()
    parts = mismatch_by_type(observed, simulated, schedule, WEIGHTS,
                             match_only_producers=False)
    assert set(parts) == {'Oil', 'Water', 'BHP'}
    assert all(v > 0 for v in parts.values())


def test_the_gas_formulation_reports_gas_instead_of_water():
    observed, simulated, schedule = _pair()
    for step in observed + simulated:
        for well in step['wellSol']:
            well['qGs'] = 5.0 * well['qOs']
    weights = dict(WEIGHTS, gas=1.0)
    parts = mismatch_by_type(observed, simulated, schedule, weights,
                             match_only_producers=False)
    assert set(parts) == {'Oil', 'Gas', 'BHP'}


def test_the_per_well_breakdown_sums_to_the_per_type_one():
    observed, simulated, schedule = _pair()
    parts = mismatch_by_type(observed, simulated, schedule, WEIGHTS,
                             match_only_producers=False)
    per_well = mismatch_by_type(observed, simulated, schedule, WEIGHTS,
                                match_only_producers=False, per_well=True)
    for name, value in parts.items():
        assert per_well[name].size == 2
        assert float(per_well[name].sum()) == pytest.approx(value, rel=1e-12)


def test_a_perfect_match_scores_zero_in_every_part():
    _observed, simulated, schedule = _pair()
    parts = mismatch_by_type(simulated, simulated, schedule, WEIGHTS,
                             match_only_producers=False)
    assert all(v == 0.0 for v in parts.values())


# --------------------------------------------------- the plotting series --

def test_well_series_lays_the_curves_out_by_step_and_well():
    observed, simulated, _schedule = _pair()
    series = well_series(observed, simulated, np.array([30.0, 60.0]) * DAY)
    assert series['wells'] == ['W0', 'W1']
    assert series['observed']['Oil'].shape == (2, 2)
    assert series['observed']['Oil'][0].tolist() == [10.0, 20.0]
    assert series['simulated']['Oil'][0].tolist() == [10.5, 19.0]


def test_water_cut_and_gor_are_derived_not_read():
    """Neither is a matched quantity, so nothing stores them; both are
    what an engineer reads off the rates."""
    observed, simulated, _schedule = _pair()
    for step in observed + simulated:
        for well in step['wellSol']:
            well['qGs'] = 5.0 * well['qOs']
    series = well_series(observed, simulated, np.array([30.0, 60.0]) * DAY)
    oil = series['observed']['Oil'][0, 0]
    water = series['observed']['Water'][0, 0]
    assert series['observed']['WaterCut'][0, 0] == pytest.approx(
        abs(water) / (abs(oil) + abs(water)))
    assert series['observed']['GOR'][0, 0] == pytest.approx(5.0)


def test_a_shut_in_step_gives_zero_rather_than_a_nan():
    """A nan would break the axis instead of drawing a gap."""
    observed, simulated, _schedule = _pair()
    for step in observed + simulated:
        for well in step['wellSol']:
            well['qOs'] = well['qWs'] = well['qGs'] = 0.0
    series = well_series(observed, simulated, np.array([30.0, 60.0]) * DAY)
    assert np.all(np.isfinite(series['observed']['WaterCut']))
    assert np.all(series['observed']['GOR'] == 0.0)


def test_every_plottable_curve_but_tracer_is_produced():
    """Tracer keeps its checkbox because FAHM has one, but nothing
    produces a tracer series: the summary reader does not carry it.
    Ticking it leaves the panel blank rather than drawing a flat zero
    that would read as a perfectly matched tracer."""
    from PRSTCore.hm.APP.fahm_app import PLOT_CURVES
    observed, simulated, _schedule = _pair()
    series = well_series(observed, simulated, np.array([30.0, 60.0]) * DAY)
    assert set(PLOT_CURVES) - set(series['observed']) == {'Tracer'}
    assert set(series['observed']) == set(series['simulated'])
