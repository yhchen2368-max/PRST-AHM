"""Tests for the port of MRST ``hm/utils/observed``."""

import datetime as dt

import numpy as np
import pytest

from PRSTCore.hm.utils.observed.addRatesObserved import (addBhpObserved,
                                                         addRatesObserved)
from PRSTCore.hm.utils.observed.addSaturationObserved import (
    _integrate_piecewise, _nonAdditivePieceWise, addSaturationObserved)
from PRSTCore.hm.utils.observed.getCellFacesDepth import getCellFacesDepth
from PRSTCore.hm.utils.observed.getNormalizationFactors import \
    getNormalizationFactors
from PRSTCore.hm.utils.observed.getObservedFromFile import getObservedFromFile
from PRSTCore.hm.utils.observed.getObservedFromSchedule import \
    getObservedFromSchedule
from PRSTCore.hm.utils.observed.readProductionHistory import (
    ATMOSPHERIC_MPA, readProductionHistory, solveKeySimilarities)

DAY = 86400.0


# ------------------------------------------------- getObservedFromSchedule --

def _schedule(nstep=3):
    W = [{'name': 'P1', 'sign': -1, 'status': True, 'qWs': -1.0, 'qOs': -2.0,
          'qGs': -3.0, 'bhp': 200.0},
         {'name': 'I1', 'sign': 1, 'status': True, 'qWs': 4.0, 'qOs': 0.0,
          'qGs': 0.0, 'bhp': 400.0}]
    return {'control': [{'W': W}],
            'step': {'val': np.full(nstep, DAY), 'control': np.ones(nstep, int)}}


def test_mrst_observed_has_one_entry_per_step():
    obs = getObservedFromSchedule(_schedule(3))
    assert len(obs) == 3
    assert obs[0]['dt'] == pytest.approx(DAY)
    assert [w['name'] for w in obs[0]['wellSol']] == ['P1', 'I1']
    assert obs[0]['wellSol'][0]['qOs'] == pytest.approx(-2.0)


def test_jutul_observed_is_arrays_keyed_by_quantity():
    out = getObservedFromSchedule(_schedule(3), simulator='Jutul')
    assert out['names'] == ['P1', 'I1']
    assert out['bhp'].shape == (3, 2)
    assert np.allclose(out['qWs'][:, 1], 4.0)


def test_unknown_simulator_is_rejected():
    with pytest.raises(ValueError, match='Unsupported'):
        getObservedFromSchedule(_schedule(), simulator='nosuch')


# --------------------------------------------------- getNormalizationFactors --

def _observed(nstep=2, qWs=-1.0, qOs=-2.0, bhp=200.0, tracer=None):
    out = []
    for _ in range(nstep):
        sol = [{'name': 'P1', 'sign': -1, 'status': True, 'qWs': qWs,
                'qOs': qOs, 'qGs': 0.0, 'bhp': bhp},
               {'name': 'I1', 'sign': 1, 'status': True, 'qWs': 9.0,
                'qOs': 0.0, 'qGs': 0.0, 'bhp': 400.0}]
        if tracer is not None:
            for s in sol:
                s['tracer'] = tracer
        out.append({'dt': DAY, 'wellSol': sol})
    return out


def test_weights_are_time_over_the_producer_integral():
    """w = sum(dt) / sum(mean|q| * dt), over producers only."""
    beta = getNormalizationFactors(_observed(2, qWs=-4.0))
    assert beta['ww'] == pytest.approx(1.0 / 4.0)


def test_a_quantity_that_never_appears_gets_zero_weight():
    beta = getNormalizationFactors(_observed(2))
    assert beta['wg'] == 0.0          # qGs is zero everywhere
    assert beta['wt'] == 0.0          # no tracer field at all


def test_injectors_do_not_enter_the_weights():
    """Only sign == -1 contributes, so the injector's 9.0 is ignored."""
    beta = getNormalizationFactors(_observed(2, qWs=-4.0))
    assert beta['ww'] == pytest.approx(1.0 / 4.0)


def test_shut_wells_contribute_zero():
    obs = _observed(1, qWs=-4.0)
    obs[0]['wellSol'][0]['status'] = False
    beta = getNormalizationFactors(obs)
    assert beta['ww'] == 0.0


def test_cumulative_time_is_differenced_into_step_lengths():
    obs = _observed(2, qWs=-4.0)
    for i, entry in enumerate(obs):
        del entry['dt']
        entry['time'] = (i + 1) * DAY
    beta = getNormalizationFactors(obs)
    assert beta['ww'] == pytest.approx(1.0 / 4.0)


def test_tracer_is_averaged_across_its_components():
    beta = getNormalizationFactors(_observed(2, tracer=np.array([2.0, 4.0])))
    assert beta['wt'] == pytest.approx(1.0 / 3.0)


# --------------------------------------------------------- add*Observed --

def _obs_and_schedule():
    W = [{'name': 'P1', 'sign': -1, 'cells': np.array([0, 1])}]
    schedule = {'control': [{'W': W}]}
    observed = [{'wellSol': [{'name': 'P1', 'sign': -1, 'qWs': 0.0,
                              'qOs': 0.0, 'qGs': 0.0, 'bhp': 0.0}]}
                for _ in range(2)]
    time_sim = np.asarray([dt.date(2020, 1, 1), dt.date(2020, 2, 1)],
                          dtype=object)
    return observed, time_sim, schedule


def test_rates_are_converted_per_day_and_signed_by_the_well():
    observed, time_sim, schedule = _obs_and_schedule()
    data = [('P1', {'date': np.asarray([dt.date(2020, 1, 1)], dtype=object),
                    'qWs': np.array([86.4]), 'qOs': np.array([172.8]),
                    'qGs': np.array([0.0])})]
    out = addRatesObserved(observed, time_sim, data, None, schedule, 'WOG')
    sol = out[0]['wellSol'][0]
    # 86.4 m^3/day / 86400 s, negated by the producer's sign.
    assert sol['qWs'] == pytest.approx(-86.4 / DAY)
    assert sol['qOs'] == pytest.approx(-172.8 / DAY)


def test_rates_only_touch_the_matching_report_step():
    observed, time_sim, schedule = _obs_and_schedule()
    data = [('P1', {'date': np.asarray([dt.date(2020, 2, 1)], dtype=object),
                    'qWs': np.array([86.4]), 'qOs': np.array([0.0]),
                    'qGs': np.array([0.0])})]
    out = addRatesObserved(observed, time_sim, data, None, schedule, 'W')
    assert out[0]['wellSol'][0]['qWs'] == 0.0
    assert out[1]['wellSol'][0]['qWs'] == pytest.approx(-86.4 / DAY)


def test_unsupported_phase_is_rejected():
    observed, time_sim, schedule = _obs_and_schedule()
    data = [('P1', {'date': np.asarray([dt.date(2020, 1, 1)], dtype=object)})]
    with pytest.raises(ValueError, match='Unsupported phase'):
        addRatesObserved(observed, time_sim, data, None, schedule, 'X')


def test_bhp_is_converted_from_mpa():
    observed, time_sim, schedule = _obs_and_schedule()
    data = [('P1', {'date': np.asarray([dt.date(2020, 1, 1)], dtype=object),
                    'bhp': np.array([25.0])})]
    out = addBhpObserved(observed, time_sim, data, None, schedule)
    assert out[0]['wellSol'][0]['bhp'] == pytest.approx(25.0e6)


# ------------------------------------------------------- depth utilities --

def test_cell_face_depths_are_the_extreme_face_centroids():
    G = {'cells': {'facePos': np.array([0, 2, 4]),
                   'faces': np.array([[0, 1], [1, 1], [2, 1], [3, 1]])},
         'faces': {'centroids': np.array([[0, 0, 10.0], [0, 0, 20.0],
                                          [0, 0, 30.0], [0, 0, 45.0]])}}
    top, bottom = getCellFacesDepth(G, [0, 1])
    assert np.allclose(top, [10.0, 30.0])
    assert np.allclose(bottom, [20.0, 45.0])


def test_piecewise_log_picks_the_first_containing_interval():
    h = np.array([0.3, 0.7])
    a, b = np.array([0.0, 10.0]), np.array([10.0, 20.0])
    got = _nonAdditivePieceWise(h, a, b, np.array([5.0, 15.0, 50.0]))
    assert np.allclose(got, [0.3, 0.7, 0.0])   # outside -> zero, not extrapolated


def test_piecewise_integral_is_exact_over_a_step_function():
    h = np.array([0.2, 0.8])
    a, b = np.array([0.0, 10.0]), np.array([10.0, 20.0])
    # Over [5, 15]: 5 units of 0.2 plus 5 of 0.8.
    assert _integrate_piecewise(h, a, b, 5.0, 15.0) == pytest.approx(5.0)


def test_saturation_profile_is_the_interval_mean():
    G = {'cells': {'facePos': np.array([0, 2]),
                   'faces': np.array([[0, 1], [1, 1]])},
         'faces': {'centroids': np.array([[0, 0, 0.0], [0, 0, 20.0]])}}
    W = [{'name': 'P1', 'cells': np.array([0])}]
    schedule = {'control': [{'W': W}]}
    observed = [{}]
    time_sim = np.asarray([dt.date(2020, 1, 1)], dtype=object)
    data = [('P1', {'date': np.asarray([dt.date(2020, 1, 1)] * 2, dtype=object),
                    'top': np.array([0.0, 10.0]),
                    'bottom': np.array([10.0, 20.0]),
                    'water': np.array([0.2, 0.8])})]
    out = addSaturationObserved(observed, time_sim, data, G, schedule, 'W')
    assert out[0]['wellsol'][0]['sw'][0, 0] == pytest.approx(0.5)


def test_saturation_given_in_percent_is_rescaled():
    G = {'cells': {'facePos': np.array([0, 2]),
                   'faces': np.array([[0, 1], [1, 1]])},
         'faces': {'centroids': np.array([[0, 0, 0.0], [0, 0, 10.0]])}}
    W = [{'name': 'P1', 'cells': np.array([0])}]
    schedule = {'control': [{'W': W}]}
    time_sim = np.asarray([dt.date(2020, 1, 1)], dtype=object)
    data = [('P1', {'date': np.asarray([dt.date(2020, 1, 1)], dtype=object),
                    'top': np.array([0.0]), 'bottom': np.array([10.0]),
                    'water': np.array([40.0])})]
    out = addSaturationObserved([{}], time_sim, data, G, schedule, 'W')
    assert out[0]['wellsol'][0]['sw'][0, 0] == pytest.approx(0.4)


# ------------------------------------------------------ production history --

def test_header_synonyms_map_to_canonical_names():
    sheet = {'井号': np.array(['P1']), '日期': np.array(['20200101']),
             '日产水量': np.array(['1.5']), '日产油量': np.array(['2.5'])}
    table = solveKeySimilarities(sheet)
    assert set(table) >= {'name', 'date', 'water', 'oil'}
    assert table['water'][0] == pytest.approx(1.5)   # text -> float


def test_production_history_reads_a_csv(tmp_path):
    path = tmp_path / 'hist.csv'
    path.write_text(
        'name,date,water,oil,gas,bhp\n'
        'P1,20200101,0,0,0,0\n'          # idle leading row, dropped
        'P1,20200201,1.0,2.0,3.0,25.0\n'
        'P2,20200201,4.0,5.0,6.0,0\n',   # zero bhp -> atmospheric
        encoding='utf-8')
    data = readProductionHistory(str(path))
    by_name = dict(data)
    assert set(by_name) == {'P1', 'P2'}
    assert by_name['P1']['water'][0] == pytest.approx(1.0)
    assert by_name['P1']['date'][0] == dt.date(2020, 2, 1)
    assert by_name['P2']['bhp'][0] == pytest.approx(ATMOSPHERIC_MPA)


def test_missing_file_is_reported():
    with pytest.raises(FileNotFoundError):
        getObservedFromFile(['nosuchfile.csv'], 'rates')


def test_unsupported_data_type_is_rejected():
    with pytest.raises(ValueError, match='Unsupported data type'):
        getObservedFromFile(['x.csv'], 'nosuchkind')
