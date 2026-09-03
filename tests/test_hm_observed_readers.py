"""Tests for the profile/tracer/monitor readers in the ``hm`` port."""

import datetime as dt

import numpy as np
import pytest

from PRSTCore.hm.utils.observed.addProfileObserved import (
    _integrate_additive, addProfileObserved, additivePieceWise)
from PRSTCore.hm.utils.observed.addTracerObserved import addTracerObserved
from PRSTCore.hm.utils.observed.processMonitorData import processMonitorData
from PRSTCore.hm.utils.observed.readProfileTest import readProfileTest
from PRSTCore.hm.utils.observed.readSaturationTest import readSaturationTest
from PRSTCore.hm.utils.observed.readTracerTest import readTracerTest

_G = {'cells': {'facePos': np.array([0, 2, 4]),
                'faces': np.array([[0, 1], [1, 1], [2, 1], [3, 1]])},
      'faces': {'centroids': np.array([[0, 0, 0.0], [0, 0, 10.0],
                                       [0, 0, 10.0], [0, 0, 20.0]])}}


# ------------------------------------------------------ additive profile --

def test_additive_log_divides_by_the_interval_width():
    h = np.array([4.0])
    a, b = np.array([0.0]), np.array([10.0])
    assert additivePieceWise(h, a, b, np.array([5.0]))[0] == pytest.approx(0.4)
    assert additivePieceWise(h, a, b, np.array([50.0]))[0] == 0.0


def test_integrating_a_full_interval_returns_the_value():
    """That is what makes the log additive rather than a density."""
    h = np.array([4.0, 6.0])
    a, b = np.array([0.0, 10.0]), np.array([10.0, 20.0])
    assert _integrate_additive(h, a, b, 0.0, 10.0) == pytest.approx(4.0)
    assert _integrate_additive(h, a, b, 0.0, 20.0) == pytest.approx(10.0)


def test_profile_splits_the_rate_and_conserves_it():
    W = [{'name': 'P1', 'cells': np.array([0, 1])}]
    schedule = {'control': [{'W': W}]}
    observed = [{'wellSol': [{'name': 'P1', 'qWs': -10.0}]}]
    time_sim = np.asarray([dt.date(2020, 1, 1)], dtype=object)
    data = [('P1', {'date': np.asarray([dt.date(2020, 1, 1)] * 2, dtype=object),
                    'top': np.array([0.0, 10.0]),
                    'bottom': np.array([10.0, 20.0]),
                    'cqW': np.array([3.0, 1.0])})]
    out = addProfileObserved(observed, time_sim, data, _G, schedule, 'W')
    cqs = out[0]['wellsol'][0]['cqs'][:, 0]
    assert cqs.sum() == pytest.approx(-10.0)      # adds back to the total
    assert cqs[0] == pytest.approx(-7.5)          # 3/(3+1) of it
    assert cqs[1] == pytest.approx(-2.5)


def test_profile_rejects_an_unsupported_phase():
    W = [{'name': 'P1', 'cells': np.array([0])}]
    schedule = {'control': [{'W': W}]}
    observed = [{'wellSol': [{'name': 'P1', 'qWs': -1.0}]}]
    data = [('P1', {'date': np.asarray([dt.date(2020, 1, 1)], dtype=object),
                    'top': np.array([0.0]), 'bottom': np.array([10.0])})]
    with pytest.raises(ValueError, match='Unsupported phase'):
        addProfileObserved(observed,
                           np.asarray([dt.date(2020, 1, 1)], dtype=object),
                           data, _G, schedule, 'X')


# ------------------------------------------------------------- readers --

def test_profile_test_splits_the_depth_interval(tmp_path):
    path = tmp_path / 'prof.csv'
    path.write_text('井号,日期,解释井段,绝对产水量\n'
                    'P1,20200101,1200 - 1250,3.0\n'
                    'P1,,1300-1360,1.0\n', encoding='utf-8')
    data = readProfileTest(str(path))
    assert len(data) == 1
    name, table = data[0]
    assert name == 'P1'
    assert np.allclose(table['top'], [1200.0, 1300.0])
    assert np.allclose(table['bottom'], [1250.0, 1360.0])
    # The blank date carries the previous survey date forward.
    assert list(table['date']) == [dt.date(2020, 1, 1)] * 2


def test_saturation_test_reads_split_depth_columns(tmp_path):
    path = tmp_path / 'sat.csv'
    path.write_text('name,date,top,bottom,sw\n'
                    'P1,20200101,1200,1250,0.4\n', encoding='utf-8')
    name, table = readSaturationTest(str(path))[0]
    assert name == 'P1'
    assert table['water'][0] == pytest.approx(0.4)
    assert table['top'][0] == pytest.approx(1200.0)


def _tracer_file(tmp_path, body):
    path = tmp_path / 'tracer.txt'
    path.write_text(body, encoding='utf-8')
    return str(path)


_FULL_RECORD = (
    '注入井号 I1\n'
    '注入层位 1200-1250 1300-1360\n'
    '注剂时间 20200101\n'
    '示踪剂类型 T1\n'
    '示踪剂用量 500\n'
    '示踪剂观测\n'
    '日期井号 P1 P2\n'
    '20200201 0.1 0.2\n'
    '20200301 0.3 0.4\n'
    '/\n'
)


def test_tracer_test_reads_a_full_record(tmp_path):
    rec = readTracerTest(_tracer_file(tmp_path, _FULL_RECORD))[0]
    assert rec['injector'] == 'I1' and rec['name'] == 'T1'
    assert rec['dosage'] == pytest.approx(500.0)
    assert np.allclose(rec['depth'], [[1200.0, 1250.0], [1300.0, 1360.0]])
    assert rec['producer'] == ['P1', 'P2']
    assert rec['output'].shape == (2, 3)
    assert rec['output'][1][2] == pytest.approx(0.4)


def test_tracer_test_reads_several_records(tmp_path):
    rec = readTracerTest(_tracer_file(tmp_path, _FULL_RECORD + _FULL_RECORD))
    assert len(rec) == 2


def test_tracer_test_reports_an_incomplete_record(tmp_path):
    with pytest.raises(ValueError, match='missing'):
        readTracerTest(_tracer_file(tmp_path, '注入井号 I1\n/\n'))


def test_tracer_test_rejects_a_duplicated_field(tmp_path):
    with pytest.raises(ValueError, match='Duplicated'):
        readTracerTest(_tracer_file(tmp_path, '注入井号 I1\n注入井号 I2\n'))


def test_tracer_test_rejects_an_unknown_keyword(tmp_path):
    with pytest.raises(ValueError, match='Unsupported keyword'):
        readTracerTest(_tracer_file(tmp_path, '没有这个关键字 1\n'))


def test_tracer_sample_row_width_is_checked(tmp_path):
    body = (
        '注入井号 I1\n注入层位 1200-1250\n注剂时间 20200101\n'
        '示踪剂类型 T1\n示踪剂用量 500\n示踪剂观测\n'
        '日期井号 P1 P2\n20200201 0.1\n/\n'
    )
    with pytest.raises(ValueError, match='one concentration per producer'):
        readTracerTest(_tracer_file(tmp_path, body))


# ------------------------------------------------------- monitor filter --

def _empty():
    return {'rates': [], 'bhp': [], 'profile': [], 'tracer': [],
            'saturation': []}


def test_monitor_data_drops_unknown_wells_with_a_warning():
    data = _empty()
    data['rates'] = [
        ('P1', {'date': np.asarray([dt.date(2020, 1, 1)], dtype=object)}),
        ('GHOST', {'date': np.asarray([dt.date(2020, 2, 1)], dtype=object)})]
    with pytest.warns(RuntimeWarning, match='no schedule data'):
        out, time = processMonitorData(data, ['P1'])
    assert [n for n, _ in out['rates']] == ['P1']
    assert list(time) == [dt.date(2020, 1, 1)]


def test_monitor_time_is_the_union_across_categories():
    data = _empty()
    data['rates'] = [('P1', {'date': np.asarray([dt.date(2020, 1, 1)],
                                                dtype=object)})]
    data['bhp'] = [('P1', {'date': np.asarray([dt.date(2020, 3, 1)],
                                              dtype=object)})]
    _, time = processMonitorData(data, ['P1'])
    assert list(time) == [dt.date(2020, 1, 1), dt.date(2020, 3, 1)]


def test_from_model_sentinel_is_left_alone():
    data = _empty()
    data['rates'] = 'fromModel'
    out, time = processMonitorData(data, ['P1'])
    assert out['rates'] == 'fromModel'
    assert time.size == 0


def test_tracer_dates_include_the_breakthrough_samples():
    data = _empty()
    data['tracer'] = [{'injector': 'I1', 'date': dt.date(2020, 1, 1),
                       'output': [[dt.date(2020, 2, 1), 0.1]]}]
    _, time = processMonitorData(data, ['I1'])
    assert list(time) == [dt.date(2020, 1, 1), dt.date(2020, 2, 1)]


# ------------------------------------------------------ addTracerObserved --

def test_tracer_observed_builds_a_source_and_records_breakthrough():
    W = [{'name': 'I1', 'cells': np.array([0, 1])},
         {'name': 'P1', 'cells': np.array([0])}]
    schedule = {'control': [{'W': W}, {'W': W}]}
    observed = [{'wellSol': []}, {'wellSol': []}]
    time_sim = np.asarray([dt.date(2020, 1, 1), dt.date(2020, 2, 1)],
                          dtype=object)
    data = [{'injector': 'I1', 'name': 'T1', 'dosage': 500.0,
             'date': dt.date(2020, 1, 1),
             'depth': np.array([[0.0, 20.0]]),
             'producer': ['P1'],
             'output': [[dt.date(2020, 2, 1), 0.42]]}]
    obs, sched = addTracerObserved(observed, time_sim, data, _G, schedule, 'WO')
    src = sched['control'][0]['src']
    assert src['rate'].size == 2
    assert np.allclose(src['sat'][0], [1.0, 0.0])
    assert obs[1]['wellsol'][1]['tracer'][0] == pytest.approx(0.42)
