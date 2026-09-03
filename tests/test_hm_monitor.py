"""Tests for the port of MRST ``getMonitorData.m``."""

import datetime as dt

import numpy as np
import pytest

from PRSTCore.hm.utils.observed.getMonitorData import getMonitorData

DAY = 86400.0
START = dt.date(2020, 1, 1)

_G = {'cells': {'facePos': np.array([0, 2, 4]),
                'faces': np.array([[0, 1], [1, 1], [2, 1], [3, 1]])},
      'faces': {'centroids': np.array([[0, 0, 0.0], [0, 0, 10.0],
                                       [0, 0, 10.0], [0, 0, 20.0]])}}


class _Model:
    G = _G

    @staticmethod
    def getPhaseNames():
        return ['W', 'O']


def _schedule(nstep=3):
    W = [{'name': 'P1', 'cells': np.array([0, 1]), 'sign': -1, 'type': 'bhp',
          'val': 1.0e7, 'qWs': -1.0, 'qOs': -2.0, 'qGs': 0.0, 'bhp': 2.0e7},
         {'name': 'I1', 'cells': np.array([0]), 'sign': 1, 'type': 'rate',
          'val': 3.0, 'qWs': 4.0, 'qOs': 0.0, 'qGs': 0.0, 'bhp': 3.0e7}]
    return {'control': [{'W': W}],
            'step': {'val': np.full(nstep, 30.0 * DAY),
                     'control': np.ones(nstep, dtype=int)}}


def test_from_model_reads_the_schedule_rates():
    obs, sched, tracers = getMonitorData(_Model(), _schedule(2), START)
    assert len(obs) == 2
    p1 = obs[0]['wellSol'][0]
    assert p1['name'] == 'P1'
    assert p1['qOs'] == pytest.approx(-2.0)
    assert p1['bhp'] == pytest.approx(2.0e7)
    assert tracers == []


def test_measured_rates_take_the_wells_sign():
    """A file gives magnitudes; the schedule supplies the direction."""
    obs, _, _ = getMonitorData(_Model(), _schedule(1), START)
    assert obs[0]['wellSol'][0]['sign'] == -1
    assert obs[0]['wellSol'][0]['qWs'] < 0      # producer
    assert obs[0]['wellSol'][1]['qWs'] > 0      # injector


def test_status_follows_nonzero_rate_or_pressure():
    sched = _schedule(1)
    for w in sched['control'][0]['W']:
        w.update({'qWs': 0.0, 'qOs': 0.0, 'qGs': 0.0, 'bhp': 0.0})
    obs, _, _ = getMonitorData(_Model(), sched, START)
    assert all(not w['status'] for w in obs[0]['wellSol'])

    sched['control'][0]['W'][0]['bhp'] = 1.0e7
    obs, _, _ = getMonitorData(_Model(), sched, START)
    assert obs[0]['wellSol'][0]['status'] is True


def test_none_option_leaves_rates_at_zero():
    obs, _, _ = getMonitorData(_Model(), _schedule(1), START,
                               Rates=None, BHP=None)
    assert obs[0]['wellSol'][0]['qWs'] == 0.0
    assert obs[0]['wellSol'][0]['bhp'] == 0.0


def test_profile_survey_refines_the_schedule(tmp_path):
    """A survey date between report steps becomes its own step boundary."""
    path = tmp_path / 'prof.csv'
    # Steps land on 2020-01-31 and 2020-03-01; survey on 2020-02-15.
    path.write_text('井号,日期,解释井段,绝对产水量\n'
                    'P1,2020-02-15,0 - 10,3.0\n'
                    'P1,2020-02-15,10-20,1.0\n', encoding='utf-8')
    obs, sched, _ = getMonitorData(_Model(), _schedule(2), START,
                                   Profile=str(path))
    assert len(obs) == 3                       # one extra boundary
    assert sched['step']['val'].size == 3
    assert sched['step']['val'].sum() == pytest.approx(60.0 * DAY)


def test_profile_survey_splits_the_rate_over_perforations(tmp_path):
    path = tmp_path / 'prof.csv'
    path.write_text('井号,日期,解释井段,绝对产水量\n'
                    'P1,2020-01-31,0 - 10,3.0\n'
                    'P1,2020-01-31,10-20,1.0\n', encoding='utf-8')
    obs, _, _ = getMonitorData(_Model(), _schedule(1), START,
                               Profile=str(path))
    cqs = obs[0]['wellSol'][0]['cqs'][:, 0]
    assert cqs.sum() == pytest.approx(-1.0)    # the well's own qWs
    assert cqs[0] == pytest.approx(-0.75)


def test_saturation_survey_reads_its_own_option(tmp_path):
    """The MATLAB splits opt.Profile in the Saturation block."""
    path = tmp_path / 'sat.csv'
    path.write_text('井号,日期,顶深,底深,含水\n'
                    'P1,2020-01-31,0,10,40\n'
                    'P1,2020-01-31,10,20,60\n', encoding='utf-8')
    obs, _, _ = getMonitorData(_Model(), _schedule(1), START,
                               Saturation=str(path))
    sw = obs[0]['wellSol'][0]['sw']
    # Given in percent, so rescaled; cell 0 spans 0-10.
    assert sw[0] == pytest.approx(0.4)
    assert sw[1] == pytest.approx(0.6)


def test_survey_for_an_unknown_well_is_dropped(tmp_path):
    path = tmp_path / 'prof.csv'
    path.write_text('井号,日期,解释井段,绝对产水量\n'
                    'GHOST,2020-01-31,0-10,3.0\n', encoding='utf-8')
    with pytest.warns(RuntimeWarning, match='no schedule data'):
        obs, _, _ = getMonitorData(_Model(), _schedule(1), START,
                                   Profile=str(path))
    assert np.allclose(obs[0]['wellSol'][0]['cqs'], 0.0)


def test_tracer_survey_records_breakthrough_at_the_right_step(tmp_path):
    path = tmp_path / 'tracer.txt'
    path.write_text(
        '注入井号 I1\n'
        '注入层位 0-20\n'
        '注剂时间 2020-01-31\n'
        '示踪剂类型 T1\n'
        '示踪剂用量 500\n'
        '示踪剂观测\n'
        '日期井号 P1\n'
        '2020-03-01 0.42\n'
        '/\n', encoding='utf-8')
    obs, sched, tracers = getMonitorData(_Model(), _schedule(2), START,
                                         Tracer=str(path))
    assert tracers == ['T1']
    # Breakthrough lands on the step whose date matches, not the first.
    values = [step['wellSol'][0]['tracer'][0] for step in obs]
    assert values[-1] == pytest.approx(0.42)
    assert all(v == 0.0 for v in values[:-1])
    assert 'src' in sched['control'][0]


def test_tracer_source_is_spread_over_the_injection_interval(tmp_path):
    path = tmp_path / 'tracer.txt'
    path.write_text(
        '注入井号 P1\n'          # two perforations, 0-10 and 10-20
        '注入层位 0-20\n'
        '注剂时间 2020-01-31\n'
        '示踪剂类型 T1\n'
        '示踪剂用量 500\n'
        '示踪剂观测\n'
        '日期井号 P1\n'
        '2020-01-31 0.1\n'
        '/\n', encoding='utf-8')
    _, sched, _ = getMonitorData(_Model(), _schedule(1), START,
                                 Tracer=str(path))
    src = sched['control'][0]['src']
    assert src['rate'].size == 2
    assert np.all(src['rate'] > 0)
    assert np.allclose(src['sat'][0], [1.0, 0.0])


def test_unsupported_phase_is_reported(tmp_path):
    class _Bad(_Model):
        @staticmethod
        def getPhaseNames():
            return ['X']

    path = tmp_path / 'prof.csv'
    path.write_text('井号,日期,解释井段,绝对产水量\n'
                    'P1,2020-01-31,0-10,3.0\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Unsupported phase'):
        getMonitorData(_Bad(), _schedule(1), START, Profile=str(path))
