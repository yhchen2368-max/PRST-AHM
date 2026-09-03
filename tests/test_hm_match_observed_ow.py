"""``matchObservedOW`` and the parameter sizing behind it.

Two things a history match cannot do without, both of which were quietly
wrong.

``matchObservedOW`` is the objective ``hm/test/HistoryMatching.m`` uses:
a weighted least-squares over water rate, oil rate and bottom-hole
pressure. MRST keeps it separate from ``matchObservedOWG`` because they
are separate objectives -- the gas term is absent, not weighted to zero.

The parameter sizing is ``setupDefaults``' ``nParam = numel(v)`` where
``v = getParameterValue(p, setup, false)`` -- read through the
parameter's *location*. PRSTCore read it by name instead, through a
switch that knew three parameters, so everything else came back as a
single zero: one tunable number where the field has 54080 cells, and an
optimiser that converges after moving almost nothing.
"""

import copy
import os

import numpy as np
import pytest

from PRSTCore.hm.utils.evaluate.matchObservedOW import matchObservedOW

DECK = 'examples/HM/QIEDIE.DATA'
NC = 52 * 52 * 20


def _sols(rates):
    """One step's well solutions from ``[(qWs, qOs, bhp), ...]``."""
    return [{'name': 'W%d' % i, 'qWs': w, 'qOs': o, 'bhp': p,
             'status': True, 'sign': -1.0}
            for i, (w, o, p) in enumerate(rates)]


@pytest.fixture
def synthetic():
    observed = [{'wellSol': _sols([(1.0, 10.0, 200e5), (2.0, 20.0, 210e5)])},
                {'wellSol': _sols([(1.5, 11.0, 201e5), (2.5, 21.0, 211e5)])}]
    schedule = {'step': {'val': np.array([30.0, 30.0]) * 86400.0,
                         'control': np.zeros(2, dtype=int)},
                'control': [{'W': []}]}
    return observed, schedule


def _total(states, observed, schedule, **weights):
    entries = matchObservedOW(None, states, schedule, observed, **weights)
    return float(np.sum([np.sum(np.asarray(e, dtype=float))
                         for e in entries]))


# ------------------------------------------------------- the objective --

def test_a_perfect_match_is_exactly_zero(synthetic):
    observed, schedule = synthetic
    states = [{'wellSol': [dict(w) for w in o['wellSol']]}
              for o in observed]
    assert _total(states, observed, schedule, WaterRateWeight=1.0,
                  OilRateWeight=1.0, BHPWeight=0.0) == 0.0


def test_the_misfit_grows_with_the_square_of_the_error(synthetic):
    """Least squares: five times the error is twenty-five times the
    misfit. A linear growth would mean the terms are not squared."""
    observed, schedule = synthetic
    out = {}
    for pct in (0.01, 0.05, 0.10):
        states = copy.deepcopy(
            [{'wellSol': [dict(w) for w in o['wellSol']]}
             for o in observed])
        for step in states:
            for well in step['wellSol']:
                well['qOs'] *= (1.0 + pct)
        out[pct] = _total(states, observed, schedule, WaterRateWeight=1.0,
                          OilRateWeight=1.0, BHPWeight=0.0)
    assert out[0.05] == pytest.approx(25 * out[0.01], rel=1e-9)
    assert out[0.10] == pytest.approx(100 * out[0.01], rel=1e-9)


def test_the_three_terms_are_water_oil_and_pressure(synthetic):
    """No gas term: this is the OW objective, not OWG with wg = 0."""
    observed, schedule = synthetic
    states = [{'wellSol': [dict(w) for w in o['wellSol']]}
              for o in observed]
    for step in states:
        for well in step['wellSol']:
            well['qGs'] = 1e6          # ignored entirely
    assert _total(states, observed, schedule, WaterRateWeight=1.0,
                  OilRateWeight=1.0, BHPWeight=1.0) == 0.0


def test_water_and_oil_share_one_default_weight(synthetic):
    """``getWeights``: both default to ``1/sum(|qWs| + |qOs|)``, so the
    two rates trade off against each other rather than each being
    normalised to its own total."""
    observed, _schedule = synthetic
    from PRSTCore.hm.utils.evaluate.matchObservedOWG import _getWeights
    qw = np.array([1.0, 2.0])
    qo = np.array([10.0, 20.0])
    ww, wo, _wg, _wp = _getWeights(qw, qo, np.zeros(2),
                                   np.array([200e5, 210e5]),
                                   None, None, 0.0, None)
    assert ww == wo == pytest.approx(1.0 / 33.0)


def test_a_zero_pressure_range_gives_a_zero_pressure_weight():
    """MRST: ``if dp == 0, wp = 0``. Dividing instead would give inf."""
    from PRSTCore.hm.utils.evaluate.matchObservedOWG import _getWeights
    _ww, _wo, _wg, wp = _getWeights(np.array([1.0]), np.array([1.0]),
                                    np.zeros(1), np.array([200e5, 200e5]),
                                    None, None, 0.0, None)
    assert wp == 0.0


# --------------------------------------------------- parameter sizing --

@pytest.fixture(scope='module')
def qiedie_setup():
    if not os.path.exists(DECK):
        pytest.skip('QIEDIE.DATA not present')
    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import \
        init_eclipse_problem_ad
    from PRSTCore.ad_props.impose_relperm_scaling import \
        impose_relperm_scaling
    from PRSTCore.hm.utils.getRelpermScalingPoints import (
        as_dict, getRelpermScalingPoints)

    state0, model, schedule, _ = init_eclipse_problem_ad(DECK)
    scaling = as_dict(getRelpermScalingPoints(model))
    md = impose_relperm_scaling(
        {'G': model.G, 'fluid': model.fluid, 'rock': model.rock}, 2,
        **scaling)
    model.rock = md['rock']
    return {'model': model, 'schedule': schedule, 'state0': state0}


@pytest.mark.parametrize('name,scaling', [
    ('porevolume', 'linear'), ('permx', 'log'), ('permy', 'log'),
    ('permz', 'log'), ('swcr', 'linear'), ('sowcr', 'linear'),
    ('krw', 'linear'), ('kro', 'linear')])
def test_a_parameter_covers_the_whole_field(qiedie_setup, name, scaling):
    """One tunable number per cell. A parameter that sizes itself at 1 is
    not an error anywhere -- the optimiser tunes that single number,
    descends, and reports success."""
    from PRSTCore.optimization.utils.parameters import add_parameter
    params = add_parameter([], qiedie_setup, name=name, scaling=scaling,
                           relative_limits=[0.5, 2.0],
                           uniform_limits=False)
    assert params[-1].n_param == NC, name


def test_the_endpoints_point_at_the_relperm_scalers(qiedie_setup):
    """``getScalerMap``'s phase and column. These only resolve once
    ``imposeRelpermScaling`` has built ``rock.krscale`` -- the two halves
    were ported but never joined."""
    from PRSTCore.optimization.utils.parameters import ModelParameter
    expected = {'swcr': ('w', 1), 'sowcr': ('ow', 1),
                'krw': ('w', 3), 'kro': ('ow', 3)}
    for name, (phase, column) in expected.items():
        p = ModelParameter(name, n_param=1, setup=qiedie_setup)
        assert p.location[:4] == ('rock', 'krscale', 'drainage', phase)
        assert p.location[4][1] == column


def test_kro_writes_both_oil_curves(qiedie_setup):
    """ECLIPSE's KRO sets the oil maximum on the water-oil *and* the
    gas-oil curve; writing one would leave half the derivative behind."""
    from PRSTCore.optimization.utils.parameters import ModelParameter
    p = ModelParameter('kro', n_param=1, setup=qiedie_setup)
    assert len(p.extra_locations) == 1
    assert p.extra_locations[0][3] == 'og'
