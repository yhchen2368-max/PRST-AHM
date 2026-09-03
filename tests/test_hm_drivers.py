"""Tests for the hm test drivers and the pieces they needed.

The drivers themselves are templates over an external simulator, so what
is checked here is that they are importable, that their configuration
surface is what MRST's is, and that the helpers they exposed as gaps
(matchObservedOWG, init_state, add_bounding_box_fields, uniform_limits)
behave.
"""

import numpy as np
import pytest

from PRSTCore.gridprocessing.add_bounding_box_fields import \
    add_bounding_box_fields
from PRSTCore.hm.utils.evaluate.matchObservedOWG import matchObservedOWG
from PRSTCore.solvers.incomp.init_state import init_state

DAY = 86400.0


# ------------------------------------------------------ matchObservedOWG --

def _sol(qWs, qOs, qGs, bhp, status=True):
    return {'name': 'W', 'qWs': qWs, 'qOs': qOs, 'qGs': qGs, 'bhp': bhp,
            'status': status, 'sign': -1.0}


def _case(sim, obs):
    schedule = {'step': {'val': np.array([DAY]), 'control': np.array([1])},
                'control': [{'W': [{'cells': [0]}]}]}
    return schedule, [{'wellSol': [sim]}], [{'wellSol': [obs]}]


def test_owg_scores_all_three_phases():
    schedule, states, observed = _case(_sol(-3, -4, -5, 200e5),
                                       _sol(-1, -1, -1, 200e5))
    obj = matchObservedOWG(None, states, schedule, observed,
                           WaterRateWeight=1.0, OilRateWeight=1.0,
                           GasRateWeight=1.0, BHPWeight=0.0)
    assert obj[0] == pytest.approx(4 + 9 + 16)


def test_owg_is_zero_for_a_perfect_match():
    s = _sol(-3, -4, -5, 200e5)
    schedule, states, observed = _case(s, dict(s))
    obj = matchObservedOWG(None, states, schedule, observed,
                           WaterRateWeight=1.0, OilRateWeight=1.0,
                           GasRateWeight=1.0, BHPWeight=1.0)
    assert obj[0] == pytest.approx(0.0)


def test_owg_zero_weight_switches_a_phase_off():
    schedule, states, observed = _case(_sol(-3, -4, -5, 200e5),
                                       _sol(-1, -1, -1, 200e5))
    obj = matchObservedOWG(None, states, schedule, observed,
                           WaterRateWeight=0.0, OilRateWeight=1.0,
                           GasRateWeight=0.0, BHPWeight=0.0)
    assert obj[0] == pytest.approx(9.0)


def test_owg_water_and_oil_share_one_normaliser():
    """MRST sets ww = wo = 1/sum(|qWs| + |qOs|), not a per-phase
    reciprocal -- so the two liquid terms stay commensurate."""
    schedule, states, observed = _case(_sol(-2, -20, 0.0, 200e5),
                                       _sol(-1, -10, 0.0, 200e5))
    obj = matchObservedOWG(None, states, schedule, observed, BHPWeight=0.0)
    rw = 1.0 + 10.0                       # |qWs_obs| + |qOs_obs|
    assert obj[0] == pytest.approx((1.0 / rw) ** 2 + (10.0 / rw) ** 2)


def test_owg_gas_gets_its_own_normaliser():
    schedule, states, observed = _case(_sol(0.0, 0.0, -200, 200e5),
                                       _sol(0.0, 0.0, -100, 200e5))
    obj = matchObservedOWG(None, states, schedule, observed, BHPWeight=0.0)
    assert obj[0] == pytest.approx(1.0)   # (100/100)^2


def test_owg_gives_no_weight_to_a_phase_that_was_not_measured():
    """A zero observed rate must not become a division by zero."""
    schedule, states, observed = _case(_sol(-3, -4, 0.0, 200e5),
                                       _sol(-1, -1, 0.0, 200e5))
    obj = matchObservedOWG(None, states, schedule, observed, BHPWeight=0.0)
    assert np.isfinite(obj[0])


def test_owg_pressure_weight_defaults_to_the_observed_spread():
    schedule = {'step': {'val': np.array([DAY]), 'control': np.array([1])},
                'control': [{'W': [{'cells': [0]}, {'cells': [1]}]}]}
    states = [{'wellSol': [_sol(0, 0, 0, 250e5), _sol(0, 0, 0, 200e5)]}]
    observed = [{'wellSol': [_sol(0, 0, 0, 300e5), _sol(0, 0, 0, 200e5)]}]
    obj = matchObservedOWG(None, states, schedule, observed)
    dp = 300e5 - 200e5
    assert obj[0] == pytest.approx((50e5 / dp) ** 2 / 2.0)


def test_owg_match_map_is_always_empty():
    """MRST declares matchMap but never assigns it."""
    schedule, states, observed = _case(_sol(-3, -4, -5, 200e5),
                                       _sol(-1, -1, -1, 200e5))
    _, match_map = matchObservedOWG(None, states, schedule, observed,
                                    WaterRateWeight=1.0, OilRateWeight=1.0,
                                    GasRateWeight=1.0, BHPWeight=0.0,
                                    return_match_map=True)
    assert match_map is None


def test_owg_weights_each_step_by_its_share_of_time():
    schedule = {'step': {'val': np.array([DAY, 3 * DAY]),
                         'control': np.array([1, 1])},
                'control': [{'W': [{'cells': [0]}]}]}
    sim = {'wellSol': [_sol(-3, 0, 0, 200e5)]}
    obs = {'wellSol': [_sol(-1, 0, 0, 200e5)]}
    obj = matchObservedOWG(None, [sim, sim], schedule, [obs, obs],
                           WaterRateWeight=1.0, OilRateWeight=0.0,
                           GasRateWeight=0.0, BHPWeight=0.0)
    assert obj[1] == pytest.approx(3.0 * obj[0])


def test_owg_can_match_producers_only():
    schedule = {'step': {'val': np.array([DAY]), 'control': np.array([1])},
                'control': [{'W': [{'cells': [0]}, {'cells': [1]}]}]}
    inj = _sol(5.0, 0.0, 0.0, 300e5)
    inj['sign'] = 1.0
    states = [{'wellSol': [_sol(-3, 0, 0, 200e5), dict(inj, qWs=9.0)]}]
    observed = [{'wellSol': [_sol(-1, 0, 0, 200e5), inj]}]
    obj = matchObservedOWG(None, states, schedule, observed,
                           WaterRateWeight=1.0, OilRateWeight=0.0,
                           GasRateWeight=0.0, BHPWeight=0.0,
                           matchOnlyProducers=True)
    assert obj[0] == pytest.approx(4.0)      # injector excluded


def test_owg_unsummed_returns_one_entry_per_term():
    schedule, states, observed = _case(_sol(-3, -4, -5, 200e5),
                                       _sol(-1, -1, -1, 200e5))
    obj = matchObservedOWG(None, states, schedule, observed,
                           WaterRateWeight=1.0, OilRateWeight=1.0,
                           GasRateWeight=1.0, BHPWeight=1.0,
                           mismatchSum=False)
    assert np.asarray(obj[0]).size == 4       # water, oil, gas, bhp


# ------------------------------------------------------------ init_state --

def _G():
    return {'cells': {'num': 4}, 'faces': {'num': 12}}


def test_init_state_sets_a_uniform_pressure():
    state = init_state(_G(), [], 200e5)
    assert np.allclose(state['pressure'], 200e5)


def test_init_state_adds_a_well_solution():
    state = init_state(_G(), [{'cells': [0, 1], 'compi': [1.0, 0.0]}], 200e5)
    assert len(state['wellSol']) == 1
    assert state['wellSol'][0]['flux'].size == 2


def test_init_state_without_wells_has_no_well_solution():
    assert 'wellSol' not in init_state(_G(), [], 200e5)


def test_init_state_rejects_a_phase_count_mismatch():
    W = [{'cells': [0], 'compi': [1.0, 0.0]}]
    with pytest.raises(AssertionError, match='does not match'):
        init_state(_G(), W, 200e5, [[0.0, 0.5, 0.5]])


def test_init_state_reports_inconsistent_well_compositions():
    with pytest.raises(ValueError, match='inconsistently specified'):
        init_state(_G(), [{'cells': [0]}], 200e5, [[0.0, 1.0]])


# ------------------------------------------------- add_bounding_box_fields --

def _grid():
    """A 2x3 rectangle. Its four edges give faces with distinguishable
    spans, so a bbox that mixed up the axes would show."""
    return {
        'griddim': 2,
        'nodes': {'coords': np.array([[0.0, 0.0], [2.0, 0.0],
                                      [2.0, 3.0], [0.0, 3.0]])},
        'faces': {'num': 4,
                  'nodePos': np.array([0, 2, 4, 6, 8]),
                  # bottom, right, top, left
                  'nodes': np.array([0, 1, 1, 2, 2, 3, 3, 0])},
    }


def test_face_bbox_is_the_span_of_its_nodes_along_each_axis():
    bbox = add_bounding_box_fields(_grid())['faces']['bbox']
    assert np.allclose(bbox[0], [2.0, 0.0])     # bottom: 2 wide, no height
    assert np.allclose(bbox[1], [0.0, 3.0])     # right: no width, 3 tall
    assert np.allclose(bbox[2], [2.0, 0.0])     # top
    assert np.allclose(bbox[3], [0.0, 3.0])     # left


def test_faces_are_added_by_default():
    assert 'bbox' in add_bounding_box_fields(_grid())['faces']


def test_faces_can_be_switched_off():
    assert 'bbox' not in add_bounding_box_fields(_grid(), faces=False)['faces']


# ---------------------------------------------------------- uniform_limits --

class _Setup(dict):
    pass


def _param_setup(values):
    class _Model:
        pass
    model = _Model()
    model._value = np.asarray(values, dtype=float)
    return {'model': model}


def test_uniform_limits_gives_every_entry_the_same_box(monkeypatch):
    from PRSTCore.optimization.utils import parameters as P
    monkeypatch.setattr(P, '_get_model_parameter_value',
                        lambda model, name: model._value)
    setup = _param_setup([1.0, 10.0])
    params = P.add_parameter([], setup, name='porevolume',
                             relative_limits=[0.5, 2.0], uniform_limits=True)
    box = params[0].box_lims
    assert np.allclose(box[0], box[1])
    assert box[0, 0] == pytest.approx(0.5)      # min * 0.5
    assert box[0, 1] == pytest.approx(20.0)     # max * 2.0


def test_per_entry_limits_scale_each_value_separately(monkeypatch):
    from PRSTCore.optimization.utils import parameters as P
    monkeypatch.setattr(P, '_get_model_parameter_value',
                        lambda model, name: model._value)
    setup = _param_setup([1.0, 10.0])
    params = P.add_parameter([], setup, name='porevolume',
                             relative_limits=[0.5, 2.0], uniform_limits=False)
    box = params[0].box_lims
    assert np.allclose(box[0], [0.5, 2.0])
    assert np.allclose(box[1], [5.0, 20.0])


def test_uniform_limits_is_the_default(monkeypatch):
    """MRST's ModelParameter defaults uniformLimits to true."""
    from PRSTCore.optimization.utils import parameters as P
    monkeypatch.setattr(P, '_get_model_parameter_value',
                        lambda model, name: model._value)
    setup = _param_setup([1.0, 10.0])
    default = P.add_parameter([], setup, name='porevolume',
                              relative_limits=[0.5, 2.0])[0].box_lims
    uniform = P.add_parameter([], setup, name='porevolume',
                              relative_limits=[0.5, 2.0],
                              uniform_limits=True)[0].box_lims
    assert np.allclose(default, uniform)


def test_a_zero_valued_entry_gets_a_unit_box(monkeypatch):
    """Relative limits scale to a degenerate [0, 0] on a zero entry, so
    MRST-0's setupDefaults overrides just those rows with [0, 1] and warns
    -- one of its `% edited by zhang` changes.  Rows that are non-zero keep
    the box the relative limits gave them."""
    from PRSTCore.optimization.utils import parameters as P
    monkeypatch.setattr(P, '_get_model_parameter_value',
                        lambda model, name: model._value)
    setup = _param_setup([0.0, 10.0])
    box = P.add_parameter([], setup, name='porevolume',
                          relative_limits=[0.5, 2.0],
                          uniform_limits=False)[0].box_lims
    assert np.allclose(box[0], [0.0, 1.0])
    assert np.allclose(box[1], [5.0, 20.0])


def test_saturation_parameters_are_bounded_by_zero_and_one(monkeypatch):
    from PRSTCore.optimization.utils import parameters as P
    monkeypatch.setattr(P, '_get_model_parameter_value',
                        lambda model, name: model._value)
    setup = _param_setup([0.3, 0.4])
    box = P.add_parameter([], setup, name='sw',
                          relative_limits=[0.5, 2.0])[0].box_lims
    assert np.allclose(box[:, 0], 0.0) and np.allclose(box[:, 1], 1.0)


# --------------------------------------------------------------- drivers --

@pytest.mark.parametrize('module', [
    'PRSTCore.hm.test.HistoryMatching',
    'PRSTCore.hm.test.CGNetTraining',
    'PRSTCore.hm.test.WellPlacementOptimization',
    'PRSTCore.hm.test.trainPolymerFlood',
    'PRSTCore.hm.test.trainSurfactantFlood',
])
def test_driver_imports(module):
    __import__(module)


@pytest.mark.parametrize('module, func, bad', [
    ('PRSTCore.hm.test.trainPolymerFlood', 'trainPolymerFlood', 'rock+water'),
    ('PRSTCore.hm.test.trainSurfactantFlood', 'trainSurfactantFlood',
     'rock+polymer'),
])
def test_train_drivers_reject_an_unknown_parameter_type(module, func, bad):
    mod = __import__(module, fromlist=[func])
    with pytest.raises(ValueError, match='ParameterType'):
        getattr(mod, func)(None, None, None, ParameterType=bad)


def test_surfactant_driver_records_what_mrst_left_disabled():
    from PRSTCore.hm.test.trainSurfactantFlood import \
        DISABLED_SURFACTANT_PARAMETERS
    assert 'adcsu' in DISABLED_SURFACTANT_PARAMETERS
    assert len(DISABLED_SURFACTANT_PARAMETERS) == 6


# ------------------------------------------- imposeRelpermScaling (MRST-0) --

def _model(nc=4):
    return {'G': {'cells': {'num': nc}}, 'fluid': {'krPts': {'w': [0, 0, 1, 1]}},
            'rock': {}}


def test_surfactant_endpoints_are_accepted():
    """trainSurfactantFlood tunes these; 2026a's shorter valid list would
    drop all six without a word."""
    from PRSTCore.ad_props.impose_relperm_scaling import VALID_SCALERS
    for name in ('SSWL', 'SSWCR', 'SSWU', 'SSOWCR', 'SKRW', 'SKRO'):
        assert name in VALID_SCALERS, name


def test_imbibition_and_residual_endpoints_are_accepted():
    from PRSTCore.ad_props.impose_relperm_scaling import VALID_SCALERS
    for name in ('ISWL', 'IKRW', 'KRWR', 'KRORW', 'KRORG', 'KRGR'):
        assert name in VALID_SCALERS, name


def test_surfactant_endpoints_land_in_the_miscible_table():
    from PRSTCore.ad_props.impose_relperm_scaling import impose_relperm_scaling
    out = impose_relperm_scaling(_model(), SSWL=0.2, SKRW=0.5)
    miscible = out['rock']['krscale']['miscible']['w']
    assert miscible[0, 0] == pytest.approx(0.2)     # L column
    assert miscible[0, 3] == pytest.approx(0.5)     # KM column


def test_imbibition_endpoints_land_in_their_own_table():
    from PRSTCore.ad_props.impose_relperm_scaling import impose_relperm_scaling
    out = impose_relperm_scaling(_model(), ISWL=0.3)
    assert out['rock']['krscale']['imbibition']['w'][0, 0] == pytest.approx(0.3)
    assert 'drainage' in out['rock']['krscale']


def test_scaling_twice_keeps_the_first_call(monkeypatch):
    """MRST-0 merges into an existing krscale; replacing it would discard
    whatever an earlier call set."""
    from PRSTCore.ad_props.impose_relperm_scaling import impose_relperm_scaling
    model = impose_relperm_scaling(_model(), SWL=0.1)
    model = impose_relperm_scaling(model, SWU=0.9)
    drainage = model['rock']['krscale']['drainage']['w']
    assert drainage[0, 0] == pytest.approx(0.1)     # still there
    assert drainage[0, 2] == pytest.approx(0.9)


def test_no_scaling_arguments_is_a_no_op():
    from PRSTCore.ad_props.impose_relperm_scaling import impose_relperm_scaling
    model = {'G': {'cells': {'num': 4}}}      # no fluid at all
    assert impose_relperm_scaling(model) is model


def test_an_unsupported_keyword_warns_rather_than_vanishing():
    from PRSTCore.ad_props.impose_relperm_scaling import impose_relperm_scaling
    with pytest.warns(RuntimeWarning, match='Unsupported'):
        impose_relperm_scaling(_model(), NOSUCH=1.0)


def test_kro_reaches_both_oil_curves():
    """One oil curve serves oil-water and oil-gas, so KRO sets both."""
    from PRSTCore.ad_props.impose_relperm_scaling import impose_relperm_scaling
    drainage = impose_relperm_scaling(_model(), KRO=0.8)['rock']['krscale']['drainage']
    assert drainage['ow'][0, 3] == pytest.approx(0.8)
    assert drainage['og'][0, 3] == pytest.approx(0.8)


def test_a_scalar_is_expanded_to_every_cell():
    from PRSTCore.ad_props.impose_relperm_scaling import impose_relperm_scaling
    out = impose_relperm_scaling(_model(nc=5), SWL=0.15)
    assert np.allclose(out['rock']['krscale']['drainage']['w'][:, 0], 0.15)


def test_a_wrong_length_array_is_rejected():
    from PRSTCore.ad_props.impose_relperm_scaling import impose_relperm_scaling
    with pytest.raises(ValueError, match='do not match grid cells'):
        impose_relperm_scaling(_model(nc=4), SWL=np.zeros(3))


# ------------------------------------------------- driver call signatures --

def test_driver_calls_match_the_real_signatures():
    """Every keyword a driver passes must actually be a parameter.

    Checking that the imports resolve is not enough: the drivers were
    first written with MATLAB's camelCase option names against ports that
    use snake_case, which resolves fine and then fails at call time. This
    walks each call and checks it against the real signature.
    """
    import ast
    import glob
    import importlib
    import inspect
    import os

    bad = []
    for path in sorted(glob.glob('PRSTCore/hm/test/*.py')) + \
            ['PRSTCore/hm/APP/fahm.py', 'PRSTCore/hm/APP/fahm_app.py']:
        if not os.path.exists(path):
            continue
        tree = ast.parse(open(path, encoding='utf-8').read())
        origin = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module \
                    and node.module.startswith('PRSTCore'):
                for alias in node.names:
                    origin[alias.asname or alias.name] = (node.module,
                                                          alias.name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or \
                    not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in origin:
                continue
            module, attr = origin[node.func.id]
            try:
                signature = inspect.signature(
                    getattr(importlib.import_module(module), attr))
            except Exception:
                continue
            params = signature.parameters
            takes_kwargs = any(p.kind == p.VAR_KEYWORD
                               for p in params.values())
            for keyword in node.keywords:
                if keyword.arg and keyword.arg not in params \
                        and not takes_kwargs:
                    bad.append('%s:%d %s(%s=...)'
                               % (os.path.basename(path), node.lineno,
                                  node.func.id, keyword.arg))
    assert not bad, 'calls that would fail at runtime:\n  ' + '\n  '.join(bad)
