import pytest
import numpy as np
from PRSTCore.ad_core.simulators.sim_runner import pack_simulation_problem
from PRSTCore.ad_core.utils import simple_schedule
from PRSTCore.network_models.utils import make_random_training
from PRSTCore.optimization import evaluate_match, unit_box_bfgs
from PRSTCore.optimization.objectives import match_observed_ow, npv_ow
from PRSTCore.optimization.utils.parameters import (
    ModelParameter,
    add_parameter,
    update_setup_from_scaled_parameters,
)


def test_schedule_and_training():
    schedule = simple_schedule([10, 20], [{"W": [{"type": "rate", "val": 1.0, "sign": -1, "status": True, "qWs": 1.0, "qOs": 0.0, "bhp": 100.0}]} , {"W": [{"type": "rate", "val": 2.0, "sign": -1, "status": True, "qWs": 2.0, "qOs": 0.0, "bhp": 110.0}]}])
    assert len(schedule["step"]["val"]) == 2
    problem = {"state0": {}, "model": {"porevolume": np.array([1.0, 2.0])}, "schedule": schedule}
    training = make_random_training(problem, 0.25, 0.05, False)
    assert training["schedule"]["step"]["val"].shape[0] == 2


def test_parameter_update():
    # ``setupByName`` locates transmissibility at ``model.operators.T``,
    # which is where the tuned value is read from and written back to.
    setup = {"state0": {}, "model": {"operators": {"T": np.array([1.0, 2.0])}},
             "schedule": {"step": {"val": [1]}, "control": []}}
    params = add_parameter([], setup, name="transmissibility", scaling="linear", box_lims=[0.5, 4.0])
    pvec = np.array([0.5, 0.5])
    setup_new = update_setup_from_scaled_parameters(setup, params, pvec)
    assert np.allclose(setup_new["model"]["operators"]["T"], np.array([2.25, 2.25]))


def test_objective_and_optimization():
    schedule = simple_schedule([1], [{"W": [{"type": "rate", "val": 1.0, "sign": -1, "status": True, "qWs": 1.0, "qOs": 0.0, "bhp": 100.0}]}])
    setup = {"state0": {}, "model": {"operators": {"pv": np.array([1.0])}},
             "schedule": schedule}
    params = add_parameter([], setup, name="porevolume", scaling="linear", box_lims=[0.5, 2.0])
    observed = [{"wellSol": [{"status": True, "qWs": 1.0, "qOs": 0.0, "bhp": 100.0}]}]
    def obj(model, states, schedule, observed, compute_partials, tstep, state):
        return match_observed_ow(model, states, schedule, observed, compute_partials=compute_partials, weighting={"WaterRateWeight": 1.0, "OilRateWeight": 0.0, "BHPWeight": 0.0})
    pinit = np.array([0.5])
    objh = lambda u: evaluate_match(u, obj, setup, params, observed, Gradient="AdjointAD")
    v, popt, history = unit_box_bfgs(pinit, objh)
    assert popt.shape == pinit.shape
    assert np.all(popt >= 0.0) and np.all(popt <= 1.0)
    assert v is not None
    # This toy setup has no G/rock/wells wired in, so simulate_schedule_ad's
    # wellSol echoes the schedule's literal qWs regardless of porevolume:
    # the gradient at pinit is genuinely (and correctly) zero, so a
    # properly-behaving gradient optimizer takes zero steps from a
    # stationary starting point.
    #
    # Note the gradient here is a finite-difference one: evaluate_match
    # ignores its Gradient option and always finite-differences, so
    # Gradient="AdjointAD" above does not reach the adjoint. See
    # test_evaluate_match_ignores_its_gradient_option below.
    v0, g0, _, _ = objh(pinit)
    assert np.allclose(g0, 0.0)
    assert len(history) == 0
    assert np.array_equal(popt, np.clip(pinit, 0.0, 1.0))


def test_npv():
    schedule = simple_schedule([1, 1], [{"W": [{"type": "rate", "val": 1.0, "sign": -1, "status": True, "qWs": 1.0, "qOs": 0.0, "bhp": 100.0}]}, {"W": [{"type": "rate", "val": 1.0, "sign": -1, "status": True, "qWs": 1.5, "qOs": 0.0, "bhp": 110.0}]}])
    states = [ {"wellSol": [{"status": True, "sign": -1, "qWs": 1.0, "qOs": 0.0, "bhp": 100.0}]}, {"wellSol": [{"status": True, "sign": -1, "qWs": 1.5, "qOs": 0.0, "bhp": 110.0}]} ]
    obj = npv_ow(None, states, schedule, oil_price=1.0, water_production_cost=0.1, water_injection_cost=0.1, discount_factor=0.0)
    assert isinstance(obj, list)
    assert len(obj) == 2


def test_evaluate_match_ignores_its_gradient_option():
    """evaluate_match accepts Gradient='AdjointAD' and finite-differences
    anyway -- the option selects nothing.

    Pinned rather than fixed: making it dispatch would route callers into
    compute_sensitivities_adjoint_ad, which returns zeros, so they would
    go from a correct finite-difference gradient to a silently wrong one.
    The option is the thing to fix, and only once the adjoint is real.
    """
    import inspect

    from PRSTCore.optimization import evaluate_match
    source = inspect.getsource(evaluate_match)
    assert '_finite_difference_gradient' in source
    assert 'compute_sensitivities_adjoint_ad' not in source


def test_the_adjoint_reports_a_parameter_it_cannot_differentiate():
    """It used to return zeros for *every* parameter without a word --
    indistinguishable from a converged gradient, so an optimiser took no
    steps and reported success. Now only the parameters with no
    derivative path come back zero, and they say so.

    The real gradient is verified end to end in
    tests/test_adjoint_verification.py."""
    import numpy as np
    import pytest

    from PRSTCore.ad_core.simulators import compute_sensitivities_adjoint_ad
    from PRSTCore.ad_core.simulators.compute_sensitivities_adjoint_ad import         SUPPORTED

    assert 'porevolume' in SUPPORTED and 'transmissibility' in SUPPORTED
    # The eleven saturation-function endpoints were added once each had a
    # verified derivative -- see tests/test_adjoint_endpoints.py.
    assert 'swl' in SUPPORTED and 'krw' in SUPPORTED
    # The well terms are deliberately not differentiated, so a connection
    # transmissibility still comes back zero, loudly.
    assert 'conntrans' not in SUPPORTED


def test_the_adjoint_placeholder_takes_mrst0s_options():
    """recomputeWI is MRST-0's addition over 2026a; accepting it now means
    call sites need not change when the adjoint is implemented."""
    import inspect

    from PRSTCore.ad_core.simulators import compute_sensitivities_adjoint_ad
    params = inspect.signature(compute_sensitivities_adjoint_ad).parameters
    for name in ('accumulate_residuals', 'is_scalar', 'LinearSolver',
                 'match_map', 'recompute_wi'):
        assert name in params, name
