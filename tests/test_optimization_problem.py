"""Self-consistency tests for OptimizationProblem
(PRSTCore.optimization.optim.optimization_problem).

No MRST-side reference implementation exists to validate against: the real
OptimizationProblem.m depends on disk-backed ResultHandler caching and
reservoir-simulator solverFun machinery that this port deliberately
excludes (see the module docstring's scope note). Validation here instead
uses closed-form optimization problems (ensemble-mean and ensemble-vertcat
quadratics, whose optimizers are exactly the centroid of the realizations'
centers) so the aggregation and optimizer-dispatch logic can be checked
against an independently-derivable answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from PRSTCore.optimization.optim.optimization_problem import OptimizationProblem

_CENTERS = {
    "r1": np.array([0.3, 0.6]),
    "r2": np.array([0.5, 0.4]),
    "r3": np.array([0.7, 0.5]),
}
_CENTROID = np.mean(list(_CENTERS.values()), axis=0)


def test_scale_unscale_round_trip_with_explicit_bounds():
    bounds = np.array([[10.0, 20.0], [-5.0, 5.0]])
    p = OptimizationProblem([], solver_fun=None, bounds=bounds)
    u = np.array([12.5, 0.0])
    us = p.scale_variables(u)
    assert np.allclose(us, [0.25, 0.5])
    assert np.allclose(p.unscale_variables(us), u)


def test_scale_unscale_identity_without_bounds():
    p = OptimizationProblem([], solver_fun=None)
    u = np.array([0.2, 0.8])
    assert np.allclose(p.scale_variables(u), u)
    assert np.allclose(p.unscale_variables(u), u)


def _quad_solver(sample, u):
    c = _CENTERS[sample]
    diff = u - c
    value = -float(diff @ diff)  # maximize -> peak at u == c
    grad = -2.0 * diff
    return value, grad


def test_ensemble_mean_bfgs_optimum_is_the_centroid_of_realization_centers():
    p = OptimizationProblem(list(_CENTERS.keys()), _quad_solver)
    u, info = p.optimize(np.array([0.1, 0.1]), optimizer="bfgs", maximize=True, grad_tol=1e-8)
    assert np.allclose(u, _CENTROID, atol=1e-3)
    # At the centroid, sum((u-c_i)^2) is minimized -> mean(-(u-c_i)^2) equals
    # -mean(||centroid-c_i||^2), the (negative) within-ensemble variance.
    expected_value = -np.mean([(_CENTROID - c) @ (_CENTROID - c) for c in _CENTERS.values()])
    assert np.isclose(info["value"], expected_value, atol=1e-3)


def _residual_solver(sample, u):
    c = _CENTERS[sample]
    return u - c, np.eye(u.size)


def test_ensemble_vertcat_lm_optimum_is_the_centroid_of_realization_centers():
    p = OptimizationProblem(list(_CENTERS.keys()), _residual_solver, objective_stat_fun="vertcat")
    u, info = p.optimize(np.array([0.1, 0.1]), optimizer="lm", verbose=False)
    assert np.allclose(u, _CENTROID, atol=1e-3)
    # LM minimizes sum(v.^2) = sum_i ||u-c_i||^2, minimized at the centroid
    # -- not zero, since the centers themselves aren't coincident.
    expected_value = sum((_CENTROID - c) @ (_CENTROID - c) for c in _CENTERS.values())
    assert np.isclose(info["value"], expected_value, atol=1e-4)


def test_out_of_bounds_initial_guess_raises():
    p = OptimizationProblem(list(_CENTERS.keys()), _quad_solver)
    with pytest.raises(ValueError):
        p.optimize(np.array([1.5, 0.1]), optimizer="bfgs")


def test_unsupported_optimizer_raises():
    p = OptimizationProblem(list(_CENTERS.keys()), _quad_solver)
    with pytest.raises(ValueError):
        p.optimize(np.array([0.1, 0.1]), optimizer="not-a-real-optimizer")
