"""Python port of the portable core of MRST's ``OptimizationProblem.m``
(mrst-2026a/autodiff/optimization/optim/OptimizationProblem.m): robust/
ensemble optimization by scaling controls to the unit box and aggregating a
per-realization objective (value[, gradient] or ``residual, jacobian``)
across an ensemble before handing it to a box-constrained optimizer.

Scope: this ports the genuinely generic algorithmic core --
``scale_variables``/``unscale_variables`` (linear box scaling,
``scaleVariables``/``unscaleVariables``), ensemble objective aggregation
(``assembleIterationObjective``/``applyStatFun``), and dispatch to
:func:`PRSTCore.optimization.optim.optimize_bound_constrained.optimize_bound_constrained`
(``unitBoxBFGS``'s port) or :func:`PRSTCore.optimization.optim.unit_box_lm.unit_box_lm`
(``unitBoxLM``'s port). NOT ported: ``BaseEnsemble``'s disk-backed
``ResultHandler`` caching, the well-control/trajectory/model-parameter deck
-update machinery (``setupProblem``'s simulator-backed ``solverFun``
variants), background/parallel optimization, and the ``ipopt`` dispatch
path -- all reservoir-simulator plumbing rather than generic optimization
logic. Callers supply their own ``solver_fun(sample, u)`` per realization
directly (playing the role of MRST's simulation-backed ``solverFun``),
decoupling this class entirely from reservoir simulation, so it has no
MRST-side reference implementation to validate against and is self-tested
only (see ``tests/test_optimization_problem.py``).
"""

from __future__ import annotations

import numpy as _np

from .optimize_bound_constrained import optimize_bound_constrained
from .unit_box_lm import unit_box_lm


class OptimizationProblem:
    """Ensemble/robust optimization over a fixed set of realizations."""

    def __init__(self, samples, solver_fun, *, bounds=None, objective_stat_fun=_np.mean):
        """
        Parameters
        ----------
        samples : sequence
            One entry per ensemble realization (opaque to this class,
            passed straight through to ``solver_fun``).
        solver_fun : callable
            ``solver_fun(sample, u) -> value`` (scalar) or ``(value,
            gradient)`` / ``(residual, jacobian)`` per realization.
        bounds : (n, 2) array or None
            Per-variable ``[lower, upper]`` control bounds. ``None`` means
            the identity scaling ``[0, 1]``, matching
            ``getControlVectorLimits``'s no-bounds fallback.
        objective_stat_fun : callable or 'vertcat'
            How per-realization results are aggregated across the
            ensemble: a reduction such as ``numpy.mean`` applied along the
            ensemble axis (matches MRST's default ``@mean``), or the
            string ``'vertcat'`` to stack per-realization residuals/
            Jacobians instead (matches MRST's ``@vertcat`` option, for
            LM/Gauss-Newton robust optimization where the stacked
            residual's sum-of-squares equals the ensemble's total
            mismatch).
        """
        self.samples = list(samples)
        self.solver_fun = solver_fun
        self.bounds = None if bounds is None else _np.asarray(bounds, dtype=float)
        self.objective_stat_fun = objective_stat_fun

    def get_control_vector_limits(self):
        if self.bounds is None:
            return 0.0, 1.0
        return self.bounds[:, 0], self.bounds[:, 1]

    def scale_variables(self, u):
        lower, upper = self.get_control_vector_limits()
        return (_np.asarray(u, dtype=float) - lower) / (upper - lower)

    def unscale_variables(self, us):
        lower, upper = self.get_control_vector_limits()
        return lower + _np.asarray(us, dtype=float) * (upper - lower)

    def _aggregate(self, results):
        if self.objective_stat_fun == "vertcat":
            return _np.concatenate([_np.atleast_1d(_np.asarray(r, dtype=float)) for r in results], axis=0)
        stacked = _np.stack([_np.atleast_1d(_np.asarray(r, dtype=float)) for r in results], axis=-1)
        agg = self.objective_stat_fun(stacked, axis=-1)
        return agg[0] if agg.shape == (1,) else agg

    def get_ensemble_objective(self, us, with_grad=True):
        """Port of ``assembleIterationObjective``: evaluate ``solver_fun``
        for every realization at the unscaled control vector, then
        aggregate with ``objective_stat_fun``."""
        u = self.unscale_variables(us)
        values, grads = [], []
        for sample in self.samples:
            result = self.solver_fun(sample, u)
            if with_grad:
                value, grad = result
                values.append(value)
                grads.append(grad)
            else:
                values.append(result)
        value = self._aggregate(values)
        if not with_grad:
            return value
        grad = self._aggregate(grads)
        return value, grad

    def optimize(self, u0, *, optimizer="bfgs", maximize=True, **optimizer_opts):
        """Port of ``optimize``: scale ``u0`` to the unit box, dispatch to
        the requested optimizer over the ensemble-aggregated objective,
        then unscale the result back to the caller's control space."""
        us0 = self.scale_variables(u0)
        if not _np.all((-1e-8 < us0) & (us0 < 1 + 1e-8)):
            raise ValueError("Initial guess is not within bounds")
        us0 = _np.clip(us0, 0.0, 1.0)

        if optimizer in ("bfgs", "default", "unitboxbfgs"):
            v, us, history = optimize_bound_constrained(
                us0, lambda us_: self.get_ensemble_objective(us_, with_grad=True),
                maximize=maximize, **optimizer_opts,
            )
        elif optimizer in ("lm", "unitboxlm"):
            v, us, history = unit_box_lm(
                us0, lambda us_: self.get_ensemble_objective(us_, with_grad=True), **optimizer_opts,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer!r}")

        u = self.unscale_variables(us)
        return u, {"value": v, "history": history}
