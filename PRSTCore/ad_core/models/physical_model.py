"""Python port of MRST's ``PhysicalModel.m`` (mrst-2026a/autodiff/ad-core/models).

Provides the template-method contract every AD model follows (``get_equations``
is the one method concrete models must implement) plus the reusable, pure
numerical primitives MRST's ``PhysicalModel`` exposes as static-ish methods:
``limit_increment`` (``updateStateFromIncrement``'s relative/absolute change
cap, combining ``limitUpdateRelative``/``limitUpdateAbsolute``) and
``cap_property`` (``capProperty``'s min/max clamp).

These primitives are deliberately narrow and side-effect-free (they take and
return plain numpy arrays, not a ``state`` dict) so they can be reused by any
model -- and unit-tested in isolation -- without depending on this class's
state-dict conventions.
"""

from __future__ import annotations

import numpy as _np


class PhysicalModel:
    """Base class for AD physical models. Mirrors MRST's ``PhysicalModel``:
    the common Newton-loop contract, plus reusable primitives subclasses use
    to build ``updateState``/``checkConvergence``.
    """

    def __init__(self, **kwargs):
        self.G = kwargs.get('G', None)
        self.operators = kwargs.get('operators', None)
        self.nonlinearTolerance = float(kwargs.get('nonlinearTolerance', 1.0e-6))
        self.verbose = bool(kwargs.get('verbose', False))
        self.stepFunctionIsLinear = bool(kwargs.get('stepFunctionIsLinear', False))

    # ------------------------------------------------------------------
    # Template methods -- concrete models must implement get_equations.
    # ------------------------------------------------------------------
    def get_equations(self, state0, state, dt, drivingForces, **kwargs):
        """Assemble the residual/Jacobian for this model. Port of MRST's
        ``getEquations`` -> ``getModelEquations``; must be implemented by
        every concrete model."""
        raise NotImplementedError(f'{type(self).__name__} must implement get_equations')

    def check_convergence_default(self, problem):
        """Generic inf-norm convergence check: every residual component must
        be smaller than ``nonlinearTolerance``. Port of the (non-CNV) branch
        of MRST ``PhysicalModel.checkConvergence``; models with a richer
        judgement (e.g. CNV/MB) override ``checkConvergence`` and fall back
        to this for the simple case."""
        values, tol, names = self._convergence_values_default(problem)
        return values < tol, values, names

    def _convergence_values_default(self, problem):
        """Port of ``PhysicalModel.getConvergenceValues`` for the simple
        inf-norm check: raw values, per-entry tolerances and names."""
        residuals = _np.atleast_1d(problem.get('Residuals', []))
        values = _np.abs(residuals)
        tol = _np.full(values.shape, self.nonlinearTolerance, dtype=float)
        names = problem.get('equationNames', ['residual'] * values.size)
        return values, tol, names

    # ------------------------------------------------------------------
    # Reusable primitives (mirror PhysicalModel.m static methods)
    # ------------------------------------------------------------------
    @staticmethod
    def limit_increment(v0, dv, rel_max=None, abs_max=None):
        """Port of MRST ``updateStateFromIncrement``'s change-capping logic
        (``limitUpdateRelative``/``limitUpdateAbsolute`` combined): scales
        ``dv`` down (never up) so that, per element,

            |dv * change| / |v0| <= rel_max   and   |dv * change| <= abs_max

        with ``change = min(1, rel_max/|dv/v0|, abs_max/|dv|)``. Elements
        where a limit does not bind (division guarded against ``v0 == 0`` /
        ``dv == 0``) are left unrestricted by that term rather than
        producing ``nan``/``inf`` in the combined result.

        Returns ``v0 + dv * change`` (the updated value), matching what
        MRST's ``updateStateFromIncrement`` assigns back into ``state``.
        """
        v0 = _np.asarray(v0, dtype=float)
        dv = _np.asarray(dv, dtype=float)
        n = v0.shape
        change = _np.ones(n, dtype=float)

        if rel_max is not None and _np.any(_np.isfinite(_np.atleast_1d(rel_max))):
            biggest = _np.abs(_np.divide(dv, v0, out=_np.zeros(n), where=v0 != 0))
            biggest = _np.where(v0 != 0, biggest, _np.where(dv != 0, _np.inf, 0.0))
            change_rel = _np.minimum(
                _np.divide(rel_max, biggest, out=_np.full(n, _np.inf), where=biggest > 0), 1.0
            )
            change = _np.minimum(change, change_rel)

        if abs_max is not None and _np.any(_np.isfinite(_np.atleast_1d(abs_max))):
            biggest = _np.abs(dv)
            change_abs = _np.minimum(
                _np.divide(abs_max, biggest, out=_np.full(n, _np.inf), where=biggest > 0), 1.0
            )
            change = _np.minimum(change, change_abs)

        return v0 + dv * change

    @staticmethod
    def cap_property(x, minvalue=-_np.inf, maxvalue=_np.inf):
        """Port of MRST ``capProperty``: clamp ``x`` into ``[minvalue, maxvalue]``."""
        return _np.clip(_np.asarray(x, dtype=float), minvalue, maxvalue)
