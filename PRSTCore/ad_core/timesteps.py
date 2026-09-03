"""Python port of MRST's pluggable timestep strategies
(mrst-2026a/autodiff/ad-core/timesteps).

``rampup_timesteps`` is a pure function (no solver state); the two selector
classes below extend the pattern already used by
:class:`PRSTCore.ad_core.solvers.nonlinear_solver.SimpleTimeStepSelector`/
``IterationCountTimeStepSelector`` (left as-is to avoid touching an
already-relied-on file) as standalone, independently testable strategies.
"""

from __future__ import annotations

import numpy as _np


def rampup_timesteps(time: float, dt: float, n: int = 8) -> _np.ndarray:
    """Port of MRST ``rampupTimesteps.m``: a geometrically increasing
    sequence of timesteps (``dt/2**n, dt/2**(n-1), ..., dt/2``) followed by
    constant-``dt`` steps, covering exactly ``time`` in total (the final
    step may be shorter than ``dt`` to land exactly on ``time``).
    """
    if time == 0:
        return _np.zeros(0)

    dt_init = dt / 2.0 ** _np.concatenate([[n], _np.arange(n, 0, -1)])
    cs_time = _np.cumsum(dt_init)
    if _np.any(cs_time > time):
        dt_init = dt_init[cs_time < time]

    dt_left = time - _np.sum(dt_init)
    nrem = int(_np.floor(dt_left / dt))
    dt_rem = _np.full(nrem, dt)
    dt_rem_final = dt_rem[-1] if nrem > 0 else 0.0

    dt_final = time - _np.sum(dt_init) - _np.sum(dt_rem)
    if dt_rem_final != 0.0 and dt_final / dt_rem_final <= 1.0e-6:
        if nrem > 0:
            dt_rem[-1] = dt_rem_final + dt_final
        dt_final = None
    elif dt_rem_final == 0.0 and dt_final <= 1.0e-6 * dt:
        dt_final = None

    parts = [dt_init, dt_rem]
    if dt_final is not None:
        parts.append(_np.array([dt_final]))
    return _np.concatenate(parts)


class StateChangeTimeStepSelector:
    """Port of MRST ``StateChangeTimeStepSelector``: targets a maximum
    relative change per step in a chosen state field (e.g. pressure or
    saturation), growing/shrinking the next step to drive the observed
    change toward the target.
    """

    def __init__(self, *, target_change: float = 0.2, dt_min: float = 1.0, dt_max: float = float("inf"),
                 growth_factor: float = 2.0, cut_factor: float = 0.5):
        self.target_change = target_change
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.growth_factor = growth_factor
        self.cut_factor = cut_factor

    def compute_next_dt(self, dt_prev: float, relative_change: float) -> float:
        """``relative_change``: the largest observed relative change in the
        tracked field over the just-completed step of size ``dt_prev``."""
        if relative_change <= 0:
            factor = self.growth_factor
        else:
            factor = min(self.growth_factor, self.target_change / relative_change)
            factor = max(factor, self.cut_factor)
        return float(_np.clip(dt_prev * factor, self.dt_min, self.dt_max))


class FactorTimeStepSelector:
    """Port of MRST ``FactorTimeStepSelector``: grows the timestep by a
    fixed factor after a converged step, shrinks it by a fixed factor after
    a failed one -- the simplest step-size heuristic, useful as a fallback
    when no richer signal (iteration count, state change) is tracked.
    """

    def __init__(self, *, growth_factor: float = 1.25, cut_factor: float = 0.5,
                 dt_min: float = 1.0, dt_max: float = float("inf")):
        self.growth_factor = growth_factor
        self.cut_factor = cut_factor
        self.dt_min = dt_min
        self.dt_max = dt_max

    def compute_next_dt(self, dt_prev: float, converged: bool) -> float:
        factor = self.growth_factor if converged else self.cut_factor
        return float(_np.clip(dt_prev * factor, self.dt_min, self.dt_max))
