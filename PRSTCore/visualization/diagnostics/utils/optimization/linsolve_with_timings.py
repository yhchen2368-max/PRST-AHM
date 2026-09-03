"""MRST ``linsolveWithTimings.m`` counterpart."""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse.linalg as spla

_TIMINGS: list[float] = []


def linsolve_with_timings(A=None, x=None, linsolve=None):
    """Solve ``A x = b`` and keep cumulative solve timings.

    Calling without arguments returns and clears collected timings, matching
    MRST's persistent-variable utility behavior.
    """
    global _TIMINGS
    if A is None and x is None:
        out = _TIMINGS.copy()
        _TIMINGS = []
        return out
    solver = linsolve or spla.spsolve
    tic = time.perf_counter()
    out = solver(A, x)
    _TIMINGS.append(time.perf_counter() - tic)
    return out


linsolveWithTimings = linsolve_with_timings

