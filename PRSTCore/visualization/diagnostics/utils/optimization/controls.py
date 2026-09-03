"""Control/well conversion helpers from MRST diagnostics optimization."""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import numpy as np


def well2control(W, *, targets: Iterable[int] | None = None, scaling=None):
    targets = _targets(targets, len(W))
    vals = np.asarray([W[int(i)].get("val", 0.0) for i in targets], dtype=float)
    if scaling is not None:
        lims = np.asarray(getattr(scaling, "boxLims", scaling.get("boxLims")), dtype=float)
        return (vals - lims[:, 0]) / (lims[:, 1] - lims[:, 0])
    return vals


def control2well(u, W, *, targets: Iterable[int] | None = None, scaling=None):
    out = deepcopy(W)
    targets = _targets(targets, len(out))
    u = np.asarray(u, dtype=float).ravel()
    for k, wno in enumerate(targets):
        value = u[k]
        if scaling is not None:
            lims = np.asarray(getattr(scaling, "boxLims", scaling.get("boxLims")), dtype=float)
            lo, hi = lims[k, :]
            value = value * (hi - lo) + lo
        out[int(wno)]["val"] = float(value)
    return out


def _targets(targets, n):
    if targets is None:
        return np.arange(n, dtype=int)
    arr = np.asarray(list(targets), dtype=int).ravel()
    if arr.size and np.min(arr) >= 1 and np.max(arr) <= n:
        arr = arr - 1
    return arr

