"""MRST private ``timeAverageWellPairDiagnostics.m`` counterpart."""

from copy import deepcopy

import numpy as np

from ..helpers import get_field


def time_average_well_pair_diagnostics(d, timeSteps=None):
    data = get_field(d, "Data", get_field(d, "data", d))
    diagnostics = get_field(data, "diagnostics")
    if timeSteps is None:
        timeSteps = np.arange(len(diagnostics), dtype=int)
    else:
        timeSteps = np.asarray(timeSteps, dtype=int).ravel()
        if timeSteps.size and np.min(timeSteps) >= 1 and np.max(timeSteps) <= len(diagnostics):
            timeSteps = timeSteps - 1
    if timeSteps.size <= 1:
        raise AssertionError("timeSteps must contain at least two entries")
    time = np.asarray(get_field(get_field(data, "time", {}), "cur", np.arange(len(diagnostics))), dtype=float)
    dt = np.diff(time[timeSteps])
    weights = dt / max(float(np.sum(dt)), np.finfo(float).eps)
    selected = [diagnostics[i] for i in timeSteps[1:]]
    WP = deepcopy(get_field(selected[0], "WP"))
    for k, weight in enumerate(weights):
        cur = get_field(selected[k], "WP")
        if k == 0:
            scale = 0.0
        else:
            scale = 1.0
        for ni in range(len(WP.inj)):
            WP.inj[ni].alloc = scale * WP.inj[ni].alloc + weight * cur.inj[ni].alloc
            WP.inj[ni].ralloc = scale * WP.inj[ni].ralloc + weight * cur.inj[ni].ralloc
        for pi in range(len(WP.prod)):
            WP.prod[pi].alloc = scale * WP.prod[pi].alloc + weight * cur.prod[pi].alloc
            WP.prod[pi].ralloc = scale * WP.prod[pi].ralloc + weight * cur.prod[pi].ralloc
        WP.vols = scale * WP.vols + weight * cur.vols
    return WP


timeAverageWellPairDiagnostics = time_average_well_pair_diagnostics

