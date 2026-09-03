"""MRST private ``getDynamicMeasures.m`` counterpart."""

import numpy as np

from PRSTCore.visualization.diagnostics import computeFandPhi, computeLorenz, computeSweep
from ..helpers import get_field


def get_dynamic_measures(d, tsel=None, wsel=None):
    del wsel
    data = get_field(d, "Data", get_field(d, "data", d))
    diagnostics = get_field(data, "diagnostics", [])
    pv = np.asarray(get_field(get_field(d, "G", {}), "cells", {}).get("PORV", get_field(data, "PORV", [])), dtype=float).ravel()
    indices = range(len(diagnostics)) if tsel is None else np.asarray(tsel, dtype=int).ravel()
    LCt, Ev, tD = [], [], []
    for i in indices:
        D = get_field(diagnostics[i], "D", get_field(diagnostics[i], "D", None))
        if D is None or pv.size == 0:
            continue
        F, Phi = computeFandPhi(pv, D.tof)
        LCt.append(computeLorenz(F, Phi))
        ev, td = computeSweep(F, Phi)
        Ev.append(ev)
        tD.append(td)
    return {"LCt": np.asarray(LCt), "Ev": Ev, "tD": tD}


getDynamicMeasures = get_dynamic_measures

