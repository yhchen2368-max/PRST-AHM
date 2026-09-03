"""Setup simulation control mappings from schedule to scaled control vector.

1:1 Python translation of MRST setupSimulationControlMappings.m
"""

import numpy as np


def setup_simulation_control_mappings(schedule, bounds, controllable_wells=None):
    """Map scaled control vector to/from schedule.

    Parameters
    ----------
    schedule : dict
        Schedule with 'control' array.
    bounds : list of dict or dict
        Bound definitions per control step.
    controllable_wells : ndarray, optional
        Boolean mask of controllable wells.

    Returns
    -------
    maps : dict
        Mapping structure.
    u : ndarray
        Initial scaled control vector.
    """
    nw = len(schedule["control"][0]["W"])
    ns = len(schedule["control"])

    if not isinstance(bounds, list):
        bounds = [bounds] * ns

    if controllable_wells is None:
        controllable_wells = np.ones(nw, dtype=bool)

    if not np.any(controllable_wells):
        return {"wellNo": np.array([]), "type": [], "stepNo": np.array([]),
                "isTarget": np.array([], dtype=bool), "bounds": np.empty((0, 2))}, np.array([])

    flds = ["bhp", "wrat", "orat", "grat", "lrat", "rate"]
    nc_total = 0
    for step in range(ns):
        for wno in range(nw):
            if controllable_wells[wno] and schedule["control"][step]["W"][wno].get("status", True):
                nc_total += 1
                # Also count limit controls
                w = schedule["control"][step]["W"][wno]
                if "lims" in w and w["lims"] is not None:
                    for f in flds:
                        if f in w["lims"] and np.isfinite(w["lims"][f]):
                            if f != w.get("type", ""):
                                nc_total += 1

    maps = {"type": [""] * nc_total, "wellNo": np.zeros(nc_total, dtype=int),
            "stepNo": np.zeros(nc_total, dtype=int), "isTarget": np.zeros(nc_total, dtype=bool),
            "bounds": np.full((nc_total, 2), np.nan)}
    u = np.full(nc_total, np.nan)

    ix = 0
    for step in range(ns):
        for wno in range(nw):
            if not controllable_wells[wno]:
                continue
            w = schedule["control"][step]["W"][wno]
            if not w.get("status", True):
                continue
            wtype = w.get("type", "")
            if wtype in flds:
                maps["type"][ix] = wtype
                maps["wellNo"][ix] = wno
                maps["stepNo"][ix] = step
                maps["isTarget"][ix] = True
                if step < len(bounds) and wtype in bounds[step]:
                    bnds = bounds[step][wtype][wno, :]
                else:
                    bnds = np.array([0.0, 1.0])
                maps["bounds"][ix, :] = bnds
                u[ix] = (w["val"] - bnds[0]) / (bnds[1] - bnds[0])
                ix += 1

            # Handle limits
            if "lims" in w and w["lims"] is not None:
                for f in flds:
                    if f in w["lims"] and np.isfinite(w["lims"][f]) and f != wtype:
                        maps["type"][ix] = f
                        maps["wellNo"][ix] = wno
                        maps["stepNo"][ix] = step
                        maps["isTarget"][ix] = False
                        if step < len(bounds) and f in bounds[step]:
                            bnds = bounds[step][f][wno, :]
                        else:
                            bnds = np.array([0.0, 1.0])
                        maps["bounds"][ix, :] = bnds
                        u[ix] = (w["lims"][f] - bnds[0]) / (bnds[1] - bnds[0])
                        ix += 1

    maps["type"] = maps["type"][:ix]
    maps["wellNo"] = maps["wellNo"][:ix]
    maps["stepNo"] = maps["stepNo"][:ix]
    maps["isTarget"] = maps["isTarget"][:ix]
    maps["bounds"] = maps["bounds"][:ix, :]
    u = u[:ix]

    return maps, u
