"""Control logic function for simulation.

1:1 Python translation of MRST controlLogicFunc.m
"""

import numpy as np


def control_logic_func(state, schedule, report, cnum, wcut_lim=None, rate_lim=None,
                       close_well_on_sign_change=False):
    """Apply control logic during simulation.

    Parameters
    ----------
    state : dict
        Current well state with 'wellSol'.
    schedule : dict
        Schedule (mutated in place conceptually).
    report : dict
        Simulation report.
    cnum : int
        Current control step number (0-based).
    wcut_lim : float, optional
        Water-cut limit for shutting in producers.
    rate_lim : float, optional
        Minimum rate for keeping wells open.
    close_well_on_sign_change : bool
        Whether to close wells on sign change.

    Returns
    -------
    tuple (state, schedule, report, is_altered)
    """
    curc = schedule["step"]["control"][cnum]
    W = schedule["control"][curc]["W"]
    ws = state["wellSol"]
    nc = len(schedule["control"])
    is_altered = False

    for k, w in enumerate(W):
        # Check water-cut in producers
        if wcut_lim is not None and w.get("sign", 0) < 0 and w.get("status", True):
            qws = ws[k].get("qWs", 0)
            qos = ws[k].get("qOs", 0)
            if (qws + qos) > 0:
                wc = qws / (qws + qos)
                if np.isfinite(wc) and wc >= wcut_lim:
                    is_altered = True
                    for c in range(curc, nc):
                        schedule["control"][c]["W"][k]["status"] = False

        # Check rates
        if rate_lim is not None and w.get("status", True):
            rate = abs(ws[k].get("qWs", 0) + ws[k].get("qOs", 0))
            if rate < rate_lim:
                is_altered = True
                schedule["control"][curc]["W"][k]["status"] = False

        # Check well sign change
        if close_well_on_sign_change:
            rate = ws[k].get("qWs", 0) + ws[k].get("qOs", 0)
            if np.sign(rate) != w.get("sign", 0):
                is_altered = True
                schedule["control"][curc]["W"][k]["status"] = False

    return state, schedule, report, is_altered
