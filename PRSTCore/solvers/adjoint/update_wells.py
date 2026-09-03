"""Update wells based on schedule time step.

1:1 Python translation of MRST solvers/adjoint/updateWells.m
"""

import numpy as np


def update_wells(W, schedule_step):
    """Update wells based on a schedule step.

    Parameters
    ----------
    W : list of dict
        Well structures.
    schedule_step : dict
        Schedule step with 'types' and 'values'.

    Returns
    -------
    list of dict
        Updated wells.
    """
    new_W = [dict(w) for w in W]
    for k, w in enumerate(new_W):
        w_type = schedule_step["types"][k]
        w_val = float(schedule_step["values"][k])
        w["type"] = w_type
        w["val"] = w_val

        if "S" in w and "RHS" in w["S"]:
            sizeB = w["S"].get("sizeB", [1])
            n = sizeB[0] if isinstance(sizeB, list) else 1
            if w_type == "bhp":
                w["S"]["RHS"]["f"] = np.full(n, -w_val)
                w["S"]["RHS"]["h"] = 0.0
            elif w_type == "rate":
                w["S"]["RHS"]["f"] = np.zeros(n)
                w["S"]["RHS"]["h"] = -w_val
    return new_W
