"""Convert schedule to control vector.

1:1 Python translation of MRST schedule2control.m
"""

import numpy as np


def schedule2control(schedule, scaling):
    """Convert schedule to scaled control vector in [0, 1].

    Parameters
    ----------
    schedule : dict
        Schedule with 'control' array of well controls.
    scaling : object
        Scaling with 'box_lims' of shape (nw, 2).

    Returns
    -------
    ndarray
        Scaled control vector.
    """
    umin = scaling.box_lims[:, 0]
    umax = scaling.box_lims[:, 1]
    u_list = []
    for ctrl in schedule["control"]:
        vals = np.array([w["val"] for w in ctrl["W"]], dtype=float)
        u_step = (vals - umin) / (umax - umin)
        u_list.append(u_step)
    return np.concatenate(u_list)


def control2schedule(u, schedule, scaling):
    """Convert scaled control vector back to schedule.

    Parameters
    ----------
    u : ndarray
        Scaled control vector in [0, 1].
    schedule : dict
        Template schedule.
    scaling : object
        Scaling with 'box_lims'.

    Returns
    -------
    dict
        Updated schedule.
    """
    umin = scaling.box_lims[:, 0]
    umax = scaling.box_lims[:, 1]
    nc = len(schedule["control"])
    nw = len(schedule["control"][0]["W"])
    c = 0
    new_schedule = {
        "step": dict(schedule["step"]),
        "control": [dict(ctrl) for ctrl in schedule["control"]],
    }
    for cs in range(nc):
        new_schedule["control"][cs]["W"] = [
            dict(w) for w in schedule["control"][cs]["W"]
        ]
        for w_idx in range(nw):
            new_schedule["control"][cs]["W"][w_idx]["val"] = (
                u[c] * (umax[w_idx] - umin[w_idx]) + umin[w_idx]
            )
            c += 1
    return new_schedule
