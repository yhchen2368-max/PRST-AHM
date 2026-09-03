"""Initialize schedule structure based on wells.

1:1 Python translation of MRST solvers/adjoint/initSchedule.m
"""

import numpy as np


def init_schedule(W, num_steps=1, total_time=1.0, time_steps=None, verbose=False):
    """Initialize schedule structure based on well W.

    Parameters
    ----------
    W : list of dict
        Well structures.
    num_steps : int
        Number of simulation time steps.
    total_time : float
        Total simulation time.
    time_steps : array-like, optional
        End times for each step (overrides num_steps/total_time).
    verbose : bool
        Display schedule.

    Returns
    -------
    list of dict
        Schedule array, each entry with timeInterval, names, types, values.
    """
    if time_steps is None:
        ts = np.linspace(total_time / num_steps, total_time, num_steps)
    else:
        ts = np.atleast_1d(time_steps).ravel()

    intervals = np.column_stack([np.concatenate([[0], ts[:-1]]), ts])

    schedule = []
    for k in range(len(ts)):
        step = {
            "timeInterval": intervals[k].tolist(),
            "names": [w.get("name", f"W{i}") for i, w in enumerate(W)],
            "types": [w.get("type", "bhp") for w in W],
            "values": [float(w.get("val", 0)) for w in W],
        }
        schedule.append(step)

    if verbose:
        disp_schedule(schedule)

    return schedule
