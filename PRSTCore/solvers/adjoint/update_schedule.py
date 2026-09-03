"""Update schedule based on controls.

1:1 Python translation of MRST solvers/adjoint/updateSchedule.m
"""

import numpy as np


def update_schedule(controls, schedule):
    """Update schedule values from control structure.

    Parameters
    ----------
    controls : dict
        Controls with 'well' array.
    schedule : list of dict
        Schedule to update.

    Returns
    -------
    list of dict
        Updated schedule.
    """
    num_steps = len(schedule)
    control_wells = [w["wellNum"] for w in controls["well"]]
    U = np.array([w["values"] for w in controls["well"]]).T  # (numSteps, numWells)

    new_schedule = [dict(s) for s in schedule]
    for k in range(num_steps):
        vals = list(new_schedule[k]["values"])
        for i, wn in enumerate(control_wells):
            vals[wn] = float(U[k, i])
        new_schedule[k]["values"] = vals
    return new_schedule
