"""Initialize control structure based on well schedule.

1:1 Python translation of MRST solvers/adjoint/initControls.m
"""

import numpy as np


def init_controls(schedule, controllable_wells=None, bhp_min_max=None,
                  rate_min_max=None, min_max=None, num_control_steps=None,
                  lin_ineq_const=None, lin_eq_const=None, verbose=False):
    """Initialize control structure.

    Parameters
    ----------
    schedule : list of dict
        Schedule from init_schedule.
    controllable_wells : list, optional
        Indices of controllable wells (default all).
    bhp_min_max : list, optional
        [min, max] for BHP wells.
    rate_min_max : list, optional
        [min, max] for rate wells.
    min_max : ndarray, optional
        Matrix of size (numControls, 2).
    num_control_steps : int, optional
        Number of control steps (default number of schedule steps).
    lin_ineq_const : dict, optional
        Linear inequality constraints {A, b}.
    lin_eq_const : dict, optional
        Linear equality constraints {A, b}.
    verbose : bool
        Display controls.

    Returns
    -------
    dict
        Controls structure with 'well' and 'linEqConst' fields.
    """
    num_wells = len(schedule[0]["names"])
    num_steps = len(schedule)

    if controllable_wells is None:
        controllable_wells = list(range(num_wells))
    if bhp_min_max is None:
        bhp_min_max = [-np.inf, np.inf]
    if rate_min_max is None:
        rate_min_max = [-np.inf, np.inf]
    if num_control_steps is None:
        num_control_steps = num_steps

    well_controls = []
    for wnum in controllable_wells:
        wtype = schedule[0]["types"][wnum]
        if wtype == "bhp":
            mm = bhp_min_max
        elif wtype == "rate":
            mm = rate_min_max
        else:
            mm = [-np.inf, np.inf]

        if min_max is not None and wnum < len(min_max):
            mm = min_max[wnum]

        values = np.array([schedule[k]["values"][wnum] for k in range(num_steps)])
        well_controls.append({
            "wellNum": wnum,
            "values": values,
            "type": wtype,
            "minMax": mm,
        })

    controls = {
        "well": well_controls,
        "linEqConst": lin_eq_const,
        "linIneqConst": lin_ineq_const,
        "numControlSteps": num_control_steps,
    }

    if verbose:
        disp_controls(controls, schedule)

    return controls
