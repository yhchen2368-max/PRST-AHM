"""Convert controls to well mappings for gradient computation.

1:1 Python translation of MRST solvers/adjoint/controls2Wells.m
"""

import numpy as np


def controls2wells(W, schedule, controls):
    """Create mappings from controls to wells.

    Parameters
    ----------
    W : list of dict
        Well structures.
    schedule : list of dict
        Schedule.
    controls : dict
        Controls structure.

    Returns
    -------
    A_N, b_N, A_D, b_D : lists
        Mappings for each step: q_tot = A_N * u + b_N, p_w = A_D * u + b_D.
    """
    num_steps = len(schedule)
    num_wells = len(W)

    rate_wells = [i for i, w in enumerate(W) if w.get("type") == "rate"]
    bhp_wells = [i for i, w in enumerate(W) if w.get("type") == "bhp"]
    num_controls = len(controls["well"])
    control_well_nums = [w["wellNum"] for w in controls["well"]]
    non_control_wells = [i for i in range(num_wells) if i not in control_well_nums]

    # Mapping from controls to wells
    A = np.zeros((num_wells, num_controls))
    for ci, wn in enumerate(control_well_nums):
        A[wn, ci] = 1.0
    b0 = np.zeros(num_wells)

    A_N_list = []
    b_N_list = []
    A_D_list = []
    b_D_list = []

    for k in range(num_steps):
        vals = np.array(schedule[k]["values"])
        b_cur = b0.copy()
        b_cur[non_control_wells] = vals[non_control_wells]

        A_N_list.append(A[np.ix_(rate_wells)])
        b_N_list.append(b_cur[rate_wells])
        A_D_list.append(A[np.ix_(bhp_wells)])
        b_D_list.append(b_cur[bhp_wells])

    return A_N_list, b_N_list, A_D_list, b_D_list
