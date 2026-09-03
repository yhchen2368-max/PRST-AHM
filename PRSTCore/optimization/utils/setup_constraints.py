"""Setup linear constraints for scaled problem across all control steps.

1:1 Python translation of MRST setupConstraints.m
"""

import numpy as np
from scipy.linalg import block_diag


def setup_constraints(lin_const, schedule, scaling):
    """Set up linear constraints for all control steps.

    Parameters
    ----------
    lin_const : dict
        Per-step constraints with 'A' and 'b'.
    schedule : dict
        Schedule with 'control' array.
    scaling : object
        Scaling with 'box_lims'.

    Returns
    -------
    dict
        Block-diagonal constraints.
    """
    umin = scaling.box_lims[:, 0]
    umax = scaling.box_lims[:, 1]
    D = np.diag(umax - umin)
    A_step = lin_const["A"] @ D
    b_step = lin_const["b"] - lin_const["A"] @ umin
    nc = len(schedule["control"])
    A_blocks = [A_step] * nc
    b_blocks = [b_step] * nc
    return {"A": block_diag(*A_blocks), "b": np.concatenate(b_blocks)}
