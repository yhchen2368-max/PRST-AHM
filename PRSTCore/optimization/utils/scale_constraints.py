"""Scale linear constraints for unit-interval.

1:1 Python translation of MRST scaleConstraints.m
"""

import numpy as np


def scale_constraints(lin_const, scaling):
    """Scale linear constraints to unit-interval [0, 1].

    Parameters
    ----------
    lin_const : dict
        Linear constraints with 'A' (matrix) and 'b' (vector).
    scaling : object
        Scaling object with 'box_lims' attribute of shape (n, 2).

    Returns
    -------
    dict
        Scaled constraints.
    """
    umin = scaling.box_lims[:, 0]
    umax = scaling.box_lims[:, 1]
    D = np.diag(umax - umin)
    A_scaled = lin_const["A"] @ D
    b_scaled = lin_const["b"] - lin_const["A"] @ umin
    return {"A": A_scaled, "b": b_scaled, "scaled": True}
