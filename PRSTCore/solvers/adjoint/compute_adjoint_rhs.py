"""Compute adjoint RHS for pressure system.

1:1 Python translation of MRST solvers/adjoint/computeAdjointRHS.m
"""

import numpy as np


def compute_adjoint_rhs(G, W, f_res):
    """Compute adjoint pressure RHS.

    Parameters
    ----------
    G : dict
        Grid.
    W : list of dict
        Well structures.
    f_res : ndarray
        Adjoint reservoir pressure conditions.

    Returns
    -------
    list
        RHS cell array [b1, b2, b3].
    """
    ncf = len(f_res)  # now matches nc (simplified)

    f_w_list = []
    h_w_list = []
    if W:
        for w in W:
            rhs = w.get("S", {}).get("RHS", {})
            f_w_list.append(np.atleast_1d(rhs.get("f", 0)).ravel())
            h_w_list.append(np.atleast_1d(rhs.get("h", 0)).ravel())

    f_w = np.concatenate(f_w_list) if f_w_list else np.array([])
    h_w = np.concatenate(h_w_list) if h_w_list else np.array([])

    nc = G["cells"]["num"]
    nf = G["faces"]["num"]

    b = [
        np.concatenate([f_res, f_w]),
        np.zeros(nc),
        np.concatenate([np.zeros(nf), h_w]),
    ]
    return b
