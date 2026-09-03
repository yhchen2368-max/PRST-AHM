"""Process adjoint gradients into structured format.

1:1 Python translation of MRST processAdjointGradients.m
"""

import numpy as np


def process_adjoint_gradients(grad, ws, control_ix=None, control_names=None):
    """Process adjoint gradients into per-control-type structure.

    Parameters
    ----------
    grad : list of ndarray
        Gradient for each control step.
    ws : list of list of dict
        Well solutions for each step.
    control_ix : list, optional
        Control step indices.
    control_names : list of str, optional
        Control type names.

    Returns
    -------
    dict
        Gradients keyed by control type name.
    """
    if control_names is None:
        control_names = ["bhp", "rate", "orat", "wrat", "grat", "lrat"]

    nw = len(ws[0]) if ws else 0
    ns = len(ws)

    # Fill in zeros for shut-in wells
    for k in range(len(grad)):
        well_status = np.array([w.get("status", True) for w in ws[k]], dtype=bool)
        if not np.all(well_status):
            tmp = np.zeros(nw)
            tmp[well_status] = np.asarray(grad[k]).ravel()
            grad[k] = tmp

    grad_matrix = np.column_stack([np.asarray(g).ravel() for g in grad])

    controls = []
    for w_step in ws:
        controls.append([w.get("type", "") for w in w_step])
    controls = np.array(controls).T  # (nw, ns)

    result = {}
    for nm in control_names:
        result[nm] = (controls == nm).astype(float) * grad_matrix.T
        if control_ix is not None:
            result[nm] = result[nm] @ np.eye(ns)[:, control_ix]

    return result
