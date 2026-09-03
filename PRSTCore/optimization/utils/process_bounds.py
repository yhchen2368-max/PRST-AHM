"""Process bounds for well controls.

1:1 Python translation of MRST processBounds.m
"""

import numpy as np


def process_bounds(W, **kwargs):
    """Process well control bounds.

    Parameters
    ----------
    W : list of dict
        Well structures.
    **kwargs : dict
        Key-value pairs of bound expressions, e.g., bhp=array, rate=array.

    Returns
    -------
    bounds : dict
        Dictionary with fields 'bhp', 'wrat', 'orat', 'grat', 'lrat', 'rate'.
        Each entry is an (nw, 2) array of [lower, upper] bounds.
    """
    nw = len(W)
    flds = ["bhp", "wrat", "orat", "grat", "lrat", "rate"]
    lower = {f: np.full(nw, np.nan) for f in flds}
    upper = {f: np.full(nw, np.nan) for f in flds}

    for expr, vals in kwargs.items():
        vals = np.atleast_2d(np.asarray(vals, dtype=float))
        if vals.shape[1] == 2:
            lower[expr] = vals[:, 0]
            upper[expr] = vals[:, 1]

    bounds = {}
    for f in flds:
        bounds[f] = np.column_stack([lower[f], upper[f]])
    return bounds
