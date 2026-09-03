"""Initialization of simple scaled ADI fluid.

1:1 Python translation of MRST initSimpleScaledADIFluid.m
"""

import numpy as np


def init_simple_scaled_adi_fluid(fluid=None, swl=0.0, swcr=0.0, sowcr=0.0, swu=1.0):
    """Create or update a simple scaled AD fluid with relperm scaling parameters.

    Parameters
    ----------
    fluid : dict, optional
        Existing fluid structure to update.
    swl : float
        Connate water saturation.
    swcr : float
        Critical water saturation.
    sowcr : float
        Critical oil-in-water saturation.
    swu : float
        Maximum water saturation.

    Returns
    -------
    dict
        Fluid structure with scaled relperm functions.
    """
    if fluid is None:
        fluid = {}

    fluid["swl"] = swl
    fluid["swcr"] = swcr
    fluid["sowcr"] = sowcr
    fluid["swu"] = swu

    def scale_sw(s):
        s = np.asarray(s, dtype=float)
        denom = swu - swcr
        if denom <= 0:
            return np.clip(s, 0, 1)
        return np.clip((s - swcr) / denom, 0, 1)

    def scale_so(s):
        s = np.asarray(s, dtype=float)
        denom = 1 - swl - sowcr
        if denom <= 0:
            return np.clip(s, 0, 1)
        return np.clip((s - sowcr) / denom, 0, 1)

    fluid["scale_sw"] = scale_sw
    fluid["scale_so"] = scale_so

    return fluid
