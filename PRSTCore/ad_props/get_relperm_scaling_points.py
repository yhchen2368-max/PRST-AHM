"""Extract relperm scaling points from a model's fluid.krPts.

This is a helper function corresponding to the pattern used in MRST examples:
    scaling = getRelpermScalingPoints(cModel);
    cModel  = imposeRelpermScaling(cModel, scaling{:});

Although not a standalone MRST function, this utility encapsulates the
logic of extracting the default scaling points from fluid.krPts.
"""

import numpy as np


def get_relperm_scaling_points(model):
    """Extract default relperm scaling points from model fluid.

    Parameters
    ----------
    model : dict
        Model with fluid.krPts field.

    Returns
    -------
    list
        Scaling arguments [kw1, val1, kw2, val2, ...] to pass
        directly to impose_relperm_scaling.
    """
    if "krPts" not in model.get("fluid", {}):
        raise ValueError("Model fluid must contain 'krPts'")

    pts = model["fluid"]["krPts"]

    # Determine which phases are present
    has_oil = model.get("oil", True)
    has_water = model.get("water", True)
    has_gas = model.get("gas", False)

    scaling = []

    if has_water:
        w_pts = pts.get("w", pts.get("ow", None))
        if w_pts is not None:
            scaling.extend([
                "SWL", w_pts[0] if len(w_pts) > 0 else 0.0,
                "SWCR", w_pts[1] if len(w_pts) > 1 else 0.0,
                "SWU", w_pts[2] if len(w_pts) > 2 else 1.0,
                "KRW", w_pts[3] if len(w_pts) > 3 else 1.0,
            ])

    if has_oil:
        o_pts = pts.get("o", pts.get("ow", None))
        if o_pts is not None and has_water:
            scaling.extend([
                "SOWCR", o_pts[1] if len(o_pts) > 1 else 0.0,
                "KRO", o_pts[3] if len(o_pts) > 3 else 1.0,
            ])
        # For gas-oil systems
        og_pts = pts.get("og", None)
        if og_pts is not None and has_gas:
            scaling.extend([
                "SOGCR", og_pts[1] if len(og_pts) > 1 else 0.0,
                "KRG", og_pts[3] if len(og_pts) > 3 else 1.0,
            ])
            if o_pts is None:
                scaling.extend([
                    "KRO", og_pts[3] if len(og_pts) > 3 else 1.0,
                ])

    if has_gas and not has_water:
        g_pts = pts.get("g", pts.get("og", None))
        if g_pts is not None:
            scaling.extend([
                "SGL", g_pts[0] if len(g_pts) > 0 else 0.0,
                "SGCR", g_pts[1] if len(g_pts) > 1 else 0.0,
                "SGU", g_pts[2] if len(g_pts) > 2 else 1.0,
                "KRG", g_pts[3] if len(g_pts) > 3 else 1.0,
            ])

    return scaling
