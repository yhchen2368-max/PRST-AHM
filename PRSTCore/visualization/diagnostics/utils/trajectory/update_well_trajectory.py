"""MRST trajectory ``updateWellTrajectory.m`` counterpart."""

from copy import deepcopy

import numpy as np


def update_well_trajectory(model, w, ws, traj):
    """Return a well copy with trajectory metadata updated."""
    del model
    w_out = deepcopy(w)
    ws_out = deepcopy(ws)
    w_out["trajectory"] = np.asarray(traj, dtype=float)
    if len(w_out["trajectory"]):
        w_out["refDepth"] = float(w_out["trajectory"][0, 2]) if w_out["trajectory"].shape[1] > 2 else w_out.get("refDepth", 0.0)
    return w_out, ws_out


updateWellTrajectory = update_well_trajectory

