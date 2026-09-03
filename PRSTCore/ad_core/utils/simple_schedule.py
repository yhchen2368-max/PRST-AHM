import numpy as np


def simple_schedule(step_vals, controls):
    """Create a simple MRST-like schedule object."""
    return {
        "step": {
            "val": np.asarray(step_vals, dtype=float),
            "control": np.arange(len(step_vals), dtype=int),
        },
        "control": controls,
    }


def perturbed_simple_schedule(base_schedule, rate_scale=0.0, bhp_scale=0.0, perturb_prob=0.5):
    """Perturb a simple schedule by applying randomized control changes."""
    schedule = {
        "step": {
            "val": np.array(base_schedule["step"]["val"], dtype=float),
            "control": np.array(base_schedule["step"]["control"], dtype=int),
        },
        "control": [dict(ctrl) for ctrl in base_schedule["control"]],
    }
    for i, ctrl in enumerate(schedule["control"]):
        if "W" not in ctrl:
            continue
        W = [dict(w) for w in ctrl["W"]]
        for w in W:
            if w.get("type") == "rate":
                w["val"] = float(w["val"]) * (1 + rate_scale * (np.random.rand() - 0.5) * 2)
                if "lims" in w and w["lims"] is not None:
                    w["lims"]["rate"] = w["val"]
            elif w.get("type") == "bhp":
                w["val"] = float(w["val"]) * (1 + bhp_scale * (np.random.rand() - 0.5) * 2)
                if "lims" in w and w["lims"] is not None:
                    w["lims"]["bhp"] = w["val"]
            if perturb_prob > 0.0 and w.get("sign", 1) < 0 and np.random.rand() > 1 - perturb_prob:
                w["status"] = False
        schedule["control"][i]["W"] = W
    return schedule
