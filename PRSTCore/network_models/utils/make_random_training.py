from copy import deepcopy
import numpy as np


def make_random_training(example, r_scale, bhp_scale, shutin):
    """Random perturbation of an existing simulation schedule.

    1:1 Python translation of MRST makeRandomTraining.m
    """
    problem = deepcopy(example)

    schedule = {
        "step": {
            "val": np.asarray(problem["schedule"]["step"]["val"], dtype=float),
            "control": np.asarray(problem["schedule"]["step"]["control"], dtype=int),
        },
        "control": list(problem["schedule"]["control"]),
    }
    control_orig = problem["schedule"]["control"]
    step_ctrl = problem["schedule"]["step"]["control"]

    # Run-length encode the control steps
    ctrl_no = []
    nstep = []
    count = 0
    for idx, c in enumerate(step_ctrl):
        if idx == 0 or c != step_ctrl[idx - 1]:
            if idx > 0:
                nstep.append(count)
            ctrl_no.append(c)
            count = 1
        else:
            count += 1
    if count > 0:
        nstep.append(count)
    inds = np.concatenate(([0], np.cumsum(nstep)))

    if not callable(r_scale):
        r_scale_fn = lambda x, y: (1 + 2 * r_scale * (x - 0.5)) * y
    else:
        r_scale_fn = r_scale
    if not callable(bhp_scale):
        bhp_scale_fn = lambda x, y: (1 + 2 * bhp_scale * (x - 0.5)) * y
    else:
        bhp_scale_fn = bhp_scale

    ctrl_ind = 0
    rng = np.random.default_rng(1)
    controls = []
    for m, n in enumerate(nstep):
        # ``ctrlVals = ceil(0.25*(1:nstep(m))') + ctrlInd``. The 0.25 is
        # the substance: report steps are grouped in *fours*, so one new
        # control covers four steps rather than one control per step.
        # Numbering them one-per-step gives four times as many controls,
        # each with its own random perturbation -- a different training
        # set, not a differently-indexed one.
        k = np.arange(1, n + 1)
        ctrl_vals = np.ceil(0.25 * k).astype(int) + ctrl_ind      # 1-based
        schedule["step"]["control"][inds[m] : inds[m + 1]] = ctrl_vals - 1

        # The first control of the block is the original, unperturbed.
        controls.append(deepcopy(control_orig[ctrl_no[m]]))

        last = int(ctrl_vals.max())
        for _ in range(ctrl_ind + 2, last + 1):
            ctrl = deepcopy(control_orig[ctrl_no[m]])
            W_new = []
            for w in ctrl.get("W", []):
                wc = dict(w)
                w_type = wc.get("type", "")
                lims = wc.get("lims")
                if w_type == "rate":
                    wc["val"] = float(r_scale_fn(rng.random(), wc["val"]))
                    # ``~isempty(lims) && ~isinf(lims.rate)``: an infinite
                    # limit means "unconstrained", and writing the new
                    # rate into it would turn that into a hard constraint
                    # at exactly the target.
                    if isinstance(lims, dict) and _is_finite(lims.get("rate")):
                        lims["rate"] = wc["val"]
                elif w_type == "bhp":
                    wc["val"] = float(bhp_scale_fn(rng.random(), wc["val"]))
                    if isinstance(lims, dict) and _is_finite(lims.get("bhp")):
                        lims["bhp"] = wc["val"]
                if shutin and (wc.get("sign", 1) < 0) and (rng.random() > 0.9):
                    wc["status"] = False
                W_new.append(wc)
            ctrl["W"] = W_new
            controls.append(ctrl)

        ctrl_ind = last

    schedule["control"] = controls

    problem["schedule"] = schedule
    problem["name"] = problem.get("name", "") + "_rand"
    return problem


def _is_finite(value):
    """MATLAB's ``~isinf(x)`` on a limit that may be absent."""
    if value is None:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False
