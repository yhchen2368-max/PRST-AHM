"""Display helper counterparts for MRST private diagnostics GUI functions."""


def show_allocation(d, src=None, ax=None, s2=None, s3=None):
    return {"d": d, "src": src, "ax": ax, "s2": s2, "s3": s3}


def show_well_communication(d, ax=None, val=None):
    return {"d": d, "ax": ax, "val": val}


def ui_pre_select_time_steps(info):
    if isinstance(info, dict) and "steps" in info:
        return info["steps"]
    return None


showAllocation = show_allocation
showWellCommunication = show_well_communication
uiPreSelectTimeSteps = ui_pre_select_time_steps

