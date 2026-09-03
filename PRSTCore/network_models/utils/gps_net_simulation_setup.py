"""GPSNet simulation setup converter.

1:1 Python translation of MRST modules/network-models/utils/gpsNetSimulationSetup.m

Converts a fine-scale schedule to be compatible with the GPSNet model.
"""

import copy

import numpy as np


def gps_net_simulation_setup(gps_net, schedule):
    """Create a simulation setup for a GPSNet model.

    Parameters
    ----------
    gps_net : GPSNet
        GPSNet model instance.
    schedule : dict
        Fine-scale simulation schedule.

    Returns
    -------
    dict
        Setup with model, schedule, state0 suitable for evaluateMatch.
    """
    # ``schedule = iSchedule``: MATLAB copies the whole struct by value,
    # so anything the caller put on the schedule beyond step/control
    # survives into the setup.
    new_schedule = copy.deepcopy(schedule)

    for ctrl in new_schedule["control"]:
        for i, network_well in enumerate(gps_net.W):
            if i >= len(ctrl["W"]):
                break
            fine = ctrl["W"][i]

            #     Wi = gpsNet.W(i);
            #     Wi.type/val/status <- the fine schedule's
            #     Wi.WI              <- sum of the fine schedule's
            #     for fn = fieldnames(Wi)', schedule.W(i).(fn) = Wi.(fn);
            #
            # The loop copies *every* field of the network well, not a
            # chosen few: the network well is the one that matches the
            # GPSNet grid, so its perforation cells, reference depth, dZ
            # and connection status all have to replace the fine-scale
            # ones. Rebuilding the well from a fixed field list instead
            # drops whatever is not on that list -- refDepth and dZ among
            # them, which the well model then defaults silently.
            Wi = dict(network_well)
            for name in ("type", "val", "status"):
                if name in fine:
                    Wi[name] = fine[name]
            if "WI" in fine:
                Wi["WI"] = float(np.sum(np.atleast_1d(
                    np.asarray(fine["WI"], dtype=float))))

            merged = dict(fine)
            merged.update(Wi)
            ctrl["W"][i] = merged

    return {
        "model": gps_net.model,
        "schedule": new_schedule,
        "state0": gps_net.state0,
    }
