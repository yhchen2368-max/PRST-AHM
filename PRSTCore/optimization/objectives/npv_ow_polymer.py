"""Net present value for oil-water-polymer systems.

1:1 Python translation of MRST NPVOWPolymer.m
"""

import numpy as np


def npv_ow_polymer(model, states, schedule, oil_price=1.0, water_production_cost=0.1,
                   water_injection_cost=0.1, polymer_injection_cost=0.1,
                   discount_factor=0.0, compute_partials=False, tstep=None,
                   state=None, from_states=True, sign_change_penalty_factor=0):
    """Compute NPV for an oil-water-polymer schedule.

    Parameters
    ----------
    model : dict
        Simulation model.
    states : list of dict
        States at each step.
    schedule : dict
        Schedule.
    oil_price, water_production_cost, water_injection_cost : float
        Economic parameters.
    polymer_injection_cost : float
        Cost per unit of polymer injected.
    discount_factor : float
        Discount factor.
    compute_partials : bool
        Whether to compute partial derivatives.
    tstep : int, optional
        Specific time step.
    state : dict, optional
        State for AD evaluation.
    from_states : bool
        Whether to get AD state from states.
    sign_change_penalty_factor : float
        Penalty for well sign changes.

    Returns
    -------
    list of ndarray
        NPV per step.
    """
    dts = np.asarray(schedule["step"]["val"], dtype=float)

    if tstep is None:
        tsteps = np.arange(len(dts))
        time = 0.0
    else:
        tsteps = np.atleast_1d(tstep) - 1
        time = np.sum(dts[:tsteps[0]]) if tsteps[0] > 0 else 0.0

    obj = []
    for step_idx, t in enumerate(tsteps):
        state_s = states[t]
        status = np.array([w.get("status", True) for w in state_s["wellSol"]], dtype=bool)
        qWs = np.array([w.get("qWs", 0.0) for w in state_s["wellSol"]])[status]
        qOs = np.array([w.get("qOs", 0.0) for w in state_s["wellSol"]])[status]
        cp = np.array([w.get("cWPoly", 0.0) for w in state_s["wellSol"]])[status]
        injectors = np.array([w.get("sign", 0) for w in state_s["wellSol"]])[status] > 0
        injecting = (qWs + qOs) > 0
        producing = ~injecting

        dt = dts[t]
        time += dt
        discount = (1 + discount_factor) ** (-time)

        penalty = sign_change_penalty_factor
        nw = len(qWs)

        if penalty == 0:
            obj.append(discount * dt * np.sum(
                -oil_price * qOs
                + (water_production_cost * producing - water_injection_cost * injecting) * qWs
            ))
            obj[-1] -= discount * dt * polymer_injection_cost * np.sum(injecting * cp * qWs)
        else:
            sgn_ch = (injectors & ~injecting) | (~injectors & injecting)
            obj.append(discount * dt * np.sum(
                -oil_price * (~injectors) * qOs
                + (water_production_cost * (~injectors) - water_injection_cost * injectors) * qWs
                - (oil_price * penalty * sgn_ch) * np.abs(qWs + qOs)
            ))
            obj[-1] -= discount * dt * polymer_injection_cost * np.sum(injecting * cp * qWs)

    return obj
