"""Net present value for black-oil systems.

1:1 Python translation of MRST NPVBlackOil.m
"""

import numpy as np


def npv_black_oil(model, states, schedule, oil_price=1.0, gas_price=0.1,
                  gas_injection_cost=0.1, water_production_cost=0.1,
                  water_injection_cost=0.1, discount_factor=0.0,
                  compute_partials=False, tstep=None):
    """Compute NPV for a black-oil schedule.

    Parameters
    ----------
    model : dict
        Simulation model.
    states : list of dict
        States at each step.
    schedule : dict
        Schedule.
    oil_price, gas_price, water_production_cost, water_injection_cost : float
        Economic parameters.
    gas_injection_cost : float
        Cost of gas injection.
    discount_factor : float
        Discount factor.
    compute_partials : bool
        Whether to compute partial derivatives.
    tstep : int, optional
        Specific time step.

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
        time = np.sum(dts[:tstep - 1]) if tstep > 1 else 0.0

    obj = []
    for step_idx, t in enumerate(tsteps):
        state = states[t]
        status = np.array([w.get("status", True) for w in state["wellSol"]], dtype=bool)
        qWs = np.array([w.get("qWs", 0.0) for w in state["wellSol"]])[status]
        qOs = np.array([w.get("qOs", 0.0) for w in state["wellSol"]])[status]
        qGs = np.array([w.get("qGs", 0.0) for w in state["wellSol"]])[status]
        inj = np.array([w.get("sign", 0) for w in state["wellSol"]])[status] > 0

        dt = dts[t] if step_idx == 0 else dts[tsteps[step_idx]]
        time += dt
        discount = (1 + discount_factor) ** (-time)

        nw = len(qWs)
        obj.append(discount * dt * np.sum(
            -oil_price * qOs
            + gas_price * (~inj) * qGs
            - gas_injection_cost * inj * qGs
            - water_production_cost * (~inj) * qWs
            - water_injection_cost * inj * qWs
        ))

    return obj
