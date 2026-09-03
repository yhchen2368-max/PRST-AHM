import numpy as np


def _vertcat_if_present(sol, fn):
    out = [w.get(fn, 0.0) for w in sol["wellSol"]]
    return np.asarray(out, dtype=float)


def npv_ow(model, states, schedule, oil_price=1.0, water_production_cost=0.1,
           water_injection_cost=0.1, discount_factor=0.0,
           compute_partials=False, tstep=None, state=None,
           from_states=True, sign_change_penalty_factor=0, **kwargs):
    dts = np.asarray(schedule["step"]["val"], dtype=float)
    obj = []
    time = 0.0
    tsteps = np.arange(len(dts)) if tstep is None else np.atleast_1d(tstep) - 1
    for t in tsteps:
        state = states[t]
        status = np.asarray([w.get("status", True) for w in state["wellSol"]], dtype=bool)
        qWs = _vertcat_if_present(state, "qWs")[status]
        qOs = _vertcat_if_present(state, "qOs")[status]
        signs = np.asarray([w.get("sign", 0) for w in state["wellSol"]], dtype=float)
        injectors = signs[status] > 0
        injecting = (qWs + qOs) > 0
        producing = ~injecting
        dt_step = dts[t]
        time += dt_step
        discount = (1 + discount_factor) ** (-time)
        if sign_change_penalty_factor == 0 or not np.any((injectors & ~injecting) | (~injectors & injecting)):
            obj.append(discount * dt_step * np.sum(
                -oil_price * qOs +
                (water_production_cost * producing - water_injection_cost * injectors) * qWs
            ))
        else:
            sgn_ch = (injectors & ~injecting) | (~injectors & injecting)
            obj.append(discount * dt_step * np.sum(
                -oil_price * (~injectors) * qOs +
                (water_production_cost * (~injectors) - water_injection_cost * injectors) * qWs -
                (oil_price * sign_change_penalty_factor * sgn_ch) * np.abs(qWs + qOs)
            ))
    return obj
