import numpy as np


def _vertcat_if_present(sol, fn):
    out = [w.get(fn, 0.0) for w in sol["wellSol"]]
    return np.asarray(out, dtype=float)


def _expand_to_full(v, status, status_obs, set_to_zero=False):
    v_full = np.zeros_like(status, dtype=float)
    v_full[status] = v
    v_obs_full = np.zeros_like(status_obs, dtype=float)
    v_obs_full[status_obs] = v
    if set_to_zero:
        ix = status != status_obs
        v_full[ix] = 0.0
        v_obs_full[ix] = 0.0
    return v_full, v_obs_full


def _get_weights(qWs, qOs, bhp, opt):
    ww = opt.get("WaterRateWeight")
    wo = opt.get("OilRateWeight")
    wp = opt.get("BHPWeight")
    rw = np.sum(np.abs(qWs) + np.abs(qOs))
    if ww is None:
        ww = 0.0 if np.sum(np.abs(qWs)) == 0 else 1.0 / rw
    if wo is None:
        wo = 0.0 if np.sum(np.abs(qOs)) == 0 else 1.0 / rw
    if wp is None:
        dp = np.max(bhp) - np.min(bhp) if bhp.size else 0.0
        wp = 0.0 if dp == 0.0 else 1.0 / dp
    return ww, wo, wp


def match_observed_ow(model, states, schedule, observed, compute_partials=False,
                      tstep=None, state=None, weighting=None, from_states=True,
                      sign_change_penalty_factor=0, match_only_producers=False,
                      mismatch_sum=True, accumulate_wells=None,
                      accumulate_types=None):
    opt = {
        "WaterRateWeight": None,
        "OilRateWeight": None,
        "BHPWeight": None,
        "ComputePartials": compute_partials,
        "tStep": tstep,
        "state": state,
        "from_states": from_states,
        "signChangePenaltyFactor": sign_change_penalty_factor,
        "matchOnlyProducers": match_only_producers,
        "mismatchSum": mismatch_sum,
        "accumulateWells": accumulate_wells,
        "accumulateTypes": accumulate_types,
    }
    if isinstance(weighting, dict):
        opt.update(weighting)
    dts = np.asarray(schedule["step"]["val"], dtype=float)
    tsteps = np.arange(len(dts)) if opt["tStep"] is None else np.atleast_1d(opt["tStep"]) - 1
    obj = []
    for t in tsteps:
        sol_obs = observed[t]
        status_obs = np.asarray([w["status"] for w in sol_obs["wellSol"]], dtype=bool)
        qWs_obs = _vertcat_if_present(sol_obs, "qWs")[status_obs]
        qOs_obs = _vertcat_if_present(sol_obs, "qOs")[status_obs]
        bhp_obs = _vertcat_if_present(sol_obs, "bhp")[status_obs]
        sol = states[t]
        status = np.asarray([w["status"] for w in sol["wellSol"]], dtype=bool)
        qWs = _vertcat_if_present(sol, "qWs")[status]
        qOs = _vertcat_if_present(sol, "qOs")[status]
        bhp = _vertcat_if_present(sol, "bhp")[status]
        if not np.all(status) or not np.all(status_obs):
            qWs, qWs_obs = _expand_to_full(qWs, status, status_obs, False)
            qOs, qOs_obs = _expand_to_full(qOs, status, status_obs, False)
            bhp, bhp_obs = _expand_to_full(bhp, status, status_obs, True)
        ww, wo, wp = _get_weights(qWs_obs, qOs_obs, bhp_obs, opt)
        dt = dts[t]
        match_cases = np.ones_like(status, dtype=bool) if not opt["matchOnlyProducers"] else (
            np.asarray([w["sign"] for w in sol["wellSol"]], dtype=float)[status] < 0
        )
        if opt["mismatchSum"]:
            denom = np.sum(match_cases) or 1
            obj.append((dt / (np.sum(dts) * denom)) * np.sum(
                (ww * match_cases * (qWs - qWs_obs)) ** 2 +
                (wo * match_cases * (qOs - qOs_obs)) ** 2 +
                (wp * match_cases * (bhp - bhp_obs)) ** 2))
        else:
            fac = dt / (np.sum(dts) * np.sum(match_cases) if np.sum(match_cases) != 0 else np.sum(dts))
            mm = [fac * (ww * match_cases * (qWs - qWs_obs)) ** 2,
                  fac * (wo * match_cases * (qOs - qOs_obs)) ** 2,
                  fac * (wp * match_cases * (bhp - bhp_obs)) ** 2]
            if accumulate_types is not None:
                tmp = [np.zeros_like(mm[0]) for _ in range(max(accumulate_types))]
                for k, ttype in enumerate(accumulate_types):
                    if ttype > 0:
                        tmp[ttype - 1] += mm[k]
                mm = tmp
            if accumulate_wells is not None:
                raise NotImplementedError("accumulate_wells is not supported")
            obj.append(np.concatenate([np.atleast_1d(x) for x in mm]))
    return obj
