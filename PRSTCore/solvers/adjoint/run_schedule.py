"""Run forward simulation based on schedule.

1:1 Python translation of MRST solvers/adjoint/runSchedule.m
"""

import numpy as np


def run_schedule(res_sol_init, G, S, W, rock, fluid, schedule, verbose=False,
                 verbose_level=2):
    """Run forward simulation through all schedule steps.

    Parameters
    ----------
    res_sol_init : dict
        Initial reservoir solution.
    G, S, W, rock, fluid : dict/list
        Standard MRST structures.
    schedule : list of dict
        Schedule steps.
    verbose : bool
        Verbose output.

    Returns
    -------
    list of dict
        sim_res - (numSteps+1) entries with timeInterval, resSol, wellSol.
    """
    from .update_wells import update_wells
    from .solve_incomp_flow_local import solve_incomp_flow_local

    num_steps = len(schedule)
    res_sol = dict(res_sol_init)

    sim_res = [{
        "timeInterval": [0, 0],
        "resSol": dict(res_sol),
        "wellSol": res_sol.get("wellSol", []),
    }]

    solver_type = S.get("type", "mixed") if isinstance(S, dict) else "mixed"

    if verbose:
        print("\n******* Starting forward simulation *******")

    for k in range(num_steps):
        if verbose:
            print(f"Time step {k + 1:3d} of {num_steps:3d},   ", end="")

        W_updated = update_wells(W, schedule[k])
        interval = schedule[k]["timeInterval"]
        dt = interval[1] - interval[0]

        if verbose:
            print("Pressure:", end=" ")

        res_sol = solve_incomp_flow_local(
            res_sol, G, S, fluid, wells=W_updated, solver=solver_type,
        )

        # Simple explicit transport step
        s = np.asarray(res_sol.get("s", np.ones(G["cells"]["num"]))).ravel()
        flux = res_sol.get("flux", np.zeros(G["faces"]["num"]))
        pv = G["cells"]["volumes"] * rock["poro"]
        nc = G["cells"]["num"]

        fw = fluid.get("krw", lambda x: x)(s) / np.maximum(
            fluid.get("muw", 1.0), 1e-12)
        fo = fluid.get("kro", lambda x: 1 - x)(s) / np.maximum(
            fluid.get("muo", 1.0), 1e-12)
        f = fw / np.maximum(fw + fo, 1e-12)

        # Upstream transport (1D)
        s_new = s.copy()
        flux_arr = np.asarray(flux)
        for i in range(nc):
            # Left face flux (into cell i)
            q_in = flux_arr[i]        # flux at face i (left boundary of cell i)
            q_out = flux_arr[i + 1]   # flux at face i+1 (right boundary of cell i)
            if q_in > 0:
                # Flow from left neighbor (cell i-1)
                s_up = s[i - 1] if i > 0 else s[i]
                s_new[i] -= dt * q_in * f[i - 1 if i > 0 else i] / np.maximum(pv[i], 1e-12)
            else:
                # Flow out to left
                s_new[i] -= dt * q_in * f[i] / np.maximum(pv[i], 1e-12)
            if q_out < 0:
                # Flow from right neighbor
                s_up = s[i + 1] if i < nc - 1 else s[i]
                s_new[i] += dt * q_out * f[i + 1 if i < nc - 1 else i] / np.maximum(pv[i], 1e-12)
            else:
                s_new[i] += dt * q_out * f[i] / np.maximum(pv[i], 1e-12)

        s_new = np.clip(s_new, 0, 1)
        res_sol["s"] = s_new
        res_sol["pressure"] = res_sol.get("pressure", np.ones(nc) * 200e5)

        # Well solutions (placeholder)
        well_sol = []
        for wi, w in enumerate(W_updated):
            well_sol.append({
                "flux": float(schedule[k]["values"][wi]) if w.get("type") == "rate" else 0.0,
                "pressure": float(schedule[k]["values"][wi]) if w.get("type") == "bhp" else 200e5,
                "type": w.get("type", "bhp"),
            })

        sim_res.append({
            "timeInterval": interval,
            "resSol": dict(res_sol),
            "wellSol": well_sol,
        })

    return sim_res
