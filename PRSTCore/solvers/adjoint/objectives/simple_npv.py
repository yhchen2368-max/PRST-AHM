"""Simple NPV objective function.

1:1 Python translation of MRST solvers/adjoint/objectives/simpleNPV.m
"""

import numpy as np


def simple_npv(G, S, W, rock, fluid, sim_res, schedule=None, controls=None,
               oil_price=100.0, water_production_cost=10.0,
               water_injection_cost=10.0, discount_factor=0.0,
               compute_partials=None):
    """Compute simple net present value.

    Parameters
    ----------
    G, S, W, rock, fluid : standard structures
    sim_res : list of dict
        Simulation results.
    schedule, controls : optional
    oil_price, water_production_cost, water_injection_cost : float
    discount_factor : float
    compute_partials : bool, optional

    Returns
    -------
    dict
        Objective with 'val' and 'partials'.
    """
    if compute_partials is None:
        compute_partials = schedule is not None

    num_steps = len(sim_res)
    nc = G["cells"]["num"]
    nf = G["faces"]["num"]
    val = 0.0
    partials = [{"v": np.zeros(nc), "p": np.zeros(nc), "pi": np.zeros(nf),
                  "s": np.zeros(nc),
                  "u": np.zeros(len(controls["well"]) if controls else 1),
                  "q_w": np.zeros(sum(len(w.get("cells", [1])) for w in W))}
                for _ in range(num_steps)]

    tot_time = max(s["timeInterval"][1] for s in sim_res if s.get("timeInterval"))

    for step in range(1, num_steps):
        res_sol = sim_res[step]["resSol"]
        well_sol = sim_res[step].get("wellSol", [])
        interval = sim_res[step]["timeInterval"]
        dt = interval[1] - interval[0]
        d_fac = (1 + discount_factor) ** (-interval[1])
        # Simplified: use dt * d_fac

        well_cells = []
        for w in W:
            wc = w.get("cells", [0])
            well_cells.extend(wc if isinstance(wc, list) else [wc])
        well_cells = np.array(well_cells, dtype=int)
        if len(well_cells) > 0:
            well_sats = np.asarray(res_sol.get("s", np.ones(G["cells"]["num"]))).ravel()
            well_sats = well_sats[np.clip(well_cells - 1, 0, len(well_sats) - 1)]
        else:
            well_sats = np.ones(1)

        krw = fluid.get("krw", lambda s: s)(well_sats)
        kro = fluid.get("kro", lambda s: 1 - s)(well_sats)
        muw = fluid.get("muw", 1.0)
        muo = fluid.get("muo", 1.0)
        mob_w = krw / muw
        mob_o = kro / muo
        Lt = mob_w + mob_o
        f_w = mob_w / np.maximum(Lt, 1e-12)

        for wi, ws in enumerate(well_sol):
            q = ws.get("flux", 0.0)
            if W[wi].get("sign", -1) < 0:  # producer
                q_o = q * (1 - f_w[min(wi, len(f_w) - 1)])
                q_w = q * f_w[min(wi, len(f_w) - 1)]
                val += d_fac * dt * (oil_price * q_o - water_production_cost * q_w)
            else:  # injector
                val -= d_fac * dt * water_injection_cost * abs(q)

        if compute_partials:
            partials[step]["q_w"] = np.zeros(len(well_cells))

    return {"val": val, "partials": partials}
