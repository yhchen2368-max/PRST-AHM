"""Recovery objective function (water volume at last step).

1:1 Python translation of MRST solvers/adjoint/objectives/recovery.m
"""

import numpy as np


def recovery(G, S, W, rock, fluid, sim_res, schedule=None, controls=None):
    """Objective: water volume at last time step.

    Parameters
    ----------
    G, S, W, rock, fluid : standard structures
    sim_res : list of dict
    schedule, controls : optional

    Returns
    -------
    dict
        Objective with 'val' and 'partials'.
    """
    num_steps = len(sim_res)
    porv = G["cells"]["volumes"] * rock["poro"]
    val = float(np.sum(porv * np.asarray(sim_res[num_steps - 1]["resSol"].get("s", 0)).ravel()))

    compute_partials = schedule is not None
    partials = []
    if compute_partials:
        ncf = G["cells"]["faces"].shape[0] if "faces" in G["cells"] else G["cells"]["num"]
        nc = G["cells"]["num"]
        nf = G["faces"]["num"]

        for k in range(num_steps):
            p = {"v": np.zeros(ncf), "p": np.zeros(nc), "pi": np.zeros(nf),
                 "q_w": np.zeros(sum(len(w.get("cells", [1])) for w in W)),
                 "s": np.zeros(nc), "u": np.zeros(len(controls["well"]) if controls else 1)}

            if k == num_steps - 1:
                p["s"] = porv.copy()
            partials.append(p)

    return {"val": val, "partials": partials}
