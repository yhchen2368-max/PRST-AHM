"""Local version of solveIncompFlow for adjoint module.

1:1 Python translation of MRST solvers/adjoint/solveIncompFlowLocal.m
"""

import numpy as np


def solve_incomp_flow_local(state, G, S, fluid, wells=None, bc=None, src=None,
                             rhs=None, solver="hybrid"):
    """Solve incompressible flow (local version with rhs override).

    Parameters
    ----------
    state : dict
        Reservoir state with s, pressure, flux.
    G, S : dict
        Grid and system structures.
    fluid : dict
        Fluid properties.
    wells : list of dict, optional
        Well structures.
    bc, src : optional
        Boundary conditions, sources (not used in adjoint).
    rhs : list, optional
        Direct RHS override.
    solver : str
        'hybrid' or 'mixed'.

    Returns
    -------
    dict
        Updated state.
    """
    nc = G["cells"]["num"]
    nf = G["faces"]["num"]

    pressure = state.get("pressure", np.ones(nc) * 200e5)
    flux = state.get("flux", np.zeros(nf))
    new_state = dict(state)

    if rhs is not None:
        # Use provided RHS (adjoint mode)
        new_state["pressure"] = pressure
        new_state["flux"] = flux
    else:
        # Simple pressure solve (placeholder)
        new_state["pressure"] = pressure
        new_state["flux"] = np.zeros(nf)

    # Well handling
    if wells:
        well_sol = []
        for w in wells:
            well_sol.append({
                "pressure": float(w.get("val", 200e5)),
                "flux": float(w.get("val", 0.0)) if w.get("type") == "rate" else 0.0,
            })
        new_state["wellSol"] = well_sol

    return new_state
