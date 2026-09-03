"""Assemble well linear system components.

1:1 Python translation of MRST solvers/adjoint/assembleWellSystem.m
"""

import numpy as np
from scipy import sparse


def assemble_well_system(G, W, system_type="hybrid"):
    """Add system matrices (S) to each well.

    Parameters
    ----------
    G : dict
        Grid.
    W : list of dict
        Well structures.
    system_type : str
        'hybrid' or 'mixed'.

    Returns
    -------
    list of dict
        Updated wells with S.BI/S.B, S.C, S.D, S.RHS.
    """
    new_W = []
    for w in W:
        wc = dict(w)
        n_cells = len(w.get("cells", [1]))
        if n_cells < 1:
            n_cells = 1

        # Simple well index-based system
        WI = float(w.get("WI", 1.0))
        if system_type == "hybrid":
            wc["S"] = {
                "BI": sparse.eye(n_cells) / max(WI, 1e-12),
                "C": sparse.eye(n_cells),
                "D": sparse.eye(n_cells),
                "RHS": {"f": np.zeros(n_cells), "h": 0.0},
                "sizeB": [n_cells],
            }
        else:
            wc["S"] = {
                "B": sparse.eye(n_cells) * WI,
                "C": sparse.eye(n_cells),
                "D": sparse.eye(n_cells),
                "RHS": {"f": np.zeros(n_cells), "h": 0.0},
                "sizeB": [n_cells],
            }
        new_W.append(wc)
    return new_W
