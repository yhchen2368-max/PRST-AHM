"""Add adjoint well fields to well structure.

1:1 Python translation of MRST solvers/adjoint/addAdjointWellFields.m
"""

import numpy as np


def add_adjoint_well_fields(CG, W, overlap_well=0, overlap_block=0):
    """Add coarse-grid fields to well structures for adjoint.

    Parameters
    ----------
    CG : dict
        Coarse grid with 'partition'.
    W : list of dict
        Well structures.
    overlap_well, overlap_block : int

    Returns
    -------
    list of dict
        Updated wells.
    """
    new_W = []
    for w in W:
        wc = dict(w)
        cells = w.get("cells", [0])
        if isinstance(cells, (list, np.ndarray)) and len(cells) > 0:
            partition = np.asarray(CG["partition"])
            coarse_cells = np.unique(partition[np.clip(np.array(cells) - 1, 0, len(partition) - 1)])
            wc["coarseCells"] = coarse_cells.tolist()
        else:
            wc["coarseCells"] = []

        wc["CS"] = {
            "overlap": overlap_block,
            "wellOverlap": overlap_well,
        }
        new_W.append(wc)
    return new_W
