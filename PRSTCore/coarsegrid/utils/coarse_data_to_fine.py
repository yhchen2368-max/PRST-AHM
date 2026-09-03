"""Convert coarse data to fine grid representation.

1:1 Python translation of MRST multiscale/coarsegrid/utils/coarseDataToFine.m
"""

import numpy as np


def coarse_data_to_fine(CG, data):
    """Map coarse-grid data back to fine grid.

    Parameters
    ----------
    CG : dict
        Coarse grid with partition.
    data : ndarray or dict
        Coarse data (first dimension = CG.cells.num).

    Returns
    -------
    ndarray or dict
        Fine-grid data.
    """
    if data is None:
        return None

    if isinstance(data, np.ndarray):
        if data.shape[0] == CG["cells"]["num"]:
            return data[CG["partition"] - 1]
        return data

    if isinstance(data, dict):
        result = {}
        for key, val in data.items():
            result[key] = coarse_data_to_fine(CG, val)
        return result

    if isinstance(data, list):
        return [coarse_data_to_fine(CG, d) for d in data]

    return data
