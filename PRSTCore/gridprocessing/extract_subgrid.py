"""Python port of MRST's ``extractSubgrid.m`` (mrst-2026a/core/gridprocessing).

Keeps only the given cells (the inverse framing of
:func:`PRSTCore.gridprocessing.remove_cells.remove_cells`, which specifies
what to *drop*): faces touching a kept cell survive, with the other side
becoming a boundary (-1) if it pointed at a cell outside the selection.
Attaches ``cells['global']``/``faces['global']``/``nodes['global']`` --
each new entity's index in the original grid ``G`` -- matching MRST's own
``extractSubgrid`` output fields.
"""

from __future__ import annotations

import numpy as np

from .remove_cells import remove_cells


def extract_subgrid(G: dict, cells) -> dict:
    """Port of MRST ``extractSubgrid.m``: keep only ``cells`` (array of
    0-based indices, or a boolean mask of length ``G['cells']['num']``).

    Returns the sub-grid ``H``, with ``H['cells']['global']`` /
    ``H['faces']['global']`` / ``H['nodes']['global']`` recording each
    surviving entity's index in ``G``.
    """
    nc = G["cells"]["num"]
    cells = np.asarray(cells)
    if cells.dtype == bool:
        if cells.size != nc:
            raise ValueError("Boolean cell mask must have length G['cells']['num']")
        keep = cells
    else:
        keep = np.zeros(nc, dtype=bool)
        keep[cells.astype(np.int64)] = True

    H, cellmap, facemap, nodemap = remove_cells(G, ~keep)

    H = dict(H)
    H["cells"] = dict(H["cells"])
    H["faces"] = dict(H["faces"])
    H["nodes"] = dict(H["nodes"])
    H["cells"]["global"] = cellmap
    H["faces"]["global"] = facemap
    H["nodes"]["global"] = nodemap
    H["type"] = list(G.get("type", [])) + ["extractSubgrid"]
    return H
