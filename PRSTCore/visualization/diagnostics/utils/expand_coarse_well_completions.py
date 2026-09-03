"""MRST ``expandCoarseWellCompletions.m`` counterpart."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .helpers import get_field


def expand_coarse_well_completions(xc: dict[str, Any], WC: list[Any], Wdf: list[Any], p):
    """Create pseudo-wells in a coarse model following fine diagnostic wells."""
    p = np.asarray(p, dtype=int).ravel()
    if p.size and np.min(p) >= 1:
        p0 = p - 1
    else:
        p0 = p
    expanded = []
    expanded_sols = []
    well_sols = get_field(xc, "wellSol", []) or []
    for fine_well in Wdf:
        fine_cells = np.asarray(get_field(fine_well, "cells", []), dtype=int).ravel()
        if fine_cells.size and np.max(fine_cells) >= p0.size and np.min(fine_cells) >= 1:
            fine_cells = fine_cells - 1
        coarse_blocks = set(p0[fine_cells].tolist()) if fine_cells.size else set()
        for j, coarse_well in enumerate(WC):
            cells = np.asarray(get_field(coarse_well, "cells", []), dtype=int).ravel()
            c0 = cells - 1 if cells.size and np.min(cells) >= 1 else cells
            loc = np.asarray([int(c) in coarse_blocks for c in c0], dtype=bool)
            if not np.any(loc):
                continue
            new_well = deepcopy(coarse_well)
            new_well["name"] = get_field(fine_well, "name", get_field(coarse_well, "name", f"W{j + 1}"))
            for field in ("cells", "r", "dir", "WI", "dZ"):
                value = get_field(coarse_well, field, None)
                if value is None:
                    continue
                arr = np.asarray(value)
                if field == "r" and arr.size == 1:
                    new_well[field] = arr.item()
                else:
                    new_well[field] = arr[loc].tolist() if arr.ndim == 1 else arr[loc, :].tolist()
            expanded.append(new_well)
            ws = deepcopy(well_sols[j]) if j < len(well_sols) else {}
            flux = get_field(ws, "flux", None)
            if flux is not None:
                ws["flux"] = np.asarray(flux, dtype=float).ravel()[loc]
            expanded_sols.append(ws)
    out_state = dict(xc)
    out_state["wellSol"] = expanded_sols
    return out_state, expanded


expandCoarseWellCompletions = expand_coarse_well_completions

