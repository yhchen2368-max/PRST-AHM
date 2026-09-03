"""MRST ``expandWellCompletions.m`` counterpart."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .helpers import get_field, well_cells
from .validate_state_for_diagnostics import validate_state_for_diagnostics


def expand_well_completions(state: dict[str, Any], W: list[Any], expansion, split=None):
    """Replace selected wells by pseudo-wells for completion diagnostics."""
    nw = len(W)
    bins = _completion_bins(W, expansion)
    if split is None:
        split = np.asarray([int(np.max(b)) if len(b) else 1 for b in bins], dtype=int)
    else:
        split = np.asarray(split, dtype=int).ravel()

    state = validate_state_for_diagnostics(state)
    expanded_wells = []
    expanded_sols = []
    well_sols = get_field(state, "wellSol", []) or []
    for iw, well in enumerate(W):
        b = np.asarray(bins[iw], dtype=int).ravel()
        nsplit = int(split[iw]) if iw < split.size else 1
        if nsplit <= 1:
            expanded_wells.append(deepcopy(well))
            expanded_sols.append(deepcopy(well_sols[iw]) if iw < len(well_sols) else {})
            continue
        ws = deepcopy(well_sols[iw]) if iw < len(well_sols) else {}
        for part in range(1, nsplit + 1):
            loc = b == part
            new_well = deepcopy(well)
            for field in ("cells", "dir", "WI", "dZ"):
                value = get_field(well, field, None)
                if value is not None:
                    arr = np.asarray(value)
                    new_well[field] = arr[loc].tolist() if arr.ndim == 1 else arr[loc, :].tolist()
            new_well["name"] = f"{get_field(well, 'name', f'W{iw + 1}')}:{part}"
            new_ws = deepcopy(ws)
            flux = get_field(new_ws, "flux", None)
            if flux is not None:
                new_ws["flux"] = np.asarray(flux, dtype=float).ravel()[loc]
            for field in ("cqs", "cdp", "cstatus"):
                value = get_field(new_ws, field, None)
                if value is not None:
                    new_ws[field] = np.asarray(value)[loc]
            if "cqs" in new_ws:
                cqs = np.asarray(new_ws["cqs"], dtype=float)
                new_ws["flux"] = np.sum(cqs, axis=1)
                for idx, name in enumerate(("qWs", "qOs", "qGs")):
                    if cqs.ndim == 2 and idx < cqs.shape[1]:
                        new_ws[name] = float(np.sum(cqs[:, idx]))
            new_ws.setdefault("name", new_well["name"])
            expanded_wells.append(new_well)
            expanded_sols.append(new_ws)
    expanded_state = dict(state)
    expanded_state["wellSol"] = expanded_sols
    return expanded_state, expanded_wells


def _completion_bins(W, expansion):
    nw = len(W)
    if isinstance(expansion, (list, tuple)) and len(expansion) == nw and not np.asarray(expansion, dtype=object).ndim == 2:
        return [np.asarray(e, dtype=int).ravel() for e in expansion]
    expansion = np.asarray(expansion, dtype=int)
    split_num = np.ones(nw, dtype=int)
    if expansion.size:
        arr = expansion.reshape((-1, 2))
        # Accept both MRST one-based well numbers and Python zero-based.
        wells = arr[:, 0].copy()
        if wells.size and np.min(wells) >= 1 and np.max(wells) <= nw:
            wells -= 1
        split_num[wells] = arr[:, 1]
    out = []
    for iw, well in enumerate(W):
        cells = np.asarray(get_field(well, "cells", []))
        M = cells.size
        B = max(int(split_num[iw]), 1)
        if B == 1 or M == 0:
            out.append(np.ones(M, dtype=int))
            continue
        b = np.arange(M, dtype=float)
        L = np.floor(M / B)
        R = np.mod(M, B)
        if L == 0:
            bins = np.arange(M) + 1
        else:
            bins = np.maximum(np.floor(b / (L + 1)), np.floor((b - R) / L)).astype(int) + 1
        out.append(bins)
    return out


expandWellCompletions = expand_well_completions

