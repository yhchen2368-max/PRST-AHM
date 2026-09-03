"""Shared helpers for MRST-style diagnostics."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


EPS = np.finfo(float).eps


def get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def set_field(obj: Any, name: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)


def num_cells(G: dict[str, Any]) -> int:
    return int(G["cells"]["num"])


def pore_volume(G: dict[str, Any], rock: dict[str, Any] | None = None) -> np.ndarray:
    nc = num_cells(G)
    if rock is not None and "poro" in rock:
        poro = np.asarray(rock["poro"], dtype=float).ravel()
        if poro.size == nc:
            volumes = np.asarray(G["cells"].get("volumes", np.ones(nc)), dtype=float).ravel()
            return volumes * poro
    if "pore_volume" in G:
        pv = np.asarray(G["pore_volume"], dtype=float).ravel()
        if pv.size == nc:
            return pv.copy()
    cells = G.get("cells", {})
    if "poreVolume" in cells:
        pv = np.asarray(cells["poreVolume"], dtype=float).ravel()
        if pv.size == nc:
            return pv.copy()
    if "volumes" in cells:
        volumes = np.asarray(cells["volumes"], dtype=float).ravel()
        if volumes.size == nc:
            if rock is not None and "poro" in rock:
                poro = np.asarray(rock["poro"], dtype=float).ravel()
                if poro.size == nc:
                    return volumes * poro
            return volumes.copy()
    return np.ones(nc, dtype=float)


def normalize_cell_indices(cells: Any, nc: int, *, one_based: bool = False) -> np.ndarray:
    arr = np.asarray(cells if cells is not None else [], dtype=int).ravel()
    if arr.size == 0:
        return arr
    # PRSTCore/Python data is normally zero-based.  MRST imported structures
    # may be one-based; shift only when explicitly requested or when an index
    # equals/exceeds ``nc`` and therefore cannot be a valid Python cell index.
    if one_based or (np.min(arr) >= 1 and np.max(arr) >= nc):
        return arr - 1
    return arr


def well_cells(well: Any, nc: int) -> np.ndarray:
    index_base = get_field(well, "index_base", get_field(well, "cell_index_base", None))
    one_based = bool(get_field(well, "cells_one_based", False))
    if isinstance(index_base, str):
        one_based = index_base.lower().replace("-", "_") in {"one", "one_based", "1", "matlab"}
    elif index_base is not None:
        one_based = int(index_base) == 1
    return normalize_cell_indices(get_field(well, "cells", []), nc, one_based=one_based)


def well_name(well: Any, index: int) -> str:
    return str(get_field(well, "name", f"W{index + 1}"))


def well_status(well: Any) -> bool:
    return bool(get_field(well, "status", True))


def well_sign(well: Any) -> float:
    return float(get_field(well, "sign", 0.0))


def well_value(well: Any) -> float:
    value = get_field(well, "val", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def completion_depths(G: dict[str, Any], well: Any) -> np.ndarray:
    cells = well_cells(well, num_cells(G))
    dz = np.asarray(get_field(well, "dZ", np.zeros(cells.size)), dtype=float).ravel()
    ref = float(get_field(well, "refDepth", 0.0) or 0.0)
    if dz.size == cells.size:
        return dz + ref
    centroids = np.asarray(G["cells"].get("centroids", np.zeros((num_cells(G), 3))), dtype=float)
    if centroids.ndim == 2 and centroids.shape[1] >= 3 and cells.size:
        return centroids[cells, 2]
    return np.full(cells.size, ref, dtype=float)


def connection_arrays(
    G: dict[str, Any],
    state: dict[str, Any],
    *,
    model: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return zero-based internal connections and face fluxes.

    Flux is positive from connection column 0 to column 1, matching MRST's
    face-neighbor convention for internal faces.
    """
    flux = get_field(state, "flux", None)
    explicit = get_field(state, "_diagnostics_connections", None)
    if explicit is not None and flux is not None:
        return np.asarray(explicit, dtype=int), np.asarray(flux, dtype=float).ravel()

    if flux is not None:
        flux_arr = np.asarray(flux, dtype=float).ravel()
        neighbors = np.asarray(G.get("faces", {}).get("neighbors", []), dtype=int)
        if neighbors.ndim == 2 and neighbors.shape[0] == flux_arr.size:
            internal = np.all(neighbors >= 0, axis=1)
            return neighbors[internal, :2].astype(int), flux_arr[internal]

    ops = getattr(model, "operators", None) if model is not None else None
    if ops is None and isinstance(model, dict):
        ops = model.get("operators")
    if isinstance(ops, dict) and "N" in ops:
        N = np.asarray(ops["N"], dtype=int)
        if N.ndim == 2 and N.shape[1] >= 2:
            if np.min(N) >= 1:
                N = N[:, :2] - 1
            else:
                N = N[:, :2]
            if "T" in ops:
                T = np.asarray(ops["T"], dtype=float).ravel()
                pressure = np.asarray(get_field(state, "pressure", np.zeros(num_cells(G))), dtype=float).ravel()
                nface = min(N.shape[0], T.size)
                N = N[:nface]
                flux_arr = T[:nface] * (pressure[N[:, 0]] - pressure[N[:, 1]])
                return N.astype(int), flux_arr

    raise ValueError("Diagnostics require state['flux'] or model.operators['N']/['T']")


def total_well_fluxes(state: dict[str, Any], W: Iterable[Any], G: dict[str, Any]) -> np.ndarray:
    well_sols = get_field(state, "wellSol", []) or []
    result = []
    nc = num_cells(G)
    for index, well in enumerate(W):
        flux = None
        if index < len(well_sols):
            flux = get_field(well_sols[index], "flux", None)
        if flux is None:
            cells = well_cells(well, nc)
            sign = well_sign(well)
            value = abs(well_value(well))
            if sign == 0.0:
                sign = 1.0 if value >= 0.0 else -1.0
            nperf = max(cells.size, 1)
            flux = np.full(nperf, sign * value / nperf, dtype=float)
        flux_arr = np.asarray(flux, dtype=float).ravel()
        if not well_status(well):
            flux_arr = np.zeros_like(flux_arr)
        result.append(float(np.sum(flux_arr)))
    return np.asarray(result, dtype=float)


def ensure_wellsol_flux(state: dict[str, Any], W: list[Any], G: dict[str, Any]) -> list[dict[str, Any]]:
    well_sols = list(get_field(state, "wellSol", []) or [])
    nc = num_cells(G)
    while len(well_sols) < len(W):
        well_sols.append({})
    for index, well in enumerate(W):
        ws = dict(well_sols[index])
        if "flux" not in ws:
            cells = well_cells(well, nc)
            sign = well_sign(well)
            value = abs(well_value(well))
            if sign == 0.0:
                sign = 1.0 if value >= 0.0 else -1.0
            ws["flux"] = np.full(max(cells.size, 1), sign * value / max(cells.size, 1), dtype=float)
        ws.setdefault("name", well_name(well, index))
        ws.setdefault("status", well_status(well))
        ws.setdefault("sign", well_sign(well))
        ws.setdefault("bhp", 0.0)
        well_sols[index] = ws
    state["wellSol"] = well_sols
    return well_sols
