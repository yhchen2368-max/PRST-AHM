"""MRST ``preprocessorGUI/utils/computePressureAndDiagnostics.m`` counterpart."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ...utils.compute_tof_and_tracer import compute_tof_and_tracer
from ...utils.compute_well_pairs import compute_well_pairs
from ...utils.helpers import (
    EPS,
    connection_arrays,
    ensure_wellsol_flux,
    get_field,
    normalize_cell_indices,
    num_cells,
    set_field,
    well_cells,
    well_name,
    well_sign,
    well_status,
    well_value,
)
from ...utils.structures import DiagnosticsStruct, TOFDiagnostics, WellPairDiagnostics


def compute_pressure_and_diagnostics(
    model: Any,
    *,
    D: TOFDiagnostics | None = None,
    state: dict[str, Any] | None = None,
    WP: WellPairDiagnostics | None = None,
    wellCommunication: np.ndarray | None = None,
    state0: dict[str, Any] | None = None,
    wells: list[Any] | None = None,
    ellipticSolver: Any | None = None,
    maxTOF: float | None = None,
    computeWellTOFs: bool = True,
    processCycles: bool = True,
    firstArrival: bool = True,
) -> tuple[dict[str, Any], DiagnosticsStruct]:
    """Compute pressure/flux and corresponding flow diagnostics.

    PRSTCore does not currently expose MRST's full ``incompTPFA`` stack, so
    this function uses an MRST-equivalent linear TPFA pressure solve whenever
    a state is not supplied.  If a state already has ``pressure`` and ``flux``
    fields, it is reused directly and only diagnostics are computed.
    """
    G = _model_field(model, "G")
    rock = _model_field(model, "rock", {})
    if wells is None:
        wells = []
    if not wells:
        raise ValueError("Empty well input, cannot compute diagnostics")

    if state is None:
        state = _solve_stationary_pressure(model, G, wells, state0=state0, solver=ellipticSolver)
    else:
        state = dict(state)
        if get_field(state, "flux", None) is None:
            connections, flux = connection_arrays(G, state, model=model)
            state["flux"] = flux
            state["_diagnostics_connections"] = connections
        else:
            # Keep explicit face ordering if it is recoverable.  This makes
            # later TOF assembly independent of grid-face storage details.
            try:
                connections, flux = connection_arrays(G, state, model=model)
                state["_diagnostics_connections"] = connections
                state["flux"] = flux
            except ValueError:
                pass
        _complete_well_solutions(state, G, wells)

    if D is None:
        D = compute_tof_and_tracer(
            state,
            G,
            rock,
            wells=wells,
            maxTOF=maxTOF,
            computeWellTOFs=computeWellTOFs,
            processCycles=processCycles,
            firstArrival=firstArrival,
            model=model,
        )
    if WP is None:
        WP = compute_well_pairs(state, G, rock, wells, D)

    diagnostics = DiagnosticsStruct(
        D=D,
        WP=WP,
        wellCommunication=_well_communication(WP) if wellCommunication is None else np.asarray(wellCommunication, dtype=float),
    )
    return state, diagnostics


def _model_field(model: Any, name: str, default: Any | None = None) -> Any:
    if isinstance(model, dict):
        return model.get(name, default)
    return getattr(model, name, default)


def _model_operators(model: Any) -> dict[str, Any]:
    ops = _model_field(model, "operators", {})
    return ops if isinstance(ops, dict) else {}


def _solve_stationary_pressure(
    model: Any,
    G: dict[str, Any],
    wells: list[Any],
    *,
    state0: dict[str, Any] | None = None,
    solver: Any | None = None,
) -> dict[str, Any]:
    nc = num_cells(G)
    ops = _model_operators(model)
    N = _connection_matrix(G, ops)
    T = _transmissibility_vector(ops, N.shape[0])
    q = _well_source_vector(G, wells)
    pref = _reference_pressure(state0, nc)

    if N.shape[0] == 0:
        pressure = np.full(nc, pref, dtype=float)
        flux = np.zeros(0, dtype=float)
    else:
        L = _tpfa_laplacian(N, T, nc).tolil()
        rhs = q.copy()
        # Pure-Neumann pressure equation needs one gauge condition.  This is
        # equivalent to MRST's pressure-reference handling for diagnostics.
        L[0, :] = 0.0
        L[0, 0] = 1.0
        rhs[0] = pref
        if solver is None:
            pressure = np.asarray(spla.spsolve(L.tocsc(), rhs), dtype=float).ravel()
        else:
            pressure = np.asarray(solver(L.tocsc(), rhs), dtype=float).ravel()
        pressure[np.isnan(pressure)] = pref
        flux = T * (pressure[N[:, 0]] - pressure[N[:, 1]])

    state: dict[str, Any] = {
        "pressure": pressure,
        "flux": flux,
        "_diagnostics_connections": N,
    }
    _complete_well_solutions(state, G, wells)
    return state


def _connection_matrix(G: dict[str, Any], ops: dict[str, Any]) -> np.ndarray:
    nc = num_cells(G)
    if "N" in ops:
        N = np.asarray(ops["N"], dtype=int)
        if N.ndim == 2 and N.shape[1] >= 2:
            out = N[:, :2]
            if out.size and np.min(out) >= 1 and np.max(out) <= nc:
                out = out - 1
            return out.astype(int)
    neighbors = np.asarray(G.get("faces", {}).get("neighbors", []), dtype=int)
    if neighbors.ndim == 2 and neighbors.shape[1] >= 2:
        if neighbors.size and np.min(neighbors) >= 1 and np.max(neighbors) <= nc:
            neighbors = neighbors - 1
        internal = np.all((neighbors[:, :2] >= 0) & (neighbors[:, :2] < nc), axis=1)
        return neighbors[internal, :2].astype(int)
    return np.zeros((0, 2), dtype=int)


def _transmissibility_vector(ops: dict[str, Any], nface: int) -> np.ndarray:
    for key in ("T_all", "T"):
        if key in ops:
            T = np.asarray(ops[key], dtype=float).ravel()
            if T.size >= nface:
                return T[:nface]
    return np.ones(nface, dtype=float)


def _tpfa_laplacian(N: np.ndarray, T: np.ndarray, nc: int) -> sp.csr_matrix:
    i = N[:, 0]
    j = N[:, 1]
    data = np.concatenate([T, T, -T, -T])
    rows = np.concatenate([i, j, i, j])
    cols = np.concatenate([i, j, j, i])
    return sp.csr_matrix((data, (rows, cols)), shape=(nc, nc))


def _well_source_vector(G: dict[str, Any], wells: list[Any]) -> np.ndarray:
    nc = num_cells(G)
    q = np.zeros(nc, dtype=float)
    for well in wells:
        if not well_status(well):
            continue
        cells = well_cells(well, nc)
        if cells.size == 0:
            continue
        value = abs(well_value(well))
        sign = well_sign(well)
        if sign == 0.0:
            sign = 1.0 if well_value(well) >= 0.0 else -1.0
        np.add.at(q, cells, sign * value / cells.size)
    # The gauge-fixed solve tolerates slight imbalance, but diagnostics behave
    # better when rates are balanced.  Put any residual on active producers.
    residual = float(np.sum(q))
    if abs(residual) > EPS:
        prod_cells = []
        for well in wells:
            if well_status(well) and well_sign(well) < 0.0:
                prod_cells.extend(well_cells(well, nc).tolist())
        target = np.asarray(prod_cells if prod_cells else list(range(nc)), dtype=int)
        if target.size:
            q[target] -= residual / target.size
    return q


def _reference_pressure(state0: dict[str, Any] | None, nc: int) -> float:
    if state0 is not None and get_field(state0, "pressure", None) is not None:
        p0 = np.asarray(get_field(state0, "pressure"), dtype=float).ravel()
        finite = p0[np.isfinite(p0)]
        if finite.size:
            return float(np.mean(finite))
    return 200.0e5


def _complete_well_solutions(state: dict[str, Any], G: dict[str, Any], wells: list[Any]) -> None:
    well_sols = ensure_wellsol_flux(state, wells, G)
    pressure = np.asarray(get_field(state, "pressure", np.zeros(num_cells(G))), dtype=float).ravel()
    nc = num_cells(G)
    for index, well in enumerate(wells):
        cells = well_cells(well, nc)
        ws = dict(well_sols[index])
        flux = np.asarray(get_field(ws, "flux", []), dtype=float).ravel()
        if flux.size == 1 and cells.size > 1:
            flux = np.full(cells.size, float(flux[0]) / cells.size)
        elif flux.size == 0:
            sign = well_sign(well)
            value = abs(well_value(well))
            if sign == 0.0:
                sign = 1.0 if value >= 0.0 else -1.0
            flux = np.full(max(cells.size, 1), sign * value / max(cells.size, 1), dtype=float)
        ws["flux"] = flux[: max(cells.size, 1)]
        cell_pressure = float(np.mean(pressure[cells])) if cells.size else 0.0
        ws_pressure = get_field(ws, "pressure", None)
        if ws_pressure is None or not np.all(np.isfinite(np.asarray(ws_pressure, dtype=float))):
            ws["pressure"] = cell_pressure
        ws_bhp = get_field(ws, "bhp", None)
        if ws_bhp is None or (
            np.asarray(ws_bhp, dtype=float).size
            and np.allclose(np.asarray(ws_bhp, dtype=float), 0.0)
            and abs(cell_pressure) > EPS
        ):
            ws["bhp"] = cell_pressure
        ws.setdefault("name", well_name(well, index))
        ws.setdefault("status", well_status(well))
        ws.setdefault("sign", well_sign(well))
        well_sols[index] = ws
    set_field(state, "wellSol", well_sols)


def _well_communication(WP: WellPairDiagnostics) -> np.ndarray:
    rows = []
    for allocation in WP.inj:
        alloc = np.asarray(allocation.alloc, dtype=float)
        if alloc.ndim == 1:
            alloc = alloc.reshape((-1, 1))
        rows.append(np.sum(alloc, axis=0))
    if not rows:
        nprod = len(WP.prod)
        return np.zeros((0, nprod), dtype=float)
    return np.vstack(rows)


computePressureAndDiagnostics = compute_pressure_and_diagnostics
