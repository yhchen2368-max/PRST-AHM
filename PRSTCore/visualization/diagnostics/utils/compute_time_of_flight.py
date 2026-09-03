"""MRST ``computeTimeOfFlight.m`` counterpart."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components, dijkstra

from .helpers import (
    EPS,
    connection_arrays,
    ensure_wellsol_flux,
    get_field,
    normalize_cell_indices,
    num_cells,
    pore_volume,
    well_cells,
)


def compute_time_of_flight(
    state: dict[str, Any],
    G: dict[str, Any],
    rock: dict[str, Any],
    *,
    wells: list[Any] | None = None,
    src: Any | None = None,
    bc: Any | None = None,
    reverse: bool = False,
    allowInf: bool = False,
    maxTOF: float | None = None,
    tracer: Iterable[Any] | None = None,
    solver: Any | None = None,
    processCycles: bool = False,
    computeWellTOFs: bool = False,
    firstArrival: bool = False,
    model: Any | None = None,
) -> tuple[np.ndarray, sp.csr_matrix, np.ndarray]:
    """Compute time-of-flight and optional stationary tracer fields.

    Returns ``(T, A, q)`` following MRST.  ``T[:, 0]`` is whole-field TOF.
    Additional columns contain tracer concentrations, optional per-tracer
    TOFs, and optional first-arrival times.
    """
    if wells is None:
        wells = []
    if src is not None:
        raise NotImplementedError("Explicit src terms are not implemented yet")
    if bc is not None:
        raise NotImplementedError("Boundary-condition diagnostics are not implemented yet")
    if not wells:
        raise AssertionError("Must have inflow described as boundary conditions, sources, or wells")

    nc = num_cells(G)
    pv = pore_volume(G, rock).astype(float)
    state = dict(state)
    ensure_wellsol_flux(state, wells, G)
    connections, flux = connection_arrays(G, state, model=model)
    if connections.size == 0:
        raise ValueError("No internal connections available for TOF")
    if connections.ndim != 2 or connections.shape[1] < 2:
        raise ValueError("connections must be an nface x 2 array")
    connections = connections[:, :2].astype(int)
    flux = np.asarray(flux, dtype=float).ravel()[: connections.shape[0]]

    q, qb = _compute_source_term(state, G, wells)
    if reverse:
        q = -q
        qb = -qb
        flux = -flux

    qp = np.maximum(q + qb, 0.0)
    c1 = connections[:, 0]
    c2 = connections[:, 1]

    out = np.minimum(flux, 0.0)
    infl = np.maximum(flux, 0.0)
    inflow = np.zeros(nc, dtype=float)
    np.add.at(inflow, c2, infl)
    np.add.at(inflow, c1, -out)
    d = inflow + qp

    if not allowInf:
        if maxTOF is None:
            total_qp = float(np.sum(qp))
            maxTOF = np.inf if total_qp <= 0.0 else 50.0 * float(np.sum(pv)) / total_qp
        max_in = max(float(np.max(d, initial=0.0)), EPS)
        above_max = (pv / np.maximum(d, EPS * max_in)) > float(maxTOF)
        if np.any(above_max):
            d[above_max] = max_in
            pv[above_max] = max_in * float(maxTOF)
            infl[above_max[c2]] = 0.0
            out[above_max[c1]] = 0.0
    else:
        max_in = max(float(np.max(d, initial=0.0)), EPS)

    A = sp.csr_matrix(
        (
            np.concatenate([infl, -out]),
            (
                np.concatenate([c2, c1]),
                np.concatenate([c1, c2]),
            ),
        ),
        shape=(nc, nc),
    )
    A = (-A + sp.diags(d, 0, shape=(nc, nc), format="csr")).tocsr()

    if (not allowInf) and processCycles and maxTOF is not None and np.isfinite(maxTOF):
        A, pv = _threshold_connected_components(A, pv, max_in=max_in, maxTOF=float(maxTOF))

    tracer_cells = _normalise_tracer_cells(tracer, nc)
    TrRHS = np.zeros((nc, len(tracer_cells)), dtype=float)
    for col, cells in enumerate(tracer_cells):
        TrRHS[cells, col] = qp[cells]

    rhs = np.column_stack([pv, TrRHS])
    T = _solve_tof_system(A, rhs, solver=solver)

    if (not allowInf) and maxTOF is not None and np.isfinite(maxTOF):
        T[T > float(maxTOF)] = float(maxTOF)

    if computeWellTOFs and tracer_cells:
        C = T[:, 1 : len(tracer_cells) + 1]
        pvi = C * pv[:, None]
        pvi[pvi < 0.0] = 0.0
        X = _solve_tof_system(A, pvi, solver=solver)
        X[X < 0.0] = 0.0
        keep = (pvi * float(maxTOF if maxTOF is not None and np.isfinite(maxTOF) else np.inf) > X) & (
            C > np.sqrt(EPS)
        )
        fill = float(maxTOF) if maxTOF is not None and np.isfinite(maxTOF) else np.nan
        X[~keep] = fill
        X[keep] = X[keep] / C[keep]
        T = np.column_stack([T, X])

        if firstArrival:
            FA = _first_arrival(connections, flux, pv, qp, tracer_cells, reverse=reverse)
            if maxTOF is not None and np.isfinite(maxTOF):
                finite = np.isfinite(FA)
                FA[finite] = np.minimum(FA[finite], float(maxTOF))
            T = np.column_stack([T, FA])

    return T, A, q


def _compute_source_term(state: dict[str, Any], G: dict[str, Any], W: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    nc = num_cells(G)
    q = np.zeros(nc, dtype=float)
    for index, well in enumerate(W):
        cells = well_cells(well, nc)
        if cells.size == 0:
            continue
        well_sols = get_field(state, "wellSol", []) or []
        flux = None
        if index < len(well_sols):
            flux = get_field(well_sols[index], "flux", None)
        if flux is None:
            continue
        flux_arr = np.asarray(flux, dtype=float).ravel()
        if flux_arr.size == 1 and cells.size > 1:
            flux_arr = np.full(cells.size, float(flux_arr[0]) / cells.size)
        flux_arr = flux_arr[: cells.size]
        np.add.at(q, cells[: flux_arr.size], flux_arr)
    qb = np.zeros(nc, dtype=float)
    return q, qb


def _normalise_tracer_cells(tracer: Iterable[Any] | None, nc: int) -> list[np.ndarray]:
    if tracer is None:
        return []
    if isinstance(tracer, np.ndarray):
        return [normalize_cell_indices(tracer, nc)]
    cells = []
    for item in tracer:
        cells.append(normalize_cell_indices(item, nc))
    return cells


def _solve_tof_system(A: sp.csr_matrix, rhs: np.ndarray, solver: Any | None = None) -> np.ndarray:
    rhs = np.asarray(rhs, dtype=float)
    if rhs.ndim == 1:
        rhs = rhs.reshape((-1, 1))
    active = np.abs(np.asarray(A.diagonal(), dtype=float)) > EPS
    solution = np.full_like(rhs, np.nan, dtype=float)
    if not np.any(active):
        return solution
    Ared = A[active][:, active].tocsc()
    bred = rhs[active, :]
    if solver is None:
        x = spla.spsolve(Ared, bred)
    else:
        x = solver(Ared, bred)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape((-1, 1))
    x[np.isnan(x)] = 0.0
    if np.any(np.isinf(x)):
        finite = x[np.isfinite(x)]
        replacement = float(np.max(finite)) if finite.size else 0.0
        x[np.isinf(x)] = replacement
    solution[active, :] = x
    return solution


def _threshold_connected_components(
    A: sp.csr_matrix,
    pv: np.ndarray,
    *,
    max_in: float,
    maxTOF: float,
) -> tuple[sp.csr_matrix, np.ndarray]:
    coo = A.tocoo()
    off = (coo.row != coo.col) & (np.abs(coo.data) > EPS)
    graph = sp.csr_matrix((np.ones(np.count_nonzero(off)), (coo.col[off], coo.row[off])), shape=A.shape)
    _, labels = connected_components(graph, directed=True, connection="strong")
    modified = A.tolil(copy=True)
    pv_out = pv.copy()
    for label in np.unique(labels):
        cells = np.flatnonzero(labels == label)
        if cells.size <= 1:
            continue
        local = A[cells][:, cells]
        q_in = float(np.sum(local.diagonal()))
        if float(np.sum(pv_out[cells])) / max(q_in, EPS * max_in) <= maxTOF:
            continue
        for cell in cells:
            modified.rows[int(cell)] = [int(cell)]
            modified.data[int(cell)] = [float(max_in)]
        pv_out[cells] = max_in * maxTOF
    return modified.tocsr(), pv_out


def _first_arrival(
    connections: np.ndarray,
    flux: np.ndarray,
    pv: np.ndarray,
    qp: np.ndarray,
    tracer_cells: list[np.ndarray],
    *,
    reverse: bool,
) -> np.ndarray:
    nc = pv.size
    edge_from = []
    edge_to = []
    weight = []
    for (c1, c2), f in zip(connections, flux, strict=False):
        if f > EPS:
            upstream, downstream = int(c1), int(c2)
            mag = float(f)
        elif f < -EPS:
            upstream, downstream = int(c2), int(c1)
            mag = float(-f)
        else:
            continue
        if reverse:
            upstream, downstream = downstream, upstream
        edge_from.append(upstream)
        edge_to.append(downstream)
        weight.append(float(pv[downstream] / max(mag, EPS)))

    super_source_base = nc
    nsrc = len(tracer_cells)
    graph_n = nc + nsrc
    for source_idx, cells in enumerate(tracer_cells):
        src_node = super_source_base + source_idx
        for cell in cells:
            rate = max(float(qp[int(cell)]), EPS)
            edge_from.append(src_node)
            edge_to.append(int(cell))
            weight.append(float(pv[int(cell)] / rate))

    graph = sp.csr_matrix((weight, (edge_from, edge_to)), shape=(graph_n, graph_n))
    result = np.full((nc, nsrc), np.inf, dtype=float)
    for source_idx in range(nsrc):
        dist = dijkstra(graph, directed=True, indices=super_source_base + source_idx)
        result[:, source_idx] = np.asarray(dist[:nc], dtype=float)
    return result


computeTimeOfFlight = compute_time_of_flight

