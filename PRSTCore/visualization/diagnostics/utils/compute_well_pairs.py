"""MRST ``computeWellPairs.m`` counterpart."""

from __future__ import annotations

from typing import Any

import numpy as np

from .helpers import (
    completion_depths,
    ensure_wellsol_flux,
    get_field,
    num_cells,
    pore_volume,
    well_cells,
    well_name,
)
from .structures import TOFDiagnostics, WellAllocation, WellPairDiagnostics


def compute_well_pairs(
    state: dict[str, Any],
    G: dict[str, Any],
    rock: dict[str, Any],
    W: list[Any],
    D: TOFDiagnostics,
) -> WellPairDiagnostics:
    """Compute pore volumes and flux allocations for injector/producer pairs.

    This follows the algebra and pair ordering from MRST:
    ``vols = D.itracer' * (poreVolume(G, rock) .* D.ptracer)`` and
    ``pairIx`` is column-major over producer groups with injector varying
    fastest.  Pair indices are zero-based inside ``D.inj`` / ``D.prod``.
    """
    state = dict(state)
    ensure_wellsol_flux(state, W, G)

    nc = num_cells(G)
    inj = np.asarray(D.inj, dtype=int).ravel()
    prod = np.asarray(D.prod, dtype=int).ravel()
    ni = inj.size
    npd = prod.size

    itracer = np.asarray(D.itracer, dtype=float).reshape((nc, ni)) if ni else np.zeros((nc, 0))
    ptracer = np.asarray(D.ptracer, dtype=float).reshape((nc, npd)) if npd else np.zeros((nc, 0))
    pv = pore_volume(G, rock).reshape((-1, 1))
    vols_matrix = itracer.T @ (pv * ptracer) if ni and npd else np.zeros((ni, npd), dtype=float)

    pairs: list[str] = []
    vols: list[float] = []
    pair_ix: list[list[int]] = []
    for prod_col, prod_well in enumerate(prod):
        for inj_col, inj_well in enumerate(inj):
            pairs.append(f"{well_name(W[int(inj_well)], int(inj_well))}, {well_name(W[int(prod_well)], int(prod_well))}")
            vols.append(float(vols_matrix[inj_col, prod_col]))
            pair_ix.append([inj_col, prod_col])

    inj_alloc = []
    for inj_col, well_index in enumerate(inj):
        well = W[int(well_index)]
        qik = _completion_fluxes(state, int(well_index), well, nc)
        if qik.size == 0:
            inj_alloc.append(
                WellAllocation(
                    alloc=np.zeros((0, npd), dtype=float),
                    ralloc=np.zeros(0, dtype=float),
                    z=np.zeros(0, dtype=float),
                    name=well_name(well, int(well_index)),
                )
            )
            continue
        cells = well_cells(well, nc)[: qik.size]
        cj = ptracer[cells, :] if npd else np.zeros((qik.size, 0), dtype=float)
        alloc = qik[:, None] * cj
        inj_alloc.append(
            WellAllocation(
                alloc=alloc,
                ralloc=qik - np.sum(alloc, axis=1),
                z=completion_depths(G, well)[: qik.size],
                name=well_name(well, int(well_index)),
            )
        )

    prod_alloc = []
    for prod_col, well_index in enumerate(prod):
        well = W[int(well_index)]
        qjk = _completion_fluxes(state, int(well_index), well, nc)
        if qjk.size == 0:
            prod_alloc.append(
                WellAllocation(
                    alloc=np.zeros((0, ni), dtype=float),
                    ralloc=np.zeros(0, dtype=float),
                    z=np.zeros(0, dtype=float),
                    name=well_name(well, int(well_index)),
                )
            )
            continue
        cells = well_cells(well, nc)[: qjk.size]
        ci = itracer[cells, :] if ni else np.zeros((qjk.size, 0), dtype=float)
        alloc = qjk[:, None] * ci
        prod_alloc.append(
            WellAllocation(
                alloc=alloc,
                ralloc=qjk - np.sum(alloc, axis=1),
                z=completion_depths(G, well)[: qjk.size],
                name=well_name(well, int(well_index)),
            )
        )

    return WellPairDiagnostics(
        pairs=pairs,
        pairIx=np.asarray(pair_ix, dtype=int).reshape((-1, 2)) if pair_ix else np.zeros((0, 2), dtype=int),
        vols=np.asarray(vols, dtype=float),
        inj=inj_alloc,
        prod=prod_alloc,
    )


def _completion_fluxes(state: dict[str, Any], well_index: int, well: Any, nc: int) -> np.ndarray:
    well_sols = get_field(state, "wellSol", []) or []
    flux = None
    if well_index < len(well_sols):
        flux = get_field(well_sols[well_index], "flux", None)
    if flux is None:
        return np.zeros(0, dtype=float)
    out = np.asarray(flux, dtype=float)
    if out.ndim > 1:
        out = np.sum(out, axis=1)
    out = out.ravel()
    cells = well_cells(well, nc)
    if out.size == 1 and cells.size > 1:
        out = np.full(cells.size, float(out[0]) / cells.size)
    return out[: cells.size]


computeWellPairs = compute_well_pairs

