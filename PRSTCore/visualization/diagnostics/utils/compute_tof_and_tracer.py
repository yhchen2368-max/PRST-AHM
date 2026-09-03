"""MRST ``computeTOFandTracer.m`` counterpart."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .compute_time_of_flight import compute_time_of_flight
from .helpers import (
    ensure_wellsol_flux,
    get_field,
    num_cells,
    total_well_fluxes,
    well_cells,
)
from .structures import TOFDiagnostics


def compute_tof_and_tracer(
    state: dict[str, Any],
    G: dict[str, Any],
    rock: dict[str, Any],
    *,
    wells: list[Any] | None = None,
    src: Any | None = None,
    bc: Any | None = None,
    tracerWells: Iterable[bool] | None = None,
    solver: Any | None = None,
    maxTOF: float | None = None,
    processCycles: bool = False,
    computeWellTOFs: bool = False,
    firstArrival: bool = False,
    splitBoundary: bool = False,
    partitionBoundary: Any | None = None,
    model: Any | None = None,
) -> TOFDiagnostics:
    """Compute TOF, reverse TOF, and stationary well tracers.

    The field layout mirrors MRST:

    ``D.inj`` / ``D.prod``
        Well numbers with positive / non-positive total well flux.  Python
        uses zero-based well indices.
    ``D.tof``
        ``ncell x 2`` array: forward and reverse time-of-flight.
    ``D.itracer`` / ``D.ptracer``
        Steady tracer concentration from injectors / producers.
    ``D.ipart`` / ``D.ppart``
        MRST-style one-based tracer partition labels, with ``0`` for cells
        not reached by any tracer.
    """
    if firstArrival:
        computeWellTOFs = True
    if src is not None:
        raise NotImplementedError("Source terms are not implemented yet")
    if bc is not None:
        # Boundary support in MRST is a sizeable branch.  For PRSTCore's
        # network use case we currently need wells only, so fail explicitly.
        raise NotImplementedError("Boundary-condition diagnostics are not implemented yet")
    if splitBoundary or partitionBoundary is not None:
        raise NotImplementedError("Boundary partition diagnostics require bc support")

    if wells is None:
        wells = []
    if not wells:
        raise AssertionError("Wells or boundary structure are required for computeTOFandTracer")

    nc = num_cells(G)
    poro = np.asarray(get_field(rock, "poro", np.ones(nc)), dtype=float).ravel()
    if poro.size != nc:
        raise AssertionError("rock['poro'] must contain one porosity per cell")
    if np.any(poro <= 0.0):
        raise AssertionError("Rock porosities must be positive numbers")

    state = dict(state)
    ensure_wellsol_flux(state, wells, G)

    if tracerWells is None:
        tracer_mask = np.ones(len(wells), dtype=bool)
    else:
        tracer_mask = np.asarray(list(tracerWells), dtype=bool).ravel()
        if tracer_mask.size != len(wells):
            raise ValueError("tracerWells must have one entry per well")

    wflux = total_well_fluxes(state, wells, G)
    iwells = wflux > 0.0
    inj = np.flatnonzero(iwells & tracer_mask)
    prod = np.flatnonzero((~iwells) & tracer_mask)

    total_abs_flux = _total_absolute_wellsol_flux(state)
    if total_abs_flux == 0.0:
        return TOFDiagnostics(
            inj=inj,
            prod=prod,
            tof=np.full((nc, 2), np.inf, dtype=float),
            itracer=np.full((nc, inj.size), np.nan, dtype=float),
            ipart=np.full(nc, np.nan, dtype=float),
            ptracer=np.full((nc, prod.size), np.nan, dtype=float),
            ppart=np.full(nc, np.nan, dtype=float),
        )

    inj_cells = [well_cells(wells[int(i)], nc) for i in inj]
    prod_cells = [well_cells(wells[int(i)], nc) for i in prod]

    tinj, _, _ = compute_time_of_flight(
        state,
        G,
        rock,
        wells=wells,
        tracer=inj_cells,
        solver=solver,
        maxTOF=maxTOF,
        processCycles=processCycles,
        computeWellTOFs=computeWellTOFs,
        firstArrival=firstArrival,
        model=model,
    )
    tof = np.full((nc, 2), np.nan, dtype=float)
    tof[:, 0] = tinj[:, 0]
    itracer = tinj[:, 1 : inj.size + 1] if inj.size else np.zeros((nc, 0), dtype=float)
    ipart = _tracer_partition(itracer)

    itof = None
    ifa = None
    if computeWellTOFs:
        cur = inj.size + 1
        itof = tinj[:, cur : cur + inj.size] if inj.size else np.zeros((nc, 0), dtype=float)
        if firstArrival:
            cur += inj.size
            ifa = tinj[:, cur : cur + inj.size] if inj.size else np.zeros((nc, 0), dtype=float)

    tprod, _, _ = compute_time_of_flight(
        state,
        G,
        rock,
        wells=wells,
        tracer=prod_cells,
        reverse=True,
        solver=solver,
        maxTOF=maxTOF,
        processCycles=processCycles,
        computeWellTOFs=computeWellTOFs,
        firstArrival=firstArrival,
        model=model,
    )
    tof[:, 1] = tprod[:, 0]
    ptracer = tprod[:, 1 : prod.size + 1] if prod.size else np.zeros((nc, 0), dtype=float)
    ppart = _tracer_partition(ptracer)

    ptof = None
    pfa = None
    if computeWellTOFs:
        cur = prod.size + 1
        ptof = tprod[:, cur : cur + prod.size] if prod.size else np.zeros((nc, 0), dtype=float)
        if firstArrival:
            cur += prod.size
            pfa = tprod[:, cur : cur + prod.size] if prod.size else np.zeros((nc, 0), dtype=float)

    return TOFDiagnostics(
        inj=inj,
        prod=prod,
        tof=tof,
        itracer=itracer,
        ipart=ipart,
        ptracer=ptracer,
        ppart=ppart,
        itof=itof,
        ptof=ptof,
        ifa=ifa,
        pfa=pfa,
    )


def _total_absolute_wellsol_flux(state: dict[str, Any]) -> float:
    total = 0.0
    for ws in get_field(state, "wellSol", []) or []:
        flux = get_field(ws, "flux", [])
        total += float(np.sum(np.abs(np.asarray(flux, dtype=float).ravel())))
    return total


def _tracer_partition(tracer: np.ndarray) -> np.ndarray:
    tracer = np.asarray(tracer, dtype=float)
    if tracer.size == 0 or tracer.shape[1] == 0:
        return np.zeros(tracer.shape[0], dtype=int)
    safe = np.nan_to_num(tracer, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.max(safe, axis=1)
    part = np.argmax(safe, axis=1).astype(int) + 1
    part[values == 0.0] = 0
    return part


computeTOFandTracer = compute_tof_and_tracer

