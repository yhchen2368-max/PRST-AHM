"""MRST ``computeTOFandTracerAverage.m`` counterpart."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .compute_tof_and_tracer import compute_tof_and_tracer
from .structures import Struct


def compute_tof_and_tracer_average(
    state: Iterable[dict[str, Any]],
    G: dict[str, Any],
    rock: dict[str, Any],
    *,
    wells: list[Any] | list[list[Any]] | None = None,
    dt: Iterable[float] | None = None,
    max_tof: float | None = None,
    min_tof: float | None = None,
    diagnostics: Iterable[Any] | None = None,
    **kwargs,
) -> Struct:
    """Execute ``computeTOFandTracer`` for multiple states and average.

    This keeps MRST's field layout and zero-based Python well indices.
    """
    states = list(state)
    nstep = len(states)
    if nstep == 0:
        return Struct(isvalid=False)

    if dt is None:
        weights = np.full(nstep, 1.0 / nstep, dtype=float)
    else:
        weights = np.asarray(list(dt), dtype=float).ravel()
        if weights.size != nstep:
            raise ValueError("dt must have one value per state")
    precomputed = list(diagnostics) if diagnostics is not None else None
    average = None
    compute_subsets = min_tof is not None and max_tof is not None

    for index, (current_state, weight) in enumerate(zip(states, weights, strict=False)):
        if precomputed is not None:
            D_new = precomputed[index]
        else:
            current_wells = _wells_for_step(wells, index)
            D_new = compute_tof_and_tracer(current_state, G, rock, wells=current_wells, **kwargs)

        if np.all(np.isinf(np.asarray(D_new.tof, dtype=float))):
            continue

        if average is None:
            nwells = _num_wells_from_diagnostics(D_new, wells)
            average = Struct(
                tof=np.zeros_like(D_new.tof, dtype=float),
                itracer=np.zeros_like(D_new.itracer, dtype=float),
                ptracer=np.zeros_like(D_new.ptracer, dtype=float),
                ipart=np.zeros_like(D_new.ipart, dtype=int),
                ppart=np.zeros_like(D_new.ppart, dtype=int),
                inj_avg=np.zeros(nwells, dtype=float),
                prod_avg=np.zeros(nwells, dtype=float),
            )
            if compute_subsets:
                average.isubset = np.zeros_like(D_new.ipart, dtype=float)
                average.psubset = np.zeros_like(D_new.ppart, dtype=float)

        average.tof += weight * np.asarray(D_new.tof, dtype=float)
        average.itracer += weight * np.asarray(D_new.itracer, dtype=float)
        average.ptracer += weight * np.asarray(D_new.ptracer, dtype=float)
        average.inj_avg[np.asarray(D_new.inj, dtype=int)] += weight
        average.prod_avg[np.asarray(D_new.prod, dtype=int)] += weight

        if compute_subsets:
            psubset = (average.tof[:, 1] >= float(min_tof)) & (average.tof[:, 1] <= float(max_tof))
            isubset = (average.tof[:, 0] >= float(min_tof)) & (average.tof[:, 0] <= float(max_tof))
            average.isubset += weight * isubset
            average.psubset += weight * psubset

    if average is None:
        return Struct(isvalid=False)

    average.ipart = _partition(average.itracer)
    average.ppart = _partition(average.ptracer)
    injectors = average.inj_avg > average.prod_avg
    average.inj = np.flatnonzero(injectors)
    average.prod = np.flatnonzero(~injectors)
    average.isvalid = True
    return average


def _wells_for_step(wells, index):
    if wells is None:
        return []
    if len(wells) == 0:
        return []
    first = wells[0]
    if isinstance(first, (list, tuple)):
        return wells[index]
    return wells


def _num_wells_from_diagnostics(D, wells) -> int:
    if wells is not None and len(wells) and not isinstance(wells[0], (list, tuple)):
        return len(wells)
    if wells is not None and len(wells) and isinstance(wells[0], (list, tuple)):
        return max(len(w) for w in wells)
    all_wells = np.concatenate([np.asarray(D.inj, dtype=int), np.asarray(D.prod, dtype=int)])
    return int(np.max(all_wells) + 1) if all_wells.size else 0


def _partition(tracer: np.ndarray) -> np.ndarray:
    if tracer.size == 0 or tracer.shape[1] == 0:
        return np.zeros(tracer.shape[0], dtype=int)
    values = np.max(tracer, axis=1)
    part = np.argmax(tracer, axis=1).astype(int) + 1
    part[values == 0.0] = 0
    return part


computeTOFandTracerAverage = compute_tof_and_tracer_average

