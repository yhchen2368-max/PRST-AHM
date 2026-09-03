"""MRST ``selectTOFRegion.m`` counterpart."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .helpers import get_field


def select_tof_region(
    D: Any,
    max_tof: float,
    min_tof: float,
    *,
    drain_wells: Iterable[int] | None = None,
    flood_wells: Iterable[int] | None = None,
    set_op: str = "union",
    near_well_max_tof: float = 0.0,
    psubset=None,
    isubset=None,
    tracer_threshold: float = 0.05,
) -> np.ndarray:
    tof = np.asarray(get_field(D, "tof"), dtype=float)
    psubset = (tof[:, 1] >= min_tof) & (tof[:, 1] <= max_tof) if psubset is None else np.asarray(psubset, dtype=bool)
    isubset = (tof[:, 0] >= min_tof) & (tof[:, 0] <= max_tof) if isubset is None else np.asarray(isubset, dtype=bool)
    ptracer = np.asarray(get_field(D, "ptracer"), dtype=float)
    itracer = np.asarray(get_field(D, "itracer"), dtype=float)
    if drain_wells is None and flood_wells is None:
        psubs = np.asarray(get_field(D, "ppart"), dtype=float) >= 0
        isubs = np.asarray(get_field(D, "ipart"), dtype=float) >= 0
    else:
        d_ix = _local(drain_wells, ptracer.shape[1])
        f_ix = _local(flood_wells, itracer.shape[1])
        psubs = np.sum(ptracer[:, d_ix], axis=1) if d_ix.size else np.zeros(tof.shape[0])
        isubs = np.sum(itracer[:, f_ix], axis=1) if f_ix.size else np.zeros(tof.shape[0])

    op = set_op.lower()
    if op == "union":
        selection = (isubset & (isubs > tracer_threshold)) | (psubset & (psubs > tracer_threshold))
    elif op == "intersection":
        selection = (isubset & psubset) & (isubs * psubs > tracer_threshold)
    elif op == "flood":
        selection = isubset & (isubs > tracer_threshold)
    elif op == "drain":
        selection = psubset & (psubs > tracer_threshold)
    else:
        selection = np.zeros(tof.shape[0], dtype=bool)
    if near_well_max_tof > 0.0:
        near = ((tof[:, 0] <= near_well_max_tof) & (isubs > 0)) | ((tof[:, 1] <= near_well_max_tof) & (psubs > 0))
        selection = selection | near
    return selection


def _local(values, count):
    if values is None:
        return np.arange(count, dtype=int)
    arr = np.asarray(list(values), dtype=int).ravel()
    if arr.size and np.min(arr) >= 1 and np.max(arr) <= count:
        arr = arr - 1
    return arr


selectTOFRegion = select_tof_region

