"""MRST ``estimateRTD.m`` counterpart."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .helpers import get_field
from .structures import Struct


def estimate_rtd(
    pv: Iterable[float],
    D: Any,
    WP: Any,
    *,
    injectorIx: Iterable[int] | None = None,
    producerIx: Iterable[int] | None = None,
    nbins: int = 100,
    match_allocation: bool = True,
) -> Struct:
    """Estimate well-pair residence-time distributions from TOF/tracers."""
    pv = np.asarray(pv, dtype=float).ravel()
    inj = np.asarray(get_field(D, "inj"), dtype=int).ravel()
    prod = np.asarray(get_field(D, "prod"), dtype=int).ravel()
    iix = _local_indices(injectorIx, len(inj))
    pix = _local_indices(producerIx, len(prod))
    nreg = len(iix) * len(pix)

    dist = Struct(
        pairIx=np.full((nreg, 2), np.nan, dtype=float),
        t=np.full((int(nbins), nreg), np.nan, dtype=float),
        volumes=np.full(nreg, np.nan, dtype=float),
        allocations=np.full(nreg, np.nan, dtype=float),
        values=np.full((int(nbins), nreg), np.nan, dtype=float),
        injectorFlux=np.full(len(iix), np.nan, dtype=float),
        producerFlux=np.full(len(pix), np.nan, dtype=float),
        creator="estimateRTD",
    )

    for k, ii in enumerate(iix):
        dist.injectorFlux[k] = float(np.sum(_alloc(get_field(WP, "inj")[ii].alloc)))
    for k, pp in enumerate(pix):
        dist.producerFlux[k] = float(np.sum(_alloc(get_field(WP, "prod")[pp].alloc)))

    itracer = np.asarray(get_field(D, "itracer"), dtype=float)
    ptracer = np.asarray(get_field(D, "ptracer"), dtype=float)
    tof = np.asarray(get_field(D, "tof"), dtype=float)
    itof_all = get_field(D, "itof", None)
    ptof_all = get_field(D, "ptof", None)
    pair_ix = np.asarray(get_field(WP, "pairIx"), dtype=int)
    vols = np.asarray(get_field(WP, "vols"), dtype=float).ravel()

    out_col = 0
    for ii in iix:
        for pp in pix:
            dist.pairIx[out_col, :] = [ii, pp]
            match = np.flatnonzero((pair_ix[:, 0] == ii) & (pair_ix[:, 1] == pp))
            if match.size:
                dist.volumes[out_col] = vols[match[0]]
            dist.allocations[out_col] = float(np.sum(_alloc(get_field(WP, "inj")[ii].alloc)[:, pp]))
            q_inj = max(float(np.sum(_alloc(get_field(WP, "inj")[ii].alloc))), np.finfo(float).eps)

            cp = itracer[:, ii] * ptracer[:, pp]
            sub = cp > 1e-5
            if not np.any(sub):
                out_col += 1
                continue

            if itof_all is not None:
                itof = np.asarray(itof_all, dtype=float)[sub, ii]
            else:
                itof = tof[sub, 0]
            if ptof_all is not None:
                ptof = np.asarray(ptof_all, dtype=float)[sub, pp]
            else:
                ptof = tof[sub, 1]
            total_tof = itof + ptof
            pvs = pv[sub] * cp[sub]
            finite = np.isfinite(total_tof) & (total_tof > 0.0) & np.isfinite(pvs) & (pvs > 0.0)
            if not np.any(finite):
                out_col += 1
                continue
            ts = total_tof[finite]
            flux = pvs[finite] / ts
            logt = np.log10(ts)
            if np.allclose(logt.max(), logt.min()):
                edges_log = np.linspace(logt.min(), logt.max() + 0.01, int(nbins) + 1)
            else:
                edges_log = np.linspace(logt.min(), logt.max(), int(nbins) + 1)
            edges = np.power(10.0, edges_log)
            bins = np.clip(np.searchsorted(edges, ts, side="right") - 1, 0, int(nbins) - 1)
            binflux = np.bincount(bins, weights=flux, minlength=int(nbins))
            totflux = float(np.sum(binflux))
            unitbinflux = np.zeros(int(nbins), dtype=float)
            widths = np.diff(edges)
            valid_width = widths > 0.0
            unitbinflux[valid_width] = binflux[valid_width] / widths[valid_width]
            if match_allocation and np.isfinite(dist.allocations[out_col]) and abs(dist.allocations[out_col]) > 0.0:
                fac = totflux / dist.allocations[out_col]
                if abs(fac) > np.finfo(float).eps:
                    unitbinflux = unitbinflux / fac
            dist.t[:, out_col] = edges[:-1]
            dist.values[:, out_col] = unitbinflux / q_inj
            out_col += 1
    return dist


def _local_indices(values: Iterable[int] | None, count: int) -> np.ndarray:
    if values is None:
        return np.arange(count, dtype=int)
    arr = np.asarray(list(values), dtype=int).ravel()
    if arr.size and np.min(arr) >= 1 and np.max(arr) <= count:
        # Convenient MRST-style caller support.
        arr = arr - 1
    return arr


def _alloc(value) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape((-1, 1))
    return arr


estimateRTD = estimate_rtd

