"""MRST ``computeRTD.m`` counterpart."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .estimate_rtd import _alloc, _local_indices
from .helpers import get_field, normalize_cell_indices, num_cells, well_cells
from .structures import Struct


def compute_rtd(
    state: dict[str, Any],
    G: dict[str, Any],
    pv: Iterable[float],
    D: Any,
    WP: Any,
    W: list[Any],
    *,
    injectorIx: Iterable[int] | None = None,
    producerIx: Iterable[int] | None = None,
    nsteps: int = 50,
    nbase: int = 5,
    computeSaturations: bool = True,
    wbHandle: Any | None = None,
    showWaitbar: bool = False,
    reverse: bool = False,
) -> Struct:
    """Compute well-pair residence-time distributions by tracer marching.

    This is a wells-only Python translation of the MRST routine.  GUI
    waitbars are intentionally ignored; ``showWaitbar`` is accepted for API
    compatibility.
    """
    del wbHandle, showWaitbar
    pv = np.asarray(pv, dtype=float).ravel()
    if nsteps < nbase:
        nsteps = nbase

    inj_all = np.asarray(get_field(D, "inj"), dtype=int).ravel()
    prod_all = np.asarray(get_field(D, "prod"), dtype=int).ravel()
    iix = _local_indices(injectorIx, len(inj_all))
    pix = _local_indices(producerIx, len(prod_all))
    nreg = len(iix) * len(pix)
    dist = Struct(
        pairIx=np.full((nreg, 2), np.nan, dtype=float),
        t=np.zeros((0, nreg), dtype=float),
        values=np.zeros((0, nreg), dtype=float),
        volumes=np.full(nreg, np.nan, dtype=float),
        allocations=np.full(nreg, np.nan, dtype=float),
        injectorFlux=np.full(len(iix), np.nan, dtype=float),
        producerFlux=np.full(len(pix), np.nan, dtype=float),
        creator="computeRTD",
        reverse=bool(reverse),
    )

    for k, ii in enumerate(iix):
        dist.injectorFlux[k] = float(np.sum(_alloc(get_field(WP, "inj")[ii].alloc)))
    for k, pp in enumerate(pix):
        dist.producerFlux[k] = float(np.sum(_alloc(get_field(WP, "prod")[pp].alloc)))

    pair_ix = np.asarray(get_field(WP, "pairIx"), dtype=int)
    wp_vols = np.asarray(get_field(WP, "vols"), dtype=float).ravel()
    col = 0
    for ii in iix:
        for pp in pix:
            dist.pairIx[col, :] = [ii, pp]
            match = np.flatnonzero((pair_ix[:, 0] == ii) & (pair_ix[:, 1] == pp))
            if match.size:
                dist.volumes[col] = wp_vols[match[0]]
            dist.allocations[col] = float(np.sum(_alloc(get_field(WP, "inj")[ii].alloc)[:, pp]))
            col += 1

    check = dist.allocations > 1e-3 * np.nansum(dist.allocations)
    if not np.any(check):
        return dist
    t1 = float(np.nanmin(dist.volumes[check] / np.maximum(dist.allocations[check], np.finfo(float).eps)))
    pvi = float(np.nansum(dist.volumes) / max(np.nansum(dist.allocations), np.finfo(float).eps))
    itracer = np.asarray(get_field(D, "itracer"), dtype=float)
    ptracer = np.asarray(get_field(D, "ptracer"), dtype=float)
    sub = np.sum(itracer[:, iix], axis=1) > 1e-6
    itof = np.asarray(get_field(D, "itof", get_field(D, "tof")[:, [0]]), dtype=float)
    itof_sub = itof[sub][:, iix] if itof.ndim == 2 else np.asarray(get_field(D, "tof"))[sub, 0]
    itof_sub = np.nan_to_num(itof_sub, nan=0.0, posinf=0.0, neginf=0.0)
    max_tof = float(np.max(itof_sub)) if np.size(itof_sub) else t1
    t_end = min(max_tof, 100.0 * pvi)
    if t1 <= 0.0 or t_end <= 0.0:
        return dist

    nperiods = int(np.ceil(np.log(t_end * (nbase - 1) / t1 + 1.0) / np.log(nbase)))
    nperiods = max(nperiods, 1)
    dts = (t1 / nsteps) * np.power(nbase, np.arange(nperiods, dtype=float))
    nsteps_vec = np.full(nperiods, int(nsteps), dtype=int)
    used_before_last = float(np.sum(dts[:-1] * nsteps_vec[:-1])) if nperiods > 1 else 0.0
    nsteps_vec[-1] = max(1, int(np.ceil((t_end - used_before_last) / dts[-1])))
    total_steps = int(np.sum(nsteps_vec))

    sysmat, qp_well, tr, pv_sub, cix = _setup_system_components(state, G, pv, W, sub, inj_all[iix], prod_all[pix], reverse)
    if tr.size == 0:
        return dist
    ni, npd = (len(iix), len(pix)) if not reverse else (len(pix), len(iix))
    vals = [np.zeros((total_steps + 1, npd), dtype=float) for _ in range(ni)]
    sats = None
    if computeSaturations and get_field(state, "s", None) is not None:
        sat = np.asarray(get_field(state, "s"), dtype=float)
        if sat.ndim == 2 and sat.shape[0] == num_cells(G):
            if not reverse:
                pv_reg = ptracer[cix][:, pix] * pv_sub[:, None]
            else:
                pv_reg = itracer[cix][:, iix] * pv_sub[:, None]
            pvw_reg = pv_reg * sat[cix, 0][:, None]
            sats = [np.zeros_like(vals[0]) for _ in range(ni)]
            for tn in range(ni):
                num = tr[:, tn].T @ pvw_reg
                den = tr[:, tn].T @ pv_reg
                sats[tn][0, :] = np.divide(num, den, out=np.zeros_like(num), where=np.abs(den) > 0)

    count = 0
    for dt, ns in zip(dts, nsteps_vec, strict=False):
        A = sysmat(float(dt))
        for _ in range(int(ns)):
            count += 1
            tr = spla.spsolve(A, tr)
            tr = np.asarray(tr, dtype=float)
            if tr.ndim == 1:
                tr = tr.reshape((-1, 1))
            for tn in range(ni):
                cur = -(tr[:, tn].T @ qp_well)
                vals[tn][count, :] = np.nan_to_num(cur, nan=0.0, posinf=0.0, neginf=0.0)
                if sats is not None:
                    num = tr[:, tn].T @ pvw_reg
                    den = tr[:, tn].T @ pv_reg
                    sats[tn][count, :] = np.divide(num, den, out=np.zeros_like(num), where=np.abs(den) > 0)

    dt_full = np.repeat(dts, nsteps_vec)
    t = np.cumsum(np.concatenate([[0.0], dt_full]))
    dist.t = np.tile(t.reshape((-1, 1)), (1, nreg))
    dist.values = np.full((total_steps + 1, nreg), np.nan, dtype=float)
    col = 0
    for ik in range(len(iix)):
        for pk in range(len(pix)):
            if not reverse:
                dist.values[:, col] = vals[ik][:, pk]
            else:
                fac = -dist.producerFlux[pk] / max(dist.injectorFlux[ik], np.finfo(float).eps)
                dist.values[:, col] = vals[pk][:, ik] * fac
            col += 1

    if sats is not None:
        dist.sw0 = np.full_like(dist.values, np.nan)
        dist.volumesW = np.full_like(dist.volumes, np.nan)
        col = 0
        sat = np.asarray(get_field(state, "s"), dtype=float)
        for ik, ii in enumerate(iix):
            for pk, pp in enumerate(pix):
                V_ip = itracer[cix, ii] * ptracer[cix, pp] * pv_sub
                dist.volumesW[col] = float(np.sum(V_ip * sat[cix, 0]))
                dist.sw0[:, col] = sats[ik][:, pk] if not reverse else sats[pk][:, ik]
                col += 1
    return dist


def _setup_system_components(state, G, pv, W, sub, inj, prod, reverse):
    if reverse:
        inj, prod = prod, inj
    nc = num_cells(G)
    N = np.asarray(G.get("faces", {}).get("neighbors", []), dtype=int)
    flux = np.asarray(get_field(state, "flux"), dtype=float).ravel()
    if N.size == 0:
        return lambda dt: sp.eye(0, format="csc"), np.zeros((0, 0)), np.zeros((0, 0)), np.zeros(0), np.zeros(0, dtype=bool)
    if N.min(initial=0) >= 1:
        N = N - 1
    if N.shape[0] != flux.size:
        internal = np.all(N >= 0, axis=1)
        N = N[internal]
    sub_ext = np.asarray(sub, dtype=bool).ravel()
    face_keep = sub_ext[N[:, 0]] & sub_ext[N[:, 1]]
    N = N[face_keep]
    v = flux[face_keep]
    cix = np.zeros(nc, dtype=bool)
    if N.size:
        cix[N.ravel()] = True
    remap = -np.ones(nc, dtype=int)
    remap[np.flatnonzero(cix)] = np.arange(np.count_nonzero(cix))
    Nr = remap[N]
    vel = -v if reverse else v
    neg = vel < 0
    Nr[neg] = Nr[neg][:, ::-1]
    vel[neg] *= -1.0

    q = np.zeros(nc, dtype=float)
    well_sols = get_field(state, "wellSol", []) or []
    for iw, well in enumerate(W):
        cells = well_cells(well, nc)
        if iw < len(well_sols):
            wf = np.asarray(get_field(well_sols[iw], "flux", np.zeros(cells.size)), dtype=float).ravel()
            if wf.size == 1 and cells.size > 1:
                wf = np.full(cells.size, wf[0] / cells.size)
            np.add.at(q, cells[: wf.size], wf[: cells.size])
    if reverse:
        q = -q
    qp = np.minimum(q, 0.0)[cix]
    ncr = int(np.count_nonzero(cix))
    if ncr == 0:
        return lambda dt: sp.eye(0, format="csc"), np.zeros((0, 0)), np.zeros((0, 0)), np.zeros(0), cix
    A = sp.csr_matrix((vel, (Nr[:, 1], Nr[:, 0])), shape=(ncr, ncr))
    d = np.asarray(A.sum(axis=0)).ravel() - qp
    A = A - sp.diags(d, 0, shape=(ncr, ncr), format="csr")
    pv_sub = pv[cix]
    A = sp.diags(1.0 / np.maximum(pv_sub, np.finfo(float).eps), 0, shape=(ncr, ncr), format="csr") @ A
    I = sp.eye(ncr, format="csc")

    qi_well = np.zeros((ncr, len(inj)), dtype=float)
    qp_well = np.zeros((ncr, len(prod)), dtype=float)
    for col, iw in enumerate(inj):
        cells = remap[well_cells(W[int(iw)], nc)]
        cells = cells[cells >= 0]
        if cells.size:
            qi_well[cells, col] = max(float(np.sum(q[well_cells(W[int(iw)], nc)])), 0.0) / cells.size
    for col, pw in enumerate(prod):
        cells = remap[well_cells(W[int(pw)], nc)]
        cells = cells[cells >= 0]
        if cells.size:
            qp_well[cells, col] = min(float(np.sum(q[well_cells(W[int(pw)], nc)])), 0.0) / cells.size
    if reverse:
        qi_well, qp_well = -qi_well, -qp_well
    denom = np.sum(qi_well, axis=0)
    weights = np.divide(qi_well, denom.reshape((1, -1)), out=np.zeros_like(qi_well), where=np.abs(denom) > 0)
    tr0 = weights / np.maximum(pv_sub[:, None], np.finfo(float).eps)
    return lambda dt: (I - float(dt) * A).tocsc(), qp_well, tr0, pv_sub, cix


computeRTD = compute_rtd

