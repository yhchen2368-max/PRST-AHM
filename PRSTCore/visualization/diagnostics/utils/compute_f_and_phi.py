"""MRST ``computeFandPhi.m`` counterpart."""

from __future__ import annotations

import numpy as np

from .helpers import get_field


def compute_f_and_phi(arg1, tof: np.ndarray | None = None, *, sum: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Compute flow-capacity/storage-capacity curves.

    This follows MRST ``computeFandPhi``.  Use either
    ``compute_f_and_phi(pv, tof)`` or ``compute_f_and_phi(rtd, sum=True)``.
    """
    if tof is None and _looks_like_rtd(arg1):
        return compute_f_and_phi_from_dist(arg1, sum=sum)

    pv = np.asarray(arg1, dtype=float).ravel()
    tof = np.asarray(tof, dtype=float)
    if tof.ndim != 2 or tof.shape[1] != 2:
        raise AssertionError("Tof input must have two columns.")
    if tof.shape[0] != pv.size:
        raise ValueError("pore volume and TOF arrays must have the same row count")

    total_time = np.sum(tof, axis=1)
    finite = np.isfinite(total_time) & np.isfinite(pv) & (pv > 0.0) & (total_time > 0.0)
    if not np.any(finite):
        return np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])

    order = np.argsort(total_time[finite])
    sorted_time = total_time[finite][order]
    sorted_volume = pv[finite][order]

    phi = np.cumsum(sorted_volume)
    phi_total = float(phi[-1])
    phi = np.concatenate(([0.0], phi / phi_total))

    flux = sorted_volume / sorted_time
    flow_capacity = np.cumsum(flux)
    flow_total = float(flow_capacity[-1])
    if flow_total <= 0.0:
        F = phi.copy()
    else:
        F = np.concatenate(([0.0], flow_capacity / flow_total))
    return F, phi


def compute_f_and_phi_from_dist(RTD, *, sum: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Compute F-Phi curves from an RTD structure.

    This mirrors MRST ``computeFandPhiFromDist.m`` for the data fields used by
    diagnostics post-processing.
    """
    creator = str(get_field(RTD, "creator", "computeRTD"))
    if creator.lower() == "estimatertd":
        normalize_alloc = True
        normalize_vol = False
    else:
        normalize_alloc = False
        normalize_vol = True

    t = np.asarray(get_field(RTD, "t"), dtype=float)
    vals = np.asarray(get_field(RTD, "values"), dtype=float)
    volumes = np.asarray(get_field(RTD, "volumes"), dtype=float).ravel()
    allocations = np.asarray(get_field(RTD, "allocations"), dtype=float).ravel()
    injector_flux = np.asarray(get_field(RTD, "injectorFlux"), dtype=float)
    pair_ix = np.asarray(get_field(RTD, "pairIx", np.zeros((allocations.size, 2), dtype=int)), dtype=int)

    if sum:
        t = t[:, [0]] if t.ndim == 2 else t.reshape((-1, 1))
        vals = np.sum(vals, axis=1, keepdims=True) if vals.ndim == 2 else vals.reshape((-1, 1))
        volumes = np.asarray([float(np.sum(volumes))])
        allocations = np.asarray([float(np.sum(allocations))])
        injflux = np.asarray([float(np.sum(injector_flux))])
    else:
        if t.ndim == 1:
            t = t.reshape((-1, 1))
        if vals.ndim == 1:
            vals = vals.reshape((-1, 1))
        if pair_ix.size and pair_ix.ndim == 2:
            inj_index = pair_ix[:, 0]
            if inj_index.size and np.min(inj_index) >= 1 and np.max(inj_index) <= injector_flux.size:
                inj_index = inj_index - 1
            injflux = injector_flux.reshape(-1)[inj_index]
        else:
            injflux = np.resize(injector_flux.reshape(-1), vals.shape[1])

    t = np.nan_to_num(t, nan=0.0)
    vals = np.nan_to_num(vals, nan=0.0)
    mt = 0.5 * t[:-1, :] + 0.5 * t[1:, :]
    mvals = vals[1:, :]
    dt = np.diff(t, axis=0)

    F = np.cumsum(dt * mvals, axis=0) * injflux.reshape((1, -1))
    denom = F[-1, :] if normalize_alloc else allocations.reshape((1, -1))
    F = F / np.maximum(denom, np.finfo(float).eps)

    Phi = np.cumsum(dt * mvals * mt, axis=0) * allocations.reshape((1, -1))
    denom_phi = Phi[-1, :] if normalize_vol else volumes.reshape((1, -1))
    Phi = Phi / np.maximum(denom_phi, np.finfo(float).eps)

    one = np.ones((1, Phi.shape[1]), dtype=float)
    zero = np.zeros_like(one)
    return np.vstack([zero, F, one]), np.vstack([zero, Phi, one])


def _looks_like_rtd(value) -> bool:
    return get_field(value, "values", None) is not None and get_field(value, "t", None) is not None


computeFandPhi = compute_f_and_phi
computeFandPhiFromDist = compute_f_and_phi_from_dist
