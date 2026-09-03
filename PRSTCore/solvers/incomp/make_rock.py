"""Python port of MRST's ``makeRock.m`` (mrst-2026a/core/params/rock)."""

from __future__ import annotations

import numpy as _np


def make_rock(G: dict, perm, poro, *, ntg=None) -> dict:
    """Port of ``makeRock.m``: build a ``{'perm', 'poro'[, 'ntg']}`` rock
    dict, broadcasting scalar/single-row ``perm``/``poro``/``ntg`` to every
    cell of ``G``.

    ``perm`` may be a scalar, a length-1/2/3 (2D) or length-1/3/6 (3D) row
    (uniform anisotropic permeability), or a full ``(nc, ncomp)`` array.
    """
    nc = int(G["cells"]["num"])

    def expand(vals, ncol_ok=None):
        vals = _np.atleast_2d(_np.asarray(vals, dtype=float))
        if vals.shape[0] == 1:
            vals = _np.tile(vals, (nc, 1))
        if vals.shape[0] != nc:
            raise ValueError("rock values must be one row per cell, or a single row")
        if ncol_ok is not None and vals.shape[1] not in ncol_ok:
            raise ValueError(f"expected {ncol_ok} columns, got {vals.shape[1]}")
        return vals

    griddim = int(G["griddim"])
    ncol_ok = {1: (1,), 2: (1, 2, 3), 3: (1, 3, 6)}[griddim]
    perm = expand(perm, ncol_ok)

    poro = expand(poro, (1,))[:, 0]
    if _np.any(poro <= 0):
        bad = _np.flatnonzero(poro <= 0)
        print(f"Warning: zero or negative porosity found in cells: {bad.tolist()}")

    rock = {"perm": perm, "poro": poro}
    if ntg is not None:
        rock["ntg"] = expand(ntg, (1,))[:, 0]
    return rock
