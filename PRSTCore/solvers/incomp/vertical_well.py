"""Python port of MRST's ``verticalWell.m`` (mrst-2026a/core/params/wells_and_bc):
insert a vertical well into a logically-Cartesian grid, using
:func:`PRSTCore.deckformat.params.compute_well_index.compute_well_index`
(Peaceman's formula) for the well index.

Scope: covers the common tutorial/example usage -- a single (I, J) column
completed over a given (or, if omitted, every) K layer, on a grid built by
:func:`PRSTCore.gridprocessing.cart_grid.cart_grid` (uniform physical cell
size derived from ``physDims / cartDims``). Not ported: ``addWell``'s full
option set (non-vertical directions, explicit per-completion WI override,
frictional/multi-lateral wells -- see
:func:`PRSTCore.deckformat.params.process_wells.process_wells` for deck-
driven COMPDAT wells on corner-point grids instead). Indices are 0-based
throughout, unlike MRST's 1-based ``I``/``J``/``K``.
"""

from __future__ import annotations

import numpy as _np

from PRSTCore.deckformat.params.compute_well_index import compute_well_index


def vertical_well(W, G: dict, rock: dict, i: int, j: int, k=None, *, type: str = "bhp",
                   val: float = 0.0, radius: float = 0.1, skin: float = 0.0,
                   comp_i=None, name: str | None = None) -> list:
    """Port of ``verticalWell.m``: append one well to ``W`` (a list of well
    dicts; pass ``[]``/``None`` for the first well) at logical column
    ``(i, j)``, completed over layers ``k`` (default: all layers).

    Returns the updated well list.
    """
    if W is None:
        W = []
    cart_dims = _np.asarray(G["cartDims"], dtype=int)
    nx, ny, nz = (int(cart_dims[d]) if cart_dims.size > d else 1 for d in range(3))
    if k is None:
        k = _np.arange(nz)
    else:
        k = _np.atleast_1d(_np.asarray(k, dtype=int))

    cells = i + nx * j + nx * ny * k
    perm = _np.atleast_2d(_np.asarray(rock["perm"], dtype=float))
    if perm.shape[0] == 1:
        perm = _np.tile(perm, (G["cells"]["num"], 1))
    if perm.shape[1] == 1:
        perm = _np.repeat(perm, 3, axis=1)
    kx, ky, kz = perm[cells, 0], perm[cells, 1], perm[cells, 2]

    # Uniform cell extents from the Cartesian grid's physical bounding box.
    coords = _np.asarray(G["nodes"]["coords"], dtype=float)
    phys = coords.max(axis=0) - coords.min(axis=0)
    dx = _np.full(k.size, phys[0] / nx)
    dy = _np.full(k.size, phys[1] / ny)
    dz = _np.full(k.size, phys[2] / nz if nz > 0 else phys[2])

    WI = compute_well_index(dx, dy, dz, kx, ky, kz, radius, direction="z", skin=skin,
                             griddim=int(G["griddim"]))

    nphase = perm.shape[1] if comp_i is not None else None
    well = {
        "name": name if name is not None else f"W{len(W) + 1}",
        "cells": cells.astype(_np.int64),
        "WI": WI,
        "dir": "z",
        "radius": _np.full(k.size, radius),
        "type": type,
        "val": val,
        "sign": 1 if type == "rate" and val >= 0 else (-1 if type == "rate" else 0),
        "compi": list(comp_i) if comp_i is not None else None,
        "cstatus": _np.ones(k.size, dtype=bool),
        "status": True,
    }
    return W + [well]
