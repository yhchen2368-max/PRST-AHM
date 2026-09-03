"""Python port of the core of MRST's ``processWells.m``
(mrst-2026a/model-io/deckformat/params/wells_and_bc): converts a deck's
``WELSPECS``/``COMPDAT`` records into MRST-format well structures (name,
completion cells, well index, reference depth, ...).

Scope: the common case -- one completion range per ``COMPDAT`` record along
a single logical (I,J) column, default TPFA well index via
:func:`PRSTCore.deckformat.params.compute_well_index.compute_well_index`
(Peaceman's formula) unless an explicit ``WI`` is given in the record, and
status/open-shut handling. Not ported: ``WELSEGS``/``COMPSEGS`` (multi-
segment wells -- see
:class:`PRSTCore.ad_core.models.multisegment_well.MultisegmentWell` for
that), non-default completion directions beyond the record's own ``Dir``
field, and KH-table overrides.
"""

from __future__ import annotations

import numpy as _np

from .compute_well_index import compute_well_index


def _strip_quotes(tok) -> str:
    s = str(tok)
    return s[1:-1] if len(s) >= 2 and s[0] == "'" and s[-1] == "'" else s


def _is_default(tok) -> bool:
    s = str(tok).strip()
    return s in ("1*", "") or (isinstance(tok, str) and tok.strip().endswith("*") and tok.strip()[:-1].strip() in ("", "1"))


def process_wells(cart_dims, welspecs: list, compdat: list, rock: dict, cell_dims, cart_to_active=None,
                   default_radius: float = 0.15) -> list[dict]:
    """Port of ``processWells.m``'s WELSPECS/COMPDAT conversion.

    Parameters
    ----------
    cart_dims : (nx, ny, nz)
    welspecs, compdat : list[list]
        Raw deck records (as parsed by ``read_eclipse_deck``/
        ``read_schedule``), e.g. ``["'INJE1'", "'P'", 24.0, 25.0, 9110.0,
        "'WATER'", 60.0]`` for WELSPECS and ``["'INJE1'", 24.0, 25.0, 11.0,
        15.0, "'OPEN'", '1*', '1*', 1.0]`` for COMPDAT (empty leading
        records from section headers are skipped automatically).
    rock : dict
        Must have ``'perm'`` ((n_active, 3) or (n_active,) array).
    cell_dims : (n_active, 3) array
        Per-active-cell (dx, dy, dz), e.g. from
        ``PRSTCore.gridprocessing.compute_geometry``'s
        ``cells['dimensions']`` if available, or derived externally.
    cart_to_active : (nx*ny*nz,) int array or None
        Maps a Fortran-order Cartesian cell index to its 0-based active-cell
        index (``-1`` for inactive); identity (no ACTNUM) if omitted.

    Returns
    -------
    list[dict]
        One dict per well: ``{'name', 'cells' (0-based active-cell indices,
        one per completion), 'WI', 'dir', 'radius', 'skin', 'refDepth',
        'status', 'cstatus'}``.
    """
    nx, ny, nz = (int(x) for x in cart_dims)
    nfull = nx * ny * nz
    if cart_to_active is None:
        cart_to_active = _np.arange(nfull, dtype=_np.int64)
    else:
        cart_to_active = _np.asarray(cart_to_active, dtype=_np.int64)

    perm = _np.atleast_2d(_np.asarray(rock["perm"], dtype=float))
    if perm.shape[1] == 1:
        perm = _np.repeat(perm, 3, axis=1)
    cell_dims = _np.atleast_2d(_np.asarray(cell_dims, dtype=float))

    ref_depth = {}
    for rec in welspecs:
        if len(rec) < 5:
            continue
        name = _strip_quotes(rec[0])
        try:
            ref_depth[name] = float(rec[4])
        except (TypeError, ValueError):
            pass

    wells: dict[str, dict] = {}
    for rec in compdat:
        if len(rec) < 6:
            continue
        name = _strip_quotes(rec[0])
        i, j = int(rec[1]), int(rec[2])
        k1, k2 = int(rec[3]), int(rec[4])
        status_open = _strip_quotes(rec[5]).upper() != "SHUT"

        wi_override = None
        if len(rec) > 7 and not _is_default(rec[7]):
            try:
                wi_override = float(rec[7])
            except (TypeError, ValueError):
                wi_override = None
        radius = default_radius
        if len(rec) > 8 and not _is_default(rec[8]):
            # COMPDAT's 9th field is wellbore *diameter*, not radius.
            try:
                radius = float(rec[8]) / 2.0
            except (TypeError, ValueError):
                pass
        skin = 0.0
        if len(rec) > 11 and not _is_default(rec[11]):
            try:
                skin = float(rec[11])
            except (TypeError, ValueError):
                pass
        direction = "z"
        if len(rec) > 13 and not _is_default(rec[13]):
            direction = _strip_quotes(rec[13]).lower()[:1]

        w = wells.setdefault(name, {
            "name": name, "cells": [], "WI": [], "cstatus": [], "dir": direction,
            "radius": radius, "skin": [], "refDepth": ref_depth.get(name, 0.0), "status": True,
        })

        for k in range(k1, k2 + 1):
            cart_idx = (i - 1) + nx * (j - 1) + nx * ny * (k - 1)
            if not (0 <= cart_idx < nfull):
                continue
            active = int(cart_to_active[cart_idx])
            if active < 0:
                continue

            if wi_override is not None:
                wi = wi_override
            else:
                dx, dy, dz = cell_dims[active]
                kx, ky, kz = perm[active]
                wi = float(compute_well_index(
                    [dx], [dy], [dz], [kx], [ky], [kz], [radius], direction=direction, skin=skin,
                )[0])

            w["cells"].append(active)
            w["WI"].append(wi)
            w["cstatus"].append(status_open)
            w["skin"].append(skin)

    out = []
    for w in wells.values():
        w["cells"] = _np.asarray(w["cells"], dtype=_np.int64)
        w["WI"] = _np.asarray(w["WI"], dtype=float)
        w["cstatus"] = _np.asarray(w["cstatus"], dtype=bool)
        w["skin"] = _np.asarray(w["skin"], dtype=float)
        w["status"] = bool(_np.any(w["cstatus"]))
        out.append(w)
    return out
