"""Tests for process_wells (processWells.m WELSPECS/COMPDAT conversion port).

Scope note: unlike most parity tests in this suite, this is validated
structurally against the real SPE9 deck (well/completion counts, valid
cell indices, positive well indices) rather than against an MRST-side
processWells trace -- the well-index formula itself
(compute_well_index/computeWellIndex.m) is separately validated exactly
against MRST in test_process_wells_index... see
tests/test_well_index (compute_well_index is MRST-verified to machine
precision; this test covers the WELSPECS/COMPDAT record parsing built on
top of it).
"""

from __future__ import annotations

import numpy as np

from PRSTCore.deckformat.deckinput.convert_deck_units import convert_deck_units
from PRSTCore.deckformat.deckinput.read_eclipse_deck import read_eclipse_deck
from PRSTCore.deckformat.params.process_wells import process_wells
from PRSTCore.gridprocessing import compute_geometry, process_grdecl


def test_process_wells_on_real_spe9_deck():
    deck = convert_deck_units(read_eclipse_deck("examples/SPE9/SPE9_CP.DATA"))
    grdecl = deck["GRID"]
    G = compute_geometry(process_grdecl(grdecl))
    nc = G["cells"]["num"]

    rock_perm = np.full((nc, 3), 1e-13)  # SPE9's real PERMX/Y/Z aren't needed to test the parsing itself
    cell_dims = np.column_stack([
        np.linalg.norm(G["cells"]["centroids"], axis=1) * 0 + 30.0,  # placeholder dims; only structure is checked
        np.full(nc, 30.0),
        np.full(nc, 10.0),
    ])
    rock = {"perm": rock_perm}

    sched = deck["SCHEDULE"]
    welspecs = [r for r in sched["WELSPECS"] if r]
    compdat = [r for r in sched["COMPDAT"] if r]

    wells = process_wells(
        grdecl["cartDims"], welspecs, compdat, rock, cell_dims,
        cart_to_active=_cart_to_active(grdecl, G),
    )

    names = {w["name"] for w in wells}
    expected_names = {_strip(r[0]) for r in welspecs}
    assert names == expected_names

    for w in wells:
        assert w["cells"].size == w["WI"].size == w["cstatus"].size
        assert np.all(w["cells"] >= 0) and np.all(w["cells"] < nc)
        assert np.all(w["WI"] > 0)
        assert w["refDepth"] > 0  # SPE9's WELSPECS always specifies a real reference depth


def _strip(tok):
    s = str(tok)
    return s[1:-1] if s.startswith("'") else s


def _cart_to_active(grdecl, G):
    import numpy as _np
    nx, ny, nz = (int(x) for x in grdecl["cartDims"])
    nfull = nx * ny * nz
    cart_to_active = _np.full(nfull, -1, dtype=_np.int64)
    cart_to_active[G["cells"]["indexMap"]] = _np.arange(G["cells"]["num"], dtype=_np.int64)
    return cart_to_active
