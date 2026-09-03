"""Self-consistency checks for process_grdecl (independent of MATLAB), plus
the MATLAB-verified SPE9/Norne parity in test_process_grdecl_mrst_parity.py.
"""

from __future__ import annotations

import numpy as np

from PRSTCore.deckformat.deckinput.convert_deck_units import convert_deck_units
from PRSTCore.deckformat.deckinput.read_eclipse_deck import read_eclipse_deck
from PRSTCore.gridprocessing import compute_geometry, process_grdecl


def _outward_normal_sum_per_cell(G: dict) -> np.ndarray:
    nc = G["cells"]["num"]
    face_pos = G["cells"]["facePos"]
    cell_faces = G["cells"]["faces"]
    counts = np.diff(face_pos)
    hf_cell = np.repeat(np.arange(nc), counts)
    hf_face = cell_faces[:, 0]
    normals = G["faces"]["normals"][hf_face]
    neighbors = G["faces"]["neighbors"]
    sign = np.where(neighbors[hf_face, 0] == hf_cell, 1.0, -1.0)
    outward = normals * sign[:, None]
    total = np.zeros((nc, 3))
    for d in range(3):
        total[:, d] = np.bincount(hf_cell, weights=outward[:, d], minlength=nc)
    return total


def test_spe9_active_cell_count_matches_actnum():
    deck = convert_deck_units(read_eclipse_deck("examples/SPE9/SPE9_CP.DATA"))
    grdecl = deck["GRID"]
    actnum = np.asarray(grdecl.get("ACTNUM", np.ones(np.prod(grdecl["cartDims"]))), dtype=int)
    G = process_grdecl(grdecl)
    assert G["cells"]["num"] == int(actnum.sum())
    assert G["cartDims"].tolist() == [int(x) for x in grdecl["cartDims"]]


def test_spe9_divergence_theorem_holds():
    deck = convert_deck_units(read_eclipse_deck("examples/SPE9/SPE9_CP.DATA"))
    G = compute_geometry(process_grdecl(deck["GRID"]))
    assert np.all(G["cells"]["volumes"] > 0)
    assert np.allclose(_outward_normal_sum_per_cell(G), 0.0, atol=1e-6)


def test_spe9_index_map_matches_active_cartesian_order():
    """MRST's processGRDECL preserves ECLIPSE's Fortran active-cell ordering
    (i fastest, then j, then k) -- downstream code (rock/PVT lookups, well
    perforation mapping) depends on this."""
    deck = convert_deck_units(read_eclipse_deck("examples/SPE9/SPE9_CP.DATA"))
    grdecl = deck["GRID"]
    actnum = np.asarray(grdecl.get("ACTNUM", np.ones(np.prod(grdecl["cartDims"]))), dtype=int).astype(bool)
    G = process_grdecl(grdecl)
    expected = np.flatnonzero(actnum)
    assert np.array_equal(G["cells"]["indexMap"], expected)


def test_missing_cartdims_raises():
    import pytest
    with pytest.raises(ValueError):
        process_grdecl({"COORD": np.zeros(6), "ZCORN": np.zeros(8)})
