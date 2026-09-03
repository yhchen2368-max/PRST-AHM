"""Tests for remove_cells/extract_subgrid (removeCells.m/extractSubgrid.m ports).

Covers both a regular cart_grid (P2's original scope) and a real corner-point
grid from process_grdecl (P1's output) with a diagonal cell removed, since
that is exactly what NWM needs: carving the near-well region out of a
background grid that may be a fault-processed corner-point grid, not just a
simple Cartesian one.
"""

from __future__ import annotations

import numpy as np

from PRSTCore.deckformat.deckinput.convert_deck_units import convert_deck_units
from PRSTCore.deckformat.deckinput.read_eclipse_deck import read_eclipse_deck
from PRSTCore.gridprocessing import (
    cart_grid, compute_geometry, extract_subgrid, process_grdecl, remove_cells,
)


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


def test_no_op_on_empty_removal():
    G = compute_geometry(cart_grid([3, 3, 3]))
    H, cellmap, facemap, nodemap = remove_cells(G, np.array([], dtype=int))
    assert H is G
    assert np.array_equal(cellmap, np.arange(G["cells"]["num"]))
    assert np.array_equal(facemap, np.arange(G["faces"]["num"]))
    assert np.array_equal(nodemap, np.arange(G["nodes"]["num"]))


def test_remove_interior_cell_from_cart_grid_preserves_geometry_and_topology():
    G = compute_geometry(cart_grid([4, 4, 4], [4.0, 4.0, 4.0]))
    nc = G["cells"]["num"]
    # (i,j,k) = (1,1,1) in Fortran cell order (i fastest): the one cell with
    # all 6 neighbors present, i.e. not touching any of the 6 domain faces.
    remove = np.array([1 + 4 * 1 + 16 * 1])

    H, cellmap, facemap, nodemap = remove_cells(G, remove)

    assert H["cells"]["num"] == nc - 1
    # cellmap[i] is H-cell i's original G index -- kept-cell geometry must be
    # copied verbatim, not recomputed.
    assert np.allclose(H["cells"]["volumes"], G["cells"]["volumes"][cellmap])
    assert np.allclose(H["cells"]["centroids"], G["cells"]["centroids"][cellmap])

    # Removing one interior cell exposes exactly its 6 faces as new boundary,
    # and its own 6 half-faces vanish -- net face count drops by 0 (6 internal
    # faces survive as boundary; nothing is fully dropped since none of its
    # neighbors were also removed).
    assert H["faces"]["num"] == G["faces"]["num"]

    # Divergence theorem must still hold on every surviving cell.
    assert np.allclose(_outward_normal_sum_per_cell(H), 0.0, atol=1e-8)

    # No dangling references.
    assert np.all(H["cells"]["faces"][:, 0] >= 0)
    assert np.all(H["cells"]["faces"][:, 0] < H["faces"]["num"])
    assert np.all(H["faces"]["nodes"] >= 0)
    assert np.all(H["faces"]["nodes"] < H["nodes"]["num"])


def test_remove_two_adjacent_cells_drops_their_shared_face():
    G = compute_geometry(cart_grid([4]))  # genuinely 1D: 4 cells, 5 faces (x0..x4)
    # Cells 1 and 2 (0-based) are adjacent; the face between them should be
    # fully dropped (both sides removed), not turned into a boundary face.
    H, cellmap, facemap, nodemap = remove_cells(G, np.array([1, 2]))
    assert H["cells"]["num"] == 2
    # Removing cells 1,2 drops the face strictly between them (x2, both sides
    # removed); the other 4 faces survive: cell0's original west boundary
    # (x0) plus its newly-exposed east side (x1), and symmetrically for cell3
    # (x3, x4).
    assert H["faces"]["num"] == 4
    assert np.array_equal(np.sort(H["faces"]["neighbors"].ravel()), [-1, -1, -1, -1, 0, 0, 1, 1])


def test_extract_subgrid_is_the_complement_of_remove_cells():
    G = compute_geometry(cart_grid([5, 4, 3], [5.0, 4.0, 3.0]))
    nc = G["cells"]["num"]
    rng = np.random.default_rng(0)
    keep = rng.choice(nc, size=nc // 3, replace=False)
    keep_mask = np.zeros(nc, dtype=bool)
    keep_mask[keep] = True

    H_extract = extract_subgrid(G, keep)
    H_remove, cellmap, facemap, nodemap = remove_cells(G, ~keep_mask)

    assert H_extract["cells"]["num"] == H_remove["cells"]["num"]
    assert H_extract["faces"]["num"] == H_remove["faces"]["num"]
    assert np.allclose(np.sort(H_extract["cells"]["volumes"]), np.sort(H_remove["cells"]["volumes"]))
    assert np.array_equal(np.sort(H_extract["cells"]["global"]), np.sort(cellmap))
    assert np.allclose(_outward_normal_sum_per_cell(H_extract), 0.0, atol=1e-8)


def test_remove_cells_on_corner_point_grdecl_grid_for_nwm_style_carve_out():
    """Exercises the exact NWM workflow: carve a near-well region (here, a
    single diagonal cell as a stand-in) out of a real corner-point background
    grid produced by process_grdecl, and verify the surviving grid is still
    geometrically sound (general -- not axis-aligned -- polyhedral cells)."""
    deck = convert_deck_units(read_eclipse_deck("examples/SPE9/SPE9_CP.DATA"))
    G = compute_geometry(process_grdecl(deck["GRID"]))
    nc = G["cells"]["num"]

    total_volume_before = float(np.sum(G["cells"]["volumes"]))
    remove = np.array([0, nc // 2, nc - 1])
    H, cellmap, facemap, nodemap = remove_cells(G, remove)

    assert H["cells"]["num"] == nc - 3
    assert np.isclose(
        float(np.sum(H["cells"]["volumes"])),
        total_volume_before - float(np.sum(G["cells"]["volumes"][remove])),
    )
    assert np.allclose(_outward_normal_sum_per_cell(H), 0.0, atol=1e-3)
    # indexMap (original ECLIPSE active-cell id) must survive the reindexing.
    assert np.array_equal(H["cells"]["indexMap"], G["cells"]["indexMap"][cellmap])
