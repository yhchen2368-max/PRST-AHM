"""Tests for PRSTCore.visualization.grid_plots (plot_grid/plot_cell_data/plot_faces),
ported from MRST's core/plotting (boundaryFaces.m, plotFaces.m, plotGrid.m,
plotCellData.m).

Structural checks (boundary-face extraction, polygon tracing) run without a
display. A couple of tests also render to a PNG via matplotlib's headless
Agg backend as an end-to-end smoke test.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from PRSTCore.gridprocessing import cart_grid, compute_geometry
from PRSTCore.visualization import boundary_faces, plot_cell_data, plot_grid, plot_well


def test_boundary_faces_whole_grid_count():
    G = compute_geometry(cart_grid([3, 2, 2]))
    faces, cells = boundary_faces(G)
    neighbors = G["faces"]["neighbors"]
    internal = (neighbors[:, 0] >= 0) & (neighbors[:, 1] >= 0)
    expected = np.count_nonzero(~internal)
    assert faces.size == expected
    # Every returned face must connect to exactly one present (all-cells) neighbor,
    # and that neighbor must be a valid cell id.
    assert np.all(cells >= 0) and np.all(cells < G["cells"]["num"])


def test_boundary_faces_single_cell_subset():
    G = compute_geometry(cart_grid([3, 3, 3]))
    # A single interior cell: all 6 of its faces should be its "boundary" since
    # none of its neighbors are in the (size-1) selection.
    center_cell = 13  # (1,1,1) 0-based in a 3x3x3 grid -> fully interior
    faces, cells = boundary_faces(G, [center_cell])
    assert faces.size == 6
    assert np.all(cells == center_cell)


def test_boundary_faces_two_adjacent_cells_share_no_internal_face():
    G = compute_geometry(cart_grid([2, 1, 1]))
    faces, cells = boundary_faces(G, [0, 1])
    # The whole grid is just these two cells, so their subset boundary == grid boundary.
    all_faces, _ = boundary_faces(G)
    assert set(faces.tolist()) == set(all_faces.tolist())


def test_cell_polygon_2d_is_a_valid_quad():
    from PRSTCore.visualization.grid_plots import _cell_polygon_2d

    G = compute_geometry(cart_grid([4, 3], [4.0, 3.0]))
    poly = _cell_polygon_2d(G, 5)
    assert poly.shape == (4, 2)
    # Shoelace formula area should match the known cell area (1x1).
    x, y = poly[:, 0], poly[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert np.isclose(area, 1.0, atol=1e-10)


def test_plot_grid_3d_smoke(tmp_path):
    G = compute_geometry(cart_grid([4, 3, 2], [4.0, 3.0, 2.0]))
    ax = plot_grid(G)
    out = tmp_path / "grid3d.png"
    ax.figure.savefig(out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_cell_data_3d_smoke_with_wells(tmp_path):
    G = compute_geometry(cart_grid([6, 5, 3], [6.0, 5.0, 3.0]))
    data = G["cells"]["centroids"][:, 0] + G["cells"]["centroids"][:, 2]
    W = [
        {"cells": [0, G["cartDims"][0] * G["cartDims"][1]], "name": "I1"},
        {"cells": [G["cells"]["num"] - 1], "name": "P1"},
    ]
    ax = plot_cell_data(G, data)
    plot_well(G, W, ax=ax)
    out = tmp_path / "celldata3d.png"
    ax.figure.savefig(out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_cell_data_2d_smoke(tmp_path):
    G = compute_geometry(cart_grid([10, 8], [10.0, 8.0]))
    data = G["cells"]["centroids"][:, 0]
    ax = plot_cell_data(G, data)
    out = tmp_path / "celldata2d.png"
    ax.figure.savefig(out)
    assert out.exists() and out.stat().st_size > 0
