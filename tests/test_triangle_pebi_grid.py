"""Tests for triangle_grid/pebi_grid (triangleGrid.m/pebi.m ports), needed
for NWM's standalone-radial-grid example (``H = pebi(triangleGrid(pP))``)."""

from __future__ import annotations

import numpy as np

from PRSTCore.gridprocessing import compute_geometry, pebi_grid, triangle_grid


def _outward_normal_sum_per_cell_2d(G: dict) -> np.ndarray:
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
    total = np.zeros((nc, 2))
    for d in range(2):
        total[:, d] = np.bincount(hf_cell, weights=outward[:, d], minlength=nc)
    return total


def _grid_points(nx=6, ny=6):
    x, y = np.meshgrid(np.linspace(0, 5, nx), np.linspace(0, 5, ny))
    rng = np.random.default_rng(0)
    pts = np.column_stack([x.ravel(), y.ravel()])
    pts += rng.uniform(-0.05, 0.05, pts.shape)  # perturb off-grid to avoid a degenerate Delaunay
    return pts


def test_triangle_grid_from_delaunay_is_geometrically_sound():
    p = _grid_points()
    G = compute_geometry(triangle_grid(p))
    assert G["cells"]["num"] > 0
    assert np.all(G["cells"]["volumes"] > 0)
    assert np.isclose(np.sum(G["cells"]["volumes"]), 5.0 * 5.0, rtol=0.05)
    assert np.allclose(_outward_normal_sum_per_cell_2d(G), 0.0, atol=1e-8)


def test_triangle_grid_explicit_triangle_list():
    p = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    t = np.array([[0, 1, 2], [1, 3, 2]])
    G = compute_geometry(triangle_grid(p, t))
    assert G["cells"]["num"] == 2
    assert np.isclose(np.sum(G["cells"]["volumes"]), 1.0)


def test_pebi_grid_of_a_regular_point_set_is_geometrically_sound():
    p = _grid_points(nx=5, ny=5)
    G = compute_geometry(pebi_grid(p))
    assert G["cells"]["num"] == p.shape[0]
    assert np.all(G["cells"]["volumes"] > 0)
    assert np.allclose(_outward_normal_sum_per_cell_2d(G), 0.0, atol=1e-6)


def test_pebi_grid_accepts_a_triangle_grid_dict_matching_mrst_idiom():
    """Mirrors MRST's ``pebi(triangleGrid(p))`` usage in NWM's standalone
    radial grid example."""
    p = _grid_points(nx=5, ny=5)
    tg = triangle_grid(p)
    G = compute_geometry(pebi_grid(tg))
    assert G["cells"]["num"] == p.shape[0]
    assert np.all(G["cells"]["volumes"] > 0)
