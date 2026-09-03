"""Self-consistency checks for PRSTCore.gridprocessing (cart_grid / tensor_grid /
compute_geometry), independent of an MRST reference.

These exercise geometric identities that only hold if the port's face
triangulation / cell tetrahedralization is implemented correctly -- most
importantly the divergence theorem (the outward-oriented face normals of a
closed cell must sum to the zero vector), which is checked on both a
regular grid and a node-perturbed non-uniform grid so the general
polyhedral algorithm is exercised, not just the trivial axis-aligned-cuboid
formula.

A MATLAB-vs-Python parity test also exists
(``tests/test_gridprocessing_mrst_parity.py``) but requires a working local
MATLAB install; see that file for the true MRST comparison.
"""

from __future__ import annotations

import numpy as np
import pytest

from PRSTCore.gridprocessing import cart_grid, compute_geometry, tensor_grid


def _outward_normal_sum_per_cell(G: dict) -> np.ndarray:
    """Sum of outward-oriented face normals for every cell; should be ~0."""
    nc = G["cells"]["num"]
    dim = G["nodes"]["coords"].shape[1]
    face_pos = G["cells"]["facePos"]
    cell_faces = G["cells"]["faces"]
    counts = np.diff(face_pos)
    hf_cell = np.repeat(np.arange(nc), counts)
    hf_face = cell_faces[:, 0]

    normals = G["faces"]["normals"][hf_face]
    neighbors = G["faces"]["neighbors"]
    sign = np.where(neighbors[hf_face, 0] == hf_cell, 1.0, -1.0)
    outward = normals * sign[:, None]

    total = np.zeros((nc, dim))
    for d in range(dim):
        total[:, d] = np.bincount(hf_cell, weights=outward[:, d], minlength=nc)
    return total


@pytest.mark.parametrize("celldim,physdim", [([4, 3, 2], [8.0, 6.0, 4.0]), ([1, 1, 1], [1.0, 1.0, 1.0])])
def test_cart_grid_3d_analytic(celldim, physdim):
    G = compute_geometry(cart_grid(celldim, physdim))
    nx, ny, nz = celldim
    dx, dy, dz = np.asarray(physdim) / np.asarray(celldim)

    cell_vol = dx * dy * dz
    assert np.allclose(G["cells"]["volumes"], cell_vol, rtol=1e-12)
    assert np.isclose(G["cells"]["volumes"].sum(), np.prod(physdim), rtol=1e-12)

    # Analytic centroid of cell (i, j, k) (0-based), in MRST's Fortran-order cell numbering.
    i, j, k = np.unravel_index(np.arange(nx * ny * nz), (nx, ny, nz), order="F")
    expected_centroids = np.column_stack([(i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz])
    assert np.allclose(G["cells"]["centroids"], expected_centroids, atol=1e-10)

    # Face area/normal-length identity.
    assert np.allclose(np.linalg.norm(G["faces"]["normals"], axis=1), G["faces"]["areas"], rtol=1e-12)

    # Divergence theorem: closed cells.
    assert np.allclose(_outward_normal_sum_per_cell(G), 0.0, atol=1e-10)

    # Structural sanity: boundary vs internal face counts.
    neighbors = G["faces"]["neighbors"]
    internal = (neighbors[:, 0] >= 0) & (neighbors[:, 1] >= 0)
    boundary = np.count_nonzero(~internal)
    expected_boundary = 2 * (ny * nz + nx * nz + nx * ny)
    assert boundary == expected_boundary


def test_tensor_grid_perturbed_divergence_theorem():
    """Non-uniform spacing + random interior-node jitter: the general
    triangulated-face / tetrahedralized-cell algorithm must still close."""
    rng = np.random.default_rng(42)
    x = np.cumsum([0, 1, 1.5, 0.7, 2, 1.2])
    y = np.cumsum([0, 1, 0.8, 1.3, 1])
    z = np.cumsum([0, 1, 1.1])

    G = tensor_grid(x, y, z)
    jitter = 0.15 * (2 * rng.random((G["nodes"]["num"], 3)) - 1)
    G["nodes"]["coords"] = G["nodes"]["coords"] + jitter
    G = compute_geometry(G)

    assert np.all(G["cells"]["volumes"] > 0), "perturbation should not invert any cell"
    # NOTE: |normal| == area is only exact for *planar* faces. Node jitter bends
    # faces slightly, so the (vector sum of sub-triangle normals) has smaller
    # magnitude than the (scalar sum of sub-triangle areas) -- true in MRST's
    # own algorithm too, not a porting bug. Just check they stay close.
    normal_len = np.linalg.norm(G["faces"]["normals"], axis=1)
    assert np.all(normal_len <= G["faces"]["areas"] * (1 + 1e-10))
    assert np.allclose(_outward_normal_sum_per_cell(G), 0.0, atol=1e-9)

    # Domain volume should be close to (but not exactly, due to jitter) the
    # unperturbed bounding box volume.
    nominal_volume = x[-1] * y[-1] * z[-1]
    assert abs(G["cells"]["volumes"].sum() - nominal_volume) / nominal_volume < 0.05


def test_cart_grid_2d_analytic():
    G = compute_geometry(cart_grid([5, 4], [10.0, 8.0]))
    dx, dy = 2.0, 2.0
    assert np.allclose(G["cells"]["volumes"], dx * dy, rtol=1e-12)
    assert np.isclose(G["cells"]["volumes"].sum(), 80.0, rtol=1e-12)
    assert np.allclose(np.linalg.norm(G["faces"]["normals"], axis=1), G["faces"]["areas"], rtol=1e-12)

    total = _outward_normal_sum_per_cell(G)
    assert np.allclose(total, 0.0, atol=1e-10)


def test_cart_grid_1d_analytic():
    G = compute_geometry(cart_grid([6], [12.0]))
    assert np.allclose(G["cells"]["volumes"], 2.0, rtol=1e-12)
    assert np.isclose(G["cells"]["volumes"].sum(), 12.0, rtol=1e-12)
    expected_centroids = (np.arange(6) + 0.5) * 2.0
    assert np.allclose(G["cells"]["centroids"].reshape(-1), expected_centroids, atol=1e-10)


def test_no_neighbor_convention_matches_prstcore_style():
    """G.faces.neighbors must use -1 (not MRST's 0) for 'no cell here', matching
    the convention already used in PRSTCore.ad_core.operators."""
    G = cart_grid([2, 2, 2])
    neighbors = G["faces"]["neighbors"]
    assert neighbors.min() == -1
    assert neighbors.max() == G["cells"]["num"] - 1
