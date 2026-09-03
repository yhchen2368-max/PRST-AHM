"""The vectorised per-cell bounding box must reproduce the per-cell loop.

``_cell_bounding_box_dims`` walked the cells one at a time, concatenating
each cell's faces' node lists and calling ``numpy.unique`` on the result.
On Norne that was 44927 iterations, and its caller runs once per schedule
control step -- 248 times -- so ``unique`` was entered eleven million times
and the grid's bounding boxes were computed 248 times over.  That was 198
of the 225 seconds a Norne set-up took.

Two things changed: the loop became one pass over the connectivity, and the
result is cached on the grid where the caller already looks for it.  Neither
may move a number, so the original loop is kept here as the oracle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import PRSTCore  # noqa: F401
from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import (
    _cell_bounding_box_dims)


def _loop_reference(G, nc):
    """The original per-cell implementation, verbatim."""
    cells = G.get('cells', {})
    faces = G.get('faces', {})
    nodes = G.get('nodes', {})
    face_pos = np.asarray(cells.get('facePos'), dtype=np.int64).ravel()
    cell_faces = np.asarray(cells.get('faces'), dtype=np.int64)
    cf0 = cell_faces[:, 0] if cell_faces.ndim == 2 else cell_faces
    node_pos = np.asarray(faces.get('nodePos'), dtype=np.int64).ravel()
    face_nodes = np.asarray(faces.get('nodes'), dtype=np.int64).ravel()
    coords = np.asarray(nodes.get('coords'), dtype=float)

    dims = np.zeros((nc, 3), dtype=float)
    for c in range(nc):
        cfaces = cf0[face_pos[c]:face_pos[c + 1]]
        node_ids = (np.concatenate([face_nodes[node_pos[f]:node_pos[f + 1]]
                                    for f in cfaces])
                    if cfaces.size else np.zeros(0, dtype=np.int64))
        if node_ids.size == 0:
            continue
        pts = coords[np.unique(node_ids)]
        dims[c] = np.maximum(pts.max(axis=0) - pts.min(axis=0), 1.0e-6)
    return dims


def _box_grid(nx, ny, nz, jitter=0.0, empty_cells=(), seed=0):
    """A hexahedral grid in the cell/face/node form the routine expects."""
    rng = np.random.default_rng(seed)
    nodes = []
    node_id = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                node_id[(i, j, k)] = len(nodes)
                offset = rng.uniform(-jitter, jitter, 3) if jitter else np.zeros(3)
                nodes.append([i + offset[0], 2.0 * j + offset[1], 0.5 * k + offset[2]])
    coords = np.asarray(nodes, dtype=float)

    face_nodes = []
    node_pos = [0]
    faces_of_cell = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                corners = [(i + a, j + b, k + c)
                           for a in (0, 1) for b in (0, 1) for c in (0, 1)]
                cell_faces = []
                # Six faces, each four corners; the exact grouping does not
                # matter to a bounding box, only that every corner appears.
                quads = [
                    [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
                    [(1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)],
                    [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
                    [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)],
                    [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
                    [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
                ]
                for quad in quads:
                    cell_faces.append(len(node_pos) - 1)
                    for a, b, c in quad:
                        face_nodes.append(node_id[(i + a, j + b, k + c)])
                    node_pos.append(len(face_nodes))
                faces_of_cell.append(cell_faces)
                assert corners  # keep the corner list meaningful

    for index in empty_cells:
        faces_of_cell[index] = []

    face_pos = np.zeros(len(faces_of_cell) + 1, dtype=np.int64)
    flat = []
    for index, cf in enumerate(faces_of_cell):
        flat.extend(cf)
        face_pos[index + 1] = len(flat)

    return {
        'cells': {'facePos': face_pos, 'faces': np.asarray(flat, dtype=np.int64)},
        'faces': {'nodePos': np.asarray(node_pos, dtype=np.int64),
                  'nodes': np.asarray(face_nodes, dtype=np.int64)},
        'nodes': {'coords': coords},
    }, len(faces_of_cell)


CASES = {
    'cartesian': dict(nx=4, ny=3, nz=2),
    'single_cell': dict(nx=1, ny=1, nz=1),
    'distorted': dict(nx=3, ny=3, nz=3, jitter=0.15, seed=7),
    'flat_column': dict(nx=1, ny=1, nz=6),
    'with_empty_cells': dict(nx=3, ny=2, nz=2, empty_cells=(0, 5, 11)),
}


@pytest.mark.parametrize('case', sorted(CASES))
def test_matches_the_per_cell_loop(case):
    G, nc = _box_grid(**CASES[case])
    np.testing.assert_array_equal(_cell_bounding_box_dims(G, nc),
                                  _loop_reference(G, nc))


def test_a_cell_with_no_faces_keeps_zero_dimensions():
    G, nc = _box_grid(nx=2, ny=2, nz=1, empty_cells=(1,))
    dims = _cell_bounding_box_dims(G, nc)
    np.testing.assert_array_equal(dims[1], np.zeros(3))
    assert np.all(dims[0] > 0)


def test_two_dimensional_cell_face_table_uses_the_first_column():
    """``cells.faces`` may carry the neighbouring cell in a second column."""
    G, nc = _box_grid(nx=2, ny=2, nz=2)
    flat = G['cells']['faces']
    G['cells']['faces'] = np.stack([flat, np.zeros_like(flat)], axis=1)
    np.testing.assert_array_equal(_cell_bounding_box_dims(G, nc),
                                  _loop_reference(G, nc))
