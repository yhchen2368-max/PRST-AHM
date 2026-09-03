"""Turn a PRSTCore grid into a VTK dataset.

The grid the simulator solves on is the grid we draw.  ``G`` is carried
through unchanged -- no second parse of ``COORD``/``ZCORN``, no separate
active-cell numbering -- so VTK cell ``k`` is row ``k`` of
``state['pressure']`` and of every other per-cell array.  That identity is
the point of building from ``G`` rather than from the deck: two readers of
the same deck have to be kept in step, and an off-by-one between them draws
a plausible picture of the wrong cells.

Two cell representations, picked from the grid itself:

* a Cartesian/tensor grid has its nodes on an ``(nx+1, ny+1, nz+1)`` lattice,
  so each cell is a ``VTK_HEXAHEDRON`` read straight off the lattice -- cheap,
  and what keeps a million-cell SPE10 usable;
* anything else (a corner-point grid from ``processGRDECL``, with faults and
  pinch-outs) becomes a ``VTK_POLYHEDRON`` built from ``G['faces']``, which
  assumes nothing about cell shape.

The hexahedron path is checked against ``G['cells']['centroids']`` before it
is used and falls back to polyhedra if it disagrees, so a grid whose node
numbering differs from the assumed one is drawn correctly rather than wrongly.
"""

from __future__ import annotations

import numpy as np

import vtk
from vtk.util.numpy_support import numpy_to_vtk


__all__ = ["grid_to_vtk", "cell_ijk", "well_block_mask"]


def cell_ijk(G: dict):
    """The zero-based ``(i, j, k)`` logical index of every cell, or ``None``.

    Taken from ``cells.indexMap`` against ``cartDims`` -- the natural-order
    index MRST keeps precisely so an active cell can be traced back to the
    box it came from.  A grid with neither (an unstructured or extracted
    subgrid) has no logical index, and slicing by I/J/K is not offered for it.
    """
    dims = G.get("cartDims")
    if dims is None:
        return None
    dims = np.asarray(dims, dtype=np.int64).ravel()
    if dims.size != 3:
        return None

    ncells = int(G["cells"]["num"])
    index_map = G["cells"].get("indexMap")
    if index_map is None:
        # An all-active grid stores no map: cell c *is* natural cell c.
        if ncells != int(np.prod(dims)):
            return None
        index_map = np.arange(ncells, dtype=np.int64)
    index_map = np.asarray(index_map, dtype=np.int64).ravel()
    if index_map.size != ncells:
        return None

    # MRST's natural order runs i fastest, then j, then k -- Fortran order.
    i, j, k = np.unravel_index(index_map, tuple(int(x) for x in dims), order="F")
    return np.stack([i, j, k], axis=1).astype(np.int32)


def _lattice_hexahedra(G: dict):
    """Hexahedral connectivity for a grid whose nodes sit on a lattice.

    Returns ``(ncells, 8)`` point ids in VTK's hexahedron order, or ``None``
    when the grid is not of that shape.  The caller verifies the result
    against the grid's own centroids before using it.
    """
    kinds = G.get("type") or []
    if not any(k in ("cartGrid", "tensorGrid") for k in kinds):
        return None

    dims = G.get("cartDims")
    if dims is None:
        return None
    dims = np.asarray(dims, dtype=np.int64).ravel()
    if dims.size != 3:
        return None
    nx, ny, nz = (int(x) for x in dims)

    ncells = int(G["cells"]["num"])
    if int(G["nodes"]["num"]) != (nx + 1) * (ny + 1) * (nz + 1):
        return None

    if ncells == nx * ny * nz:
        cells = np.arange(ncells, dtype=np.int64)
    else:
        # A tensor grid with inactive cells still numbers its nodes over the
        # full lattice; indexMap is what says which boxes survived.
        index_map = G["cells"].get("indexMap")
        if index_map is None:
            return None
        cells = np.asarray(index_map, dtype=np.int64).ravel()
        if cells.size != ncells:
            return None

    ci, cj, ck = np.unravel_index(cells, (nx, ny, nz), order="F")

    def node(di, dj, dk):
        return ((ci + di) + (nx + 1) * ((cj + dj) + (ny + 1) * (ck + dk))).astype(np.int64)

    # VTK_HEXAHEDRON: the k face first, going round it, then the k+1 face
    # directly above in the same order.
    return np.stack([
        node(0, 0, 0), node(1, 0, 0), node(1, 1, 0), node(0, 1, 0),
        node(0, 0, 1), node(1, 0, 1), node(1, 1, 1), node(0, 1, 1),
    ], axis=1)


def _hexahedra_agree_with_centroids(G: dict, hexes, rtol: float = 0.05) -> bool:
    """Check assumed connectivity against the grid's own geometry.

    The mean of a hexahedron's eight corners is not the exact centroid of a
    distorted cell, so this compares loosely, against the grid's overall
    size.  It is here to catch a *different node numbering*, which is wrong
    by a whole cell or more, not to validate the geometry.
    """
    centroids = G["cells"].get("centroids")
    if centroids is None:
        return True  # No geometry computed to check against; trust the type.
    coords = np.asarray(G["nodes"]["coords"], dtype=float)
    centroids = np.asarray(centroids, dtype=float)

    sample = np.arange(hexes.shape[0])
    if sample.size > 2000:
        sample = np.linspace(0, hexes.shape[0] - 1, 2000).astype(np.int64)

    corner_mean = coords[hexes[sample]].mean(axis=1)
    extent = coords.max(axis=0) - coords.min(axis=0)
    scale = float(np.linalg.norm(extent)) or 1.0
    return bool(np.all(np.linalg.norm(corner_mean - centroids[sample], axis=1) < rtol * scale))


def _insert_polyhedra(ug, G: dict) -> None:
    """Insert every cell as a ``VTK_POLYHEDRON`` described by its own faces."""
    face_pos = np.asarray(G["cells"]["facePos"], dtype=np.int64).ravel()
    cell_faces = np.asarray(G["cells"]["faces"], dtype=np.int64)
    if cell_faces.ndim == 2:
        cell_faces = cell_faces[:, 0]
    cell_faces = cell_faces.ravel()
    node_pos = np.asarray(G["faces"]["nodePos"], dtype=np.int64).ravel()
    face_nodes = np.asarray(G["faces"]["nodes"], dtype=np.int64).ravel()

    ncells = int(G["cells"]["num"])
    ug.Allocate(ncells)

    # The cell's faces go in as a vtkCellArray.  The older overload that takes
    # a flat face stream is unusable from Python: the wrapper reads the
    # ``nfaces`` argument as the length of the stream and rejects the call.
    face_array = vtk.vtkCellArray()
    for c in range(ncells):
        faces = cell_faces[face_pos[c]:face_pos[c + 1]]

        face_array.Reset()
        seen = {}
        for f in faces:
            nodes = face_nodes[node_pos[f]:node_pos[f + 1]]
            face_array.InsertNextCell(len(nodes))
            for n in nodes:
                n = int(n)
                face_array.InsertCellPoint(n)
                seen[n] = None  # a dict keeps insertion order; a set does not.

        point_ids = list(seen)
        ug.InsertNextCell(vtk.VTK_POLYHEDRON, len(point_ids), point_ids, face_array)


def grid_to_vtk(G: dict, cell_data: dict | None = None):
    """Build a ``vtkUnstructuredGrid`` whose cell order is ``G``'s cell order.

    Parameters
    ----------
    G : dict
        A PRSTCore grid (``cart_grid``, ``tensor_grid`` or ``process_grdecl``,
        with or without ``compute_geometry`` having been run).
    cell_data : dict, optional
        Per-cell arrays to attach, ``{name: values}``, each of length
        ``G['cells']['num']``.

    Returns
    -------
    vtk.vtkUnstructuredGrid
        Carrying cell arrays ``I``, ``J``, ``K`` (zero-based logical indices)
        whenever the grid has a logical index, plus anything in ``cell_data``.
    """
    coords = np.ascontiguousarray(np.asarray(G["nodes"]["coords"], dtype=float))
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(coords, deep=True))

    ug = vtk.vtkUnstructuredGrid()
    ug.SetPoints(points)

    hexes = _lattice_hexahedra(G)
    if hexes is not None and not _hexahedra_agree_with_centroids(G, hexes):
        hexes = None

    if hexes is not None:
        ncells = hexes.shape[0]
        cells = vtk.vtkCellArray()
        # (npts, p0..p7) per cell, the layout SetCells reads.
        flat = np.empty((ncells, 9), dtype=np.int64)
        flat[:, 0] = 8
        flat[:, 1:] = hexes
        id_array = numpy_to_vtk(flat.ravel(), deep=True, array_type=vtk.VTK_ID_TYPE)
        cells.SetCells(ncells, id_array)
        ug.SetCells(vtk.VTK_HEXAHEDRON, cells)
    else:
        _insert_polyhedra(ug, G)

    ijk = cell_ijk(G)
    if ijk is not None:
        for axis, name in enumerate(("I", "J", "K")):
            array = numpy_to_vtk(np.ascontiguousarray(ijk[:, axis]), deep=True)
            array.SetName(name)
            ug.GetCellData().AddArray(array)

    for name, values in (cell_data or {}).items():
        values = np.ascontiguousarray(np.asarray(values, dtype=float).ravel())
        if values.size != ug.GetNumberOfCells():
            raise ValueError(
                "cell array %r has %d values but the grid has %d cells"
                % (name, values.size, ug.GetNumberOfCells()))
        array = numpy_to_vtk(values, deep=True)
        array.SetName(name)
        ug.GetCellData().AddArray(array)

    return ug


def well_block_mask(G: dict, W) -> np.ndarray:
    """A 0/1 per-cell flag marking every cell perforated by a well.

    GeoView carries the same array under the name ``WELL_BLOCKS`` and makes it
    one more stage of the threshold chain, which is how "show only the cells
    the wells actually see" is expressed without a second pipeline.
    """
    mask = np.zeros(int(G["cells"]["num"]), dtype=float)
    for w in (W or []):
        cells = np.asarray(w["cells"], dtype=np.int64).ravel()
        mask[cells] = 1.0
    return mask
