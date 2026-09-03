"""Python port of MRST's core plotting layer (mrst-2026a/core/plotting):
``boundaryFaces.m``, ``plotFaces.m``, ``plotGrid.m``, ``plotCellData.m``.

This is the piece PRSTCore was missing entirely: a way to actually *look*
at a grid and the scalar fields (pressure, saturation, porosity, ...)
computed on it. Built directly on
:mod:`PRSTCore.gridprocessing` (``cart_grid``/``tensor_grid``/``compute_geometry``).

For 3D grids, cell data is rendered on the *boundary faces of the selected
cell subset* (matching MRST's ``plotCellData`` semantics exactly: a face is
part of the boundary if exactly one of its two neighboring cells is in the
selection). For 2D grids, each selected cell's own polygon is filled with
its data value, matching MRST's ``griddim <= 2`` branch in ``plotCellData``.
"""

from __future__ import annotations

import numpy as np


def boundary_faces(G: dict, cells=None):
    """Port of MRST ``boundaryFaces.m``.

    Returns ``(faces, connected_cell)``: the faces bounding the given
    subset of cells (default: all cells, i.e. the whole grid boundary),
    and for each such face the (0-based) selected cell it is attached to.
    """
    nc = G["cells"]["num"]
    if cells is None:
        cells = np.arange(nc)
    else:
        cells = np.asarray(cells, dtype=int).ravel()

    present = np.zeros(nc, dtype=bool)
    present[cells] = True

    neighbors = np.asarray(G["faces"]["neighbors"])
    a, b = neighbors[:, 0], neighbors[:, 1]
    pa = (a >= 0) & present[np.clip(a, 0, nc - 1)]
    pb = (b >= 0) & present[np.clip(b, 0, nc - 1)]

    is_boundary = pa ^ pb
    f = np.nonzero(is_boundary)[0]
    c = np.where(pa[f], a[f], b[f])
    return f, c


def _face_polygon_3d(G: dict, face: int) -> np.ndarray:
    node_pos = G["faces"]["nodePos"]
    nodes = G["faces"]["nodes"][node_pos[face] : node_pos[face + 1]]
    return np.asarray(G["nodes"]["coords"])[nodes]


def _cell_polygon_2d(G: dict, cell: int) -> np.ndarray:
    """Trace the ordered boundary loop of a 2D cell from its (unordered)
    half-face edges. Handles arbitrary polygonal cells, not just quads."""
    face_pos = G["cells"]["facePos"]
    cell_faces = G["cells"]["faces"]
    hf = cell_faces[face_pos[cell] : face_pos[cell + 1], 0]

    face_nodes = np.asarray(G["faces"]["nodes"]).reshape(-1, 2)
    neighbors = G["faces"]["neighbors"]
    coords = np.asarray(G["nodes"]["coords"])

    edges = {}
    node_set = set()
    for f in hf:
        n0, n1 = face_nodes[f]
        if neighbors[f, 1] == cell:
            n0, n1 = n1, n0
        edges[int(n0)] = int(n1)
        node_set.add(int(n0))
        node_set.add(int(n1))

    try:
        start = next(iter(edges))
        loop = [start]
        cur = edges[start]
        steps = 0
        while cur != start:
            loop.append(cur)
            cur = edges[cur]
            steps += 1
            if steps > len(edges) + 1:
                raise KeyError("non-simple cell boundary (loop did not close)")
        return coords[loop]
    except KeyError:
        # Degenerate/non-simple cell (e.g. a topology-repair artifact):
        # fall back to an angular sort of its node set around the
        # centroid, which always yields a drawable (if not necessarily
        # topologically exact) polygon.
        nodes = np.array(sorted(node_set))
        pts = coords[nodes]
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        return pts[np.argsort(angles)]


def plot_faces(G: dict, faces, data=None, *, ax=None, cmap="viridis", edgecolor="k",
                facecolor="0.85", linewidth=0.3, alpha=1.0, colorbar=True, **patch_kwargs):
    """Port of MRST ``plotFaces.m``: draw the given (3D) faces as patches,
    optionally colored by a per-face scalar ``data`` array."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    faces = np.asarray(faces, dtype=int).ravel()
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

    if faces.size == 0:
        return ax

    verts = [_face_polygon_3d(G, f) for f in faces]
    pc = Poly3DCollection(verts, edgecolor=edgecolor, linewidths=linewidth, alpha=alpha, **patch_kwargs)

    if data is not None:
        data = np.asarray(data, dtype=float).ravel()
        pc.set_array(data)
        pc.set_cmap(cmap)
        if colorbar:
            ax.figure.colorbar(pc, ax=ax, shrink=0.7)
    else:
        pc.set_facecolor(facecolor)

    ax.add_collection3d(pc)
    _autoscale_3d(ax, G)
    return ax


def plot_grid(G: dict, cells=None, *, ax=None, facecolor="0.85", edgecolor="k", linewidth=0.3, alpha=1.0):
    """Port of MRST ``plotGrid.m``: wireframe/flat-shaded outline of the
    grid (or a cell subset), with no data coloring."""
    if int(G["griddim"]) == 3:
        faces, _ = boundary_faces(G, cells)
        return plot_faces(G, faces, data=None, ax=ax, facecolor=facecolor,
                           edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, colorbar=False)

    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    if ax is None:
        _, ax = plt.subplots()

    cell_ids = np.arange(G["cells"]["num"]) if cells is None else np.asarray(cells, dtype=int).ravel()
    polys = [_cell_polygon_2d(G, c) for c in cell_ids]
    pc = PolyCollection(polys, facecolor=facecolor, edgecolor=edgecolor, linewidths=linewidth, alpha=alpha)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    return ax


def plot_cell_data(G: dict, data, cells=None, *, ax=None, cmap="viridis", edgecolor="k",
                     linewidth=0.3, alpha=1.0, colorbar=True, **kwargs):
    """Port of MRST ``plotCellData.m``: color the grid by a per-cell scalar
    field. 3D grids render the boundary faces of the (sub)domain; 1D/2D
    grids render the cells themselves."""
    data = np.asarray(data, dtype=float).ravel()
    griddim = int(G["griddim"])

    if griddim == 3:
        faces, c = boundary_faces(G, cells)
        return plot_faces(G, faces, data[c], ax=ax, cmap=cmap, edgecolor=edgecolor,
                           linewidth=linewidth, alpha=alpha, colorbar=colorbar, **kwargs)

    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    if ax is None:
        _, ax = plt.subplots()

    cell_ids = np.arange(G["cells"]["num"]) if cells is None else np.asarray(cells, dtype=int).ravel()

    if griddim == 2:
        polys = [_cell_polygon_2d(G, c) for c in cell_ids]
        pc = PolyCollection(polys, edgecolor=edgecolor, linewidths=linewidth, alpha=alpha, **kwargs)
        pc.set_array(data[cell_ids])
        pc.set_cmap(cmap)
        ax.add_collection(pc)
        if colorbar:
            ax.figure.colorbar(pc, ax=ax, shrink=0.7)
        ax.autoscale_view()
        ax.set_aspect("equal")
        return ax

    # griddim == 1: MRST plots data(c) vs. cell centroid x-coordinate.
    x = np.asarray(G["cells"]["centroids"]).reshape(-1)[cell_ids]
    ax.plot(x, data[cell_ids], **kwargs)
    return ax


def plot_well(G: dict, W, *, ax=None, radius=1.0, height=5.0, color="r",
              color2=None, cylpts=10, fontsize=16, ambstr=0.8, linewidth=2.0,
              label_wells=True):
    """1:1 port of MRST ``plotWell.m`` (mrst-2026a/core/plotting/plotWell.m).

    Every geometric decision mirrors MRST exactly -- nothing is invented:

    * the bore runs through ``W.cells`` in the *stored order*, one cylinder
      per consecutive pair, oriented along the well's ``W.dir`` entry
      (``'z'``/``'x'``/``'y'``); a single-cell well is forced to ``'s'``;
    * each segment's radius comes from the *first* cell of the pair
      (``cno = W.cells(i)``): ``rl = radius*0.25*sqrt(Amax)`` and
      ``rv = radius*0.25*V/Amax``, with ``Amax`` the cell's largest face area
      (``G.cells.facePos``/``G.cells.faces``/``G.faces.areas``);
    * a single-cell well is drawn as a sphere (ellipsoid) of those radii;
    * the label sits at ``ztop = min(zcoord) - height`` -- ``height`` above
      the top of the model -- at the first cell's (x, y), with a leader line
      from ``ztop`` down through all perforation centroids
      (``c = c([1 1:end],:); c(1,3) = ztop``);
    * ``color2`` (when given) is used for producers (``W.sign < 0``).

    Keyword arguments mirror MRST's ``prm`` exactly: ``radius`` (default 1),
    ``height`` (5), ``color`` ('r'), ``color2`` (None), ``cylpts`` (10),
    ``fontsize`` (16), ``ambstr`` (0.8), ``linewidth`` (2).
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

    if int(G["griddim"]) != 3:
        # MRST asserts 3D; keep a plain-marker fallback for flat grids so
        # callers that feed 2D grids still get well positions.
        centroids = np.asarray(G["cells"]["centroids"])
        for w in W:
            wcells = np.asarray(w.get("cells", []), dtype=int).ravel()
            wcells = wcells[(wcells >= 0) & (wcells < len(centroids))]
            if wcells.size == 0:
                continue
            pts = centroids[wcells]
            ax.plot(pts[:, 0], pts[:, 1], color=color, marker="o")
            if label_wells:
                ax.text(pts[0, 0], pts[0, 1], w.get("name", ""))
        return ax

    centroids = np.asarray(G["cells"]["centroids"])
    face_pos = np.asarray(G["cells"]["facePos"], dtype=int)
    cell_faces = np.asarray(G["cells"]["faces"], dtype=int)[:, 0]
    face_areas = np.asarray(G["faces"]["areas"], dtype=float)
    volumes = np.asarray(G["cells"].get("volumes", np.ones(len(centroids))))
    node_z = np.asarray(G["nodes"]["coords"], dtype=float)[:, 2]
    # MRST: ztop = min(zcoord) - prm.height   (label height above the model top)
    ztop = float(node_z.min()) - height

    def cell_max_face_area(cno):
        """MRST: area = max(G.faces.areas(G.cells.faces(fno))) for cell ``cno``."""
        faces = cell_faces[face_pos[cno]:face_pos[cno + 1]]
        return float(face_areas[faces].max())

    def cell_radii(cno):
        area = cell_max_face_area(cno)
        rl = radius * 0.25 * np.sqrt(area)         # MRST: prm.radius*0.25*sqrt(area)
        rv = radius * 0.25 * volumes[cno] / area   # MRST: prm.radius*0.25*V/area
        return rl, rv

    th = np.linspace(0.0, 2.0 * np.pi, cylpts + 1)  # ring angle; last == first

    for w in W:
        cells = np.asarray(w.get("cells", []), dtype=int).ravel()
        cells = cells[(cells >= 0) & (cells < len(centroids))]
        if cells.size == 0:
            continue
        c = centroids[cells]
        dirc = list(w.get("dir", "z"))
        if cells.size == 1:
            dirc = ["s"]                            # MRST forces dir='s' for one cell
        name = w.get("name", "")
        sign = float(w.get("sign", -1.0))
        wcolor = color2 if (sign < 0 and color2) else (color or "r")

        # Assemble the bore surface as MRST's XW/YW/ZW (rings stacked by row).
        XW, YW, ZW = [], [], []
        # 1) spheres for 's' entries (MRST's first loop: for i=1:numel(dir))
        for i, d in enumerate(dirc):
            if d != "s":
                continue
            rl, rv = cell_radii(cells[i])
            az = np.linspace(0.0, 2.0 * np.pi, cylpts + 1)
            pol = np.linspace(0.0, np.pi, cylpts + 1)
            azm, polm = np.meshgrid(az, pol)
            XW.append(rl * np.sin(polm) * np.cos(azm) + c[i, 0])
            YW.append(rl * np.sin(polm) * np.sin(azm) + c[i, 1])
            ZW.append(rv * np.cos(polm) + c[i, 2])
        # 2) one cylinder per consecutive cell pair along dirc[i] (MRST's
        #    second loop); radius from the first cell of the pair.
        n = cylpts + 1
        for i in range(len(cells) - 1):
            d = dirc[i]
            if d == "s":
                continue
            rl, rv = cell_radii(cells[i])
            ct, st = np.cos(th), np.sin(th)
            c0, c1 = c[i], c[i + 1]
            if d == "z":                          # cylinder([rl;rl]), axis z
                X = rl * ct + np.array([c0[0], c1[0]])[:, None]
                Y = rl * st + np.array([c0[1], c1[1]])[:, None]
                Z = np.zeros((2, n)) + np.array([c0[2], c1[2]])[:, None]
            elif d == "x":                        # [zw,yw,xw]=cyl; xw=0 -> axis x, R=rv
                X = np.zeros((2, n)) + np.array([c0[0], c1[0]])[:, None]
                Y = rv * ct + np.array([c0[1], c1[1]])[:, None]
                Z = rv * st + np.array([c0[2], c1[2]])[:, None]
            elif d == "y":                        # [xw,zw,yw]=cyl; yw=0 -> axis y, R=rv
                X = rv * ct + np.array([c0[0], c1[0]])[:, None]
                Y = np.zeros((2, n)) + np.array([c0[1], c1[1]])[:, None]
                Z = rv * st + np.array([c0[2], c1[2]])[:, None]
            else:                                 # generic: R=rl along z, zw=rv*(zw-0.5)
                zoff = (rv * (np.array([0.0, 1.0]) - 0.5))[:, None]
                X = rl * ct + np.array([c0[0], c1[0]])[:, None]
                Y = rl * st + np.array([c0[1], c1[1]])[:, None]
                Z = np.broadcast_to(zoff, (2, n)) + np.array([c0[2], c1[2]])[:, None]
            XW.append(X)
            YW.append(Y)
            ZW.append(Z)
        if not XW:
            continue
        XW = np.vstack(XW)
        YW = np.vstack(YW)
        ZW = np.vstack(ZW)
        m, n = XW.shape                      # m rings, n == cylpts + 1 columns
        verts = np.column_stack([XW.ravel(), YW.ravel(), ZW.ravel()])

        # side faces -- MRST's `faces(sz, I(:), J(:))` quad table
        quads = []
        for i in range(m - 1):
            for j in range(n - 1):
                quads.append([i * n + j, (i + 1) * n + j,
                              (i + 1) * n + j + 1, i * n + j + 1])
        ax.add_collection3d(Poly3DCollection(
            [verts[q] for q in quads], color=wcolor, edgecolor="none",
            alpha=ambstr))

        # top and bottom caps (MRST draws them with EdgeColor 'k'); the pole
        # rings of a sphere are degenerate (single point), skip those.
        for ring in (range(0, n), range((m - 1) * n, m * n)):
            pts = verts[list(ring)]
            if np.ptp(pts, axis=0).max() > 1e-12:
                ax.add_collection3d(Poly3DCollection(
                    [pts], color=wcolor, edgecolor="k", alpha=ambstr))

        # leader line from ztop through all perforation centroids
        # (MRST: c = c([1 1:end],:); c(1,3) = ztop; plot3(c(:,1),c(:,2),c(:,3)))
        path = np.vstack([np.array([c[0, 0], c[0, 1], ztop]), c])
        ax.plot(path[:, 0], path[:, 1], path[:, 2],
                color=wcolor, linewidth=linewidth)

        # label at the first cell's (x, y), bottom-aligned at ztop
        if label_wells and fontsize > 0:
            ax.text(c[0, 0], c[0, 1], ztop, name, fontsize=fontsize,
                    color=wcolor, verticalalignment="bottom", zorder=6)
    return ax


def plot_face_data(G: dict, data, cells=None, *, ax=None, cmap="viridis", colorbar=True, **patch_kwargs):
    """Port of MRST ``plotFaceData.m``: color the *exterior* (boundary)
    faces of ``G`` (or a cell subset) by a per-face scalar field ``data``
    (length ``G['faces']['num']``)."""
    assert int(G["griddim"]) == 3, "plotFaceData is only supported in 3D"
    data = np.asarray(data, dtype=float).ravel()
    faces, _ = boundary_faces(G, cells)
    patch_kwargs.setdefault("edgecolor", "none")
    return plot_faces(G, faces, data[faces], ax=ax, cmap=cmap, colorbar=colorbar, **patch_kwargs)


def plot_grid_volumes(G: dict, values, *, ax=None, n=20, vmin=None, vmax=None, mesh=None,
                        cmap="jet", basealpha=1.0, extrudefaces=True):
    """Port of MRST ``plotGridVolumes.m``: partially-transparent isosurfaces
    of a scattered per-cell field ``values``, interpolated onto a regular
    voxel grid over the domain's bounding box and rendered with
    ``skimage.measure.marching_cubes`` (MRST's own ``isosurface``/``patch``
    pipeline has no direct Python equivalent).

    Functionally redundant with :func:`plot_cell_data` (both visualize a
    per-cell scalar field) -- this gives an interpolated volumetric-shell
    view instead of a per-cell-face flat-shaded one.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    from skimage import measure

    assert int(G["griddim"]) == 3, "plotGridVolumes is only supported in 3D"
    values = np.asarray(values, dtype=float).ravel()
    gc = np.asarray(G["cells"]["centroids"], dtype=float)

    if extrudefaces:
        bf, bc = boundary_faces(G)
        gfc = np.asarray(G["faces"]["centroids"])[bf]
        gc = np.vstack([gc, gfc])
        values = np.concatenate([values, values[bc]])

    v = values
    if vmin is not None:
        v = v[v >= vmin]
    if vmax is not None:
        v = v[v <= vmax]
    counts, edges = np.histogram(v, bins=n)
    binc = 0.5 * (edges[:-1] + edges[1:])

    node_coords = np.asarray(G["nodes"]["coords"], dtype=float)
    m = np.asarray(G["cartDims"], dtype=int) + 2 if mesh is None else np.asarray(mesh, dtype=int)
    lo = np.maximum(gc.min(axis=0), node_coords.min(axis=0))
    hi = np.minimum(gc.max(axis=0), node_coords.max(axis=0))
    axes = [np.linspace(lo[d], hi[d], int(m[d]) + 1) for d in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")

    try:
        interp = LinearNDInterpolator(gc, values)
        vol = interp(X, Y, Z)
        nan_mask = np.isnan(vol)
        if np.any(nan_mask):
            vol[nan_mask] = NearestNDInterpolator(gc, values)(X[nan_mask], Y[nan_mask], Z[nan_mask])
    except Exception:
        vol = NearestNDInterpolator(gc, values)(X, Y, Z)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

    colors = plt.get_cmap(cmap)(np.linspace(0, 1, n))
    spacing = tuple((axes[d][-1] - axes[d][0]) / m[d] for d in range(3))
    origin = np.array([axes[0][0], axes[1][0], axes[2][0]])
    for i, level in enumerate(binc):
        if not (vol.min() < level < vol.max()):
            continue
        try:
            verts, faces, _, _ = measure.marching_cubes(vol, level=level, spacing=spacing)
        except (ValueError, RuntimeError):
            continue
        verts = verts + origin
        pc = Poly3DCollection(verts[faces], facecolor=colors[i], edgecolor="none",
                               alpha=min(basealpha * (i + 1) / (2 * n), 1.0))
        ax.add_collection3d(pc)

    _autoscale_3d(ax, G)
    return ax


def _autoscale_3d(ax, G: dict) -> None:
    coords = np.asarray(G["nodes"]["coords"])
    mins, maxs = coords.min(axis=0), coords.max(axis=0)
    center = (mins + maxs) / 2
    radius = max((maxs - mins).max() / 2, 1e-9)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _logical_ijk(G: dict):
    """Zero-based ``(i, j, k)`` logical index per cell, or ``None``.

    Kept here (not in :mod:`~PRSTCore.visualization.vtk_grid`) so this
    matplotlib-only module never pulls VTK in.  Same convention as
    ``vtk_grid.cell_ijk``: MRST's natural order runs i fastest (Fortran).
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
        if ncells != int(np.prod(dims)):
            return None
        index_map = np.arange(ncells, dtype=np.int64)
    index_map = np.asarray(index_map, dtype=np.int64).ravel()
    if index_map.size != ncells:
        return None
    i, j, k = np.unravel_index(index_map, tuple(int(x) for x in dims), order="F")
    return np.stack([i, j, k], axis=1).astype(np.int32)


def slice_cell_polygons(G: dict, axis: int, index: int):
    """2D polygons of the cells cut by a slice plane.

    Parameters
    ----------
    G : dict
        A 3D PRSTCore grid.
    axis : int
        0 = I plane, 1 = J plane, 2 = K plane.
    index : int
        1-based slice index.

    Returns
    -------
    (polys, cell_ids) : list of ``(n, 2)`` arrays in cell order, and the
    active cell indices whose face lies in the slice plane.
    """
    ijk = _logical_ijk(G)
    if ijk is None:
        return [], np.zeros(0, dtype=int)
    selected = np.nonzero(ijk[:, axis] == int(index) - 1)[0]
    if selected.size == 0:
        return [], selected

    coords = np.asarray(G["nodes"]["coords"], dtype=float)
    face_pos = np.asarray(G["cells"]["facePos"], dtype=int).ravel()
    cell_faces = np.asarray(G["cells"]["faces"], dtype=int)
    if cell_faces.ndim == 2:
        cell_faces = cell_faces[:, 0]
    cell_faces = cell_faces.ravel()
    node_pos = np.asarray(G["faces"]["nodePos"], dtype=int).ravel()
    face_nodes = np.asarray(G["faces"]["nodes"], dtype=int).ravel()
    other_axes = [a for a in (0, 1, 2) if a != axis]

    polygons = []
    for cell in selected:
        best = None                       # (mean along axis, face id)
        for face in cell_faces[face_pos[cell]:face_pos[cell + 1]]:
            nodes = face_nodes[node_pos[face]:node_pos[face + 1]]
            if len(nodes) < 3:
                continue
            values = coords[nodes, axis]
            mean = float(values.mean())
            if values.max() - values.min() <= 1e-6 * (abs(mean) + 1.0):
                if best is None or mean > best[0]:
                    best = (mean, face)
        if best is not None:
            nodes = face_nodes[node_pos[best[1]]:node_pos[best[1] + 1]]
            polygons.append(coords[nodes][:, other_axes])
    return polygons, selected


def plot_slice(G: dict, data, axis: int, index: int, *, wells=None, ax=None,
               cmap="viridis", edgecolor="0.25", linewidth=0.3,
               colorbar=True, label_wells=True):
    """Plot a 2D slice of a 3D grid, coloured by a per-cell scalar.

    The plane is a logical index cut (0=I, 1=J, 2=K) at 1-based ``index``;
    each cut cell is drawn as its face lying in that plane, exactly the shape
    an MRST ``plotCellData`` slice would show.  Wells whose completions touch
    the slice are drawn as markers with their name.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    if ax is None:
        _, ax = plt.subplots()
    polygons, cell_ids = slice_cell_polygons(G, axis, index)
    if not polygons:
        ax.text(0.5, 0.5, "no cells in this slice", ha="center",
                va="center", transform=ax.transAxes)
        return ax

    values = np.asarray(data, dtype=float).ravel()[cell_ids]
    collection = PolyCollection(polygons, edgecolor=edgecolor,
                                linewidths=linewidth)
    collection.set_array(values)
    collection.set_cmap(cmap)
    ax.add_collection(collection)

    if wells:
        order = {int(c): i for i, c in enumerate(cell_ids)}
        for well in wells:
            cells = np.asarray(well.get("cells", []), dtype=int).ravel()
            for cell in cells:
                i = order.get(int(cell))
                if i is None:
                    continue
                point = np.asarray(polygons[i]).mean(axis=0)
                ax.plot(point[0], point[1], marker="o", color="k",
                        markersize=4, zorder=4)
                if label_wells:
                    ax.annotate(str(well.get("name", "")), (point[0], point[1]),
                                xytext=(4, 4), textcoords="offset points",
                                fontsize=8, fontweight="bold", color="white",
                                path_effects=[__import__(
                                    "matplotlib.patheffects", fromlist=[
                                        "withStroke"]).withStroke(
                                    linewidth=2, foreground="black")],
                                zorder=5)
                break

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_axisbelow(True)
    if colorbar:
        ax.figure.colorbar(collection, ax=ax, shrink=0.7)
    return ax
