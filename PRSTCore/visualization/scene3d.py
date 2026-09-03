"""The 3D reservoir scene, lifted out of GeoView's ``view_3d``/``processing``.

GeoView draws a reservoir with a chain of ``vtkThreshold`` filters -- I, then
J, then K, then well blocks, then the value range -- feeding one geometry
filter and one actor.  That is the whole idea, and it is worth keeping: the
slicing, the "only cells the wells see" toggle and the value cut-off are the
same mechanism applied five times, so there is one pipeline to rebuild and
one actor to swap rather than a filter per control.

What is *not* kept is the transport.  In GeoView every one of these functions
is a ``@state.change`` callback and the picture leaves over a websocket to a
browser.  Here the same VTK calls sit on a plain object with ordinary methods,
so the pipeline can be driven from a native window (see
:mod:`PRSTCore.visualization.qt_viewer`), from a script, or from a test.

The scene owns no data of its own: cell arrays come from a PRSTCore grid
``G`` via :func:`~PRSTCore.visualization.vtk_grid.grid_to_vtk`, and scalar
fields are handed in as plain arrays indexed exactly like ``G``'s cells.
"""

from __future__ import annotations

import numpy as np

import vtk
from vtk.util.numpy_support import numpy_to_vtk

from .vtk_grid import grid_to_vtk, cell_ijk, well_block_mask

# vtkThreshold tries the GPU-accelerated vtkmThreshold first; for grids whose
# cells are not a single supported type it logs a "falling back" WARN on every
# rebuild (dragging an I/J/K slider re-runs it, flooding the console with
# hundreds of lines).  The fallback is harmless and the plain implementation
# always produces the same result, so quiet the logger down to error level.
try:
    vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_ERROR)
except Exception:  # pragma: no cover - older VTK has no such knob
    pass


__all__ = ["ReservoirScene"]


#: The array the threshold chain and the mapper both read.  One name, rewritten
#: in place whenever the field or the timestep changes, so nothing downstream
#: has to be rebuilt to point at a different array.
ACTIVE = "ActiveScalars"

#: The extra thresholds, in the order GeoView chains them.
_WELL_BLOCKS = "WELL_BLOCKS"


class ReservoirScene:
    """A renderer, a grid and the threshold chain that filters it.

    Parameters
    ----------
    G : dict
        A PRSTCore grid.  Cell ``k`` here is cell ``k`` in every field.
    W : list[dict], optional
        PRSTCore wells (``{'name', 'cells', ...}``).  Drawn as a track through
        the centroids of the cells they perforate, with a stem up to above the
        reservoir and a label, as GeoView does.
    static_fields : dict, optional
        Per-cell arrays that do not vary in time, ``{name: (ncells,)}`` --
        porosity, permeability, region numbers.
    equalize_axes : bool
        Scale the actors so the model's three extents are equal, which is what
        GeoView does and what stops a reservoir far wider than it is thick
        from rendering as a sheet.  The scaling is applied to the actors, not
        to the data, so picked coordinates stay in model units.
    """

    def __init__(self, G, W=None, static_fields=None, equalize_axes=True):
        self.G = G
        self.W = list(W or [])

        cell_data = dict(static_fields or {})
        if self.W:
            cell_data[_WELL_BLOCKS] = well_block_mask(G, self.W)

        self.grid = grid_to_vtk(G, cell_data)
        self.has_ijk = cell_ijk(G) is not None
        self.has_well_blocks = _WELL_BLOCKS in cell_data

        #: ``{name: array}``; an array is either ``(ncells,)`` (static) or
        #: ``(nsteps, ncells)`` (a state variable through time).
        self.fields = {name: np.asarray(v, dtype=float)
                       for name, v in (static_fields or {}).items()}

        self.dims = self._logical_dims()
        self._active = None
        self._step = 0
        self._value_range = (0.0, 1.0)
        self._full_range = (0.0, 1.0)
        self._slices = self._full_slices()
        self._well_blocks_only = False
        self._colormap = "viridis"
        self._opacity = 1.0
        self._show_scalars = True
        self._show_wireframe = True

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1, 1, 1)

        self._equalize = bool(equalize_axes)
        self._z_factor = 1.0
        self.scales = self._axis_scales() if equalize_axes else np.ones(3)

        self.actor = vtk.vtkActor()
        self.actor.SetScale(*self.scales)
        self.renderer.AddActor(self.actor)

        self.scalar_bar = vtk.vtkScalarBarActor()
        self.scalar_bar.SetNumberOfLabels(5)
        # Left to itself the bar takes a quarter of the window; pin it to a
        # slim strip down the right-hand edge.
        self.scalar_bar.SetWidth(0.08)
        self.scalar_bar.SetHeight(0.55)
        self.scalar_bar.SetPosition(0.90, 0.22)
        self.scalar_bar.GetLabelTextProperty().SetColor(0, 0, 0)
        self.scalar_bar.GetLabelTextProperty().SetFontSize(11)
        self.scalar_bar.GetTitleTextProperty().SetColor(0, 0, 0)
        self.scalar_bar.GetTitleTextProperty().SetFontSize(12)
        self.scalar_bar.SetUnconstrainedFontSize(True)
        self.renderer.AddActor2D(self.scalar_bar)

        self.well_actors = []
        if self.W:
            self._add_wells()

        self._render_window = None

    # ---------------------------------------------------------------- setup

    def _logical_dims(self):
        dims = self.G.get("cartDims")
        if dims is None:
            return None
        dims = np.asarray(dims, dtype=np.int64).ravel()
        return tuple(int(x) for x in dims) if dims.size == 3 else None

    def _full_slices(self):
        if self.dims is None:
            return None
        return [[1, self.dims[0]], [1, self.dims[1]], [1, self.dims[2]]]

    def _axis_scales(self):
        bounds = np.asarray(self.grid.GetBounds(), dtype=float)
        extent = np.abs(bounds[1::2] - bounds[0::2])
        extent[extent == 0] = 1.0
        return extent.max() / extent

    def set_axis_scaling(self, equalize=None, z_factor=None):
        """How much the vertical is stretched.

        ``equalize`` is GeoView's default: scale each axis so the model's
        three extents match, which turns a reservoir far wider than it is
        thick into something you can actually see layering in -- at the cost
        of showing a 10:1 model as a cube.  Turn it off for true proportions,
        and use ``z_factor`` for a chosen exaggeration on top of either.
        """
        if equalize is not None:
            self._equalize = bool(equalize)
        if z_factor is not None:
            self._z_factor = float(z_factor)

        scales = self._axis_scales() if self._equalize else np.ones(3)
        scales = np.asarray(scales, dtype=float).copy()
        scales[2] *= self._z_factor
        self.scales = scales

        self.actor.SetScale(*self.scales)
        for actor in self.well_actors:
            if isinstance(actor, vtk.vtkActor):
                actor.SetScale(*self.scales)
        # Label anchors are 2D: they carry no transform of their own, so the
        # scaling has to be baked into their coordinates again.
        self._place_well_labels()

        self.reset_camera()
        return self

    def attach(self, render_window):
        """Bind the scene to a render window so changes repaint themselves."""
        self._render_window = render_window
        render_window.AddRenderer(self.renderer)
        return self

    def render(self):
        if self._render_window is not None:
            self._render_window.Render()

    # --------------------------------------------------------------- fields

    def add_field(self, name, values):
        """Register a scalar field.

        ``values`` is either ``(ncells,)`` for something static or
        ``(nsteps, ncells)`` for a state variable through time.  Which one it
        is decides whether the timestep control does anything for this field.
        """
        values = np.asarray(values, dtype=float)
        ncells = int(self.G["cells"]["num"])
        if values.ndim == 1:
            if values.size != ncells:
                raise ValueError("field %r has %d values for %d cells"
                                 % (name, values.size, ncells))
        elif values.ndim == 2:
            if values.shape[1] != ncells:
                raise ValueError("field %r has %d columns for %d cells"
                                 % (name, values.shape[1], ncells))
        else:
            raise ValueError("field %r must be 1- or 2-dimensional" % (name,))
        self.fields[name] = values
        return self

    def add_states(self, states, keys=("pressure", "s")):
        """Register the usual per-step arrays out of a list of PRST states.

        ``s`` is the saturation matrix, one column per phase; its columns are
        split out as ``SW``/``SO``/``SG`` following PRSTCore's phase order for
        a two- or three-phase model.  Anything in ``keys`` that a state does
        not carry is skipped rather than faked.
        """
        states = list(states)
        if not states:
            return self

        phase_names = {1: ("S",), 2: ("SW", "SO"), 3: ("SW", "SO", "SG")}
        for key in keys:
            if key not in states[0]:
                continue
            stack = np.asarray([np.asarray(s[key], dtype=float) for s in states])
            if stack.ndim == 2:
                self.add_field(key.upper(), stack)
            elif stack.ndim == 3:
                names = phase_names.get(stack.shape[2])
                for p in range(stack.shape[2]):
                    label = names[p] if names else "%s%d" % (key.upper(), p + 1)
                    self.add_field(label, stack[:, :, p])
        return self

    @property
    def field_names(self):
        return sorted(self.fields)

    def n_steps(self, name=None):
        """How many timesteps the named field has (1 if it is static)."""
        name = name or self._active
        values = self.fields.get(name)
        return int(values.shape[0]) if values is not None and values.ndim == 2 else 1

    # --------------------------------------------------------- active field

    def set_active_field(self, name, reset_range=True):
        """Show ``name``, and by default re-fit the value range to it."""
        if name not in self.fields:
            raise KeyError("no such field: %r" % (name,))
        self._active = name
        self._step = min(self._step, self.n_steps(name) - 1)
        self._write_active_scalars(update_range=reset_range)
        self.scalar_bar.SetTitle(name)
        self._rebuild()
        return self

    def set_step(self, step):
        """Move to a report step.  A static field simply ignores it."""
        self._step = max(0, min(int(step), self.n_steps() - 1))
        if self._active is None:
            return self
        self._write_active_scalars(update_range=False)
        self._rebuild()
        return self

    @property
    def step(self):
        return self._step

    def _write_active_scalars(self, update_range):
        """Port of GeoView's ``common.set_active_scalars``.

        One deliberate difference: the colour scale is held at the field's
        range over *all* timesteps.  GeoView re-fits the mapper to the visible
        step (``mapper.SetScalarRange(vtk_grid.GetScalarRange())``), which
        means a colour means something different in every frame and an
        animation of a depleting field looks flat.  The range over all steps
        keeps frames comparable; ``set_value_range`` still cuts cells away.
        """
        values = self.fields[self._active]
        if values.ndim == 2:
            if update_range:
                self._full_range = (float(values.min()), float(values.max()))
            data = values[self._step]
        else:
            if update_range:
                self._full_range = (float(values.min()), float(values.max()))
            data = values

        if update_range:
            self._value_range = self._full_range

        array = numpy_to_vtk(np.ascontiguousarray(data), deep=True)
        array.SetName(ACTIVE)
        self.grid.GetCellData().AddArray(array)
        self.grid.GetCellData().SetActiveScalars(ACTIVE)

    @property
    def full_range(self):
        """The min/max of the active field over every timestep."""
        return self._full_range

    # ------------------------------------------------------------ filtering

    def _make_threshold(self, limits, attr, upstream=None, ijk=False):
        """Port of GeoView's ``view_3d.make_threshold``.

        ``ijk`` selects the one-based inclusive convention the I/J/K sliders
        use; everything else thresholds on the value directly.
        """
        threshold = vtk.vtkThreshold()
        if upstream is None:
            threshold.SetInputData(self.grid)
        else:
            threshold.SetInputConnection(upstream.GetOutputPort())

        if ijk:
            lo, hi = int(limits[0]), int(limits[1])
            threshold.SetLowerThreshold(lo - 1)
            # A single-index selection needs a half-open top, or vtkThreshold
            # keeps nothing when the two bounds land on the same integer.
            threshold.SetUpperThreshold(hi - 0.5 if lo == hi else hi - 1)
        else:
            threshold.SetLowerThreshold(float(limits[0]))
            threshold.SetUpperThreshold(float(limits[1]))

        threshold.SetInputArrayToProcess(0, 0, 0, 1, attr)  # 1 = cell data
        return threshold

    def _rebuild(self):
        """Rebuild the threshold chain and re-point the actor at its output."""
        if self._active is None:
            return

        stage = None
        if self.has_ijk and self._slices is not None:
            for axis, name in enumerate(("I", "J", "K")):
                stage = self._make_threshold(self._slices[axis], name,
                                             upstream=stage, ijk=True)
        if self.has_well_blocks:
            stage = self._make_threshold(
                [0.5, 1.5] if self._well_blocks_only else [-0.5, 1.5],
                _WELL_BLOCKS, upstream=stage)
        stage = self._make_threshold(self._value_range, ACTIVE, upstream=stage)
        stage.Update()

        surface = vtk.vtkGeometryFilter()
        surface.SetInputData(stage.GetOutput())

        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(surface.GetOutputPort())
        mapper.SetScalarRange(*self._full_range)
        self.actor.SetMapper(mapper)

        self._apply_colormap()
        self._apply_visibility()
        self.actor.GetProperty().SetOpacity(self._opacity)
        self.render()

    def set_slices(self, i=None, j=None, k=None):
        """Restrict the visible box by one-based inclusive I/J/K ranges."""
        if self._slices is None:
            return self
        for axis, value in enumerate((i, j, k)):
            if value is not None:
                self._slices[axis] = [int(value[0]), int(value[1])]
        self._rebuild()
        return self

    @property
    def slices(self):
        return None if self._slices is None else [list(s) for s in self._slices]

    def set_value_range(self, lo, hi):
        """Hide cells whose value falls outside ``[lo, hi]``."""
        self._value_range = (float(lo), float(hi))
        self._rebuild()
        return self

    def set_well_blocks_only(self, only):
        self._well_blocks_only = bool(only)
        self._rebuild()
        return self

    # ------------------------------------------------------------ appearance

    def set_colormap(self, name):
        self._colormap = name
        self._apply_colormap()
        self.render()
        return self

    def _apply_colormap(self):
        """Port of ``view_3d.update_cmap``: a matplotlib colormap as a LUT."""
        mapper = self.actor.GetMapper()
        if mapper is None:
            return
        if not self._show_scalars:
            mapper.ScalarVisibilityOff()
            return

        from matplotlib import colormaps

        cmap = colormaps[self._colormap]
        colors = cmap(np.arange(cmap.N))
        table = vtk.vtkLookupTable()
        table.SetNumberOfTableValues(len(colors))
        for i, rgba in enumerate(colors):
            table.SetTableValue(i, rgba[0], rgba[1], rgba[2])
        table.Build()
        mapper.SetLookupTable(table)
        mapper.SetScalarRange(*self._full_range)
        self.scalar_bar.SetLookupTable(table)

    def set_opacity(self, opacity):
        self._opacity = float(opacity)
        self.actor.GetProperty().SetOpacity(self._opacity)
        self.render()
        return self

    def show_scalars(self, show):
        self._show_scalars = bool(show)
        self._apply_visibility()
        self._apply_colormap()
        self.render()
        return self

    def show_wireframe(self, show):
        self._show_wireframe = bool(show)
        self._apply_visibility()
        self.render()
        return self

    def _apply_visibility(self):
        """Port of ``change_field_visibility``/``change_wireframe_visibility``.

        The two toggles interact: scalars off but wireframe on draws the mesh
        without colour, and both off hides the actor entirely.
        """
        prop = self.actor.GetProperty()
        mapper = self.actor.GetMapper()
        if self._show_scalars:
            prop.SetRepresentationToSurface()
            self.actor.SetVisibility(True)
            if mapper is not None:
                mapper.ScalarVisibilityOn()
            self.scalar_bar.SetVisibility(True)
        elif self._show_wireframe:
            prop.SetRepresentationToWireframe()
            self.actor.SetVisibility(True)
            if mapper is not None:
                mapper.ScalarVisibilityOff()
            self.scalar_bar.SetVisibility(False)
        else:
            self.actor.SetVisibility(False)
            self.scalar_bar.SetVisibility(False)

    def show_wells(self, show):
        for actor in self.well_actors:
            actor.SetVisibility(bool(show))
        self.render()
        return self

    # ---------------------------------------------------------------- wells

    def _well_track(self, w):
        """The perforated cells' centroids, ordered top-down.

        PRSTCore wells carry completion cells rather than a survey, so the
        track is the centroids of the cells the well actually opens -- the
        same thing ``grid_plots.plot_well`` draws in matplotlib.

        Completions are collapsed to one point per layer first.  A well that
        opens several cells in the same layer (a deviated or multi-column
        completion) otherwise draws as a zigzag, because sorting by depth
        alone puts cells at equal depth in arbitrary order.
        """
        centroids = self.G["cells"].get("centroids")
        if centroids is None:
            return None
        cells = np.asarray(w["cells"], dtype=np.int64).ravel()
        if cells.size == 0:
            return None
        pts = np.asarray(centroids, dtype=float)[cells]

        ijk = cell_ijk(self.G)
        if ijk is not None:
            layer = ijk[cells, 2]
            order = np.argsort(layer, kind="stable")
            layer, pts = layer[order], pts[order]
            edges = np.flatnonzero(np.diff(layer)) + 1
            pts = np.array([group.mean(axis=0)
                            for group in np.split(pts, edges)])

        return pts[np.argsort(pts[:, 2])]

    def _add_wells(self):
        """Port of ``processing.add_wells``, against PRSTCore's well dicts."""
        colors = vtk.vtkNamedColors()
        bounds = np.asarray(self.grid.GetBounds(), dtype=float)
        z_min = bounds[4] - 0.1 * (bounds[5] - bounds[4])

        points, lines = vtk.vtkPoints(), vtk.vtkCellArray()
        stem_points, stem_lines = vtk.vtkPoints(), vtk.vtkCellArray()
        labels = vtk.vtkStringArray()
        labels.SetName("labels")

        heads = []
        drawn = 0
        for w in self.W:
            track = self._well_track(w)
            if track is None:
                continue

            ids = [points.InsertNextPoint(*row) for row in track]
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(len(ids))
            for n, pid in enumerate(ids):
                line.GetPointIds().SetId(n, pid)
            lines.InsertNextCell(line)

            # A stem from the top completion up to above the model, so the
            # label does not sit buried inside the grid.
            head = np.array([track[0, 0], track[0, 1], z_min])
            stem = [stem_points.InsertNextPoint(*head),
                    stem_points.InsertNextPoint(*track[0])]
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(2)
            for n, pid in enumerate(stem):
                line.GetPointIds().SetId(n, pid)
            stem_lines.InsertNextCell(line)

            heads.append(head)
            labels.InsertNextValue(str(w.get("name", "")))
            drawn += 1

        if drawn == 0:
            return

        track_poly = vtk.vtkPolyData()
        track_poly.SetPoints(points)
        track_poly.SetLines(lines)
        self.well_actors.append(
            self._line_actor(track_poly, colors.GetColor3d("Red"), width=3))

        stem_poly = vtk.vtkPolyData()
        stem_poly.SetPoints(stem_points)
        stem_poly.SetLines(stem_lines)
        self.well_actors.append(
            self._line_actor(stem_poly, colors.GetColor3d("Green"), width=2))

        self._label_heads = np.asarray(heads, dtype=float)
        self._label_poly = vtk.vtkPolyData()
        self._label_poly.GetPointData().AddArray(labels)
        self._place_well_labels()

        label_mapper = vtk.vtkLabeledDataMapper()
        label_mapper.SetInputData(self._label_poly)
        label_mapper.SetFieldDataName("labels")
        label_mapper.SetLabelModeToLabelFieldData()
        label_actor = vtk.vtkActor2D()
        label_actor.SetMapper(label_mapper)
        self.renderer.AddActor(label_actor)
        self.well_actors.append(label_actor)

    def _place_well_labels(self):
        """Re-anchor the well labels for the current axis scaling."""
        heads = getattr(self, "_label_heads", None)
        if heads is None or len(heads) == 0:
            return
        points = vtk.vtkPoints()
        for head in heads * self.scales:
            points.InsertNextPoint(*head)
        self._label_poly.SetPoints(points)
        self._label_poly.Modified()

    def _line_actor(self, poly, color, width):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.SetScale(*self.scales)
        actor.GetProperty().SetLineWidth(width)
        actor.GetProperty().SetColor(color)
        self.renderer.AddActor(actor)
        return actor

    # --------------------------------------------------------------- camera

    def reset_camera(self):
        """Port of ``common.reset_camera``: look down on the model from a
        corner, with +z pointing *down* because reservoir depth increases
        downward and an unflipped camera shows the model upside down."""
        self.renderer.ResetCamera()
        camera = self.renderer.GetActiveCamera()
        position = np.asarray(camera.GetPosition(), dtype=float)
        focal = np.asarray(camera.GetFocalPoint(), dtype=float)
        distance = float(np.linalg.norm(position - focal))
        offset = distance / np.sqrt(3.0)
        camera.SetPosition(focal[0] - offset, focal[1] - offset, focal[2] - offset)
        camera.SetViewUp(0, 0, -1)
        self.renderer.ResetCamera()
        self.render()
        return self

    def set_background(self, rgb):
        self.renderer.SetBackground(*rgb)
        text = (0, 0, 0) if sum(rgb) > 1.5 else (1, 1, 1)
        self.scalar_bar.GetLabelTextProperty().SetColor(*text)
        self.scalar_bar.GetTitleTextProperty().SetColor(*text)
        self.render()
        return self

    def default_view(self):
        """Port of ``view_3d.default_view``: every control back to its start."""
        self._slices = self._full_slices()
        self._value_range = self._full_range
        self._opacity = 1.0
        self._show_scalars = True
        self._show_wireframe = True
        self._well_blocks_only = False
        self._step = 0
        if self._active is not None:
            self._write_active_scalars(update_range=True)
        self._rebuild()
        self.show_wells(True)
        self.reset_camera()
        return self
