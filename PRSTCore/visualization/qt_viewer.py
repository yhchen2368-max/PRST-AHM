"""A native window around :class:`~PRSTCore.visualization.scene3d.ReservoirScene`.

GeoView puts these controls in a browser and ships every frame over a
websocket.  The controls themselves are the useful part, not the transport,
so here they are Qt widgets driving the same VTK pipeline in-process: no
server, no browser, no client/server copy of the grid.

Qt binding: whichever one VTK's ``QVTKRenderWindowInteractor`` picks, which is
PySide6 when it is installed.  Importing that module first and taking ``Qt``
from what it selected is what keeps the two from disagreeing -- loading a
second binding into the same process is the usual way this crashes.

Run it from a script::

    from PRSTCore.visualization import view_reservoir
    view_reservoir(G, W=W, states=states)
"""

from __future__ import annotations

import sys

import numpy as np

# QVTKRenderWindowInteractor decides the binding; import it before Qt itself
# so that decision is the one the whole process follows.  The name it settled
# on is a global of that module -- it is not written back to ``vtkmodules.qt``,
# which still reads ``None`` afterwards.
import vtkmodules.qt.QVTKRenderWindowInteractor as _qvtk

QVTKRenderWindowInteractor = _qvtk.QVTKRenderWindowInteractor
_BINDING = _qvtk.PyQtImpl

if _BINDING == "PyQt5":
    from PyQt5 import QtCore, QtWidgets
elif _BINDING == "PyQt6":
    from PyQt6 import QtCore, QtWidgets
elif _BINDING == "PySide2":
    from PySide2 import QtCore, QtWidgets
elif _BINDING == "PySide6":
    from PySide6 import QtCore, QtWidgets
else:
    raise ImportError(
        "VTK's Qt widget selected an unsupported binding: %r. Install PySide6."
        % (_BINDING,))

#: PySide spells it ``Signal``, PyQt ``pyqtSignal``; both behave the same here.
_Signal = getattr(QtCore, "Signal", None) or QtCore.pyqtSignal

from .scene3d import ReservoirScene


__all__ = ["ReservoirWindow", "view_reservoir"]


#: Colour maps offered in the picker.  Sequential ones for magnitudes,
#: diverging for anything read against a middle value.
COLORMAPS = ("viridis", "turbo", "plasma", "inferno", "magma", "cividis",
             "jet", "coolwarm", "RdBu_r", "seismic", "terrain", "gist_earth")


class _RangeRow(QtWidgets.QWidget):
    """A labelled pair of spin boxes holding an inclusive ``[lo, hi]`` range.

    Two boxes rather than a two-handled slider: Qt ships no range slider, and
    for I/J/K the exact index matters more than the drag.  The boxes clamp
    against each other so ``lo`` can never pass ``hi``.
    """

    changed = _Signal(int, int)

    def __init__(self, label, minimum, maximum, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel(label))

        self.lo = QtWidgets.QSpinBox()
        self.hi = QtWidgets.QSpinBox()
        for box, value in ((self.lo, minimum), (self.hi, maximum)):
            box.setRange(minimum, maximum)
            box.setValue(value)
            layout.addWidget(box)

        self.lo.valueChanged.connect(self._on_change)
        self.hi.valueChanged.connect(self._on_change)

    def _on_change(self, _value):
        if self.lo.value() > self.hi.value():
            # Push the other box rather than refuse the edit, so dragging a
            # bound past its partner collapses the range instead of sticking.
            if self.sender() is self.lo:
                self.hi.setValue(self.lo.value())
            else:
                self.lo.setValue(self.hi.value())
            return
        self.changed.emit(self.lo.value(), self.hi.value())

    def set_values(self, lo, hi):
        was = self.blockSignals(True)
        self.lo.setValue(lo)
        self.hi.setValue(hi)
        self.blockSignals(was)


class ReservoirWindow(QtWidgets.QMainWindow):
    """The 3D view plus the controls that drive it."""

    def __init__(self, scene: ReservoirScene, title="PRSTCore 3D", parent=None):
        super().__init__(parent)
        self.scene = scene
        self.setWindowTitle(title)
        self.resize(1280, 820)

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.setCentralWidget(self.vtk_widget)

        render_window = self.vtk_widget.GetRenderWindow()
        scene.attach(render_window)
        self.interactor = render_window.GetInteractor()
        self.interactor.SetInteractorStyle(
            __import__("vtk").vtkInteractorStyleTrackballCamera())

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._advance)

        self._building = True
        self._build_controls()
        self._building = False

        if scene.field_names:
            self.field_box.setCurrentText(scene.field_names[0])
            self._on_field(scene.field_names[0])

    # ------------------------------------------------------------- controls

    def _build_controls(self):
        dock = QtWidgets.QDockWidget("Controls", self)
        dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
                             | QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        panel = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(panel)
        form.setSpacing(8)

        # --- what is drawn
        form.addWidget(self._heading("Field"))
        self.field_box = QtWidgets.QComboBox()
        self.field_box.addItems(self.scene.field_names)
        self.field_box.currentTextChanged.connect(self._on_field)
        form.addWidget(self.field_box)

        self.cmap_box = QtWidgets.QComboBox()
        self.cmap_box.addItems(COLORMAPS)
        self.cmap_box.currentTextChanged.connect(self.scene.set_colormap)
        form.addWidget(self.cmap_box)

        # --- time
        form.addWidget(self._heading("Report step"))
        step_row = QtWidgets.QHBoxLayout()
        self.step_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.step_slider.setMinimum(0)
        self.step_slider.valueChanged.connect(self._on_step)
        self.step_label = QtWidgets.QLabel("0")
        self.step_label.setMinimumWidth(56)
        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._on_play)
        step_row.addWidget(self.step_slider, 1)
        step_row.addWidget(self.step_label)
        step_row.addWidget(self.play_button)
        form.addLayout(step_row)

        speed_row = QtWidgets.QHBoxLayout()
        speed_row.addWidget(QtWidgets.QLabel("Frame (ms)"))
        self.speed_box = QtWidgets.QSpinBox()
        self.speed_box.setRange(20, 2000)
        self.speed_box.setSingleStep(20)
        self.speed_box.setValue(200)
        self.speed_box.valueChanged.connect(
            lambda ms: self._timer.setInterval(int(ms)))
        speed_row.addWidget(self.speed_box)
        form.addLayout(speed_row)

        # --- slicing
        if self.scene.dims is not None:
            form.addWidget(self._heading("Index range"))
            nx, ny, nz = self.scene.dims
            self.i_row = _RangeRow("I", 1, nx)
            self.j_row = _RangeRow("J", 1, ny)
            self.k_row = _RangeRow("K", 1, nz)
            self.i_row.changed.connect(lambda lo, hi: self.scene.set_slices(i=(lo, hi)))
            self.j_row.changed.connect(lambda lo, hi: self.scene.set_slices(j=(lo, hi)))
            self.k_row.changed.connect(lambda lo, hi: self.scene.set_slices(k=(lo, hi)))
            for row in (self.i_row, self.j_row, self.k_row):
                form.addWidget(row)
        else:
            self.i_row = self.j_row = self.k_row = None

        # --- value cut-off
        form.addWidget(self._heading("Value range"))
        value_row = QtWidgets.QHBoxLayout()
        self.value_lo = QtWidgets.QDoubleSpinBox()
        self.value_hi = QtWidgets.QDoubleSpinBox()
        for box in (self.value_lo, self.value_hi):
            box.setDecimals(4)
            box.setRange(-1e30, 1e30)
            box.valueChanged.connect(self._on_value_range)
            value_row.addWidget(box)
        form.addLayout(value_row)

        # --- appearance
        form.addWidget(self._heading("Appearance"))
        opacity_row = QtWidgets.QHBoxLayout()
        opacity_row.addWidget(QtWidgets.QLabel("Opacity"))
        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(5, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(
            lambda v: self.scene.set_opacity(v / 100.0))
        opacity_row.addWidget(self.opacity_slider)
        form.addLayout(opacity_row)

        scale_row = QtWidgets.QHBoxLayout()
        scale_row.addWidget(QtWidgets.QLabel("Vertical ×"))
        self.zscale_box = QtWidgets.QDoubleSpinBox()
        self.zscale_box.setRange(0.1, 100.0)
        self.zscale_box.setSingleStep(0.5)
        self.zscale_box.setValue(1.0)
        self.zscale_box.valueChanged.connect(
            lambda v: self.scene.set_axis_scaling(z_factor=v))
        scale_row.addWidget(self.zscale_box)
        form.addLayout(scale_row)

        self.equalize_check = self._check(
            "Equalize axes", True,
            lambda on: self.scene.set_axis_scaling(equalize=on), form)

        self.scalars_check = self._check("Coloured surface", True,
                                         self.scene.show_scalars, form)
        self.wireframe_check = self._check("Wireframe when uncoloured", True,
                                           self.scene.show_wireframe, form)
        self.wells_check = self._check("Wells", True, self.scene.show_wells, form)
        self.blocks_check = self._check("Only well blocks", False,
                                        self.scene.set_well_blocks_only, form)
        self.blocks_check.setEnabled(self.scene.has_well_blocks)

        dark = self._check("Dark background", False, self._on_theme, form)
        dark.setChecked(False)

        # --- camera
        form.addWidget(self._heading("View"))
        reset_row = QtWidgets.QHBoxLayout()
        camera_button = QtWidgets.QPushButton("Reset camera")
        camera_button.clicked.connect(self.scene.reset_camera)
        default_button = QtWidgets.QPushButton("Reset all")
        default_button.clicked.connect(self._on_default_view)
        reset_row.addWidget(camera_button)
        reset_row.addWidget(default_button)
        form.addLayout(reset_row)

        form.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(300)
        dock.setWidget(scroll)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)

    @staticmethod
    def _heading(text):
        label = QtWidgets.QLabel(text)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        return label

    def _check(self, text, checked, slot, layout):
        box = QtWidgets.QCheckBox(text)
        box.setChecked(checked)
        box.toggled.connect(slot)
        layout.addWidget(box)
        return box

    # -------------------------------------------------------------- handlers

    def _on_field(self, name):
        if not name:
            return
        self.scene.set_active_field(name)

        steps = self.scene.n_steps(name)
        self.step_slider.setMaximum(max(0, steps - 1))
        self.step_slider.setEnabled(steps > 1)
        self.play_button.setEnabled(steps > 1)
        self._sync_step_label()

        lo, hi = self.scene.full_range
        span = (hi - lo) or 1.0
        was = self._building
        self._building = True
        for box in (self.value_lo, self.value_hi):
            box.setRange(lo - span, hi + span)
            box.setSingleStep(span / 100.0)
        self.value_lo.setValue(lo)
        self.value_hi.setValue(hi)
        self._building = was

    def _on_step(self, step):
        self.scene.set_step(step)
        self._sync_step_label()

    def _sync_step_label(self):
        self.step_label.setText("%d / %d" % (self.scene.step,
                                             max(0, self.scene.n_steps() - 1)))

    def _on_value_range(self, _value):
        if self._building:
            return
        lo, hi = self.value_lo.value(), self.value_hi.value()
        if lo > hi:
            return
        self.scene.set_value_range(lo, hi)

    def _on_play(self, playing):
        self.play_button.setText("Pause" if playing else "Play")
        if playing:
            self._timer.start(int(self.speed_box.value()))
        else:
            self._timer.stop()

    def _advance(self):
        last = self.scene.n_steps() - 1
        if last <= 0:
            self.play_button.setChecked(False)
            return
        self.step_slider.setValue(0 if self.scene.step >= last
                                  else self.scene.step + 1)

    def _on_theme(self, dark):
        self.scene.set_background((0, 0, 0) if dark else (1, 1, 1))

    def _on_default_view(self):
        self.scene.default_view()
        if self.i_row is not None:
            nx, ny, nz = self.scene.dims
            self.i_row.set_values(1, nx)
            self.j_row.set_values(1, ny)
            self.k_row.set_values(1, nz)
        self.opacity_slider.setValue(100)
        self.step_slider.setValue(0)
        self._on_field(self.field_box.currentText())

    # ----------------------------------------------------------- life cycle

    def start(self):
        """Show the window and hand the render window its interactor."""
        self.show()
        self.interactor.Initialize()
        self.scene.reset_camera()
        return self

    def closeEvent(self, event):
        # Without this the VTK interactor keeps the process alive after the
        # window is gone.
        self._timer.stop()
        self.vtk_widget.Finalize()
        super().closeEvent(event)


def view_reservoir(G, W=None, states=None, static_fields=None, fields=None,
                   title="PRSTCore 3D", equalize_axes=True, block=True):
    """Open the 3D viewer on a PRSTCore grid.

    Parameters
    ----------
    G : dict
        The grid, as the simulator has it.
    W : list[dict], optional
        Wells, drawn as tracks through their completions.
    states : list[dict], optional
        Per-report-step states; ``pressure`` and the saturation columns are
        picked up and become time-varying fields.
    static_fields : dict, optional
        Time-invariant per-cell arrays -- ``{'PORO': poro, 'PERMX': ...}``.
    fields : dict, optional
        Extra fields, each either ``(ncells,)`` or ``(nsteps, ncells)``.
    block : bool
        Run the Qt event loop and return when the window closes.  Pass
        ``False`` from an environment that already runs one, and drive the
        returned window yourself.

    Returns
    -------
    ReservoirWindow
    """
    scene = ReservoirScene(G, W=W, static_fields=static_fields,
                           equalize_axes=equalize_axes)
    if states:
        scene.add_states(states)
    for name, values in (fields or {}).items():
        scene.add_field(name, values)

    if not scene.field_names:
        # Nothing to colour by would render a blank window; a cell index at
        # least shows the grid and proves the geometry arrived.
        scene.add_field("CELL", np.arange(int(G["cells"]["num"]), dtype=float))

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = ReservoirWindow(scene, title=title).start()
    if block:
        app.exec()
    return window
