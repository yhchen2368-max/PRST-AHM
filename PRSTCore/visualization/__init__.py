"""Visualization utilities ported from MRST, plus a VTK/Qt 3D viewer.

Two layers, deliberately separate:

* :mod:`~PRSTCore.visualization.grid_plots` -- the MRST plotting core
  (``plotGrid``, ``plotCellData``, ...) on matplotlib, for figures;
* :mod:`~PRSTCore.visualization.scene3d` and
  :mod:`~PRSTCore.visualization.qt_viewer` -- an interactive 3D reservoir
  view on VTK, for looking around a model.

The VTK half is imported lazily.  It needs ``vtk`` and a Qt binding (``vtk``
has no PyPI wheel for CPython 3.14, but conda-forge ships it -- ``conda
install -c conda-forge vtk`` -- and PySide6 wheels cover 3.14), and a hard
import here would take the matplotlib half down with it on any interpreter
that only runs the solver, which does not need VTK.
"""

from .grid_plots import (boundary_faces, plot_cell_data, plot_face_data, plot_faces,
                          plot_grid, plot_grid_volumes, plot_slice, plot_well,
                          slice_cell_polygons)

__all__ = ["boundary_faces", "plot_cell_data", "plot_face_data", "plot_faces", "plot_grid",
           "plot_grid_volumes", "plot_slice", "slice_cell_polygons", "plot_well",
           # VTK/Qt, resolved on first use by __getattr__ below.
           "grid_to_vtk", "cell_ijk", "well_block_mask",
           "ReservoirScene", "ReservoirWindow", "view_reservoir",
           # Well-solution curves (matplotlib), port of MRST plotWellSols.
           "load_well_rates", "well_sol_field_names", "well_field_label",
           "plot_well_sols", "plot_well_rates",
           # Integrated load->configure->run->watch workbench (Qt + matplotlib).
           "SimulatorWindow", "run_simulator"]

_LAZY = {
    "grid_to_vtk": ".vtk_grid",
    "cell_ijk": ".vtk_grid",
    "well_block_mask": ".vtk_grid",
    "ReservoirScene": ".scene3d",
    "ReservoirWindow": ".qt_viewer",
    "view_reservoir": ".qt_viewer",
    "load_well_rates": ".well_curves",
    "well_sol_field_names": ".well_curves",
    "well_field_label": ".well_curves",
    "plot_well_sols": ".well_curves",
    "plot_well_rates": ".well_curves",
    "SimulatorWindow": ".simulator_gui",
    "run_simulator": ".simulator_gui",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    from importlib import import_module

    return getattr(import_module(module, __name__), name)


def __dir__():
    return sorted(__all__)
