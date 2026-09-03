"""Python port of MRST's ``tessellationGrid.m`` (mrst-2026a/core/gridprocessing).

Builds a 2D grid from a set of points and a list of polygons (each row/entry
a list of node indices forming one cell) -- the general form ``triangleGrid``
and ``pebi`` are built on top of. Delegates to the existing, validated port
in :mod:`PRSTCore.nwm._core` (used there to build the NWM Volume-of-Interest
grid) rather than duplicating it, since its CSR output already matches this
package's ``cart_grid``/``process_grdecl`` conventions (0-based indices, -1
marks a boundary face).
"""

from __future__ import annotations

from PRSTCore.nwm._core import tessellationGrid as _tessellation_grid_impl


def tessellation_grid(p, t):
    """Port of MRST ``tessellationGrid.m``.

    Parameters
    ----------
    p : (n, 2) array
        Node coordinates.
    t : array or sequence of arrays
        One polygon per cell, as node indices into ``p`` (0-based). A 2D
        array works for a fixed polygon size (e.g. all triangles); a list of
        1D arrays supports mixed polygon sizes.
    """
    return _tessellation_grid_impl(p, t)
