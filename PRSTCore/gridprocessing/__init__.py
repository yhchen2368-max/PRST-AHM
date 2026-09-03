"""Grid generation and geometry, ported from mrst-2026a/core/gridprocessing.

Public API mirrors MRST's split between topology construction and geometry:

    >>> from PRSTCore.gridprocessing import cart_grid, compute_geometry
    >>> G = compute_geometry(cart_grid([4, 3, 2], [40, 30, 20]))
"""

from .cart_grid import cart_grid
from .compute_geometry import compute_geometry
from .extract_subgrid import extract_subgrid
from .pebi_grid import pebi_grid
from .process_grdecl import process_grdecl
from .remove_cells import remove_cells
from .tensor_grid import tensor_grid
from .tessellation_grid import tessellation_grid
from .triangle_grid import triangle_grid

__all__ = [
    "cart_grid", "tensor_grid", "compute_geometry", "process_grdecl",
    "remove_cells", "extract_subgrid", "tessellation_grid", "triangle_grid", "pebi_grid",
]
