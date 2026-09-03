"""MRST coarsegrid utils Python migration.

1:1 translation of multiscale/coarsegrid/utils/ MATLAB functions.
"""

from .coarsen_geometry import coarsen_geometry
from .fine_to_coarse_sign import fine_to_coarse_sign
from .coarsen_bc import coarsen_bc
from .coarsen_flux import coarsen_flux
from .coarse_data_to_fine import coarse_data_to_fine
from .invert_partition import invert_partition

__all__ = [
    "coarsen_geometry",
    "fine_to_coarse_sign",
    "coarsen_bc",
    "coarsen_flux",
    "coarse_data_to_fine",
    "invert_partition",
]
