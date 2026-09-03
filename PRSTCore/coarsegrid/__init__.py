"""MRST coarsegrid Python migration.

1:1 translation of multiscale/coarsegrid/ MATLAB functions.
"""

from .partition_layers import partition_layers
from .process_partition import process_partition, compress_partition, partition_ui
from .partition_cart_grid import partition_cart_grid
from .cell_partition_to_face_partition import cell_partition_to_face_partition
from .generate_coarse_grid import generate_coarse_grid
from .process_face_partition import process_face_partition
from .sub_faces import sub_faces

__all__ = [
    "partition_layers",
    "process_partition",
    "compress_partition",
    "partition_ui",
    "partition_cart_grid",
    "cell_partition_to_face_partition",
    "generate_coarse_grid",
    "process_face_partition",
    "sub_faces",
]
