"""JutulDarcy integration for PRSTCore (self-contained, no GeoCode).

The Julia drivers (``jutul_run.jl`` / ``jutul_optimize.jl``) and their Julia
environment (``Project.toml`` / ``Manifest.toml``) ship with this package;
:mod:`PRSTCore.jutul.driver` launches them through a direct ``julia``
subprocess.  Results use the same unified HDF5 schema as the PRST engine
(``states.h5`` / ``wells.h5`` / ``cell_indices.h5`` / ``manifest.json``),
readable by :mod:`PRSTCore.visualization.h5_results`.
"""

from .driver import (  # noqa: F401
    JULIA_HINT, available, find_julia, run_optimize, run_simulate)

__all__ = ["find_julia", "available", "run_simulate", "run_optimize",
           "JULIA_HINT"]
