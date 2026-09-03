"""PRSTCore package organized along MRST module boundaries.

Import functions from their MRST-style subpackages, e.g.
``PRSTCore.ad_core.simulators.simulate_schedule_ad`` or
``PRSTCore.optimization.utils.parameters``.
"""

# Runs before numpy/scipy are imported anywhere below this package, which is
# what a conda environment invoked without ``conda activate`` needs in order
# for MKL to find its OpenMP runtime.  See the module docstring: without it a
# three-dimensional sparse solve kills the interpreter outright.
from ._dll_directories import ensure_native_dll_path as _ensure_native_dll_path

_ensure_native_dll_path()

del _ensure_native_dll_path
