"""MRST adjoint objectives.

1:1 translation of solvers/adjoint/objectives/
"""

from .simple_npv import simple_npv
from .recovery import recovery

__all__ = ["simple_npv", "recovery"]
