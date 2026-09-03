"""Controls to RHS mapping (alias for controls2wells in adjoint context).

1:1 Python translation of MRST solvers/adjoint/controls2RHS.m
"""

# controls2RHS.m is identical in logic to controls2Wells.m
# Re-export for MRST compatibility
from .controls2wells import controls2wells as controls2rhs
