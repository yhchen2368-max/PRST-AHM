"""MRST solvers/adjoint Python migration.

1:1 translation of solvers/adjoint/ MATLAB code.
"""

from .init_schedule import init_schedule
from .init_controls import init_controls
from .update_schedule import update_schedule
from .update_wells import update_wells
from .controls2wells import controls2wells
from .controls2rhs import controls2rhs
from .compute_gradient import compute_gradient
from .compute_numerical_gradient import compute_numerical_gradient
from .compute_adjoint_rhs import compute_adjoint_rhs
from .generate_upstream_transport_matrix import generate_upstream_transport_matrix
from .solve_adjoint_pressure_system import solve_adjoint_pressure_system
from .solve_adjoint_transport_system import solve_adjoint_transport_system
from .solve_incomp_flow_local import solve_incomp_flow_local
from .run_schedule import run_schedule
from .run_adjoint import run_adjoint
from .optimize_objective import optimize_objective
from .line_search_agr import line_search_agr
from .project_gradient import project_gradient
from .disp_controls import disp_controls, disp_schedule
from .add_adjoint_well_fields import add_adjoint_well_fields
from .adjoint_fluid_fields import adjoint_fluid_fields
from .assemble_well_system import assemble_well_system

__all__ = [
    "init_schedule",
    "init_controls",
    "update_schedule",
    "update_wells",
    "controls2wells",
    "controls2rhs",
    "compute_gradient",
    "compute_numerical_gradient",
    "compute_adjoint_rhs",
    "generate_upstream_transport_matrix",
    "solve_adjoint_pressure_system",
    "solve_adjoint_transport_system",
    "solve_incomp_flow_local",
    "run_schedule",
    "run_adjoint",
    "optimize_objective",
    "line_search_agr",
    "project_gradient",
    "disp_controls",
    "disp_schedule",
    "add_adjoint_well_fields",
    "adjoint_fluid_fields",
    "assemble_well_system",
]
