"""MRST optimization.utils Python migration.

1:1 translation of autodiff/optimization/utils/ MATLAB functions.

Core parameter helpers live here, matching MRST's
autodiff/optimization/utils layout.
"""

from .parameters import ModelParameter, add_parameter, update_setup_from_scaled_parameters, get_scaled_parameter_vector
from .process_bounds import process_bounds
from .scale_constraints import scale_constraints
from .setup_constraints import setup_constraints
from .schedule2control import schedule2control, control2schedule
from .control_logic_func import control_logic_func
from .eval_objective import eval_objective
from .evaluate_objective import evaluate_objective
from .init_simple_scaled_adi_fluid import init_simple_scaled_adi_fluid
from .process_adjoint_gradients import process_adjoint_gradients
from .setup_simulation_control_mappings import setup_simulation_control_mappings
from .simulation_solver_fun import simulation_solver_fun
from .compute_ensemble_simulation_objective import compute_ensemble_simulation_objective

__all__ = [
    "process_bounds",
    "scale_constraints",
    "setup_constraints",
    "schedule2control",
    "control2schedule",
    "control_logic_func",
    "eval_objective",
    "evaluate_objective",
    "init_simple_scaled_adi_fluid",
    "process_adjoint_gradients",
    "setup_simulation_control_mappings",
    "simulation_solver_fun",
    "compute_ensemble_simulation_objective",
    "ModelParameter",
    "add_parameter",
    "update_setup_from_scaled_parameters",
    "get_scaled_parameter_vector",
]
