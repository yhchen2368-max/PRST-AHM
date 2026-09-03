"""Optimization helpers mirroring ``diagnostics/utils/optimization``."""

from .compute_default_basis import computeDefaultBasis, compute_default_basis
from .controls import control2well, well2control
from .diagnostics_npv import DiagnosticsNPV
from .diagnostics_objective import DiagnosticsObjective
from .diagnostics_solver_fun import diagnosticsSolverFun, diagnostics_solver_fun
from .eval_objective_diagnostics import evalObjectiveDiagnostics, eval_objective_diagnostics
from .get_objective_diagnostics import getObjectiveDiagnostics, get_objective_diagnostics
from .linsolve_with_timings import linsolveWithTimings, linsolve_with_timings
from .optimize_diagnostics_bfgs import optimizeDiagnosticsBFGS, optimize_diagnostics_bfgs
from .optimize_tof import optimizeTOF, optimize_tof
from .optimize_well_placement_diagnostics import (
    optimizeWellPlacementDiagnostics,
    optimize_well_placement_diagnostics,
)
from .optim_place_simple import optimPlaceSimple, optim_place_simple
from .plot_well_rates import plotWellRates, plot_well_rates
from .plot_wells_print import plotWellsPrint, plot_wells_print
from .solve_stationary_pressure import solveStationaryPressure, solve_stationary_pressure
from .tof_robust_fix import tofRobustFix, tof_robust_fix

__all__ = [
    "DiagnosticsNPV",
    "DiagnosticsObjective",
    "computeDefaultBasis",
    "compute_default_basis",
    "control2well",
    "diagnosticsSolverFun",
    "diagnostics_solver_fun",
    "evalObjectiveDiagnostics",
    "eval_objective_diagnostics",
    "getObjectiveDiagnostics",
    "get_objective_diagnostics",
    "linsolveWithTimings",
    "linsolve_with_timings",
    "optimizeDiagnosticsBFGS",
    "optimize_diagnostics_bfgs",
    "optimizeTOF",
    "optimize_tof",
    "optimizeWellPlacementDiagnostics",
    "optimize_well_placement_diagnostics",
    "optimPlaceSimple",
    "optim_place_simple",
    "plotWellRates",
    "plot_well_rates",
    "plotWellsPrint",
    "plot_wells_print",
    "solveStationaryPressure",
    "solve_stationary_pressure",
    "tofRobustFix",
    "tof_robust_fix",
    "well2control",
]

