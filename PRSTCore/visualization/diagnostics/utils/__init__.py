"""Utilities mirroring ``mrst-2026a/visualization/diagnostics/utils``."""

from .compute_f_and_phi import (
    computeFandPhi,
    computeFandPhiFromDist,
    compute_f_and_phi,
    compute_f_and_phi_from_dist,
)
from .compute_lorenz import computeLorenz, compute_lorenz
from .compute_sweep import computeSweep, compute_sweep
from .compute_time_of_flight import computeTimeOfFlight, compute_time_of_flight
from .compute_rtd import computeRTD, compute_rtd
from .compute_tof_and_tracer_average import (
    computeTOFandTracerAverage,
    compute_tof_and_tracer_average,
)
from .compute_tof_and_tracer import computeTOFandTracer, compute_tof_and_tracer
from .compute_well_pairs import computeWellPairs, compute_well_pairs
from .estimate_rtd import estimateRTD, estimate_rtd
from .expand_coarse_well_completions import (
    expandCoarseWellCompletions,
    expand_coarse_well_completions,
)
from .expand_well_completions import expandWellCompletions, expand_well_completions
from .interactive_diagnostics import interactiveDiagnostics, interactive_diagnostics
from .plot_tof_arrival import plotTOFArrival, plot_tof_arrival
from .plot_tracer_blend import plotTracerBlend, plot_tracer_blend
from .plot_well_allocation_comparison import (
    plotWellAllocationComparison,
    plot_well_allocation_comparison,
)
from .plot_well_allocation_panel import plotWellAllocationPanel, plot_well_allocation_panel
from .plot_well_pair_connections import plotWellPairConnections, plot_well_pair_connections
from .postprocess_diagnostics import PostProcessDiagnostics, post_process_diagnostics
from .postprocess_diagnostics_eclipse import (
    PostProcessDiagnosticsECLIPSE,
    post_process_diagnostics_eclipse,
)
from .postprocess_diagnostics_mrst import PostProcessDiagnosticsMRST, post_process_diagnostics_mrst
from .select_tof_region import selectTOFRegion, select_tof_region
from .structures import DiagnosticsStruct, TOFDiagnostics, WellAllocation, WellPairDiagnostics
from .validate_state_for_diagnostics import validateStateForDiagnostics, validate_state_for_diagnostics

# Match the MRST location of computePressureAndDiagnostics while also making
# it available from ``diagnostics.utils`` for convenient Python imports.
from ..preprocessorGUI.utils.compute_pressure_and_diagnostics import (  # noqa: E402
    computePressureAndDiagnostics,
    compute_pressure_and_diagnostics,
)

__all__ = [
    "DiagnosticsStruct",
    "TOFDiagnostics",
    "WellAllocation",
    "WellPairDiagnostics",
    "computeFandPhi",
    "computeFandPhiFromDist",
    "computeLorenz",
    "computePressureAndDiagnostics",
    "computeRTD",
    "computeSweep",
    "computeTOFandTracer",
    "computeTOFandTracerAverage",
    "computeTimeOfFlight",
    "computeWellPairs",
    "estimateRTD",
    "expandCoarseWellCompletions",
    "expandWellCompletions",
    "interactiveDiagnostics",
    "plotTOFArrival",
    "plotTracerBlend",
    "plotWellAllocationComparison",
    "plotWellAllocationPanel",
    "plotWellPairConnections",
    "PostProcessDiagnostics",
    "PostProcessDiagnosticsECLIPSE",
    "PostProcessDiagnosticsMRST",
    "selectTOFRegion",
    "validateStateForDiagnostics",
    "compute_f_and_phi",
    "compute_f_and_phi_from_dist",
    "compute_lorenz",
    "compute_pressure_and_diagnostics",
    "compute_rtd",
    "compute_sweep",
    "compute_time_of_flight",
    "compute_tof_and_tracer",
    "compute_tof_and_tracer_average",
    "compute_well_pairs",
    "estimate_rtd",
    "expand_coarse_well_completions",
    "expand_well_completions",
    "interactive_diagnostics",
    "plot_tof_arrival",
    "plot_tracer_blend",
    "plot_well_allocation_comparison",
    "plot_well_allocation_panel",
    "plot_well_pair_connections",
    "post_process_diagnostics",
    "post_process_diagnostics_eclipse",
    "post_process_diagnostics_mrst",
    "select_tof_region",
    "validate_state_for_diagnostics",
]
