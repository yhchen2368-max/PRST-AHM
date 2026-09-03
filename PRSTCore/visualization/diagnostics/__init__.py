"""MRST-style flow diagnostics.

The package mirrors the useful parts of
``mrst-2026a/visualization/diagnostics`` while using PRSTCore/Python data
structures.  Public functions keep MRST names as aliases for downstream code
that follows the MATLAB examples.
"""

from .utils import (
    DiagnosticsStruct,
    WellPairDiagnostics,
    computeFandPhi,
    computeLorenz,
    computePressureAndDiagnostics,
    computeRTD,
    computeSweep,
    computeTOFandTracer,
    computeTOFandTracerAverage,
    computeTimeOfFlight,
    computeWellPairs,
    estimateRTD,
    expandCoarseWellCompletions,
    expandWellCompletions,
    interactiveDiagnostics,
    plotTOFArrival,
    plotTracerBlend,
    plotWellAllocationComparison,
    plotWellAllocationPanel,
    plotWellPairConnections,
    PostProcessDiagnostics,
    PostProcessDiagnosticsECLIPSE,
    PostProcessDiagnosticsMRST,
    selectTOFRegion,
    validateStateForDiagnostics,
)

__all__ = [
    "DiagnosticsStruct",
    "WellPairDiagnostics",
    "computeFandPhi",
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
]
