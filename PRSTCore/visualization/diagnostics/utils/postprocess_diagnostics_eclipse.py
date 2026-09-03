"""MRST ``PostProcessDiagnosticsECLIPSE.m`` counterpart."""

from .postprocess_diagnostics import PostProcessDiagnostics


def post_process_diagnostics_eclipse(*args, **kwargs):
    """Create a diagnostics post-processor for ECLIPSE-derived data."""
    return PostProcessDiagnostics(*args, **kwargs)


PostProcessDiagnosticsECLIPSE = post_process_diagnostics_eclipse

