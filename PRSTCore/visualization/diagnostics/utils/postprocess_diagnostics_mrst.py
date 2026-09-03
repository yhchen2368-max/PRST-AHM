"""MRST ``PostProcessDiagnosticsMRST.m`` counterpart."""

from .postprocess_diagnostics import PostProcessDiagnostics


def post_process_diagnostics_mrst(problem, *args, **kwargs):
    """Create a diagnostics post-processor for a packed MRST/PRST problem."""
    return PostProcessDiagnostics(Data={"problem": problem}, options={"args": args, **kwargs})


PostProcessDiagnosticsMRST = post_process_diagnostics_mrst

