"""MRST ``interactiveDiagnostics.m`` counterpart."""

from .postprocess_diagnostics import PostProcessDiagnostics


def interactive_diagnostics(G, rock, W, **kwargs):
    """Return a non-interactive diagnostics post-processor container.

    The MATLAB function opens a GUI.  PRSTCore keeps a script-friendly object
    with the same data instead.
    """
    return PostProcessDiagnostics(G=G, Data={"rock": rock, "W": W}, options=kwargs)


interactiveDiagnostics = interactive_diagnostics

