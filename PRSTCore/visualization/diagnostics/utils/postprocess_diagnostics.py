"""Python counterpart for MRST ``PostProcessDiagnostics.m``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PostProcessDiagnostics:
    """Lightweight non-interactive post-processor for diagnostics data.

    MRST's original class is a GUI-heavy MATLAB handle class.  This Python
    version preserves the data container role and exposes the common fields
    needed by scripts/tests.
    """

    G: Any | None = None
    Data: Any | None = None
    Gs: Any | None = None
    valid_ix: Any | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def get_data(self):
        return self.Data

    def set_data(self, data):
        self.Data = data
        return self

    def get_global_communication_matrix(self):
        diagnostics = None
        if isinstance(self.Data, dict):
            diagnostics = self.Data.get("diagnostics")
        else:
            diagnostics = getattr(self.Data, "diagnostics", None)
        if diagnostics is None:
            return None
        if isinstance(diagnostics, (list, tuple)):
            matrices = [getattr(d, "wellCommunication", None) if not isinstance(d, dict) else d.get("wellCommunication") for d in diagnostics]
            matrices = [m for m in matrices if m is not None]
            if not matrices:
                return None
            import numpy as np

            return np.max(np.stack(matrices, axis=-1), axis=-1)
        return getattr(diagnostics, "wellCommunication", None) if not isinstance(diagnostics, dict) else diagnostics.get("wellCommunication")

    def show(self):
        raise NotImplementedError("The MRST interactive diagnostics GUI is not implemented in PRSTCore/Python.")


def post_process_diagnostics(*args, **kwargs):
    return PostProcessDiagnostics(*args, **kwargs)

