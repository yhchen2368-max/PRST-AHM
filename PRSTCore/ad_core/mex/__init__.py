"""Compiled kernels, and the pure-Python paths that stand in for them.

Every extension here is optional.  Nothing imports one directly: the code
that wants it asks this module, gets ``None`` when it was not built, and
falls back.  A missing kernel must cost speed and nothing else -- a silent
change of answer would be far worse than a slow one.

Build them with ``scripts/build_kernels.py``.
"""

from __future__ import annotations


def load_discrete_divergence():
    """The compiled divergence assembler, or ``None`` if it is not built."""
    try:
        from . import discrete_divergence_ext
        return discrete_divergence_ext
    except ImportError:
        return None


def load_face_operators():
    """The compiled face arithmetic, or ``None`` if it is not built."""
    try:
        from . import face_operators_ext
        return face_operators_ext
    except ImportError:
        return None
