"""MRST ad-core/upscale Python migration.

1:1 translation of autodiff/ad-core/upscale/ MATLAB code.
"""

from .upscale_model_tpfa import upscale_model_tpfa
from .upscale_state import upscale_state
from .upscale_schedule import upscale_schedule

__all__ = [
    "upscale_model_tpfa",
    "upscale_state",
    "upscale_schedule",
]
