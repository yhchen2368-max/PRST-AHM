"""MRST ad-props Python migration.

1:1 translation of autodiff/ad-props/ MATLAB functions.
"""

from .impose_relperm_scaling import impose_relperm_scaling
from .get_relperm_scaling_points import get_relperm_scaling_points
from .get_normalization_factors import get_normalization_factors

__all__ = [
    "impose_relperm_scaling",
    "get_relperm_scaling_points",
    "get_normalization_factors",
]
