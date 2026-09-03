"""MRST optimization.objectives Python migration.

1:1 translation of autodiff/optimization/objectives/ MATLAB functions.
"""

from .match_observed_ow import match_observed_ow
from .npv_ow import npv_ow
from .npv_black_oil import npv_black_oil
from .npv_ow_polymer import npv_ow_polymer

__all__ = [
    "match_observed_ow",
    "npv_ow",
    "npv_black_oil",
    "npv_ow_polymer",
]
