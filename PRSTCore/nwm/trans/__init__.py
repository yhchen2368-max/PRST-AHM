"""Transmissibility / interface routines of the ``nwm`` module, ported 1:1
from MRST (``mrst-2026a/modules/nwm/trans``).  Function names preserved.
"""
from .computeRadTransFactor import computeRadTransFactor
from .handleMatchingFaces import handleMatchingFaces
from .handleNonMatchingFaces import handleNonMatchingFaces

__all__ = ['computeRadTransFactor', 'handleMatchingFaces', 'handleNonMatchingFaces']
