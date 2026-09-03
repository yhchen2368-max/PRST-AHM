"""Model classes of the ``nwm`` module, ported 1:1 from MRST
(``mrst-2026a/modules/nwm/models``).  Class names preserved.
"""
from .HorWellRegion import HorWellRegion
from .MultiSegWellNWM import MultiSegWellNWM
from .NearWellboreModel import NearWellboreModel
from .VolumeOfInterest import VolumeOfInterest

__all__ = ['HorWellRegion', 'MultiSegWellNWM', 'NearWellboreModel',
           'VolumeOfInterest']
