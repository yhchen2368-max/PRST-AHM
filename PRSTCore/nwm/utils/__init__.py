"""Utility routines of the ``nwm`` module, ported 1:1 from MRST
(``mrst-2026a/modules/nwm/utils``).  Function names are preserved.
"""
from .arrayfunUniOut import arrayfunUniOut
from .bisection import bisection
from .cellfunUniOut import cellfunUniOut
from .circleCross import circleCross
from .computeCentroids import computeCentroids
from .computePD import computePD
from .convertTo3DPlane import convertTo3DPlane
from .convertToColumn import convertToColumn
from .convertToXYPlane import convertToXYPlane
from .dispInfo import dispInfo
from .euclideanDistance import euclideanDistance
from .getDZ import getDZ
from .getUnitDisVectors import getUnitDisVectors
from .polyintersect import polyintersect
from .sortPtsClockWise import sortPtsClockWise
from .sortPtsCounterClockWise import sortPtsCounterClockWise
from .tabulate_NWM import tabulate_NWM
from .tri_area import tri_area

__all__ = [
    'arrayfunUniOut', 'bisection', 'cellfunUniOut', 'circleCross',
    'computeCentroids', 'computePD', 'convertTo3DPlane', 'convertToColumn',
    'convertToXYPlane', 'dispInfo', 'euclideanDistance', 'getDZ',
    'getUnitDisVectors', 'polyintersect', 'sortPtsClockWise',
    'sortPtsCounterClockWise', 'tabulate_NWM', 'tri_area',
]
