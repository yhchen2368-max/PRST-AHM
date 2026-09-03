"""Grid construction routines of the ``nwm`` module, ported 1:1 from MRST
(``mrst-2026a/modules/nwm/gridding``).  Function names preserved.
"""
from .assembleGrids import assembleGrids
from .buildRadialGrid import buildRadialGrid
from .distmesh_2d_nwm import distmesh_2d_nwm
from .extractBdyNodesCells import extractBdyNodesCells
from .generateHWGridNodes import generateHWGridNodes
from .generateVOIGridNodes import generateVOIGridNodes
from .getConnListAndBdyNodeWR2D import getConnListAndBdyNodeWR2D
from .makeConnListFromMat import makeConnListFromMat
from .makeLayeredGridNWM import makeLayeredGridNWM
from .passToDistmesh import passToDistmesh
from .pointsSingleWellNode import pointsSingleWellNode
from .radCartHybridGrid import radCartHybridGrid

__all__ = [
    'assembleGrids', 'buildRadialGrid', 'distmesh_2d_nwm',
    'extractBdyNodesCells', 'generateHWGridNodes', 'generateVOIGridNodes',
    'getConnListAndBdyNodeWR2D', 'makeConnListFromMat',
    'makeLayeredGridNWM', 'passToDistmesh', 'pointsSingleWellNode',
    'radCartHybridGrid',
]
