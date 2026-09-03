"""Near-wellbore modeling (nwm) module, ported 1:1 from MRST.

Routines for the implementation and manipulation of the near-wellbore
modeling (NWM) methodology (author: Lin Zhao, China University of
Petroleum (Beijing) / SINTEF Digital).

The package mirrors the MRST module layout exactly:

    PRSTCore/nwm/
        gridding/   -- grid construction (assembleGrids, buildRadialGrid, ...)
        models/     -- classes (NearWellboreModel, MultiSegWellNWM,
                       VolumeOfInterest, HorWellRegion)
        trans/      -- transmissibility / interface handling
        utils/      -- small helper routines
        data/       -- ECLIPSE-style data files
        examples/   -- example scripts

Indexing conventions (identical to the rest of PRSTCore, differing from
MRST): cells / faces / nodes are 0-based and ``G['faces']['neighbors']``
uses ``-1`` (not ``0``) to mark a boundary face.
"""
