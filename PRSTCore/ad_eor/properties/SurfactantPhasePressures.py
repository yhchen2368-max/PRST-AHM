"""Port of MRST ``SurfactantPhasePressures.m``.

Per-phase pressure from the reference pressure plus each phase's capillary
pressure (``None``/empty entries in ``pc`` pass the reference pressure
through unchanged, matching ``isempty(pc{i})`` in the ``.m`` source).
"""


def SurfactantPhasePressures(p, pc):
    return [p if pc_i is None else p + pc_i for pc_i in pc]
