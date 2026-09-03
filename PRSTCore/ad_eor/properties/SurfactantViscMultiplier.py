"""Port of MRST ``SurfactantViscMultiplier.m``.

Surfactant viscosity multiplier: multiplies the water viscosity, normalized
by the reference viscosity multiplier ``fluid.muWr``.
"""


def SurfactantViscMultiplier(fluid, cs):
    mSft = fluid['muWSft'](cs)
    return mSft / float(fluid['muWr'])
