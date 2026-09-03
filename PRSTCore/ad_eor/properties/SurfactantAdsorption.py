"""Port of MRST ``SurfactantAdsorption.m``.

Surfactant adsorption; affects the mass conservation equation for surfactant.
"""

from .._adcompat import amax as _amax


def SurfactantAdsorption(fluid, cs, csmax):
    if int(fluid.get('adsInxSft', 1)) == 2:
        ce = _amax(cs, csmax)
    else:
        ce = cs
    return fluid['surfads'](ce)
