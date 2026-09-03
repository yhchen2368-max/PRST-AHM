"""Port of MRST ``PolymerEffViscMult.m``.

Effective water viscosity multiplier under a Todd-Longstaff mixing model:
blends the fully-mixed viscosity multiplier ``a = muWMult(cpmax)^(1-mixpar)``
with the concentration-dependent multiplier, only applied to the water phase.
"""

import numpy as _np

from .._adcompat import value as _value


def PolymerEffViscMult(fluid, cp):
    cpmax = _np.full_like(_value(cp), float(fluid['cpmax']))
    mult = fluid['muWMult'](cp)
    multMax = fluid['muWMult'](cpmax)
    mixpar = float(fluid['mixPar'])
    cpbar = cp / float(fluid['cpmax'])
    a = multMax ** (1.0 - mixpar)
    b = 1.0 / (1.0 - cpbar + cpbar / a)
    return b * mult ** mixpar
