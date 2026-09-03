"""Port of MRST ``PolymerViscMult.m``.

Polymer viscosity multiplier used for the polymer's own transport velocity
(as opposed to the *effective* water viscosity multiplier, see
``PolymerEffViscMult``). A Todd-Longstaff model blends the fully-mixed
(``fluid.muWMult`` at ``cpmax``) and unmixed viscosities.
"""

import numpy as _np

from .._adcompat import value as _value


def PolymerViscMult(fluid, cp):
    cpmax = _np.full_like(_value(cp), float(fluid['cpmax']))
    mult = fluid['muWMult'](cp)
    multMax = fluid['muWMult'](cpmax)
    mixpar = float(fluid['mixPar'])
    return mult ** mixpar * multMax ** (1.0 - mixpar)
