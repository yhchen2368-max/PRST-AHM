"""Port of MRST ``PolymerAdsorption.m``.

Polymer adsorption. The quantity of adsorbed polymer affects the mass
conservation equation for polymer and the relative permeability of water
(see ``PolymerPermReduction``).
"""

from .._adcompat import amax as _amax


def PolymerAdsorption(fluid, cp, cpmax):
    """``evaluateOnDomain``: ``ads = fluid.ads(ce)`` where ``ce`` is ``cp``
    or ``max(cp, cpmax)`` depending on ``fluid['adsInx']`` (1 = no
    desorption hysteresis, 2 = irreversible/max-history adsorption)."""
    if int(fluid.get('adsInx', 1)) == 2:
        ce = _amax(cp, cpmax)
    else:
        ce = cp
    return fluid['ads'](ce)
