"""Port of MRST ``SurfactantCapillaryPressure.m``.

Scales the water-oil capillary pressure by the interfacial-tension ratio:
``pcow(Sw, c) = pcow(Sw) * ift(c) / ift(0)``.
"""


def SurfactantCapillaryPressure(fluid, pcow, cs):
    """``pcow`` is the black-oil ``pcOW(sW)`` evaluated upstream (this
    mirrors ``evaluateOnDomain@BlackOilCapillaryPressure`` being called
    first in the ``.m`` source); pass ``None``/0 if the fluid has no
    water-oil capillary pressure table (``pc{iW}`` empty in MRST)."""
    if pcow is None:
        return None
    ift0 = float(fluid['ift'](0.0))
    return pcow * fluid['ift'](cs) / ift0
