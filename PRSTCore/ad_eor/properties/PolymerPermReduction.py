"""Port of MRST ``PolymerPermReduction.m``.

Polymer permeability-reduction factor: divides the water relative
permeability to account for adsorbed polymer reducing the effective pore
throat size (residual resistance factor ``fluid.rrf``).
"""


def PolymerPermReduction(fluid, ads):
    rrf = float(fluid['rrf'])
    adsMax = float(fluid['adsMax'])
    return 1.0 + ((rrf - 1.0) / adsMax) * ads
