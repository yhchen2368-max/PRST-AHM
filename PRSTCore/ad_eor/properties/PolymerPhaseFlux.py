"""Port of MRST ``PolymerPhaseFlux.m``.

Face-valued polymer flux under the Todd-Longstaff mixing model: the polymer
travels at a fraction of the water face flux ``vW``, set by the
concentration-dependent mixing factor.
"""


def PolymerPhaseFlux(fluid, vW, cpf):
    mixpar = float(fluid['mixPar'])
    cpbar = cpf / float(fluid['cpmax'])
    a = fluid['muWMult'](float(fluid['cpmax'])) ** (1.0 - mixpar)
    return vW / (a + (1.0 - a) * cpbar) * cpf
