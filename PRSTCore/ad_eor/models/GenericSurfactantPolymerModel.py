"""Port of MRST ``GenericSurfactantPolymerModel.m``.

In MRST this class is a thin ``ThreePhaseSurfactantPolymerModel &
GenericReservoirModel`` mix-in: it does not add new physics of its own, it
just lets ``ThreePhaseSurfactantPolymerModel`` run through the modern
``Component``/``FlowDiscretization``/``FacilityModel`` StateFunction graph
(``ImmiscibleComponent``/``OilComponent``/``GasComponent`` plus
``PolymerComponent``/``SurfactantComponent`` from
``model.validateModel``) instead of the legacy procedural
``equationsThreePhaseSurfactantPolymer.m`` path, and adds one genuinely
new piece of physics: ``getConvergenceValues``' dimensionless EOR-equation
scaling (``scale = dt/(pv*cmax*rhoWS)``, tolerance ``toleranceEOR = 1e-3``),
which every ``ad_eor`` model class here already carries (see
``OilWaterPolymerModel.checkConvergence`` and friends).

PRSTCore's ``GenericBlackOilModel`` does not have that StateFunction graph
(see ``ad_eor`` package docstring), so there is no PRSTCore counterpart to
literally subclass. This class instead reproduces
``GenericSurfactantPolymerModel``'s *user-facing* role -- one constructor
that accepts any water/oil/gas + polymer/surfactant combination and returns
a working model -- as a thin factory dispatching to whichever procedural
``ad_eor`` model class matches the requested phases/EOR combination.

Combinations not yet implemented (``equationsThreePhaseBlackOilSurfactant``/
``equationsThreePhaseSurfactantPolymer`` are not ported, see the
``ad_eor.utils`` package) raise ``NotImplementedError`` rather than silently
returning the wrong physics.
"""

from .OilWaterPolymerModel import OilWaterPolymerModel
from .OilWaterSurfactantModel import OilWaterSurfactantModel
from .ThreePhaseBlackOilPolymerModel import ThreePhaseBlackOilPolymerModel


def GenericSurfactantPolymerModel(G=None, rock=None, fluid=None, *args,
                                   water=True, oil=True, gas=False,
                                   polymer=False, surfactant=False, **kwargs):
    """Factory returning the ``ad_eor`` model matching ``(water, oil, gas,
    polymer, surfactant)``. See module docstring for the (procedural vs.
    StateFunction-graph) difference from MRST's actual class."""
    if not (water and oil):
        raise NotImplementedError(
            'GenericSurfactantPolymerModel: only water+oil(+gas) combinations '
            'are wired in the ad_eor port (water/oil are both required)')
    if polymer and surfactant:
        raise NotImplementedError(
            'GenericSurfactantPolymerModel: combined polymer+surfactant is not '
            'ported (equationsThreePhaseSurfactantPolymer.m has no ad_eor '
            'counterpart yet)')
    if surfactant and gas:
        raise NotImplementedError(
            'GenericSurfactantPolymerModel: gas+surfactant is not ported '
            '(equationsThreePhaseBlackOilSurfactant.m has no ad_eor '
            'counterpart yet)')
    if surfactant:
        return OilWaterSurfactantModel(G=G, rock=rock, fluid=fluid, *args, **kwargs)
    if polymer and gas:
        return ThreePhaseBlackOilPolymerModel(G=G, rock=rock, fluid=fluid, *args, **kwargs)
    if polymer:
        return OilWaterPolymerModel(G=G, rock=rock, fluid=fluid, *args, **kwargs)
    if gas:
        raise NotImplementedError(
            'GenericSurfactantPolymerModel: plain water+oil+gas with no EOR '
            'component should use GenericBlackOilModel directly, not this '
            'EOR factory')
    raise NotImplementedError(
        'GenericSurfactantPolymerModel: water+oil with neither polymer nor '
        'surfactant active should use GenericBlackOilModel directly')
