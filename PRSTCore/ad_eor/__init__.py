"""Python port of MRST's ``autodiff/ad-eor`` module (Enhanced Oil Recovery:
polymer, surfactant and combined surfactant-polymer flooding).

Directory layout mirrors MRST 1:1:

- ``properties/`` -- pure physics functions (adsorption, viscosity
  multipliers, capillary-desaturation relative permeability, ...),
  ported from the ``StateFunction`` subclasses of the same name.
- ``utils/`` -- residual-equation assembly (``equations*.py``) and
  supporting helpers (shear-thinning solves, TPFA velocity operators).
- ``models/`` -- thin model classes matching MRST's model hierarchy
  (``OilWaterPolymerModel``, ``ThreePhaseBlackOilPolymerModel``,
  ``OilWaterSurfactantModel``, ``ThreePhaseBlackOilSurfactantModel``,
  ``ThreePhaseSurfactantPolymerModel``).

MRST's newer ``GenericSurfactantPolymerModel`` and its supporting
``StateFunction``/``Component`` classes (``properties/EORViscosity.m``,
``properties/EORRelativePermeability.m``, ``properties/PhaseMultipliers.m``,
``properties/ComponentPhaseFluxWithPolymer.m``, ``properties/FaceConcentration.m``,
``properties/PerforationMobilityEOR.m``, ``properties/PerforationComponentPhaseDensityEOR.m``,
``models/GenericSurfactantPolymerModel.m``, ``models/components/*.m``) are not
ported: they exist purely to plug into MRST's StateFunction dependency-graph
architecture, which PRSTCore's ``GenericBlackOilModel`` does not use (a prior,
deliberate architectural decision in this project). Their physics is instead
folded directly into the procedural ``equations*.py`` functions below,
mirroring how MRST's own legacy ``equationsOilWaterPolymer.m`` computes
adsorption/viscosity terms inline rather than through the StateFunction graph.
"""
