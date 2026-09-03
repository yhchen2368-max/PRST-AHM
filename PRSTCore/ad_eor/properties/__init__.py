"""Port of MRST ``autodiff/ad-eor/properties``.

MRST implements these as ``StateFunction`` subclasses that plug into the
``FlowPropertyFunctions``/``PVTPropertyFunctions`` dependency graph (accessed
via ``model.getProp(state, 'PolymerAdsorption')`` and friends). PRSTCore's
``GenericBlackOilModel`` has no such graph, so each class here becomes a
plain function taking the physical inputs the ``evaluateOnDomain`` method
would have fetched via ``model.getProps``/``prop.getEvaluatedDependencies``.
Function names, argument names and the arithmetic itself match the ``.m``
source 1:1; only the calling convention (explicit arguments instead of a
state-function graph) differs.

Not ported: ``EORViscosity.m``, ``EORRelativePermeability.m``,
``PhaseMultipliers.m``, ``ComponentPhaseFluxWithPolymer.m``,
``FaceConcentration.m``, ``PerforationMobilityEOR.m``,
``PerforationComponentPhaseDensityEOR.m`` -- these are pure StateFunction
dependency-graph plumbing (container/wiring classes) with no physics of
their own; their effect is reproduced directly in ``ad_eor/utils/equations*.py``.
"""
