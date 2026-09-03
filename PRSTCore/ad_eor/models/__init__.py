"""Port of MRST ``autodiff/ad-eor/models``.

Each class subclasses :class:`PRSTCore.ad_core.models.generic_black_oil_model.GenericBlackOilModel`
and overrides ``get_equations``/``updateState``/``checkConvergence`` exactly
where MRST's model classes override ``getEquations``/``updateState``/
``updateAfterConvergence`` -- the Newton driver lives in
``GenericBlackOilModel`` itself (see its ``_mrst_generic_newton_ministep``-style
method, invoked as ``self.get_equations``/``self.checkConvergence``/
``self.updateState``), so overriding these three methods is the correct and
sufficient integration point, matching how ``_get_equations_mrst_generic_ow``
et al. already plug into that same driver for the plain black-oil model.

Not ported: ``GenericSurfactantPolymerModel.m`` and ``components/*.m`` (see
``PRSTCore.ad_eor`` package docstring for why).
"""
