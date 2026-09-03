"""Port of MRST ``autodiff/ad-eor/utils``: residual-equation assembly
(``equations*.py``) and supporting helpers.

``equations*.py`` function/argument names mirror the ``.m`` source; the
``model`` argument is a PRSTCore reservoir model exposing the same
``operators``/``fluid``/relative-permeability/PVT helpers used elsewhere in
``PRSTCore.ad_core.models.generic_black_oil_model`` (see
``_mrst_generic_adi_residual_ow`` for the established calling pattern this
module follows).
"""
