"""Port of MRST ``getFluxAndPropsWaterPolymer_BO.m``.

Given pressure, water saturation and polymer concentration, computes the
water/polymer face fluxes and supporting cell-valued properties used by
``equationsOilWaterPolymer``/``equationsThreePhaseBlackOilPolymer``.

MRST's ``model.operators`` (``s.Grad``, ``s.faceAvg``,
``s.splitFaceCellValue``) are generic-grid operators; PRSTCore's residual
assembly (see ``GenericBlackOilModel._mrst_generic_adi_residual_ow``) works
directly with the internal-connection neighbor arrays ``c1``/``c2`` and
transmissibility ``T`` instead, so this port takes those explicitly rather
than a ``model``/``operators`` object.
"""

import numpy as _np


def getFluxAndPropsWaterPolymer_BO(fluid, pO, sW, cp, ads, krW, T, c1, c2, gdz):
    """
    Parameters
    ----------
    pO : ADI/ndarray (cell)      Oil-phase pressure.
    sW : ADI/ndarray (cell)      Water saturation.
    cp : ADI/ndarray (cell)      Polymer concentration.
    ads : ADI/ndarray (cell)     Effective adsorption (``effads``).
    krW : ADI/ndarray (cell)     Water relative permeability.
    T : ndarray (face)           Face transmissibility.
    c1, c2 : ndarray[int] (face) Upstream/downstream neighbor cell indices.
    gdz : ndarray (face)         Geometric gravity term ``g . (z[c2]-z[c1])``.

    Returns
    -------
    vW, vP, bW, muWeffMult, mobW, mobP, rhoW, pW, upcw, a
    """
    pcOW = 0.0
    if fluid.get('pcOW') is not None:
        pcOW = fluid['pcOW'](sW)
    pW = pO - pcOW
    muW = fluid['muW'](pW)

    mixpar = float(fluid['mixPar'])
    cpbar = cp / float(fluid['cpmax'])
    a = float(fluid['muWMult'](float(fluid['cpmax']))) ** (1.0 - mixpar)
    b = 1.0 / (1.0 - cpbar + cpbar / a)
    muWeffMult = b * fluid['muWMult'](cp) ** mixpar
    permRed = 1.0 + ((float(fluid['rrf']) - 1.0) / float(fluid['adsMax'])) * ads
    muWMult = muWeffMult * permRed

    bW = fluid['bW'](pO)
    rhoW = bW * float(fluid['rhoWS'])
    muWeff = muWMult * muW

    # MRST: rhoWf = s.faceAvg(rhoW), which carries rhoW's derivative
    # through -- rhoW = bW(pO)*rhoWS depends on pressure. Freezing it at
    # the current iterate (as this did with _value) drops d(gravity)/dp
    # from the Jacobian. Harmless where gdz == 0 (a horizontal 1D case),
    # an incomplete Jacobian anywhere else. SparseADI supports both the
    # row indexing and the arithmetic, so one expression serves both.
    rhoWf = 0.5 * (rhoW[c1] + rhoW[c2])
    dpW = (pW[c2] - pW[c1]) - rhoWf * gdz

    upcw = _value(dpW) <= 0.0
    upstream = _np.where(upcw, c1, c2)

    krWf = krW[upstream]
    muWeff_f = muWeff[upstream]
    mobW = krW / muWeff

    vW = -(krWf / muWeff_f) * T * dpW
    if _np.any(_value(bW) < 0):
        import warnings
        warnings.warn('Negative water compressibility present!')

    muPeff = muWeff * (a + (1.0 - a) * cpbar)
    muPeff_f = muPeff[upstream]
    cpf = cp[upstream]
    mobP = krW / muPeff

    vP = -(krWf / muPeff_f) * cpf * T * dpW

    return vW, vP, bW, muWeffMult, mobW, mobP, rhoW, pW, upcw, a



def _value(x):
    from .._adcompat import value
    return value(x)
