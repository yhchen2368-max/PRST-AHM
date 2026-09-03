"""Extend fluid with adjoint derivative fields.

1:1 Python translation of MRST solvers/adjoint/adjointFluidFields.m
"""

import numpy as np


def adjoint_fluid_fields(fluid):
    """Add dLtInv, d2LtInv, d2fw to fluid for 2nd-order adjoint.

    Parameters
    ----------
    fluid : dict
        Fluid with krw, kro, muw, muo.

    Returns
    -------
    dict
        Updated fluid.
    """
    f = dict(fluid)

    def _dLtInv(state):
        s = np.asarray(state.get("s", np.ones(1))).ravel()
        krw = f.get("krw", lambda x: x)(s)
        kro = f.get("kro", lambda x: 1 - x)(s)
        dkrw = f.get("dkrw", lambda x: np.ones_like(x))(s)
        dkro = f.get("dkro", lambda x: -np.ones_like(x))(s)
        muw = f.get("muw", 1.0)
        muo = f.get("muo", 1.0)
        mob_w = krw / muw
        mob_o = kro / muo
        dmob_w = dkrw / muw
        dmob_o = dkro / muo
        Lt = mob_w + mob_o
        dLt = dmob_w + dmob_o
        return -dLt / np.maximum(Lt**2, 1e-12)

    def _d2LtInv(state):
        s = np.asarray(state.get("s", np.ones(1))).ravel()
        krw, dkrw, d2krw = _get_mobilities(f, s)
        muw, muo = f.get("muw", 1.0), f.get("muo", 1.0)
        mob_w = krw / muw
        mob_o = kro / muo
        dmob_w = dkrw / muw
        dmob_o = dkro / muo
        d2mob_w = d2krw / muw
        d2mob_o = d2kro / muo
        Lt = mob_w + mob_o
        dLt = dmob_w + dmob_o
        d2Lt = d2mob_w + d2mob_o
        return -d2Lt / Lt**2 + 2 * dLt**2 / Lt**3

    def _d2fw(state):
        s = np.asarray(state.get("s", np.ones(1))).ravel()
        krw, dkrw, d2krw = _get_mobilities(f, s)
        kro, dkro, d2kro = _get_mobilities_o(f, s)
        muw, muo = f.get("muw", 1.0), f.get("muo", 1.0)
        mob_w, mob_o = krw / muw, kro / muo
        dmob_w, dmob_o = dkrw / muw, dkro / muo
        d2mob_w, d2mob_o = d2krw / muw, d2kro / muo
        Lt = mob_w + mob_o
        dLt = dmob_w + dmob_o
        d2Lt = d2mob_w + d2mob_o
        fw = mob_w / Lt
        dfw = (dmob_w * Lt - mob_w * dLt) / Lt**2
        return (d2mob_w * Lt - mob_w * d2Lt - 2 * dfw * dLt) / Lt**2

    def _get_mobilities(f, s):
        krw = np.asarray(f.get("krw", lambda x: x)(s))
        dkrw = np.asarray(f.get("dkrw", lambda x: np.ones_like(x))(s))
        d2krw = np.asarray(f.get("d2krw", lambda x: np.zeros_like(x))(s))
        return krw, dkrw, d2krw

    def _get_mobilities_o(f, s):
        return _get_mobilities(f, s)

    f["dLtInv"] = _dLtInv
    f["d2LtInv"] = _d2LtInv
    f["d2fw"] = _d2fw
    return f
