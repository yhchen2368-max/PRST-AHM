"""Port of MRST ``SurfactantRelativePermeability.m``.

Capillary-desaturation relative permeability: rescales the saturation axis
between "without surfactant" (base) and "with surfactant" (fully
desaturated, using ``fluid.krPts.*`` with the surfactant region's residual
saturations) endpoints, interpolated by a miscibility factor
``m = fluid.miscfact(log10(Nc))`` that depends on the capillary number.

MRST dispatches ``krW``/``krO``/``krG`` per-cell through ``satreg``/
``surfreg`` SATNUM/surfactant-region indices via
``StateFunction.evaluateFunctionOnDomainWithArguments``: the *base*
(no-surfactant) curve is evaluated with the cell's ``SATNUM``-selected
table, the *fully-desaturated* curve with its ``SURFNUM``-selected table --
in general two genuinely different tables, not just different residual
saturations (see e.g. ``SURFACTANT1D.DATA``'s two stacked ``SWOF`` blocks).
PRSTCore's relative-permeability tables (``ad_props/relperm_tables.py``)
have no per-cell SATNUM/SURFNUM dispatch, so this port takes the resolved
region tables directly as ``fluid_base``/``fluid_surf`` (each a dict of
table-evaluation callables) rather than doing per-cell region lookup --
i.e. it supports one uniform SATNUM/SURFNUM assignment across all cells
(exactly what the bundled decks use), not per-cell heterogeneous regions.
"""

import numpy as _np


def SurfactantRelativePermeability(fluid, sW, sO, sG, cs, Nc, krPts_base, krPts_surf, has_gas,
                                    fluid_base=None, fluid_surf=None):
    """
    Parameters
    ----------
    fluid : dict
        Must provide ``miscfact``. Also used as a fallback for
        ``fluid_base``/``fluid_surf`` when they are not given (single
        shared table, matching earlier single-region test fixtures).
    fluid_base, fluid_surf : dict, optional
        Must each provide ``krW``, ``krO`` (or ``krOW``), and (if
        ``has_gas``) ``krG``/``krOG`` table-evaluation callables -- the
        SATNUM-region and SURFNUM-region tables respectively.
    krPts_base, krPts_surf : dict
        Residual-saturation endpoints for the base (no-surfactant) and
        surfactant-desaturated tables: keys ``'w'``, ``'ow'`` and (if
        ``has_gas``) ``'og'``, ``'g'``, each the column-2 (MATLAB
        ``fluid.krPts.*(reg, 2)``) residual saturation value.
    """
    fluid_base = fluid_base if fluid_base is not None else fluid
    fluid_surf = fluid_surf if fluid_surf is not None else fluid
    m = _np.zeros_like(_value(cs))
    if _np.count_nonzero(_value(cs) > 0) > 0:
        logNc = _np.log(Nc) / _np.log(10.0)
        logNc = _np.minimum(_np.maximum(-20.0, logNc), 20.0)
        m = fluid['miscfact'](logNc)

    sWc_ns, sOWr_ns = float(krPts_base['w']), float(krPts_base['ow'])
    sWc_s, sOWr_s = float(krPts_surf['w']), float(krPts_surf['ow'])

    sNcWc = m * sWc_s + (1.0 - m) * sWc_ns
    sNcOWr = m * sOWr_s + (1.0 - m) * sOWr_ns
    sNcWeff = (sW - sNcWc) / (1.0 - sNcWc - sNcOWr)
    sNcOWeff = (sO - sNcOWr) / (1.0 - sNcWc - sNcOWr)

    sW_ns = (1.0 - sWc_ns - sOWr_ns) * sNcWeff + sWc_ns
    sOW_ns = (1.0 - sWc_ns - sOWr_ns) * sNcOWeff + sOWr_ns
    krW_ns = fluid_base['krW'](sW_ns)
    if 'krO' in fluid_base:
        krO_ns = fluid_base['krO'](sOW_ns)
    else:
        krO_ns = fluid_base['krOW'](sOW_ns)

    sW_s = (1.0 - sWc_s - sOWr_s) * sNcWeff + sWc_s
    sOW_s = (1.0 - sWc_s - sOWr_s) * sNcOWeff + sOWr_s
    krW_s = fluid_surf['krW'](sW_s)
    if 'krO' in fluid_surf:
        krO_s = fluid_surf['krO'](sOW_s)
    else:
        krO_s = fluid_surf['krOW'](sOW_s)

    if has_gas:
        sOGr_ns, sGr_ns = float(krPts_base['og']), float(krPts_base['g'])
        sOGr_s, sGr_s = float(krPts_surf['og']), float(krPts_surf['g'])

        sNcOGr = m * sOGr_s + (1.0 - m) * sOGr_ns
        sNcGr = m * sGr_s + (1.0 - m) * sGr_ns
        sNcGeff = (sG - sNcGr) / (1.0 - sNcGr - sNcOGr)
        sNcOGeff = (sO - sNcOGr) / (1.0 - sNcGr - sNcOGr)

        sG_ns = (1.0 - sGr_ns - sOGr_ns) * sNcGeff + sGr_ns
        sOG_ns = (1.0 - sGr_ns - sOGr_ns) * sNcOGeff + sOGr_ns
        krOG_ns = fluid_base['krOG'](sOG_ns)
        krG_ns = fluid_base['krG'](sG_ns)

        sWc_ns_c = _amin(sWc_ns, _value(sW_ns) - 1e-5)
        d = (sG_ns - sGr_ns + sW_ns - sWc_ns_c)
        ww = (sW_ns - sWc_ns_c) / d
        wg = 1.0 - ww
        krO_ns = wg * krOG_ns + ww * krO_ns

        sOG_s = (1.0 - sGr_s - sOGr_s) * sNcOGeff + sOGr_s
        sG_s = (1.0 - sGr_s - sOGr_s) * sNcGeff + sGr_s
        krOG_s = fluid_surf['krOG'](sOG_s)
        krG_s = fluid_surf['krG'](sG_s)

        sWc_s_c = _amin(sWc_s, _value(sW_s) - 1e-5)
        d = (sG_s - sGr_s + sW_s - sWc_s_c)
        ww = (sW_s - sWc_s_c) / d
        wg = 1.0 - ww
        krO_s = wg * krOG_s + ww * krO_s
        krG = m * krG_s + (1.0 - m) * krG_ns

    krW = m * krW_s + (1.0 - m) * krW_ns
    krO = m * krO_s + (1.0 - m) * krO_ns
    if has_gas:
        return krW, krO, krG
    return krW, krO


def _value(x):
    from .._adcompat import value
    return value(x)


def _amin(a, b):
    from .._adcompat import amin
    return amin(a, b)
