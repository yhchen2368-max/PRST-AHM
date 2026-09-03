"""Python port of MRST's ``wellBoreFriction.m`` (mrst-2026a/autodiff/ad-core/
models/facilities): the empirical Moody-type wellbore friction pressure-drop
model used by multi-segment well (MSW) segment equations.

Splits the Reynolds-number range into laminar / transitional / turbulent
regimes (matching ECLIPSE's convention), using the Colebrook-White-style
turbulent friction factor and a linear interpolation across the transitional
band (2000 <= Re <= 4000).
"""

from __future__ import annotations

import numpy as _np


def well_bore_friction(v, rho, mu, D, L, roughness, *, flowtype: str = "massRate",
                        assume_turbulent: bool = False):
    """Port of MRST ``wellBoreFriction.m``.

    Parameters
    ----------
    v : array
        Flow rate (interpretation set by ``flowtype``: velocity, volumetric
        rate, or mass rate) per segment.
    rho, mu : array
        Fluid density / viscosity per segment.
    D : array or (array, array)
        Segment diameter, or ``(inner, outer)`` diameters for an annulus.
    L : array
        Segment length.
    roughness : array
        Pipe wall roughness.
    flowtype : {'velocity', 'volumeRate', 'massRate'}
    assume_turbulent : bool
        If True, skip the laminar/transitional regime split and always use
        the turbulent correlation (matches MRST's ``assumeTurbulent`` flag).

    Returns
    -------
    dp : ndarray
        Frictional pressure drop per segment (signed with the flow direction).
    """
    v = _np.asarray(v, dtype=float).copy()
    rho = _np.asarray(rho, dtype=float)
    mu = _np.asarray(mu, dtype=float)
    L = _np.asarray(L, dtype=float)
    roughness = _np.asarray(roughness, dtype=float)

    # Accept either a plain outer-diameter array/scalar, or an (Di, Do) pair
    # (matching MRST's ``numel(D) == 2`` check for an annulus).
    if isinstance(D, (tuple, list)) and len(D) == 2:
        Di, Do = _np.asarray(D[0], dtype=float), _np.asarray(D[1], dtype=float)
    else:
        Di, Do = 0.0, _np.asarray(D, dtype=float)

    if flowtype == "velocity":
        pass
    elif flowtype == "volumeRate":
        v = v / (_np.pi * ((Do / 2.0) ** 2 - (Di / 2.0) ** 2))
    elif flowtype == "massRate":
        v = v / (_np.pi * rho * ((Do / 2.0) ** 2 - (Di / 2.0) ** 2))
    else:
        raise ValueError(f"Unknown flow type: {flowtype!r}")

    is_zero = v == 0.0
    v = _np.where(is_zero, _np.finfo(float).eps, v)

    re = _np.abs(rho * v * (Do - Di) / mu)

    f = (-3.6 * _np.log10(6.9 / re + (roughness / (3.7 * Do)) ** (10.0 / 9.0))) ** (-2)

    if not assume_turbulent:
        re1, re2 = 2000.0, 4000.0
        lam = re <= re1
        tur = re >= re2
        inter = ~(lam | tur)
        f1 = 16.0 / re1
        f2 = (-3.6 * _np.log10(6.9 / re2 + (roughness / (3.7 * Do)) ** (10.0 / 9.0))) ** (-2)
        f = _np.where(lam, 16.0 / _np.maximum(re, 1e-300), f)
        f = _np.where(inter, f1 + ((f2 - f1) / (re2 - re1)) * (re - re1), f)

    dp = -(2.0 * _np.sign(v) * L / (Do - Di)) * (f * rho * v**2)
    dp = _np.where(is_zero, 0.0, dp)
    return dp


def well_bore_friction_adi(v, rho, mu, D, L, roughness, *, flowtype: str = "massRate",
                            assume_turbulent: bool = False):
    """ADI-differentiable counterpart of :func:`well_bore_friction`.

    Differentiates the friction pressure drop through ``v``/``rho`` (either
    may be a :class:`~PRSTCore.ad_core.adi.SparseADI` or a plain array/
    scalar; falls back to :func:`well_bore_friction` when neither is ADI).
    ``mu``/``D``/``L``/``roughness`` are always treated as fixed segment
    properties, matching the base function.

    Only the laminar/transitional/turbulent *regime classification* (and
    the zero-flow fallback) is evaluated at the frozen current-iterate
    value -- the same convention this codebase already uses for other
    status/classification flags (e.g. the gas-oil-ratio saturation status
    in :meth:`PRSTCore.ad_core.models.multisegment_well.MultisegmentWell.compute_node_mix`).
    Within each frozen regime, the Moody friction-factor formula itself
    (``16/Re`` laminar, the Colebrook-White-style turbulent correlation, and
    the linear transitional interpolation between them) is evaluated with
    full ADI arithmetic, so the resulting pressure drop is properly
    differentiated through ``v``/``rho`` rather than added to a residual as
    a Newton-frozen constant.
    """
    from PRSTCore.ad_core.adi import SparseADI as _S, ad_abs, ad_select

    if not isinstance(v, _S) and not isinstance(rho, _S):
        return well_bore_friction(v, rho, mu, D, L, roughness, flowtype=flowtype,
                                   assume_turbulent=assume_turbulent)

    nvar = v.nvar if isinstance(v, _S) else rho.nvar
    n = v.val.size if isinstance(v, _S) else rho.val.size
    v = v if isinstance(v, _S) else _S.constant(_np.broadcast_to(_np.asarray(v, dtype=float), (n,)), nvar)
    rho = rho if isinstance(rho, _S) else _S.constant(_np.broadcast_to(_np.asarray(rho, dtype=float), (n,)), nvar)

    mu = _np.asarray(mu, dtype=float)
    L = _np.asarray(L, dtype=float)
    roughness = _np.asarray(roughness, dtype=float)
    if isinstance(D, (tuple, list)) and len(D) == 2:
        Di, Do = _np.asarray(D[0], dtype=float), _np.asarray(D[1], dtype=float)
    else:
        Di, Do = _np.zeros(n), _np.asarray(D, dtype=float)

    area = _np.pi * ((Do / 2.0) ** 2 - (Di / 2.0) ** 2)
    if flowtype == "velocity":
        vv = v
    elif flowtype == "volumeRate":
        vv = v / area
    elif flowtype == "massRate":
        vv = v / (rho * area)
    else:
        raise ValueError(f"Unknown flow type: {flowtype!r}")

    is_zero = vv.val == 0.0
    sign_v = _np.where(is_zero, 0.0, _np.sign(vv.val))
    if _np.any(is_zero):
        vv = _S(_np.where(is_zero, _np.finfo(float).eps, vv.val), vv.jac)

    re = ad_abs(rho * vv * (Do - Di) / mu)
    rough_term = (roughness / (3.7 * Do)) ** (10.0 / 9.0)
    inv_ln10 = 1.0 / _np.log(10.0)

    def _turb_factor(re_):
        inner = 6.9 / re_ + rough_term
        log10_inner = inner.log() * inv_ln10
        return (-3.6 * log10_inner) ** (-2.0)

    f_turb = _turb_factor(re)

    if assume_turbulent:
        f = f_turb
    else:
        re1, re2 = 2000.0, 4000.0
        re_val = re.val
        lam = re_val <= re1
        tur = re_val >= re2

        f_lam = 16.0 / re
        f1 = 16.0 / re1
        f2_val = (-3.6 * _np.log10(6.9 / re2 + rough_term)) ** (-2.0)
        slope = (f2_val - f1) / (re2 - re1)
        f_inter = f1 + slope * (re - re1)

        f = ad_select(lam, f_lam, ad_select(tur, f_turb, f_inter))

    prefactor = -(2.0 * sign_v * L / (Do - Di))
    dp = prefactor * (f * rho * (vv ** 2.0))
    if _np.any(is_zero):
        dp = ad_select(is_zero, _S.constant(_np.zeros(n), dp.nvar), dp)
    return dp
