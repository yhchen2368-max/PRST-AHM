"""Port of MRST ``equationsOilWaterSurfactant.m``.

Water/oil/surfactant conservation equations, following the same
``SparseADI``/``c1``-``c2`` conventions as ``equationsOilWaterPolymer``
(see that module's docstring). Surfactant modifies the water-oil capillary
pressure (``SurfactantCapillaryPressure``) and relative permeability
(``SurfactantRelativePermeability``, via the capillary number from
``CapillaryNumber``) rather than adding its own flux law -- it is
transported passively with the water phase (``vSft = faceUpstr(cs)*vW``),
unlike polymer's Todd-Longstaff-mixed flux.

Well/surfactant coupling uses the same simplification as
``equationsOilWaterPolymer``'s polymer well source (implicit in
p/sW/bhp/qWs, not in a separate ``qWSft`` unknown).

Requires ``model.operators['sqVeloc']`` (built by
``ad_eor.utils.computeSqVelocTPFA`` from generic half-face connectivity,
``G['cells']['facePos']``/``G['cells']['faces']``) for the capillary
number's velocity reconstruction; raises clearly if unavailable, since
PRSTCore's structured black-oil grid does not currently build that
connectivity.

Simplification: MRST's ``CapillaryNumber`` differentiates ``Nc`` through
the ADI pressure gradient (it affects the Newton Jacobian via
``SurfactantRelativePermeability``'s ``m = fluid.miscfact(log10(Nc))``
interpolation weight). Here ``Nc`` is evaluated from the *current iterate's
value* of the pressure gradient only (no derivative propagated through
``sqVeloc``'s sparse-matrix chain), i.e. it is Newton-lagged rather than
fully implicit -- a common simplification for this kind of secondary,
smoothly-varying auxiliary quantity; it does not change the converged
solution, only the Newton convergence rate.
"""

import numpy as _np
import scipy.sparse as _sp

from PRSTCore.ad_core.adi import SparseADI as _SparseADI
from PRSTCore.ad_core.adi import ad_maximum as _ad_maximum
from PRSTCore.ad_core.adi import ad_select as _ad_select
from PRSTCore.ad_core.models.facility_model import FacilityModel as _FacilityModel

from ..properties.CapillaryNumber import CapillaryNumber
from ..properties.SurfactantAdsorption import SurfactantAdsorption
from ..properties.SurfactantCapillaryPressure import SurfactantCapillaryPressure
from ..properties.SurfactantRelativePermeability import SurfactantRelativePermeability


def equationsOilWaterSurfactant(model, state0, state, dt, drivingForces, wells,
                                 krPts_base, krPts_surf, fluid_base=None, fluid_surf=None):
    """
    Parameters
    ----------
    krPts_base, krPts_surf : dict
        Residual-saturation endpoints, see
        ``ad_eor.properties.SurfactantRelativePermeability``.
    fluid_base, fluid_surf : dict, optional
        SATNUM-region / SURFNUM-region ``krW``/``krOW`` table callables (see
        ``SurfactantRelativePermeability``); default to ``model.fluid``
        itself (single shared table).
    """
    fluid = model.fluid
    nc = model._num_cells()
    nw = len(wells)
    nvar = 3 * nc + 3 * nw

    p = _SparseADI.variable(state['pressure'], nvar, 0)
    sw = _SparseADI.variable(state['sW'], nvar, nc)
    cs = _SparseADI.variable(state['surfactant'], nvar, 2 * nc)
    zero = _SparseADI.constant(_np.zeros(nc), nvar)
    so = 1.0 - sw

    ops = model.operators or {}
    # Shared with GenericBlackOilModel: operators['N'] arrives 0-based or
    # 1-based depending on which operator builder produced it, and guessing
    # from min(N) misreads a 0-based list whose cell 0 is unconnected.
    c1, c2, T = model._internal_connections()
    nface = c1.size
    centroids = _np.asarray(model.G.get('cells', {}).get('centroids', _np.zeros((nc, 3))), dtype=float)
    z = centroids[:, 2] if centroids.ndim == 2 and centroids.shape[1] >= 3 else _np.zeros(nc)
    grav = _np.asarray(getattr(model, 'gravity', [0.0, 0.0, 9.80665]), dtype=float).ravel()
    g = float(grav[-1]) if grav.size else 9.80665
    gdz = g * (z[c2] - z[c1]) if nface else _np.zeros(0)
    C = (_sp.csr_matrix((_np.r_[_np.ones(nface), -_np.ones(nface)],
                        (_np.r_[c1, c2], _np.r_[_np.arange(nface), _np.arange(nface)])),
                       shape=(nc, nface)) if nface else _sp.csr_matrix((nc, 0)))

    sqVeloc = ops.get('sqVeloc')
    if sqVeloc is None:
        raise NotImplementedError(
            "equationsOilWaterSurfactant requires model.operators['sqVeloc'] "
            '(build via ad_eor.utils.computeSqVelocTPFA on a grid providing '
            "cells.facePos/cells.faces half-face connectivity)")
    gradp = (p[c2] - p[c1]) if nface else _SparseADI.constant(_np.zeros(0), nvar)
    # Nc is deliberately evaluated from plain (Newton-lagged) values only,
    # per the module docstring: fluid['ift'](cs) would otherwise carry cs's
    # derivative through Nc (SparseADI's __rtruediv__), but
    # SurfactantRelativePermeability's log(Nc)/log10 needs the SparseADI
    # ``.log()`` *method* rather than plain ``numpy.log`` -- simplest to
    # keep Nc fully un-differentiated instead of special-casing that.
    cs_val = cs.val if isinstance(cs, _SparseADI) else _np.asarray(cs)
    Nc = CapillaryNumber(fluid, gradp.val if nface else _np.zeros(0), T, cs_val, sqVeloc)

    krW, krO = SurfactantRelativePermeability(fluid, sw, so, None, cs, Nc, krPts_base, krPts_surf, False,
                                               fluid_base=fluid_base, fluid_surf=fluid_surf)

    pcOW_base = fluid['pcOW'](sw) if fluid.get('pcOW') is not None else None
    pcOW = SurfactantCapillaryPressure(fluid, pcOW_base, cs)
    pW = p - pcOW if pcOW is not None else p
    pO = p

    bW = fluid['bW'](pW)
    bO = fluid['bO'](pO)
    muW = fluid['muW'](pW)
    muO = fluid['muO'](pO)
    lamW = krW / _ad_maximum(muW, 1.0e-30)
    lamO = krO / _ad_maximum(muO, 1.0e-30)
    rhoWS, rhoOS, _ = model._mrst_surface_densities()
    rhoW = bW * rhoWS
    rhoO = bO * rhoOS

    p0 = _np.asarray(state0['pressure'], dtype=float).ravel()
    sw0 = _np.asarray(state0['sW'], dtype=float).ravel()
    so0 = 1.0 - sw0
    cs0 = _np.asarray(state0['surfactant'], dtype=float).ravel()
    csmax0 = _np.asarray(state0.get('surfactantmax', cs0), dtype=float).ravel()
    csmax = _np.asarray(state.get('surfactantmax', state['surfactant']), dtype=float).ravel()
    pcOW0_base = fluid['pcOW'](sw0) if fluid.get('pcOW') is not None else None
    pcOW0 = SurfactantCapillaryPressure(fluid, pcOW0_base, cs0)
    pW0 = p0 - pcOW0 if pcOW0 is not None else p0
    bW0 = fluid['bW'](pW0)
    bO0 = fluid['bO'](p0)

    def phase_flux(phase_pressure, lam, rho, component_density):
        if not nface:
            empty = _SparseADI.constant(_np.zeros(0), nvar)
            return _SparseADI.constant(_np.zeros(nc), nvar), empty, _np.zeros(0, dtype=int)
        potential = phase_pressure[c2] - phase_pressure[c1] - (rho[c1] + rho[c2]) * (0.5 * gdz)
        upstream = _np.where(potential.val <= 0.0, c1, c2)
        q = potential * (-T) * lam[upstream]
        flux = q * component_density[upstream]
        return flux.linear_map(C), q, upstream

    divW, qW_face, upW = phase_flux(pW, lamW, rhoW, bW * rhoWS)
    divO, _, _ = phase_flux(pO, lamO, rhoO, bO * rhoOS)

    # Surfactant is carried passively with the water phase (no independent
    # flux law, unlike polymer's Todd-Longstaff mixing): vSft = cs_f * vW,
    # reusing the raw (non-mass-scaled) water potential flow qW_face.
    csf = cs[upW] if nface else _SparseADI.constant(_np.zeros(0), nvar)
    bWf = bW[upW] if nface else _SparseADI.constant(_np.zeros(0), nvar)
    divS = (bWf * csf * qW_face).linear_map(C) if nface else _SparseADI.constant(_np.zeros(nc), nvar)

    ads = SurfactantAdsorption(fluid, cs, csmax)
    ads0 = SurfactantAdsorption(fluid, cs0, csmax0)
    poro = _np.asarray(model.rock.get('poro', _np.zeros(nc)), dtype=float).ravel() \
        if isinstance(model.rock, dict) else _np.zeros(nc)
    rhoRSft = float(fluid.get('rhoRSft', 0.0))

    pv, pv0 = model._mrst_pore_volume_adi(p), model._mrst_pore_volume(p0)
    invdt = 1.0 / max(float(dt), 1.0e-30)

    resW = (pv * bW * sw * rhoWS - pv0 * bW0 * sw0 * rhoWS) * invdt + divW
    resO = (pv * bO * so * rhoOS - pv0 * bO0 * so0 * rhoOS) * invdt + divO
    # equationsOilWaterSurfactant.m:
    #   ads_term   = rhoRSft.*((1-poro)./poro).*(ads - ads0);
    #   surfactant = (1/dt).*(pv.*bW.*sW.*cs - pv0.*bW0.*sW0.*cs0)
    #              + (op.pv/dt).*ads_term + Div(bWvSft)
    # op.pv (the base pore volume) multiplies the adsorbed term:
    # pv*(1-poro)/poro is the rock volume, times rhoRSft the rock mass,
    # times ads the adsorbed surfactant per unit rock mass.
    pv_base = _np.asarray(model._porevolume_vector(), dtype=float).ravel()
    resS = ((pv * bW * sw * cs - pv0 * bW0 * sw0 * cs0) * invdt
            + (pv_base * rhoRSft * ((1.0 - poro) / _np.maximum(poro, 1.0e-12))
               * (ads - ads0)) * invdt
            + divS)
    diag_cs = _np.asarray(resS.jac[:, 2 * nc:3 * nc].diagonal()).ravel()
    eps_tol = _np.sqrt(1.0e-8) * _np.mean(_np.abs(diag_cs))
    if eps_tol == 0.0:
        eps_tol = 1.0e-8
    bad = _np.abs(diag_cs) < eps_tol
    if _np.any(bad):
        resS = _ad_select(bad, cs, resS)

    qws = _SparseADI.variable(state.get('facility_qWs', _np.zeros(nw)), nvar, 3 * nc)
    qos = _SparseADI.variable(state.get('facility_qOs', _np.zeros(nw)), nvar, 3 * nc + nw)
    bhp = _SparseADI.variable(state.get('facility_bhp', _np.zeros(nw)), nvar, 3 * nc + 2 * nw)

    def component_mass_fn(c, qph):
        qW, qO = qph
        return [qW * rhoWS * bW[c], qO * rhoOS * bO[c]]

    (srcW, srcO), (surface_w, surface_o), perf_phase_all, perf_component_all = \
        model.FacilityModel.compute_well_contributions(
            wells=wells, state=state, p=p, bhp=bhp,
            lam_phases=[lamW, lamO], rhoS_phases=[rhoWS, rhoOS],
            component_mass_fn=component_mass_fn, nc=nc, nw=nw, nvar=nvar, n_component_phases=2,
        )
    state['facility_perforation_phase_flux'] = [
        _np.pad(a, ((0, 0), (0, 1))) if a.size else _np.zeros((0, 3)) for a in perf_phase_all]
    state['facility_perforation_component_flux'] = [
        _np.pad(a, ((0, 0), (0, 1))) if a.size else _np.zeros((0, 3)) for a in perf_component_all]

    resW = resW - srcW
    resO = resO - srcO
    resS = resS - _surfactant_well_source(wells, perf_component_all, cs, rhoWS, nvar, nc)

    rhoW0 = rhoWS * bW0
    rhoO0 = rhoOS * bO0
    scaleW, scaleO = rhoWS / _np.mean(rhoW0), rhoOS / _np.mean(rhoO0)
    fW, fO = (qws - surface_w) * scaleW, (qos - surface_o) * scaleO
    closure = _FacilityModel.compute_control_equations(
        wells, qs_phases={'w': qws, 'o': qos}, bhp=bhp, phase_order=['w', 'o'],
    )
    residual = _SparseADI.concat(
        (resW, resO, resS, fW, fO, _SparseADI.concat(closure) if closure else _SparseADI.constant(_np.zeros(0), nvar)))
    return residual, {'rho0': (rhoW0, rhoO0)}


def _surfactant_well_source(wells, perf_component_all, cs, rhoWS, nvar, nc):
    """Same well-coupling simplification as
    ``equationsOilWaterPolymer._polymer_well_source``, for surfactant."""
    if not wells:
        return _SparseADI.constant(_np.zeros(nc), nvar)
    src = _np.zeros(nc)
    cs_val = cs.val if isinstance(cs, _SparseADI) else _np.asarray(cs)
    for iw, w in enumerate(wells):
        cells = _np.atleast_1d(w.get('cells', [])).astype(int)
        n = cells.size
        if n == 0:
            continue
        perf_comp = perf_component_all[iw] if iw < len(perf_component_all) else _np.zeros((n, 2))
        cqWs = perf_comp[:, 0] / max(rhoWS, 1.0e-30) if perf_comp.size else _np.zeros(n)
        isInj = cqWs > 0
        conc_ctrl = float(w.get('surfactant', 0.0))
        conc = _np.where(isInj, conc_ctrl, cs_val[cells])
        _np.add.at(src, cells, conc * cqWs)
    return _SparseADI.constant(src, nvar)
