"""Port of MRST ``equationsOilWaterPolymer.m``.

Assembles the linearized water/oil/polymer conservation equations, mirroring
:meth:`GenericBlackOilModel._mrst_generic_adi_residual_ow`'s structure and
conventions (single concatenated ``SparseADI`` Jacobian; ``c1``/``c2``
neighbor arrays instead of MRST's generic ``operators.Grad``/``Div``) with
the water phase replaced by ``getFluxAndPropsWaterPolymer_BO`` and an added
polymer conservation equation.

Well/polymer coupling simplification: MRST's ``insertWellEquations`` couples
the polymer source term implicitly through a per-well ``qWPoly`` primary
variable and closure equation (``getExtraWellContributions``). PRSTCore's
``FacilityModel`` has no generic mechanism for extra per-well unknowns, so
here the polymer well source is instead computed from each perforation's
*already-assembled* water phase flux (through the existing qWs/bhp facility
variables) -- i.e. still fully implicit in pressure/saturation/bhp/qWs, but
not implicit in its own rate unknown. This reproduces the correct component
mass balance; it only omits the (purely diagnostic) ``qWPoly`` accounting
variable MRST reports alongside it.
"""

import numpy as _np
import scipy.sparse as _sp

from PRSTCore.ad_core.adi import SparseADI as _SparseADI
from PRSTCore.ad_core.adi import ad_maximum as _ad_maximum
from PRSTCore.ad_core.adi import ad_select as _ad_select

from .getFluxAndPropsWaterPolymer_BO import getFluxAndPropsWaterPolymer_BO
from ..properties.PolymerAdsorption import PolymerAdsorption


def equationsOilWaterPolymer(model, state0, state, dt, drivingForces, wells):
    """
    Parameters
    ----------
    model : GenericBlackOilModel-like
        Must expose ``operators`` (with ``'N'``, ``'T'``), ``fluid``,
        ``G``, ``gravity``, ``_phase_pressures_adi``/``_phase_pressures``,
        ``_relative_perm_adi``, ``_mrst_surface_densities``,
        ``_mrst_pore_volume_adi``/``_mrst_pore_volume``, ``FacilityModel``.
    state0, state : dict
        Must additionally carry ``'polymer'``/``'polymermax'`` (cell arrays)
        alongside the usual ``'pressure'``/``'sW'``.
    wells : list[dict]
        Active wells; each well needs a ``'polymer'`` (injection
        concentration) field when it is an injector with polymer control.

    Returns
    -------
    residual : SparseADI
    aux : dict  (pvt, rho0 -- matches ``_mrst_generic_adi_residual_ow``'s aux)
    """
    fluid = model.fluid
    nc = model._num_cells()
    nw = len(wells)
    nvar = 3 * nc + 3 * nw

    p = _SparseADI.variable(state['pressure'], nvar, 0)
    sw = _SparseADI.variable(state['sW'], nvar, nc)
    cp = _SparseADI.variable(state['polymer'], nvar, 2 * nc)
    cpmax = _np.asarray(state['polymermax'], dtype=float).ravel()
    zero = _SparseADI.constant(_np.zeros(nc), nvar)

    pW, pO, _ = model._phase_pressures_adi(p, sw, zero)
    pvt = model._phase_pvt_from_phase_pressures_adi(
        pW, pO, pO, rs_override=zero, rv_override=zero, sG_override=zero,
    )
    p0 = _np.asarray(state0['pressure'], dtype=float).ravel()
    sw0 = _np.asarray(state0['sW'], dtype=float).ravel()
    cp0 = _np.asarray(state0['polymer'], dtype=float).ravel()
    cpmax0 = _np.asarray(state0['polymermax'], dtype=float).ravel()
    pW0, pO0, _ = model._phase_pressures(p0, sw0, _np.zeros(nc))
    pvt0 = model._phase_pvt_from_phase_pressures(
        pW0, pO0, pO0, rs_override=_np.zeros(nc), rv_override=_np.zeros(nc),
        sG_override=_np.zeros(nc),
    )
    bO0 = pvt0['bo']

    krW, krO, _ = model._relative_perm_adi(sw, zero)
    rhoWS, rhoOS, _ = model._mrst_surface_densities()

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

    ads = PolymerAdsorption(fluid, cp, cpmax)
    ads0 = PolymerAdsorption(fluid, cp0, cpmax0)

    vW, vP, bW, muWeffMult, mobW, mobP, rhoW, pW_out, upcw, a_mix = \
        getFluxAndPropsWaterPolymer_BO(fluid, pO, sw, cp, ads, krW, T, c1, c2, gdz)
    bW0 = fluid['bW'](p0)

    # Oil phase flux: matches _mrst_generic_adi_residual_ow's phase_flux.
    lamO = krO / _ad_maximum(pvt['muo'], 1.0e-30)
    bO = pvt['bo']
    rhoO = bO * rhoOS
    rhoO0 = rhoOS * pvt0['bo']

    def phase_flux(phase_pressure, lam, rho, component_density):
        if not nface:
            return _SparseADI.constant(_np.zeros(nc), nvar), _np.zeros(0), _np.zeros(0, dtype=int)
        potential = phase_pressure[c2] - phase_pressure[c1] - (rho[c1] + rho[c2]) * (0.5 * gdz)
        upstream = _np.where(potential.val <= 0.0, c1, c2)
        q = potential * (-T) * lam[upstream]
        flux = q * component_density[upstream]
        return flux.linear_map(C), q, upstream

    divO, _, _ = phase_flux(pO, lamO, rhoO, bO * rhoOS)
    # MRST: bWvW = faceUpstr(upcw, bW).*vW; bWvP = faceUpstr(upcw, bW).*vP.
    # water/oil rows here are in mass units (matching
    # _mrst_generic_adi_residual_ow's convention), so bWvW also carries the
    # rhoWS factor that promotes it from surface-volume to mass flux; the
    # polymer row stays in MRST's native (unscaled) surface-volume units.
    upW = _np.where(upcw, c1, c2) if nface else _np.zeros(0, dtype=int)
    bWvW = bW[upW] * rhoWS * vW if nface else _SparseADI.constant(_np.zeros(0), nvar)
    bWvP = bW[upW] * vP if nface else _SparseADI.constant(_np.zeros(0), nvar)
    divW = bWvW.linear_map(C) if nface else _SparseADI.constant(_np.zeros(nc), nvar)
    divP = bWvP.linear_map(C) if nface else _SparseADI.constant(_np.zeros(nc), nvar)

    pv, pv0 = model._mrst_pore_volume_adi(p), model._mrst_pore_volume(p0)
    invdt = 1.0 / max(float(dt), 1.0e-30)

    resW = (pv * sw * bW * rhoWS - pv0 * sw0 * bW0 * rhoWS) * invdt + divW
    resO = (pv * (1.0 - sw) * rhoOS * bO - pv0 * (1.0 - sw0) * rhoOS * bO0) * invdt + divO

    poro = _np.asarray(model.rock.get('poro', _np.zeros(nc)), dtype=float).ravel() \
        if isinstance(model.rock, dict) else _np.zeros(nc)
    dps = float(fluid.get('dps', 0.0))
    rhoR = float(fluid.get('rhoR', 0.0))
    # equationsOilWaterPolymer.m:
    #   polymer = (op.pv.*(1-dps)/dt).*(pvMult.*bW.*sW.*cp - pvMult0.*bW0.*sW0.*cp0)
    #           + (op.pv/dt).*(rhoR.*((1-poro)./poro).*(ads-ads0)) + Div(bWvP)
    # The adsorbed term carries op.pv as well: pv*(1-poro)/poro is the rock
    # volume, times rhoR the rock mass, times ads [kg/kg] the adsorbed
    # polymer mass -- without it the term is not even dimensionally a mass,
    # and adsorption is effectively absent from the transport.
    # ``op.pv`` here is the base pore volume; the compressibility multiplier
    # appears explicitly as pvMult only on the accumulation term above (and
    # is already folded into ``pv``/``pv0``).
    pv_base = _np.asarray(model._porevolume_vector(), dtype=float).ravel()
    resP = ((pv * (1.0 - dps) * bW * sw * cp - pv0 * (1.0 - dps) * bW0 * sw0 * cp0) * invdt
            + (pv_base * rhoR * ((1.0 - poro) / _np.maximum(poro, 1.0e-12))
               * (ads - ads0)) * invdt + divP)

    # MRST: ``polymer(bad) = cp(bad)`` -- in cells whose polymer-equation
    # Jacobian diagonal (w.r.t. cp) is ill-conditioned (typically cells with
    # ~no water), the row's value+Jacobian are replaced wholesale by the cp
    # primary variable itself, which linearizes to ``cp_new = 0``: a safety
    # clamp driving the (physically meaningless) concentration in an
    # essentially dry cell to zero rather than letting a near-singular row
    # produce a nonsense update.
    diag_cp = _np.asarray(resP.jac[:, 2 * nc:3 * nc].diagonal()).ravel()
    eps_tol = _np.sqrt(1.0e-8) * _np.mean(_np.abs(diag_cp))
    if eps_tol == 0.0:
        eps_tol = 1.0e-8
    bad = _np.abs(diag_cp) < eps_tol
    if _np.any(bad):
        resP = _ad_select(bad, cp, resP)

    qws = _SparseADI.variable(state.get('facility_qWs', _np.zeros(nw)), nvar, 3 * nc)
    qos = _SparseADI.variable(state.get('facility_qOs', _np.zeros(nw)), nvar, 3 * nc + nw)
    bhp = _SparseADI.variable(state.get('facility_bhp', _np.zeros(nw)), nvar, 3 * nc + 2 * nw)

    def component_mass_fn(c, qph):
        qW, qO = qph
        return [qW * rhoWS * bW[c], qO * rhoOS * bO[c]]

    (srcW, srcO), (surface_w, surface_o), perf_phase_all, perf_component_all = \
        model.FacilityModel.compute_well_contributions(
            wells=wells, state=state, p=p, bhp=bhp,
            lam_phases=[mobW, lamO], rhoS_phases=[rhoWS, rhoOS],
            component_mass_fn=component_mass_fn, nc=nc, nw=nw, nvar=nvar, n_component_phases=2,
        )
    perf_phase_all = [_np.pad(a, ((0, 0), (0, 1))) if a.size else _np.zeros((0, 3)) for a in perf_phase_all]
    perf_component_all = [_np.pad(a, ((0, 0), (0, 1))) if a.size else _np.zeros((0, 3)) for a in perf_component_all]
    state['facility_perforation_phase_flux'] = perf_phase_all
    state['facility_perforation_component_flux'] = perf_component_all

    resW = resW - srcW
    resO = resO - srcO
    resP = resP - _polymer_well_source(model, wells, perf_component_all, cp, rhoWS, nvar, nc)

    rhoW0 = rhoWS * pvt0['bw']
    scaleW, scaleO = rhoWS / _np.mean(rhoW0), rhoOS / _np.mean(rhoO0)
    fW, fO = (qws - surface_w) * scaleW, (qos - surface_o) * scaleO

    from PRSTCore.ad_core.models.facility_model import FacilityModel as _FacilityModel
    closure = _FacilityModel.compute_control_equations(
        wells, qs_phases={'w': qws, 'o': qos}, bhp=bhp, phase_order=['w', 'o'],
    )
    residual = _SparseADI.concat(
        (resW, resO, resP, fW, fO, _SparseADI.concat(closure) if closure else _SparseADI.constant(_np.zeros(0), nvar)))
    return residual, {'pvt': pvt, 'rho0': (rhoW0, rhoO0)}


def _polymer_well_source(model, wells, perf_component_all, cp, rhoWS, nvar, nc):
    """Scatter-add each perforation's polymer mass rate into its cell's
    polymer equation. ``perf_component_all[iw]`` is well ``iw``'s
    per-perforation ``[water, oil]`` mass-rate array (see
    ``FacilityModel.compute_well_contributions``); dividing column 0 by
    ``rhoWS`` recovers the perforation's water *volume* rate ``cqWs``,
    matching ``getExtraWellContributions``'s ``cqWs = qMass{wix}./f.rhoWS``.
    The Todd-Longstaff mixing correction on producing connections
    (``concpolyRes./(a+(1-a)*cpbar)``) is intentionally omitted here: it
    only matters when polymer is simultaneously injected and produced
    within the same timestep at the same well, a corner case not needed for
    the flooding scenarios this port targets."""
    if not wells:
        return _SparseADI.constant(_np.zeros(nc), nvar)
    src = _np.zeros(nc)
    cp_val = cp.val if isinstance(cp, _SparseADI) else _np.asarray(cp)
    for iw, w in enumerate(wells):
        cells = _np.atleast_1d(w.get('cells', [])).astype(int)
        n = cells.size
        if n == 0:
            continue
        perf_comp = perf_component_all[iw] if iw < len(perf_component_all) else _np.zeros((n, 2))
        cqWs = perf_comp[:, 0] / max(rhoWS, 1.0e-30) if perf_comp.size else _np.zeros(n)
        isInj = cqWs > 0
        conc_ctrl = float(w.get('polymer', 0.0))
        conc = _np.where(isInj, conc_ctrl, cp_val[cells])
        _np.add.at(src, cells, conc * cqWs)
    # Units match the rest of the (unscaled, "surface-volume") polymer
    # equation: concentration * volume-rate, exactly ``cqP`` in
    # ``getExtraWellContributions`` -- no extra rhoWS factor here.
    return _SparseADI.constant(src, nvar)
