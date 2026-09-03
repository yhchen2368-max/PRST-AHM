"""Port of MRST ``equationsThreePhaseBlackOilPolymer.m``.

Three-phase black-oil + polymer conservation equations, mirroring
:meth:`GenericBlackOilModel._mrst_generic_adi_residual`'s structure
(component masses, Rs-driven status switching, vapoil support) with the
water phase replaced by ``getFluxAndPropsWaterPolymer_BO`` and an added
polymer conservation equation -- i.e. the three-phase extension of
``equationsOilWaterPolymer`` exactly as the ``.m`` sources relate to each
other.

Same well/polymer coupling simplification as ``equationsOilWaterPolymer``
(see its module docstring): the well polymer source is computed from the
already-assembled qWs/bhp facility variables rather than through a separate
``qWPoly`` primary variable/closure equation.

PLYSHEAR/PLYSHLOG shear-thinning: only the reservoir internal-face
correction is applied (``vW``, ``vP`` divided by ``computeShearMult(Log)``
evaluated at the face water velocity). The ``.m`` source's *well*-side shear
correction (``VwW``, based on each perforation's representative radius
``W.rR`` and perforation thickness) is not ported: it needs well geometry
fields PRSTCore's well dict does not currently carry.
"""

import numpy as _np
import scipy.sparse as _sp

from PRSTCore.ad_core.adi import SparseADI as _SparseADI
from PRSTCore.ad_core.adi import ad_maximum as _ad_maximum
from PRSTCore.ad_core.adi import ad_select as _ad_select
from PRSTCore.ad_core.models.facility_model import FacilityModel as _FacilityModel

from .getFluxAndPropsWaterPolymer_BO import getFluxAndPropsWaterPolymer_BO
from .private.computeShearMult import computeShearMult
from .private.computeShearMultLog import computeShearMultLog
from ..properties.PolymerAdsorption import PolymerAdsorption


def equationsThreePhaseBlackOilPolymer(model, state0, state, dt, drivingForces, wells):
    if not (model.water and model.oil and model.gas):
        raise NotImplementedError('equationsThreePhaseBlackOilPolymer requires all three active phases')
    fluid = model.fluid
    nc = model._num_cells()
    nw = len(wells)
    status = model._mrst_blackoil_status(state)
    st1, st2, st3 = [_np.asarray(x, dtype=bool).ravel() for x in status]
    nvar = 4 * nc + 4 * nw

    p = _SparseADI.variable(state['pressure'], nvar, 0)
    sw = _SparseADI.variable(state['sW'], nvar, nc)
    x = _SparseADI.variable(model._mrst_pack_primary(state, drivingForces)[0][2 * nc:3 * nc], nvar, 2 * nc)
    cp = _SparseADI.variable(state['polymer'], nvar, 3 * nc)
    cpmax = _np.asarray(state['polymermax'], dtype=float).ravel()

    sg = (1.0 - sw) * st2 + x * st3
    pW, pO, pG = model._phase_pressures_adi(p, sw, sg, state.get('pcowScale'))
    pvt_sat = model._phase_pvt_adi(pO)
    rs = (pvt_sat['rs'] * (~st1)) + x * st1 if model.disgas else _SparseADI.constant(state.get('rs', _np.zeros(nc)), nvar)
    if model.vapoil:
        deck_pvt = getattr(model, '_blackoil_pvt', None)
        if deck_pvt is None or not hasattr(deck_pvt, 'rv_sat_adi'):
            raise NotImplementedError('Deck AD VAPOIL assembly requires DeckBlackOilPVT.rv_sat_adi')
        rv_sat = deck_pvt.rv_sat_adi(pG)
        rv = rv_sat * (~st2) + x * st2
    else:
        rv = _SparseADI.constant(state.get('rv', _np.zeros(nc)), nvar)
    so = 1.0 - sw - sg
    pvt = model._phase_pvt_from_phase_pressures_adi(
        pW, pO, pG, rs_override=rs, rv_override=rv, sG_override=sg,
        oil_saturated_override=(sg.val > 0.0),
        gas_saturated_override=(so.val > 0.0),
    )
    bO, bG = pvt['bo'], pvt['bg']
    muO, muG = pvt['muo'], pvt['mug']
    krW, krO, krG = model._relative_perm_adi(sw, sg)

    rhoWS, rhoOS, rhoGS = model._mrst_surface_densities()
    rhoO = bO * (rhoOS + rs * rhoGS)
    rhoG = bG * (rhoGS + rv * rhoOS)
    rhoG_component = bG * rhoGS

    p0 = _np.asarray(state0['pressure'], dtype=float).ravel()
    sw0 = _np.asarray(state0['sW'], dtype=float).ravel()
    sg0 = _np.asarray(state0['sG'], dtype=float).ravel()
    so0 = 1.0 - sw0 - sg0
    rs0 = _np.asarray(state0['rs'], dtype=float).ravel()
    rv0 = _np.asarray(state0.get('rv', _np.zeros(nc)), dtype=float).ravel()
    cp0 = _np.asarray(state0['polymer'], dtype=float).ravel()
    cpmax0 = _np.asarray(state0['polymermax'], dtype=float).ravel()
    pW0, pO0, pG0 = model._phase_pressures(p0, sw0, sg0, state0.get('pcowScale'))
    pvt0 = model._phase_pvt_from_phase_pressures(
        pW0, pO0, pG0, rs_override=rs0, rv_override=rv0, sG_override=sg0,
        oil_saturated_override=(sg0 > 0.0),
        gas_saturated_override=(so0 > 0.0),
    )
    rhoO0 = pvt0['bo'] * (rhoOS + rs0 * rhoGS)
    rhoG_phase0 = pvt0['bg'] * (rhoGS + rv0 * rhoOS)
    rhoG_component0 = pvt0['bg'] * rhoGS
    bW0 = fluid['bW'](p0)

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

    lamO = krO / _ad_maximum(muO, 1.0e-30)
    lamG = krG / _ad_maximum(muG, 1.0e-30)

    def phase_flux(phase_pressure, lam, rho, component_density):
        if not nface:
            return _SparseADI.constant(_np.zeros(nc), nvar), _SparseADI.constant(_np.zeros(0), nvar), _np.zeros(0, dtype=int)
        potential = phase_pressure[c2] - phase_pressure[c1] - (rho[c1] + rho[c2]) * (0.5 * gdz)
        upstream = _np.where(potential.val <= 0.0, c1, c2)
        q = potential * (-T) * lam[upstream]
        flux = q * component_density[upstream]
        return flux.linear_map(C), q, upstream

    divO, qOface, upO = phase_flux(pO, lamO, rhoO, bO * rhoOS)
    divG, qGface, upG = phase_flux(pG, lamG, rhoG, bG * rhoGS)
    if nface:
        divG = divG + (qOface * (rs[upO] * rhoGS * bO[upO])).linear_map(C)
        if model.vapoil:
            divO = divO + (qGface * (rv[upG] * rhoOS * bG[upG])).linear_map(C)

    upW = _np.where(upcw, c1, c2) if nface else _np.zeros(0, dtype=int)
    vW_used, vP_used = vW, vP

    if getattr(model, 'usingShear', False) or getattr(model, 'usingShearLog', False):
        # Reservoir-side shear correction only (see module docstring). MRST
        # normalizes the Darcy face flux to an interstitial pore velocity
        # (``Vw = vW./(poroFace.*faceA)``); that normalization is skipped
        # here since PRSTCore's ``operators`` dict does not expose face
        # areas aligned with the internal-connection ordering, so the raw
        # Darcy flux magnitude is used instead. This shifts where the
        # PLYSHEAR/PLYSHLOG table's velocity threshold triggers but keeps
        # the shear-thinning mechanism itself intact.
        Vw = _np.abs(vW.val if isinstance(vW, _SparseADI) else vW)
        muWMultf = muWeffMult.val[upW] if isinstance(muWeffMult, _SparseADI) else _np.asarray(muWeffMult)[upW]
        if getattr(model, 'usingShear', False):
            shearMultf = computeShearMult(fluid, Vw, muWMultf)
        else:
            shearMultf = computeShearMultLog(fluid, Vw, muWMultf)
        vW_used = vW / shearMultf
        vP_used = vP / shearMultf

    bWvW = bW[upW] * rhoWS * vW_used if nface else _SparseADI.constant(_np.zeros(0), nvar)
    bWvP = bW[upW] * vP_used if nface else _SparseADI.constant(_np.zeros(0), nvar)
    divW = bWvW.linear_map(C) if nface else _SparseADI.constant(_np.zeros(nc), nvar)
    divP = bWvP.linear_map(C) if nface else _SparseADI.constant(_np.zeros(nc), nvar)

    qws = _SparseADI.variable(state.get('facility_qWs', _np.zeros(nw)), nvar, 4 * nc)
    qos = _SparseADI.variable(state.get('facility_qOs', _np.zeros(nw)), nvar, 4 * nc + nw)
    qgs = _SparseADI.variable(state.get('facility_qGs', _np.zeros(nw)), nvar, 4 * nc + 2 * nw)
    bhp = _SparseADI.variable(state.get('facility_bhp', _np.zeros(nw)), nvar, 4 * nc + 3 * nw)

    def component_mass_fn(c, qph):
        qphW, qphO, qphG = qph
        cmassW = qphW * rhoWS * bW[c]
        cmassO = qphO * rhoOS * bO[c]
        if model.vapoil:
            cmassO = cmassO + qphG * rv[c] * rhoOS * bG[c]
        cmassG = qphG * rhoGS * bG[c] + qphO * rs[c] * rhoGS * bO[c]
        return [cmassW, cmassO, cmassG]

    (srcW, srcO, srcG), (surface_w, surface_o, surface_g), perf_phase_all, perf_component_all = \
        model.FacilityModel.compute_well_contributions(
            wells=wells, state=state, p=p, bhp=bhp,
            lam_phases=[mobW, lamO, lamG], rhoS_phases=[rhoWS, rhoOS, rhoGS],
            component_mass_fn=component_mass_fn, nc=nc, nw=nw, nvar=nvar, n_component_phases=3,
        )
    state['facility_perforation_phase_flux'] = perf_phase_all
    state['facility_perforation_component_flux'] = perf_component_all

    pv = model._mrst_pore_volume_adi(p)
    pv0 = model._mrst_pore_volume(p0)
    invdt = 1.0 / max(float(dt), 1.0e-30)

    resW = (pv * sw * bW * rhoWS - pv0 * sw0 * bW0 * rhoWS) * invdt + divW - srcW
    if model.vapoil:
        resO = (pv * (so * bO + rv * bG * sg) * rhoOS -
                pv0 * (so0 * pvt0['bo'] + rv0 * pvt0['bg'] * sg0) * rhoOS) * invdt + divO - srcO
    else:
        resO = (pv * so * rhoOS * bO - pv0 * so0 * rhoOS * pvt0['bo']) * invdt + divO - srcO
    resG = (pv * (sg * rhoG_component + so * rs * rhoGS * bO) -
            pv0 * (sg0 * rhoG_component0 + so0 * rs0 * rhoGS * pvt0['bo'])) * invdt + divG - srcG

    poro = _np.asarray(model.rock.get('poro', _np.zeros(nc)), dtype=float).ravel() \
        if isinstance(model.rock, dict) else _np.zeros(nc)
    dps = float(fluid.get('dps', 0.0))
    rhoR = float(fluid.get('rhoR', 0.0))
    # equationsThreePhaseBlackOilPolymer.m carries op.pv on the adsorbed
    # term too: pv*(1-poro)/poro is the rock volume, times rhoR the rock
    # mass, times ads [kg/kg] the adsorbed polymer mass. See the identical
    # note in equationsOilWaterPolymer.py.
    pv_base = _np.asarray(model._porevolume_vector(), dtype=float).ravel()
    resP = ((pv * (1.0 - dps) * bW * sw * cp - pv0 * (1.0 - dps) * bW0 * sw0 * cp0) * invdt
            + (pv_base * rhoR * ((1.0 - poro) / _np.maximum(poro, 1.0e-12))
               * (ads - ads0)) * invdt + divP)
    diag_cp = _np.asarray(resP.jac[:, 3 * nc:4 * nc].diagonal()).ravel()
    eps_tol = _np.sqrt(1.0e-8) * _np.mean(_np.abs(diag_cp))
    if eps_tol == 0.0:
        eps_tol = 1.0e-8
    bad = _np.abs(diag_cp) < eps_tol
    if _np.any(bad):
        resP = _ad_select(bad, cp, resP)
    resP = resP - _polymer_well_source(wells, perf_component_all, cp, rhoWS, nvar, nc)

    rhoW0 = rhoWS * bW0
    rho_scale = _np.asarray([rhoWS / _np.mean(rhoW0), rhoOS / _np.mean(rhoO0), rhoGS / _np.mean(rhoG_phase0)])
    fW = (qws - surface_w) * rho_scale[0]
    fO = (qos - surface_o) * rho_scale[1]
    fG = (qgs - surface_g) * rho_scale[2]
    closure = _FacilityModel.compute_control_equations(
        wells, qs_phases={'w': qws, 'o': qos, 'g': qgs}, bhp=bhp, phase_order=['w', 'o', 'g'],
    )
    closure_adi = _SparseADI.concat(closure) if closure else _SparseADI.constant(_np.zeros(0), nvar)
    residual = _SparseADI.concat((resW, resO, resG, resP, fW, fO, fG, closure_adi))
    return residual, {'status': status, 'pvt': pvt}


def _polymer_well_source(wells, perf_component_all, cp, rhoWS, nvar, nc):
    """See ``equationsOilWaterPolymer._polymer_well_source`` (identical
    logic; duplicated rather than shared to keep each equations module a
    faithful, self-contained 1:1 counterpart of its ``.m`` source)."""
    if not wells:
        return _SparseADI.constant(_np.zeros(nc), nvar)
    src = _np.zeros(nc)
    cp_val = cp.val if isinstance(cp, _SparseADI) else _np.asarray(cp)
    for iw, w in enumerate(wells):
        cells = _np.atleast_1d(w.get('cells', [])).astype(int)
        n = cells.size
        if n == 0:
            continue
        perf_comp = perf_component_all[iw] if iw < len(perf_component_all) else _np.zeros((n, 3))
        cqWs = perf_comp[:, 0] / max(rhoWS, 1.0e-30) if perf_comp.size else _np.zeros(n)
        isInj = cqWs > 0
        conc_ctrl = float(w.get('polymer', 0.0))
        conc = _np.where(isInj, conc_ctrl, cp_val[cells])
        _np.add.at(src, cells, conc * cqWs)
    return _SparseADI.constant(src, nvar)
