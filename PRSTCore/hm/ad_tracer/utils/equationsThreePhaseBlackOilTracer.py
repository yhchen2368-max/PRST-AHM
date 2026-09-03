"""Port of MRST ``equationsThreePhaseBlackOilTracer.m``
(mrst-2026a/hm/ad-tracer/utils).

Three-phase black oil (with disgas/vapoil) plus any number of passive
water-borne tracers.  The reservoir rows are the same component-mass
balances ``GenericBlackOilModel._mrst_generic_adi_residual`` assembles; each
tracer adds

    (1/dt)*(pv*bW*sW*t - pv0*bW0*sW0*t0) + Div(faceUpstr(upcw, bW)*vT)

with ``vT = faceUpstr(upcw, t)*vW``.
"""

import numpy as _np
import scipy.sparse as _sp

from PRSTCore.ad_core.adi import SparseADI as _SparseADI
from PRSTCore.ad_core.adi import ad_maximum as _ad_maximum
from PRSTCore.ad_core.adi import ad_select as _ad_select
from PRSTCore.ad_core.models.facility_model import FacilityModel as _FacilityModel

from .equationsOilWaterTracer import _tracer, _tracer_well_source


def equationsThreePhaseBlackOilTracer(model, state0, state, dt, drivingForces, wells):
    if not (model.water and model.oil and model.gas):
        raise NotImplementedError(
            'equationsThreePhaseBlackOilTracer requires all three active phases')
    nc = model._num_cells()
    nw = len(wells)
    nt = model.getNumberOfTracers()
    status = model._mrst_blackoil_status(state)
    st1, st2, st3 = [_np.asarray(x, dtype=bool).ravel() for x in status]
    nvar = (3 + nt) * nc + 4 * nw

    p = _SparseADI.variable(state['pressure'], nvar, 0)
    sw = _SparseADI.variable(state['sW'], nvar, nc)
    x = _SparseADI.variable(
        model._mrst_pack_primary(state, drivingForces)[0][2 * nc:3 * nc], nvar, 2 * nc)
    tracers = [_SparseADI.variable(_tracer(state, i), nvar, (3 + i) * nc)
               for i in range(nt)]
    tracers0 = [_np.asarray(_tracer(state0, i), dtype=float).ravel() for i in range(nt)]

    sg = (1.0 - sw) * st2 + x * st3
    pW, pO, pG = model._phase_pressures_adi(p, sw, sg, state.get('pcowScale'))
    pvt_sat = model._phase_pvt_adi(pO)
    rs = ((pvt_sat['rs'] * (~st1)) + x * st1 if model.disgas
          else _SparseADI.constant(state.get('rs', _np.zeros(nc)), nvar))
    if model.vapoil:
        deck_pvt = getattr(model, '_blackoil_pvt', None)
        if deck_pvt is None or not hasattr(deck_pvt, 'rv_sat_adi'):
            raise NotImplementedError(
                'Deck AD VAPOIL assembly requires DeckBlackOilPVT.rv_sat_adi')
        rv = deck_pvt.rv_sat_adi(pG) * (~st2) + x * st2
    else:
        rv = _SparseADI.constant(state.get('rv', _np.zeros(nc)), nvar)
    so = 1.0 - sw - sg

    pvt = model._phase_pvt_from_phase_pressures_adi(
        pW, pO, pG, rs_override=rs, rv_override=rv, sG_override=sg,
        oil_saturated_override=(sg.val > 0.0),
        gas_saturated_override=(so.val > 0.0))
    bW, bO, bG = pvt['bw'], pvt['bo'], pvt['bg']
    krW, krO, krG = model._relative_perm_adi(sw, sg)
    lamW = krW / _ad_maximum(pvt['muw'], 1.0e-30)
    lamO = krO / _ad_maximum(pvt['muo'], 1.0e-30)
    lamG = krG / _ad_maximum(pvt['mug'], 1.0e-30)

    rhoWS, rhoOS, rhoGS = model._mrst_surface_densities()
    rhoW = bW * rhoWS
    rhoO = bO * (rhoOS + rs * rhoGS)
    rhoG = bG * (rhoGS + rv * rhoOS)
    rhoG_component = bG * rhoGS

    p0 = _np.asarray(state0['pressure'], dtype=float).ravel()
    sw0 = _np.asarray(state0['sW'], dtype=float).ravel()
    sg0 = _np.asarray(state0['sG'], dtype=float).ravel()
    so0 = 1.0 - sw0 - sg0
    rs0 = _np.asarray(state0['rs'], dtype=float).ravel()
    rv0 = _np.asarray(state0.get('rv', _np.zeros(nc)), dtype=float).ravel()
    pW0, pO0, pG0 = model._phase_pressures(p0, sw0, sg0, state0.get('pcowScale'))
    pvt0 = model._phase_pvt_from_phase_pressures(
        pW0, pO0, pG0, rs_override=rs0, rv_override=rv0, sG_override=sg0,
        oil_saturated_override=(sg0 > 0.0), gas_saturated_override=(so0 > 0.0))
    bW0 = pvt0['bw']
    rhoW0 = rhoWS * bW0
    rhoO0 = pvt0['bo'] * (rhoOS + rs0 * rhoGS)
    rhoG_phase0 = pvt0['bg'] * (rhoGS + rv0 * rhoOS)
    rhoG_component0 = pvt0['bg'] * rhoGS

    c1, c2, T = model._internal_connections()
    nface = c1.size
    centroids = _np.asarray(model.G.get('cells', {}).get('centroids', _np.zeros((nc, 3))),
                            dtype=float)
    z = centroids[:, 2] if centroids.ndim == 2 and centroids.shape[1] >= 3 else _np.zeros(nc)
    grav = _np.asarray(getattr(model, 'gravity', [0.0, 0.0, 9.80665]), dtype=float).ravel()
    g = float(grav[-1]) if grav.size else 9.80665
    C = (_sp.csr_matrix((_np.r_[_np.ones(nface), -_np.ones(nface)],
                         (_np.r_[c1, c2], _np.r_[_np.arange(nface), _np.arange(nface)])),
                        shape=(nc, nface)) if nface else _sp.csr_matrix((nc, 0)))

    def phase_flux(phase_pressure, lam, rho, component_density):
        if not nface:
            zero_c = _SparseADI.constant(_np.zeros(nc), nvar)
            return zero_c, _SparseADI.constant(_np.zeros(0), nvar), _np.zeros(0, dtype=int)
        potential = (phase_pressure[c2] - phase_pressure[c1]
                     - (rho[c1] + rho[c2]) * (0.5 * g * (z[c2] - z[c1])))
        upstream = _np.where(potential.val <= 0.0, c1, c2)
        q = potential * (-T) * lam[upstream]
        flux = q * component_density[upstream]
        return flux.linear_map(C), q, upstream

    divW, qWface, upW = phase_flux(pW, lamW, rhoW, bW * rhoWS)
    divO, qOface, upO = phase_flux(pO, lamO, rhoO, bO * rhoOS)
    divG, qGface, upG = phase_flux(pG, lamG, rhoG, bG * rhoGS)
    if nface:
        divG = divG + (qOface * (rs[upO] * rhoGS * bO[upO])).linear_map(C)
        if model.vapoil:
            divO = divO + (qGface * (rv[upG] * rhoOS * bG[upG])).linear_map(C)

    pv = model._mrst_pore_volume_adi(p)
    pv0 = model._mrst_pore_volume(p0)
    invdt = 1.0 / max(float(dt), 1.0e-30)

    resW = (pv * sw * rhoW - pv0 * sw0 * rhoW0) * invdt + divW
    if model.vapoil:
        resO = (pv * (so * bO + rv * bG * sg) * rhoOS
                - pv0 * (so0 * pvt0['bo'] + rv0 * pvt0['bg'] * sg0) * rhoOS) * invdt + divO
    else:
        resO = (pv * so * rhoOS * bO - pv0 * so0 * rhoOS * pvt0['bo']) * invdt + divO
    resG = (pv * (sg * rhoG_component + so * rs * rhoGS * bO)
            - pv0 * (sg0 * rhoG_component0 + so0 * rs0 * rhoGS * pvt0['bo'])) * invdt + divG

    res_t = []
    for i in range(nt):
        t, t0 = tracers[i], tracers0[i]
        if nface:
            bWvT = bW[upW] * (t[upW] * qWface)
            divT = bWvT.linear_map(C)
        else:
            divT = _SparseADI.constant(_np.zeros(nc), nvar)
        res_t.append((pv * bW * sw * t - pv0 * bW0 * sw0 * t0) * invdt + divT)

    for i in range(nt):
        col = (3 + i) * nc
        diag = _np.asarray(res_t[i].jac[:, col:col + nc].diagonal()).ravel()
        eps_tol = _np.sqrt(1.0e-8) * _np.mean(_np.abs(diag))
        if eps_tol == 0.0:
            eps_tol = 1.0e-8
        bad = _np.abs(diag) < eps_tol
        if _np.any(bad):
            res_t[i] = _ad_select(bad, tracers[i], res_t[i])

    base = (3 + nt) * nc
    qws = _SparseADI.variable(state.get('facility_qWs', _np.zeros(nw)), nvar, base)
    qos = _SparseADI.variable(state.get('facility_qOs', _np.zeros(nw)), nvar, base + nw)
    qgs = _SparseADI.variable(state.get('facility_qGs', _np.zeros(nw)), nvar, base + 2 * nw)
    bhp = _SparseADI.variable(state.get('facility_bhp', _np.zeros(nw)), nvar, base + 3 * nw)

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
            lam_phases=[lamW, lamO, lamG], rhoS_phases=[rhoWS, rhoOS, rhoGS],
            component_mass_fn=component_mass_fn, nc=nc, nw=nw, nvar=nvar,
            n_component_phases=3)

    state['facility_perforation_phase_flux'] = perf_phase_all
    state['facility_perforation_component_flux'] = perf_component_all

    resW = resW - srcW
    resO = resO - srcO
    resG = resG - srcG
    for i in range(nt):
        res_t[i] = res_t[i] - _tracer_well_source(
            wells, perf_component_all, tracers[i], i, rhoWS, nvar, nc)

    rho_scale = _np.asarray([rhoWS / _np.mean(rhoW0), rhoOS / _np.mean(rhoO0),
                             rhoGS / _np.mean(rhoG_phase0)])
    fW = (qws - surface_w) * rho_scale[0]
    fO = (qos - surface_o) * rho_scale[1]
    fG = (qgs - surface_g) * rho_scale[2]
    closure = _FacilityModel.compute_control_equations(
        wells, qs_phases={'w': qws, 'o': qos, 'g': qgs}, bhp=bhp,
        phase_order=['w', 'o', 'g'])
    closure_adi = (_SparseADI.concat(closure) if closure
                   else _SparseADI.constant(_np.zeros(0), nvar))

    residual = _SparseADI.concat(
        tuple([resW, resO, resG] + res_t + [fW, fO, fG, closure_adi]))
    return residual, {'status': status, 'pvt': pvt}
