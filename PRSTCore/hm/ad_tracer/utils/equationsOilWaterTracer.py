"""Port of MRST ``equationsOilWaterTracer.m`` (mrst-2026a/hm/ad-tracer/utils).

Two-phase oil/water flow with any number of passive water-borne tracers.
Each tracer adds one conservation equation

    (1/dt) * (pv*bW*sW*t - pv0*bW0*sW0*t0) + Div(faceUpstr(upcw, bW) * vT)

with ``vT = faceUpstr(upcw, t) * vW`` -- the tracer rides the water phase
and does not feed back into it, which is why
``ThreePhaseBlackOilTracerModel`` sets ``stepFunctionIsLinear = true``.

Structural conventions follow ``ad_eor.utils.equationsOilWaterPolymer``
(single concatenated ``SparseADI`` Jacobian, ``c1``/``c2`` neighbour arrays
rather than MRST's generic ``operators.Grad``/``Div``), so the two read the
same way.
"""

import numpy as _np
import scipy.sparse as _sp

from PRSTCore.ad_core.adi import SparseADI as _SparseADI
from PRSTCore.ad_core.adi import ad_maximum as _ad_maximum
from PRSTCore.ad_core.adi import ad_select as _ad_select
from PRSTCore.ad_core.models.facility_model import FacilityModel as _FacilityModel


def equationsOilWaterTracer(model, state0, state, dt, drivingForces, wells):
    """Assemble water/oil/tracer residuals.

    Returns ``(residual, aux)`` matching the contract
    ``GenericBlackOilModel._mrst_generic_adi_residual_ow`` establishes.
    """
    nc = model._num_cells()
    nw = len(wells)
    nt = model.getNumberOfTracers()
    # Primary variables: pressure, sW, tracer_1..tracer_nt, then the three
    # facility unknowns -- MRST's
    # primaryVars = [{'pressure'}, {'sW'}, tracerVarNames, wellVarNames].
    nvar = (2 + nt) * nc + 3 * nw

    p = _SparseADI.variable(state['pressure'], nvar, 0)
    sw = _SparseADI.variable(state['sW'], nvar, nc)
    tracers = [_SparseADI.variable(_tracer(state, i), nvar, (2 + i) * nc)
               for i in range(nt)]
    tracers0 = [_np.asarray(_tracer(state0, i), dtype=float).ravel() for i in range(nt)]
    zero = _SparseADI.constant(_np.zeros(nc), nvar)

    pW, pO, _ = model._phase_pressures_adi(p, sw, zero)
    pvt = model._phase_pvt_from_phase_pressures_adi(
        pW, pO, pO, rs_override=zero, rv_override=zero, sG_override=zero)

    p0 = _np.asarray(state0['pressure'], dtype=float).ravel()
    sw0 = _np.asarray(state0['sW'], dtype=float).ravel()
    pW0, pO0, _ = model._phase_pressures(p0, sw0, _np.zeros(nc))
    pvt0 = model._phase_pvt_from_phase_pressures(
        pW0, pO0, pO0, rs_override=_np.zeros(nc), rv_override=_np.zeros(nc),
        sG_override=_np.zeros(nc))

    bW, bO = pvt['bw'], pvt['bo']
    bW0, bO0 = pvt0['bw'], pvt0['bo']
    rhoWS, rhoOS, _ = model._mrst_surface_densities()
    rhoW, rhoO = bW * rhoWS, bO * rhoOS
    rhoW0, rhoO0 = rhoWS * bW0, rhoOS * bO0

    krW, krO, _ = model._relative_perm_adi(sw, zero)
    lamW = krW / _ad_maximum(pvt['muw'], 1.0e-30)
    lamO = krO / _ad_maximum(pvt['muo'], 1.0e-30)

    c1, c2, T = model._internal_connections()
    nface = c1.size
    centroids = _np.asarray(model.G.get('cells', {}).get('centroids', _np.zeros((nc, 3))),
                            dtype=float)
    z = centroids[:, 2] if centroids.ndim == 2 and centroids.shape[1] >= 3 else _np.zeros(nc)
    grav = _np.asarray(getattr(model, 'gravity', [0.0, 0.0, 9.80665]), dtype=float).ravel()
    g = float(grav[-1]) if grav.size else 9.80665
    gdz = g * (z[c2] - z[c1]) if nface else _np.zeros(0)
    C = (_sp.csr_matrix((_np.r_[_np.ones(nface), -_np.ones(nface)],
                         (_np.r_[c1, c2], _np.r_[_np.arange(nface), _np.arange(nface)])),
                        shape=(nc, nface)) if nface else _sp.csr_matrix((nc, 0)))

    def phase_flux(phase_pressure, lam, rho, component_density):
        """MRST PhaseFlux + PhaseUpwindFlag: v = -T*mob_face*dp, flag = dp<=0."""
        if not nface:
            zero_c = _SparseADI.constant(_np.zeros(nc), nvar)
            return zero_c, _SparseADI.constant(_np.zeros(0), nvar), _np.zeros(0, dtype=int)
        potential = phase_pressure[c2] - phase_pressure[c1] - (rho[c1] + rho[c2]) * (0.5 * gdz)
        upstream = _np.where(potential.val <= 0.0, c1, c2)
        q = potential * (-T) * lam[upstream]
        flux = q * component_density[upstream]
        return flux.linear_map(C), q, upstream

    divW, qWface, upW = phase_flux(pW, lamW, rhoW, bW * rhoWS)
    divO, _, _ = phase_flux(pO, lamO, rhoO, bO * rhoOS)

    pv = model._mrst_pore_volume_adi(p)
    pv0 = model._mrst_pore_volume(p0)
    invdt = 1.0 / max(float(dt), 1.0e-30)

    # water/oil rows are in mass units, matching
    # _mrst_generic_adi_residual_ow; the tracer rows keep MRST's native
    # (unscaled) surface-volume units.
    resW = (pv * sw * bW * rhoWS - pv0 * sw0 * bW0 * rhoWS) * invdt + divW
    resO = (pv * (1.0 - sw) * bO * rhoOS - pv0 * (1.0 - sw0) * bO0 * rhoOS) * invdt + divO

    res_t = []
    for i in range(nt):
        t, t0 = tracers[i], tracers0[i]
        if nface:
            # vT = faceUpstr(upcw, t).*vW ; bWvT = faceUpstr(upcw, bW).*vT
            vT = t[upW] * qWface
            bWvT = bW[upW] * vT
            divT = bWvT.linear_map(C)
        else:
            divT = _SparseADI.constant(_np.zeros(nc), nvar)
        acc = (pv * bW * sw * t - pv0 * bW0 * sw0 * t0) * invdt
        res_t.append(acc + divT)

    # MRST: eqs_t{i}(bad) = tracers{i}(bad) -- where the tracer equation's
    # own Jacobian diagonal is ill-conditioned (a cell with essentially no
    # water), replace the row wholesale by the tracer primary variable,
    # which linearizes to t_new = 0 rather than letting a near-singular row
    # produce a nonsense update.
    for i in range(nt):
        col = (2 + i) * nc
        diag = _np.asarray(res_t[i].jac[:, col:col + nc].diagonal()).ravel()
        eps_tol = _np.sqrt(1.0e-8) * _np.mean(_np.abs(diag))
        if eps_tol == 0.0:
            eps_tol = 1.0e-8
        bad = _np.abs(diag) < eps_tol
        if _np.any(bad):
            res_t[i] = _ad_select(bad, tracers[i], res_t[i])

    qws = _SparseADI.variable(state.get('facility_qWs', _np.zeros(nw)), nvar, (2 + nt) * nc)
    qos = _SparseADI.variable(state.get('facility_qOs', _np.zeros(nw)), nvar, (2 + nt) * nc + nw)
    bhp = _SparseADI.variable(state.get('facility_bhp', _np.zeros(nw)), nvar, (2 + nt) * nc + 2 * nw)

    def component_mass_fn(c, qph):
        qW, qO = qph
        return [qW * rhoWS * bW[c], qO * rhoOS * bO[c]]

    (srcW, srcO), (surface_w, surface_o), perf_phase_all, perf_component_all = \
        model.FacilityModel.compute_well_contributions(
            wells=wells, state=state, p=p, bhp=bhp,
            lam_phases=[lamW, lamO], rhoS_phases=[rhoWS, rhoOS],
            component_mass_fn=component_mass_fn, nc=nc, nw=nw, nvar=nvar,
            n_component_phases=2)

    state['facility_perforation_phase_flux'] = perf_phase_all
    state['facility_perforation_component_flux'] = perf_component_all

    resW = resW - srcW
    resO = resO - srcO
    for i in range(nt):
        res_t[i] = res_t[i] - _tracer_well_source(
            wells, perf_component_all, tracers[i], i, rhoWS, nvar, nc)

    rho_scale = _np.asarray([rhoWS / _np.mean(rhoW0), rhoOS / _np.mean(rhoO0)])
    fW = (qws - surface_w) * rho_scale[0]
    fO = (qos - surface_o) * rho_scale[1]
    closure = _FacilityModel.compute_control_equations(
        wells, qs_phases={'w': qws, 'o': qos}, bhp=bhp, phase_order=['w', 'o'])
    closure_adi = (_SparseADI.concat(closure) if closure
                   else _SparseADI.constant(_np.zeros(0), nvar))

    residual = _SparseADI.concat(tuple([resW, resO] + res_t + [fW, fO, closure_adi]))
    return residual, {'pvt': pvt, 'rho0': (rhoW0, rhoO0)}


def _tracer(state, index):
    """Read tracer ``index`` from ``state['tracer']`` (list or ``nc x nt``)."""
    tr = state['tracer']
    if isinstance(tr, (list, tuple)):
        return _np.asarray(tr[index], dtype=float).ravel()
    tr = _np.asarray(tr, dtype=float)
    return tr[:, index].ravel() if tr.ndim == 2 else tr.ravel()


def _tracer_well_source(wells, perf_component_all, tracer, index, rhoWS, nvar, nc):
    """Port of ``addComponentContributions``' tracer branch.

        qW = phaseMass{1}/rhoWS;  isInj = qW > 0
        qC = (isInj.*c + ~isInj.*component(cells)).*qW

    i.e. an injector delivers the well's declared tracer concentration and a
    producer withdraws the cell's own.
    """
    src = _SparseADI.constant(_np.zeros(nc), nvar)
    for iw, w in enumerate(wells):
        if iw >= len(perf_component_all):
            continue
        comp = _np.asarray(perf_component_all[iw], dtype=float)
        if comp.size == 0:
            continue
        cells = _np.atleast_1d(_np.asarray(w.get('cells', []), dtype=int)).ravel()
        qW = comp[:, 0] / rhoWS
        n = min(cells.size, qW.size)
        if n == 0:
            continue
        cells, qW = cells[:n], qW[:n]
        conc = _well_tracer_concentration(w, index)
        injecting = qW > 0.0
        for k in range(n):
            c = int(cells[k])
            if injecting[k]:
                src = src + _SparseADI.scatter(
                    [c], _SparseADI.constant(_np.array([conc * qW[k]]), nvar), nc)
            else:
                src = src + _SparseADI.scatter([c], tracer[c] * float(qW[k]), nc)
    return src


def _well_tracer_concentration(w, index):
    """Injected tracer concentration for tracer ``index`` of well ``w``."""
    conc = w.get('tracer')
    if conc is None:
        return 0.0
    conc = _np.atleast_1d(_np.asarray(conc, dtype=float)).ravel()
    if conc.size == 0:
        return 0.0
    return float(conc[index]) if index < conc.size else float(conc[0])
