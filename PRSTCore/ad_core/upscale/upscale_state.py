"""Create an upscaled state by simple processing of values.

1:1 Python translation of MRST autodiff/ad-core/upscale/upscaleState.m
"""

import numpy as np


def upscale_state(coarse_model, model, state):
    """Convert a fine state to a coarse state.

    Parameters
    ----------
    coarse_model : dict
        Coarse model with G.partition.
    model : dict
        Fine model with operators.pv.
    state : dict
        Fine state with pressure, s, optional rs, rv, flux, T, components.

    Returns
    -------
    dict
        Coarse state.
    """
    CG = _model_get(coarse_model, "G")
    p = np.asarray(CG["partition"])
    ncoarse = CG["cells"]["num"]

    ops = _model_get(model, "operators", {}) or {}
    pv_fine = np.asarray(ops["pv"]).ravel()
    pv_coarse = np.bincount(p, weights=pv_fine, minlength=ncoarse + 1)[1:]
    pv_coarse = np.maximum(pv_coarse, 1e-12)

    counts = np.bincount(p, minlength=ncoarse + 1)[1:]
    state_f = state
    state_c = dict(state)

    # Saturations (PV-weighted)
    if "s" in state:
        s_fine = np.asarray(state["s"])
        if s_fine.ndim == 1:
            s_fine = s_fine.reshape(-1, 1)
        nph = s_fine.shape[1]
        pvs = s_fine * pv_fine.reshape(-1, 1)
        s_coarse = np.zeros((ncoarse, nph))
        for i in range(nph):
            s_coarse[:, i] = np.bincount(p, weights=pvs[:, i],
                                          minlength=ncoarse + 1)[1:] / pv_coarse
        if nph > 1:
            # The last phase is set by closure rather than averaged. The
            # pore-volume averages need not sum to exactly one -- rounding
            # alone will see to that -- and a coarse state whose
            # saturations do not close is not a valid state.
            s_coarse[:, -1] = 1.0 - np.sum(s_coarse[:, :-1], axis=1)
        state_c["s"] = s_coarse

    # Concentrations travel with the fluid, so they are pore-volume
    # weighted like the saturations.
    for name, flag in (("cs", "surfactant"), ("cp", "polymer")):
        if name in state and _has_prop(model, flag):
            values = np.asarray(state[name], dtype=float).ravel()
            state_c[name] = np.bincount(p, weights=values * pv_fine,
                                        minlength=ncoarse + 1)[1:] / pv_coarse

    # Compositional (if present)
    if "components" in state:
        _upscale_components(state_c, state_f, model, coarse_model, p, pv_fine, pv_coarse, ncoarse)

    # Dissolved gas-oil ratio
    if "rs" in state and _has_prop(model, "disgas"):
        sO_fine = _get_prop(model, state_f, "sO")
        sO_coarse = _get_prop(coarse_model, state_c, "sO")
        rs_coarse = np.bincount(p, weights=sO_fine * state["rs"] * pv_fine,
                                 minlength=ncoarse + 1)[1:] / np.maximum(sO_coarse * pv_coarse, 1e-12)
        rs_coarse[~np.isfinite(rs_coarse)] = 0
        state_c["rs"] = rs_coarse

    # Vaporized oil ratio
    if "rv" in state and _has_prop(model, "vapoil"):
        sG_fine = _get_prop(model, state_f, "sG")
        sG_coarse = _get_prop(coarse_model, state_c, "sG")
        rv_coarse = np.bincount(p, weights=sG_fine * state["rv"] * pv_fine,
                                 minlength=ncoarse + 1)[1:] / np.maximum(sG_coarse * pv_coarse, 1e-12)
        rv_coarse[~np.isfinite(rv_coarse)] = 0
        state_c["rv"] = rv_coarse

    # Pressure (PV-weighted average)
    if "pressure" in state:
        state_c["pressure"] = np.bincount(p, weights=pv_fine * state["pressure"],
                                           minlength=ncoarse + 1)[1:] / pv_coarse

    # Temperature
    if "T" in state:
        state_c["T"] = np.bincount(p, weights=state["T"],
                                    minlength=ncoarse + 1)[1:] / np.maximum(counts, 1)

    # Flux
    if "flux" in state and "connPos" in CG["faces"] and "fconn" in CG["faces"]:
        cfacesno = np.repeat(np.arange(CG["faces"]["num"]),
                             np.diff(CG["faces"]["connPos"]))
        # sign: +1 if first neighbor matches, -1 otherwise
        N = CG["faces"]["neighbors"]
        cfsign = np.ones(len(cfacesno))
        # For each coarse face connection, determine sign
        for fc in range(len(cfacesno)):
            cf = cfacesno[fc]
            if N[cf, 0] != 0:
                fine_face = CG["faces"]["fconn"][fc]
                fine_nbrs = CG["parent"]["faces"]["neighbors"][fine_face]
                if fine_nbrs[0] > 0:
                    fb = CG["partition"][fine_nbrs[0] - 1]
                    if fb == cf + 1:
                        cfsign[fc] = 1
                    else:
                        cfsign[fc] = -1

        nph_flux = 1 if np.ndim(state["flux"]) == 1 else state["flux"].shape[1]
        flux_coarse = np.zeros((CG["faces"]["num"], max(nph_flux, 1)))
        for i in range(flux_coarse.shape[1]):
            col = state["flux"][CG["faces"]["fconn"], i] if state["flux"].ndim > 1 else state["flux"][CG["faces"]["fconn"]]
            flux_coarse[:, i] = np.bincount(cfacesno, weights=col * cfsign,
                                             minlength=CG["faces"]["num"] + 1)[:CG["faces"]["num"]]
        if state["flux"].ndim == 1:
            flux_coarse = flux_coarse.ravel()
        state_c["flux"] = flux_coarse

    # Remove FlowProps
    if "FlowProps" in state_c:
        del state_c["FlowProps"]

    return state_c


def _upscale_components(state_c, state_f, model, coarse_model, p, pv_fine, pv_coarse, ncoarse):
    """Upscale compositional variables."""
    # Placeholder for full compositional upscaling
    ncomp = state_f["components"].shape[1]
    N_c = np.zeros((ncoarse, ncomp))
    # Simplified: use component mass * pv
    for i in range(ncomp):
        N_c[:, i] = np.bincount(p, weights=state_f["components"][:, i] * pv_fine,
                                 minlength=ncoarse + 1)[1:] / pv_coarse
    state_c["components"] = N_c / np.maximum(N_c.sum(axis=1, keepdims=True), 1e-12)
    for f in ["L", "x", "y", "K", "Z_L", "Z_V", "mixing", "flag", "eos"]:
        if f in state_c:
            del state_c[f]


def _has_prop(model, prop):
    """Check if model has a property."""
    return model.get(prop, False) if isinstance(model, dict) else getattr(model, prop, False)


def _get_prop(model, state, prop):
    """Get phase property from state."""
    if "s" not in state:
        return np.ones(1)
    s = np.asarray(state["s"])
    if s.ndim == 1:
        s = s.reshape(-1, 1)
    if prop == "sO":
        return s[:, 1] if s.shape[1] > 1 else 1 - s[:, 0]
    if prop == "sG":
        return s[:, 2] if s.shape[1] > 2 else np.zeros(s.shape[0])
    if prop == "sW":
        return s[:, 0]
    return np.ones(s.shape[0])


def _model_get(model, name, default=None):
    if isinstance(model, dict):
        return model.get(name, default)
    return getattr(model, name, default)
