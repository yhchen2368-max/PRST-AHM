"""Python port of MRST's ``AquiferModel`` (mrst-2026a/autodiff/ad-core/models/
aquifers/AquiferModel.m) plus ``computeInitAquifer.m``
(mrst-2026a/autodiff/ad-blackoil/utils): the Fetkovich analytic aquifer
model, coupled to the reservoir's water-phase mass balance through
:func:`PRSTCore.deckformat.params.process_aquifer.process_aquifer`'s
per-connection table.

Coupling scope (matches MRST's own semi-implicit scheme, not a
reformulation): the aquifer flux at each connection is affine in the
*current* reservoir-side water pressure/mobility (so it does carry a
correct Jacobian contribution back into the water equation during a
Newton solve), but each aquifer's own pressure/volume are read as fixed
values from the *last converged* state -- they are updated once per
timestep by :meth:`AquiferModel.update_after_convergence`, not resolved
as coupled AD unknowns within the Newton iteration. This mirrors
``equationsBlackOil.m``/``equationsOilWater.m``'s ``addAquifersContribution``
call and ``ReservoirModel.m``'s ``updateAfterConvergence`` hook. Wiring
this into :mod:`PRSTCore.ad_core.models.generic_black_oil_model` and
:func:`PRSTCore.ad_core.simulators.simulate_schedule_ad.simulate_schedule_ad`
is left to the caller (see the module docstring's "Scope" note) -- this
module provides the validated, self-contained physics.

Fixes a real bug in MRST's own ``computeInitAquifer.m``: its last line
references an undefined variable ``V_aq`` (should be the function's own
``initaqvolumes`` parameter, matching the temp state a few lines above --
almost certainly a copy/paste typo, since MRST's own function would raise
an "undefined variable" error if ever called as written).
"""

from __future__ import annotations

import numpy as _np

from PRSTCore.ad_core.adi import SparseADI as _SparseADI


class AquiferModel:
    """One Fetkovich analytic aquifer system, built from
    :func:`PRSTCore.deckformat.params.process_aquifer.process_aquifer`'s
    output (``aquifers``/``aquind``/``aquiferprops``/``initval``)."""

    def __init__(self, aquifers: _np.ndarray, aquind: dict, aquiferprops: dict, initval: dict):
        self.aquifers = _np.asarray(aquifers, dtype=float)
        self.aquind = aquind
        self.aquiferprops = aquiferprops
        self.initvals = {
            'pressures': _np.asarray(initval['pressures'], dtype=float).copy(),
            'volumes': _np.asarray(initval['volumes'], dtype=float).copy(),
        }
        # Backward-compatible spelling used by the earlier isolated port.
        self.initval = self.initvals
        self.n_aquifers = int(_np.max(self.aquifers[:, aquind["aquid"]]))

    def _aquid2conn(self) -> _np.ndarray:
        """Dense (nconn, naq) 0/1 dispatch matrix: row i has a 1 in column
        (aquid[i] - 1) (1-based aquifer ids from the deck, 0-based here)."""
        aquid = self.aquifers[:, self.aquind["aquid"]].astype(int)
        nconn = aquid.size
        M = _np.zeros((nconn, self.n_aquifers))
        M[_np.arange(nconn), aquid - 1] = 1.0
        return M

    def init_state_aquifer(self) -> dict:
        """Port of ``initStateAquifer``: initial per-aquifer pressure/volume
        from the deck's ``AQUFETP`` (``self.initval``)."""
        return {
            "pressure": self.initvals["pressures"].copy(),
            "volume": self.initvals["volumes"].copy(),
        }

    def compute_aquifer_fluxes(self, *, p_aq, v_aq, pW_conn, bW_conn, rhoWS: float,
                                gravity: float = 9.80665, dt: float = 0.0):
        """Port of ``computeAquiferFluxes``: per-connection volumetric flux
        from the aquifer into the reservoir (positive = into the
        reservoir), one entry per row of ``self.aquifers``.

        Parameters
        ----------
        p_aq, v_aq : (n_aquifers,) array or SparseADI
            Current per-aquifer pressure / volume. Ordinarily numeric --
            these are the *lagged* values from the last converged state
            (see the module docstring) -- but :meth:`compute_init_aquifer`
            passes ``p_aq`` as a SparseADI variable to differentiate the
            flux with respect to it.
        pW_conn, bW_conn : array or SparseADI, (nconn,)
            Reservoir water phase pressure / formation-volume factor at
            each connection cell (``self.aquifers[:, aquind['conn']]``).
            May be ``SparseADI`` to carry the Jacobian back into the water
            equation.
        rhoWS : float
            Water surface density (kg/m^3).
        gravity : float
            Gravitational acceleration (m/s^2), z-component convention
            matching MRST's ``model.gravity(3)``.
        dt : float
            Timestep length (s); ``dt=0`` matches MRST's steady-state flux
            (used by ``compute_init_aquifer``).
        """
        ix = self.aquind
        alpha = self.aquifers[:, ix["alpha"]]
        J = self.aquifers[:, ix["J"]]
        depthconn = self.aquifers[:, ix["depthconn"]]
        depthaq = self.aquifers[:, ix["depthaq"]]
        C = self.aquifers[:, ix["C"]]

        aquid2conn = self._aquid2conn()
        p_aq_conn = p_aq.linear_map(aquid2conn) if isinstance(p_aq, _SparseADI) \
            else aquid2conn @ _np.asarray(p_aq, dtype=float)
        v_aq_conn = v_aq.linear_map(aquid2conn) if isinstance(v_aq, _SparseADI) \
            else aquid2conn @ _np.asarray(v_aq, dtype=float)

        rhoW_conn = bW_conn * rhoWS
        driving = p_aq_conn - pW_conn + rhoW_conn * (gravity * (depthconn - depthaq))

        # Tc/coef only ever depend on v_aq, which compute_init_aquifer
        # (the one caller that needs AD here) always passes as numeric --
        # so this stays plain NumPy even when p_aq is a SparseADI variable.
        Tc = C * _np.asarray(v_aq_conn.val if isinstance(v_aq_conn, _SparseADI) else v_aq_conn) / J
        if dt == 0:
            coef = _np.ones_like(Tc)
        else:
            coef = (1.0 - _np.exp(-dt / Tc)) / (dt / Tc)

        return (alpha * J * coef) * driving

    def update_after_convergence(self, aquifer_sol: dict, q: _np.ndarray, dt: float) -> dict:
        """Port of ``updateAfterConvergence``: advance each aquifer's
        pressure/volume by the (numeric) converged-timestep flux ``q``
        (from :meth:`compute_aquifer_fluxes`, evaluated at the converged
        state). Returns a new ``{'pressure', 'volume'}`` dict."""
        aquid2conn = self._aquid2conn()
        Q = dt * (aquid2conn.T @ _np.asarray(q, dtype=float))
        C = _np.asarray(self.aquiferprops["C"], dtype=float)
        p0 = _np.asarray(aquifer_sol["pressure"], dtype=float)
        vol0 = _np.asarray(aquifer_sol["volume"], dtype=float)
        p = p0 - Q / (C * vol0)
        vol = vol0 - Q
        return {"pressure": p, "volume": vol}

    def add_aquifer_contribution(self, water_eq, q):
        """Port of ``addAquifersContribution``: subtract the (ADI or
        numeric) per-connection flux ``q`` from the water equation
        residual at the connected cells (``water_eq`` is the water-phase
        mass-balance :class:`SparseADI`, sized per reservoir cell)."""
        conn = self.aquifers[:, self.aquind["conn"]].astype(int)
        if isinstance(q, _SparseADI):
            scatter = _SparseADI.scatter(conn, q, water_eq.val.size)
            return water_eq - scatter
        contrib = _np.zeros(water_eq.val.size)
        _np.add.at(contrib, conn, _np.asarray(q, dtype=float))
        return water_eq - contrib

    def compute_init_aquifer(self, *, pW_conn, bW_conn, rhoWS: float, initaqvolumes,
                              gravity: float = 9.80665) -> dict:
        """Port of ``computeInitAquifer.m`` (with its ``V_aq`` typo fixed --
        see the module docstring): solve for the per-aquifer initial
        pressure that minimizes the sum of squared steady-state
        (``dt=0``) connection fluxes, given the reservoir's initial water
        pressure/b-factor at each connection and a supplied initial
        aquifer volume.

        Since ``q(p_aq)`` is affine in ``p_aq`` (``q = M @ p_aq + r``),
        this is a linear least-squares problem solved via the normal
        equations, exactly as MRST's own ADI-Jacobian-based approach:
        ``p_aq = -(M^T M)^{-1} M^T r``.
        """
        p_aq0 = _SparseADI.variable(_np.zeros(self.n_aquifers), self.n_aquifers, 0)
        q = self.compute_aquifer_fluxes(
            p_aq=p_aq0, v_aq=_np.asarray(initaqvolumes, dtype=float),
            pW_conn=pW_conn, bW_conn=bW_conn, rhoWS=rhoWS, gravity=gravity, dt=0.0,
        )
        r = q.val
        M = q.jac.toarray()
        p_aq = -_np.linalg.solve(M.T @ M, M.T @ r)
        return {"pressure": p_aq, "volume": _np.asarray(initaqvolumes, dtype=float).copy()}
