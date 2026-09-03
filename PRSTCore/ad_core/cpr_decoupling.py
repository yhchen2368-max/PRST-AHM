"""How a black-oil system's component balances are combined into a pressure equation.

Constrained Pressure Residual preconditions the pressure part of the system
with multigrid, and that only works if there *is* a pressure part.  A
deck-derived Jacobian has none to begin with: its equations are component
mass balances, one per phase per cell, and its leading rows are the water
balance differentiated with respect to pressure -- not an elliptic operator.
Handing that to multigrid converges nowhere; measured on SPE9, two hundred
iterations at a relative residual of 0.8.

The fix, and MRST's, is to replace each cell's first balance by a weighted
sum of all of them, chosen so the sum depends on pressure and (as nearly as
possible) not on saturation.  Which weights to use is the decoupling
strategy, and MRST offers three:

``'trueIMPES'``
    The rigorous weights, from ``BlackOilPressureReductionFactors``: the
    inverse phase densities corrected for dissolved gas and vaporised oil,
    divided by pore volume.  MRST's default.  Needs the model's fluid
    state, not just the matrix.

``'quasiIMPES'``
    Solve ``D^T w = e_pressure`` on each cell's own diagonal block, so the
    combined equation has unit pressure coefficient and zero saturation
    coefficients *for that cell*.  Needs only the matrix.

``'none'``
    Unit weights -- the plain sum of the balances.

plus a fourth that MRST reaches through the same switch:

``'simple'``
    ``1/b_phase`` evaluated once at the mean pressure.  The analytical
    approximation MRST falls back to for models that cannot produce
    pressure reduction factors.

Orthogonally, ``'mrst_drs'`` (Gries et al, SPE-163608-PA) drops from each
cell's sum the balances that are not locally elliptic, judged by diagonal
dominance and by how strongly they couple to pressure.

Everything here works in the *grouped* ordering deck-derived systems use --
every cell's water balance, then every cell's oil balance -- not the
interleaved one.  Getting that backwards does not raise; it builds a
pressure block out of the wrong entries and the solve quietly stops
converging.
"""

from __future__ import annotations

import numpy as _np

try:
    import scipy.sparse as _sp
except Exception:  # pragma: no cover - scipy is a hard dependency elsewhere
    _sp = None

#: The strategies :func:`decoupling_weights` accepts.
STRATEGIES = ('trueimpes', 'quasiimpes', 'simple', 'none')


def _normalise(name):
    return str(name).strip().lower().replace('-', '').replace('_', '')


def decoupling_weights(strategy, A, nc, ncomp, model=None, state=None):
    """Per-cell weights ``(nc, ncomp)`` for combining the component balances.

    ``A`` is the assembled Jacobian in grouped ordering.  ``model`` and
    ``state`` are only needed by the fluid-based strategies; asking for one
    of those without them is an error rather than a silent downgrade,
    because the downgrade would be a different preconditioner wearing the
    requested name.
    """
    key = _normalise(strategy)
    if key not in STRATEGIES:
        raise ValueError('unknown CPR decoupling %r; expected one of %s'
                         % (strategy, ', '.join(STRATEGIES)))

    if key == 'none':
        return _np.ones((nc, ncomp), dtype=float)
    if key == 'quasiimpes':
        return quasi_impes_weights(A, nc, ncomp)
    if model is None:
        raise ValueError(
            'CPR decoupling %r needs the model to evaluate fluid properties; '
            'pass model=, or use quasiIMPES which works from the matrix alone'
            % strategy)
    if key == 'trueimpes':
        return true_impes_weights(model, state, nc, ncomp)
    return simple_weights(model, state, nc, ncomp)


# ----------------------------------------------------------------- matrix --
def cell_diagonal_blocks(A, nc, ncomp):
    """``D[c]`` -- how cell ``c``'s balances vary with cell ``c``'s unknowns.

    ``D[c, i, j]`` is d(balance ``i`` in cell ``c``)/d(variable ``j`` in cell
    ``c``), read out of the grouped ordering by taking the diagonal of each
    ``nc`` by ``nc`` sub-block.
    """
    csr = A.tocsr()
    blocks = _np.zeros((nc, ncomp, ncomp))
    for i in range(ncomp):
        rows = csr[i * nc:(i + 1) * nc, :]
        for j in range(ncomp):
            blocks[:, i, j] = rows[:, j * nc:(j + 1) * nc].diagonal()
    return blocks


def quasi_impes_weights(A, nc, ncomp):
    """Port of ``getScalingInternalCPR``'s ``quasiimpes`` branch.

    Solves ``D_c^T w = e_pressure`` per cell, which makes the combined
    equation's coefficient on that cell's pressure exactly one and on its
    own saturations exactly zero.

    A pseudo-inverse rather than a solve, so a rank-deficient cell needs no
    special case: a deck that declares a phase and holds none of it -- SPE10
    model 2 declares gas and has none -- leaves that balance identically
    zero, and solving through the singular block returned weights of order
    1e14 that made the decoupled system unusable.  Where the block has full
    rank the pseudo-inverse *is* the solve.

    ``rcond`` stays near machine precision on purpose.  These blocks are
    legitimately ill-conditioned -- SPE10's transmissibilities span eleven
    orders of magnitude -- so a comfortable-looking cut-off discards real
    coupling; at 1e-10, SPE10 model 1 stopped converging entirely.
    """
    blocks = cell_diagonal_blocks(A, nc, ncomp)
    transposed = _np.swapaxes(blocks, 1, 2)
    rhs = _np.zeros((nc, ncomp))
    rhs[:, 0] = 1.0                      # pressure is the first variable group
    weights = (_np.linalg.pinv(transposed, rcond=1e-14) @ rhs[..., None])[..., 0]

    bad = ~_np.isfinite(weights).all(axis=1)
    if _np.any(bad):
        weights[bad] = 0.0
        weights[bad, 0] = 1.0
    empty = _np.abs(weights).max(axis=1) <= 0.0
    if _np.any(empty):
        weights[empty, 0] = 1.0
    return weights


# ------------------------------------------------------------------ fluid --
def _phase_properties(model, state, nc):
    """``b``, ``rs``, ``rv`` and the surface densities at the current state."""
    pressure = _np.asarray(state['pressure'], dtype=float).ravel()
    rs = _np.asarray(state.get('rs', _np.zeros(nc)), dtype=float).ravel()
    rv = _np.asarray(state.get('rv', _np.zeros(nc)), dtype=float).ravel()
    pvt = model._phase_pvt(pressure,
                           rs_override=rs if getattr(model, 'disgas', False) else None,
                           rv_override=rv if getattr(model, 'vapoil', False) else None)
    surface = _np.asarray(model._mrst_surface_densities(), dtype=float).ravel()
    return pvt, rs, rv, surface


def _pore_volume(model, state, nc):
    pv = getattr(model, 'porevolume', None)
    if pv is None:
        return _np.ones(nc)
    pv = _np.asarray(pv, dtype=float).ravel()
    return pv if pv.size == nc else _np.ones(nc)


def _active_component_order(model):
    """The component balances, in the order the equations are assembled."""
    order = []
    if getattr(model, 'water', False):
        order.append('water')
    if getattr(model, 'oil', False):
        order.append('oil')
    if getattr(model, 'gas', False):
        order.append('gas')
    return order


def true_impes_weights(model, state, nc, ncomp):
    """Port of ``BlackOilPressureReductionFactors.evaluateOnDomain``.

    ``w_phase = f_phase / pore_volume``, with

    * water: ``1 / (b_w * rho_wS)``
    * oil:   ``(alpha / rho_oS) * (1/b_o - rs/b_g)`` when gas dissolves,
      otherwise ``1 / (b_o * rho_oS)``
    * gas:   ``(alpha / rho_gS) * (1/b_g - rv/b_o)`` when oil vaporises,
      otherwise ``1 / (b_g * rho_gS)``

    and ``alpha = 1/(1 - rs*rv)``, which is one unless both dissolution and
    vaporisation are active.  The pore-volume division is what puts the
    combined equation on the same footing from cell to cell.
    """
    pvt, rs, rv, surface = _phase_properties(model, state, nc)
    pv = _pore_volume(model, state, nc)
    disgas = bool(getattr(model, 'disgas', False))
    vapoil = bool(getattr(model, 'vapoil', False))

    bw = _np.asarray(pvt['bw'], dtype=float).ravel()
    bo = _np.asarray(pvt['bo'], dtype=float).ravel()
    bg = _np.asarray(pvt['bg'], dtype=float).ravel()
    tiny = _np.finfo(float).tiny
    bw = _np.where(_np.abs(bw) > tiny, bw, 1.0)
    bo = _np.where(_np.abs(bo) > tiny, bo, 1.0)
    bg = _np.where(_np.abs(bg) > tiny, bg, 1.0)

    rho_ws, rho_os, rho_gs = (surface.tolist() + [1.0, 1.0, 1.0])[:3]
    alpha = 1.0 / (1.0 - (rs * rv if (disgas and vapoil) else 0.0))

    factors = {}
    factors['water'] = 1.0 / (bw * rho_ws)
    if disgas:
        factors['oil'] = (alpha / rho_os) * (1.0 / bo - rs / bg)
    else:
        factors['oil'] = 1.0 / (bo * rho_os)
    if vapoil:
        factors['gas'] = (alpha / rho_gs) * (1.0 / bg - rv / bo)
    else:
        factors['gas'] = 1.0 / (bg * rho_gs)

    if disgas and not vapoil:
        _apply_undersaturated_correction(model, state, factors, bo, rs,
                                         rho_os, rho_gs, nc)

    order = _active_component_order(model)
    weights = _np.zeros((nc, ncomp), dtype=float)
    for index, name in enumerate(order[:ncomp]):
        weights[:, index] = factors.get(name, 0.0) / pv
    return weights


def _apply_undersaturated_correction(model, state, factors, bo, rs,
                                     rho_os, rho_gs, nc):
    """MRST's ``useUndersaturated`` branch, for cells with no free gas.

    Where the gas saturation is zero the oil's shrinkage factor still varies
    with dissolved gas, and the weights that ignore that make the combined
    equation depend on Rs.  The correction needs one derivative,
    ``d b_o / d rs``, which is taken by seeding Rs as the differentiation
    variable and reading it back -- the same route MRST takes with
    ``initVariablesAD_diagonal``.

    Silently skipped when the model cannot differentiate with respect to Rs;
    the uncorrected weights are still a valid pressure equation, only a
    slightly worse one.
    """
    saturation = _np.asarray(state.get('sG', _np.zeros(nc)), dtype=float).ravel()
    undersaturated = saturation == 0.0
    if not _np.any(undersaturated):
        return

    dbo_drs = _shrinkage_derivative_wrt_rs(model, state, nc)
    if dbo_drs is None:
        return

    subset = undersaturated
    bou = bo[subset]
    rsu = rs[subset]
    dbo = dbo_drs[subset]
    factors['oil'] = _np.array(factors['oil'], dtype=float, copy=True)
    factors['gas'] = _np.array(factors['gas'], dtype=float, copy=True)
    factors['oil'][subset] = (1.0 / (rho_os * bou)) * (1.0 + (rsu / bou) * dbo)
    factors['gas'][subset] = -(1.0 / (rho_gs * bou ** 2)) * dbo


def _shrinkage_derivative_wrt_rs(model, state, nc):
    """``d b_o / d rs`` at the current state, or ``None`` if unavailable."""
    pvt = getattr(model, '_blackoil_pvt', None)
    if pvt is None or not hasattr(pvt, 'eval_adi'):
        return None
    try:
        from PRSTCore.ad_core.adi import SparseADI

        pressure = _np.asarray(state['pressure'], dtype=float).ravel()
        rs = _np.asarray(state.get('rs', _np.zeros(nc)), dtype=float).ravel()
        rv = _np.asarray(state.get('rv', _np.zeros(nc)), dtype=float).ravel()
        # One variable, Rs, so the Jacobian is a single diagonal and the
        # derivative is its diagonal entries.
        rs_ad = SparseADI.variable(rs, nc, 0)
        pressure_ad = SparseADI.constant(pressure, nc)
        out = pvt.eval_adi(pressure_ad, rs_override=rs_ad,
                           rv_override=rv,
                           oil_saturated_override=_np.zeros(nc, dtype=bool))
        bo = out['bo']
        if not hasattr(bo, 'jac'):
            return None
        return _np.asarray(bo.jac.diagonal(), dtype=float).ravel()
    except Exception:
        return None


def simple_weights(model, state, nc, ncomp):
    """Port of ``getScalingFactorsCPR``'s analytical branch.

    ``1/b_phase`` at the mean pressure -- one number per phase for the whole
    grid, which is why MRST calls it the simplified weighting.  Cheap, and
    enough when the fluid does not vary much over the field.
    """
    pressure = _np.asarray(state['pressure'], dtype=float).ravel()
    mean_pressure = _np.full(1, float(_np.mean(pressure)))
    disgas = bool(getattr(model, 'disgas', False))
    vapoil = bool(getattr(model, 'vapoil', False))
    rs = _np.asarray(state.get('rs', _np.zeros(nc)), dtype=float).ravel()
    rv = _np.asarray(state.get('rv', _np.zeros(nc)), dtype=float).ravel()
    pvt = model._phase_pvt(
        mean_pressure,
        rs_override=_np.full(1, float(_np.mean(rs))) if disgas else None,
        rv_override=_np.full(1, float(_np.mean(rv))) if vapoil else None)

    by_name = {'water': 'bw', 'oil': 'bo', 'gas': 'bg'}
    order = _active_component_order(model)
    weights = _np.zeros((nc, ncomp), dtype=float)
    for index, name in enumerate(order[:ncomp]):
        b = float(_np.asarray(pvt[by_name[name]], dtype=float).ravel()[0])
        weights[:, index] = 1.0 / b if abs(b) > _np.finfo(float).tiny else 1.0
    return weights


# -------------------------------------------------------- dynamic row sum --
def apply_dynamic_row_sum(weights, A, nc, ncomp, diagonal_tol=1e-2,
                          coupling_tol=0.0):
    """Port of the ``mrst_drs`` filter (Gries et al, SPE-163608-PA).

    Drops from a cell's sum the balances that are not locally elliptic in
    pressure: the pressure diagonal must dominate that row's other pressure
    entries by ``diagonal_tol``, and must be coupled to the neighbouring
    cells' pressures by at least ``coupling_tol``.  A cell that would lose
    every balance keeps its first, so the combination stays invertible.
    """
    csr = A.tocsr()
    keep = _np.zeros((nc, ncomp), dtype=bool)
    for i in range(ncomp):
        rows = csr[i * nc:(i + 1) * nc, :]
        pressure_block = rows[:, 0:nc].tocsr()
        diagonal = _np.abs(pressure_block.diagonal())
        off = _np.abs(pressure_block).sum(axis=1).A.ravel() - diagonal
        dominant = (diagonal >= diagonal_tol * off if _np.isfinite(diagonal_tol)
                    else _np.ones(nc, dtype=bool))
        coupled = (off >= coupling_tol * diagonal if _np.isfinite(coupling_tol)
                   else _np.ones(nc, dtype=bool))
        keep[:, i] = dominant & coupled

    orphan = ~keep.any(axis=1)
    if _np.any(orphan):
        keep[orphan, 0] = True
    return weights * keep


# ------------------------------------------------------------- assembling --
def decoupling_operator(weights, n, nc, ncomp):
    """The sparse ``M`` with ``M @ A`` decoupled.

    Row ``c`` becomes the weighted sum of cell ``c``'s balances; every other
    row, well rows included, is left alone.  ``M`` is block diagonal with a
    unit tail, so it is invertible whenever a cell keeps a nonzero weight --
    which :func:`quasi_impes_weights` and :func:`apply_dynamic_row_sum` both
    guarantee -- and ``M A x = M b`` therefore has the solution ``A x = b``
    has.  The transformation changes which equations are solved, not which
    unknowns, so nothing needs undoing afterwards.
    """
    if _sp is None:
        raise RuntimeError('scipy is required to build the decoupling operator')
    cells = _np.arange(nc)
    tail = _np.arange(nc, n, dtype=int)
    rows = _np.concatenate([cells] * ncomp + [tail])
    cols = _np.concatenate([cells + i * nc for i in range(ncomp)] + [tail])
    vals = _np.concatenate([weights[:, i] for i in range(ncomp)]
                           + [_np.ones(tail.size)])
    return _sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
