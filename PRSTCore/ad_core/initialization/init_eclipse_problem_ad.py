"""Port of MRST initEclipseProblemAD to PRSTCore (scaffold + partial implementation).

This module implements the high-level setup logic and calls into
deck/grid/rock/fluid/schedule initializers. Several deep helpers are left
as minimal implementations so the packed problem wrapper can operate.
"""

from typing import Tuple, Any
from datetime import datetime as _datetime
from copy import deepcopy as _deepcopy
from PRSTCore.deckformat.deckinput import read_eclipse_deck, convert_deck_units
try:
    from PRSTCore.ad_core.initialization.init_deck_adi_fluid import init_deck_adi_fluid as _init_deck_adi_fluid_impl
except Exception:
    _init_deck_adi_fluid_impl = None
# MRST's own grid path: processGRDECL for corner-point, tensorGrid for a
# block-centred deck, then computeGeometry. The older init_eclipse_grid
# is a lightweight stand-in that returns a different shape per branch and
# carries no cell-to-face topology, so computeTrans and computeWellIndex
# -- both faithfully ported -- had no deck-derived grid to run on.
from PRSTCore.deckformat.grid.eclipse_grid import init_eclipse_grid
from PRSTCore.deckformat.params.rock.init_eclipse_rock import init_eclipse_rock
from PRSTCore.ad_core.models.generic_black_oil_model import make_generic_black_oil_model
from PRSTCore.ad_core.solvers import NonLinearSolver, IterationCountTimeStepSelector


def init_eclipse_problem_ad(deck, **opt) -> Tuple[Any, Any, Any, Any]:
    """Set up state0, model, schedule, nonlinear solver from an ECLIPSE deck.
    """
    if isinstance(deck, str):
        deck = read_eclipse_deck(deck)
    deck = convert_deck_units(deck)

    opts = {
        'useMex': False,
        'useMexGeometry': None,
        'useMexProcessGrid': None,
        'TimestepStrategy': 'iteration',
        'useCPR': True,
        'rowMajorAD': False,
        'AutoDiffBackend': None,
        'UniformFacilityModel': False,
        'maxIterations': 12,
        'useRelaxation': True,
        'model': None,
        'G': None,
        'getSchedule': True,
        'getInitialState': True,
        # MRST's initEclipseGrid default (false). A deck can mark cells
        # active through ACTNUM and then give them no pore volume through
        # PORV, PORO, NTG or MULTPV; those cells contribute an identically
        # zero conservation equation, so the Jacobian comes out
        # structurally singular and no linear solver can start. T142 does
        # this for 242115 of its 433104 active cells. Enabling this drops
        # them, at the cost of no longer matching a run that kept them.
        'RemoveZeroPoreVolume': False,
        # ECLIPSE MINPV, imposed from outside. The deck's own MINPV/MINPVV
        # is honoured whether or not this is set; this is for a deck that
        # states none and still needs one.
        'MinPoreVolume': None,
        # MRST's ``setupSPE10_AD`` option of the same name:
        # ``rock.poro(rock.poro < minporo) = minporo`` (its default is
        # 0.01; ``spe10.m`` uses 1e-3).  Every MRST SPE10 setup floors the
        # porosity, and the reason is the CPR weighting rather than the
        # physics: BlackOilPressureReductionFactors divides by pore volume,
        # so a field carrying the SPE10 decks' 1e-7 filler porosity hands
        # the pressure equation row weights spanning six orders of
        # magnitude and the multigrid stage stops converging.  Off by
        # default -- it moves pore volume, so it changes the answer, and
        # that is the caller's decision, not the reader's.
        'minporo': None,
    }
    opts.update(opt)

    if opts['model'] is None:
        model = _initialize_model(deck, opts)
    else:
        model = opts['model']

    if opts.get('getInitialState', True):
        state0 = _init_state_deck(model, deck)
    else:
        state0 = None

    if opts.get('getSchedule', True):
        # Extract G and rock from model for well completion processing
        G = getattr(model, 'G', None)
        rock = getattr(model, 'rock', None)
        schedule = _convert_deck_schedule_to_mrst(model, deck, G=G, rock=rock)
    else:
        schedule = None

    nonlinear = _get_non_linear_solver(model, opts)

    return state0, model, schedule, nonlinear


def _initialize_model(deck, opt):
    rock = init_eclipse_rock(deck)
    minporo = opt.get('minporo')
    if minporo is not None and isinstance(rock, dict) and rock.get('poro') is not None:
        # setupSPE10_AD.m: rock.poro(rock.poro < opt.minporo) = opt.minporo
        import numpy as _np
        poro = _np.asarray(rock['poro'], dtype=float)
        rock['poro'] = _np.where(poro < float(minporo), float(minporo), poro)
    if 'ACTNUM' not in deck.get('GRID', {}):
        nc = 1
        if 'cartDims' in deck.get('RUNSPEC', {}):
            dims = deck['RUNSPEC']['cartDims']
            nc = int(dims[0] * dims[1] * dims[2])
        deck['GRID']['ACTNUM'] = [1] * nc

    G = init_eclipse_grid(deck, SplitDisconnected=opt.get('SplitDisconnected', False),
                          useMex=opt.get('useMexProcessGrid', False),
                          removeZeroPV=bool(opt.get('RemoveZeroPoreVolume', False)),
                          minPoreVolume=opt.get('MinPoreVolume'))

    if 'cells' not in G:
        nc = int(G.get('cartDims', [1, 1, 1])[0] * G.get('cartDims', [1, 1, 1])[1] * G.get('cartDims', [1, 1, 1])[2])
        import numpy as _np
        G['cells'] = {'indexMap': _np.arange(nc, dtype=int), 'num': nc, 'centroids': _np.zeros((nc, 3))}

    _restrict_rock_to_grid(rock, G, deck)

    fluid = _init_deck_adi_fluid(deck, G, useMex=opt.get('useMex', False))
    model = _select_model_from_deck(G, rock, fluid, deck, UseLegacyModels=opt.get('useLegacyModels', False))
    # setup_operators is the logical-Cartesian fast path and only accepts
    # the grid dicts init_eclipse_grid builds.  Falling back to an *empty*
    # operators dict, as this used to on any failure, is not a safe
    # degradation: with no N/T the flux term is identically zero, so the
    # model silently solves a no-flow problem instead of reporting that it
    # could not discretise the grid.  Fall back to the general-grid port of
    # setupOperatorsTPFA, which works from face topology and therefore
    # covers every grid compute_geometry produces, and let a genuine
    # failure surface.
    from PRSTCore.ad_core.operators import setup_operators
    operators = None
    try:
        operators = setup_operators(G, rock)
    except Exception:
        operators = None
    if not operators:
        from PRSTCore.ad_core.operators_tpfa import setup_operators_tpfa
        operators = setup_operators_tpfa(G, rock)
    setattr(model, 'operators', operators)
    if getattr(model, 'surfactant', False):
        from PRSTCore.ad_eor.deck import cartesian_sq_veloc
        model.operators['sqVeloc'] = cartesian_sq_veloc(model)
    model.dpMaxRel = 0.2
    # poreVolume.m is poro .* G.cells.volumes .* ntg (NTG only when
    # present).  Leaving model.porevolume as None is also correct -- the
    # model's own _porevolume_vector applies the identical rule -- so this
    # is purely a precomputation, and it must not disagree with it.
    import numpy as _np
    vols = G.get('cell_volumes') if isinstance(G, dict) else None
    if vols is None and isinstance(G, dict):
        vols = G.get('cells', {}).get('volumes')
    if vols is not None and isinstance(rock, dict) and rock.get('poro') is not None:
        pv = _np.asarray(vols, dtype=float).ravel() * _np.asarray(rock['poro'], dtype=float).ravel()
        ntg = rock.get('ntg')
        if ntg is not None:
            ntg = _np.asarray(ntg, dtype=float).ravel()
            if ntg.size == 1:
                pv = pv * float(ntg[0])
            elif ntg.size == pv.size:
                pv = pv * ntg
        model.porevolume = pv
    return model


def _init_state_deck(model, deck):
    """Initialize the direct-assignment branch of MRST ``initStateDeck``.

    SPE1 supplies PRESSURE/SWAT/SGAS/RS directly.  MRST therefore uses
    those deck vectors verbatim (after unit conversion); it does *not*
    replace them with a constant pressure or with saturated PVT Rs values.
    """
    import numpy as _np
    # Get cell count from model object (GenericBlackOilModel) or fallback from grid
    if hasattr(model, 'G') and isinstance(model.G, dict) and 'cells' in model.G:
        nc = int(model.G['cells'].get('num', 1))
    elif hasattr(model, '_num_cells'):
        nc = model._num_cells()
    else:
        G = getattr(model, 'G', None)
        if isinstance(G, dict) and 'cartDims' in G:
            dims = G['cartDims']
            nc = int(dims[0] * dims[1] * dims[2]) if len(dims) >= 3 else 1
        else:
            nc = 1
    state = {}
    sol = deck.get('SOLUTION', {})
    index_map = _np.arange(nc, dtype=int)
    if hasattr(model, 'G') and isinstance(model.G, dict):
        try:
            index_map = _np.asarray(model.G['cells'].get('indexMap', index_map), dtype=int).ravel()
        except Exception:
            pass
    if index_map.size != nc:
        index_map = _np.arange(nc, dtype=int)

    def deck_vector(name, default=None):
        if not isinstance(sol, dict) or name not in sol:
            return default
        try:
            values = _np.asarray(sol[name], dtype=float).ravel()
        except Exception:
            return default
        if values.size == 0:
            return default
        # MRST directAssignment indexes full cartesian deck vectors through
        # G.cells.indexMap.  Fully active grids simply take the first nc.
        if index_map.size and values.size > int(index_map.max(initial=-1)):
            return values[index_map]
        if values.size == nc:
            return values.copy()
        return default

    pressure = deck_vector('PRESSURE')
    if pressure is None:
        equilibrium_state = _init_equilibrium_state_deck(model, deck, nc)
        if equilibrium_state is not None:
            return _apply_swatinit_deck(model, deck, equilibrium_state, index_map)
        # Keep a clear fallback for incomplete decks; direct SPE1-style
        # decks always take the branch above.
        pressure = _np.full((nc,), 1.0e7, dtype=float)
    state['pressure'] = pressure

    sw = deck_vector('SWAT')
    sg = deck_vector('SGAS')
    so = deck_vector('SOIL')
    if sw is None:
        if so is not None:
            sw = 1.0 - so - (_np.zeros((nc,)) if sg is None else sg)
        else:
            sw = _np.zeros((nc,), dtype=float)
    if sg is None:
        sg = _np.zeros((nc,), dtype=float) if so is None else 1.0 - sw - so
    state['sW'] = _np.asarray(sw, dtype=float).ravel()
    state['sG'] = _np.asarray(sg, dtype=float).ravel()

    # Direct RS/RV assignment is the exact branch in initStateDeck.m.
    rs = deck_vector('RS')
    if rs is None:
        rs = _np.zeros((nc,), dtype=float)
    state['rs'] = _np.asarray(rs, dtype=float).ravel()
    rv = deck_vector('RV')
    state['rv'] = _np.zeros((nc,), dtype=float) if rv is None else _np.asarray(rv, dtype=float).ravel()
    state['time'] = 0.0
    state['wellSol'] = []
    return state


def _apply_swatinit_deck(model, deck, state, index_map):
    """Apply MRST ``initStateDeck``'s post-EQUIL SWATINIT assignment."""
    import numpy as _np
    props = deck.get('PROPS', {})
    swatinit = props.get('SWATINIT') if isinstance(props, dict) else None
    if swatinit is None:
        return state
    try:
        swatinit = _np.asarray(swatinit, dtype=float).ravel()
    except (TypeError, ValueError):
        return state
    if swatinit.size <= int(index_map.max(initial=-1)):
        return state
    swat = swatinit[index_map]
    sg = _np.asarray(state.get('sG', _np.zeros_like(swat)), dtype=float).ravel()
    so = 1.0 - swat - sg
    if _np.any(so < 0.0):
        # initStateDeck warns then clamps the implicit oil saturation.
        so = _np.maximum(so, 0.0)
    state['sW'] = swat
    # ``initStateDeck.m`` first obtains the phase-pressure matrix from
    # ``initStateBlackOilAD`` and then rescales the OW capillary curve so
    # the explicitly supplied SWATINIT remains in hydrostatic equilibrium.
    # Preserve the pre-SWATINIT water pressure supplied by the equilibrium
    # initializer for this exact ``dp./pcow`` operation.
    pw_equil = state.pop('_mrst_equilibrium_water_pressure', None)
    if pw_equil is not None:
        try:
            pw_equil = _np.asarray(pw_equil, dtype=float).ravel()
            pressure = _np.asarray(state['pressure'], dtype=float).ravel()
            if pw_equil.size == pressure.size:
                pW, pO, _ = model._phase_pressures(
                    _np.zeros_like(pressure), state['sW'], state['sG'])
                pcow = _np.asarray(pO - pW, dtype=float).ravel()
                dp = pressure - pw_equil
                with _np.errstate(divide='ignore', invalid='ignore'):
                    scale = dp / pcow
                scale[dp <= 0.0] = 1.0
                state['pcowScale'] = scale
        except Exception:
            pass
    return state


def _build_swof_table(props):
    """Return a 4-column ``[Sw, Krw, Krow, Pcow]`` table.

    Thin wrapper around :func:`PRSTCore.ad_props.relperm_tables.
    build_swof_sgof_tables` (the single source of truth also used by
    ``GenericBlackOilModel``'s relperm evaluation), keeping this module's
    original zero-row-array-on-failure contract.
    """
    import numpy as _np
    from PRSTCore.ad_props.relperm_tables import build_swof_sgof_tables
    swof, _sgof = build_swof_sgof_tables(props)
    return _np.zeros((0, 4)) if swof is None else swof


def _init_equilibrium_state_deck(model, deck, nc):
    """Port the applicable ``initStateDeck`` EQUIL branch for EGG.

    MRST routes EQUIL through ``initStateBlackOilAD``.  The bundled EGG
    case is its two-phase, no-capillary-pressure PVCDO special case:
    ``assignPVCDO.m`` gives an exponential oil shrinkage factor, for which
    the hydrostatic ODE in ``initializeEquilibriumPressures.m`` has the
    closed-form solution used below.  Other EQUIL variants deliberately
    remain unhandled here until their matching MRST region/PVT paths are
    ported, rather than falling back to an invented initialization.
    """
    import numpy as _np
    sol = deck.get('SOLUTION', {})
    equil = sol.get('EQUIL') if isinstance(sol, dict) else None
    if equil is None:
        return None
    try:
        eql = _np.asarray(equil, dtype=float)
        eql = eql[0] if eql.ndim > 1 else eql
    except Exception:
        return None
    if eql.size < 4 or not (getattr(model, 'water', False) and getattr(model, 'oil', False)):
        return None
    if getattr(model, 'gas', False):
        if getattr(model, 'vapoil', False):
            return _init_vapoil_equilibrium_state(model, deck, nc)
        return _init_spe9_equilibrium_state(model, deck, eql, nc)
    pvt = getattr(model, '_blackoil_pvt', None)
    pvcdo = getattr(pvt, 'pvcdo', None)
    if pvcdo is None:
        return None
    props = deck.get('PROPS', {})
    swof = _build_swof_table(props)
    if swof.ndim != 2 or swof.shape[1] < 4 or not _np.allclose(swof[:, 3], 0.0):
        return None
    centroids = _np.asarray(getattr(model, 'G', {}).get('cells', {}).get('centroids', _np.zeros((nc, 3))), dtype=float)
    if centroids.shape[0] != nc or centroids.shape[1] < 3:
        return None
    datum_depth, datum_pressure, water_contact = float(eql[0]), float(eql[1]), float(eql[2])
    por, bor, co, _, _ = [float(v) for v in pvcdo]
    density = _np.asarray(props.get('DENSITY', []), dtype=float).ravel()
    if density.size < 1:
        return None
    rho_os = float(density[0])
    g = float(_np.asarray(getattr(model, 'gravity', [0.0, 0.0, 9.80665]), dtype=float).ravel()[-1])
    z = centroids[:, 2]
    if abs(co) < 1.0e-30:
        pressure = datum_pressure + g * rho_os / bor * (z - datum_depth)
    else:
        # dp/dz = g*rho_os*exp(co*(p-por))/bor, integrated exactly.
        e0 = _np.exp(-co * (datum_pressure - por))
        e = e0 - co * g * rho_os / bor * (z - datum_depth)
        if _np.any(e <= 0.0):
            raise ValueError('PVCDO equilibrium hydrostatic solution left its physical branch')
        pressure = por - _np.log(e) / co
    # initializeEquilibriumSaturations.m selects SWOF's lower endpoint
    # when p_w-p_o equals the constant zero Pc curve.
    sw = _np.full(nc, float(swof[0, 0]))
    return {
        'pressure': pressure,
        'sW': sw,
        'sG': _np.zeros(nc),
        'rs': _np.zeros(nc),
        'rv': _np.zeros(nc),
        'time': 0.0,
        'wellSol': [],
    }


def _init_vapoil_equilibrium_state(model, deck, nc):
    """MRST multi-region EQUIL initialization for PVTO/PVTG black oil.

    This follows ``getInitializationRegionsDeck``,
    ``initializeEquilibriumPressures`` and
    ``initializeEquilibriumSaturations`` for the PVT/SAT region combination
    used by Norne.  In particular, RSVD is a table per EQL region and PVTG
    is evaluated with MRST's ``linshift`` interpolation through
    :class:`DeckBlackOilPVT`.
    """
    import numpy as _np
    try:
        from scipy.integrate import solve_ivp
    except Exception as exc:
        raise RuntimeError('EQUIL initialization requires scipy.integrate') from exc

    sol = deck.get('SOLUTION', {})
    regions = deck.get('REGIONS', {})
    props = deck.get('PROPS', {})
    pvt = getattr(model, '_blackoil_pvt', None)
    equil = sol.get('EQUIL') if isinstance(sol, dict) else None
    eqlnum = regions.get('EQLNUM') if isinstance(regions, dict) else None
    density = _np.asarray(props.get('DENSITY', []), dtype=float).ravel()
    swof = _build_swof_table(props)
    sgof = _np.asarray(props.get('SGOF', []), dtype=float)
    if (pvt is None or equil is None or density.size < 3 or
            swof.ndim != 2 or swof.shape[1] < 4 or
            sgof.ndim != 2 or sgof.shape[1] < 4):
        return None
    try:
        equil = _np.asarray(equil, dtype=float)
    except (TypeError, ValueError):
        return None
    if equil.ndim == 1:
        equil = equil.reshape((1, -1))
    if equil.ndim != 2 or equil.shape[1] < 8:
        return None

    index_map = _np.asarray(model.G.get('cells', {}).get('indexMap', _np.arange(nc)), dtype=int).ravel()
    if index_map.size != nc:
        return None
    if eqlnum is None:
        # getRegionMap.m: EQLNUM defaults to region 1 everywhere when the
        # deck has a single equilibration region (by far the common case --
        # SPE3 has one EQUIL record and never sets EQLNUM explicitly).
        eqlnum = _np.ones(nc, dtype=int)
    else:
        eqlnum = _np.asarray(eqlnum, dtype=int).ravel()
        if eqlnum.size <= int(index_map.max(initial=-1)):
            return None
        eqlnum = eqlnum[index_map]
    centroids = _np.asarray(model.G.get('cells', {}).get('centroids', _np.zeros((nc, 3))), dtype=float)
    if centroids.shape != (nc, 3):
        return None

    rho_os, rho_ws, rho_gs = (float(density[0]), float(density[1]), float(density[2]))
    gravity = float(_np.asarray(model.gravity, dtype=float).ravel()[-1])
    z_all = centroids[:, 2]
    state_p = _np.zeros(nc, dtype=float)
    state_pw = _np.zeros(nc, dtype=float)
    state_sw = _np.zeros(nc, dtype=float)
    state_sg = _np.zeros(nc, dtype=float)
    state_rs = _np.zeros(nc, dtype=float)
    state_rv = _np.zeros(nc, dtype=float)
    touched = _np.zeros(nc, dtype=bool)

    def linear_nearest(x, y, q):
        # griddedInterpolant(x, y, 'linear', 'nearest') in getRegion.m.
        return _np.interp(_np.asarray(q, dtype=float), _np.asarray(x, dtype=float),
                          _np.asarray(y, dtype=float), left=float(y[0]), right=float(y[-1]))

    def interp_pc(table, saturation):
        # assignSWOF/assignSGOF extend the endpoint values before interp1d.
        return _np.interp(_np.asarray(saturation, dtype=float), table[:, 0], table[:, 3],
                          left=table[0, 3], right=table[-1, 3])

    def solve_saturation(dp, table, sign, s_min, s_max):
        # initializeEquilibriumSaturations.solveSaturations exactly uses a
        # 0.01 bounds grid followed by a 0.0001 inversion grid.
        sat_bounds = _np.arange(0.0, 1.0000001, 1.0e-2)
        pc_bounds = sign * interp_pc(table, sat_bounds)
        dp = _np.asarray(dp, dtype=float)
        s_min = _np.asarray(s_min, dtype=float)
        s_max = _np.asarray(s_max, dtype=float)
        out = _np.zeros_like(dp)
        to_max = dp > float(pc_bounds.max())
        to_min = dp <= float(pc_bounds.min())
        out[to_min] = s_min if s_min.size == 1 else s_min[to_min]
        out[to_max] = s_max if s_max.size == 1 else s_max[to_max]
        middle = ~(to_min | to_max)
        if _np.any(middle):
            sat = _np.arange(0.0, 1.0000001, 1.0e-4)
            pc = sign * interp_pc(table, sat)
            if float(pc[0]) == float(pc[-1]):
                out[middle] = s_max if s_max.size == 1 else s_max[middle]
            else:
                # MATLAB unique(pc, 'last') retains the final plateau point.
                pc_unique, from_end = _np.unique(pc[::-1], return_index=True)
                sat_unique = sat[pc.size - 1 - from_end]
                lower = s_min if s_min.size == 1 else s_min[middle]
                out[middle] = _np.maximum(_np.interp(dp[middle], pc_unique, sat_unique), lower)
        return out

    def integrate_pressure(rhs, p_datum, z_datum, z, contacts):
        zmin = min(float(z.min()), *[float(value) for value in contacts])
        zmax = max(float(z.max()), *[float(value) for value in contacts])
        ode_options = dict(rtol=5.0e-8, atol=1.0e-10, dense_output=True, method='RK45')
        upward = (solve_ivp(rhs, (z_datum, zmin), [p_datum], **ode_options)
                  if zmin < z_datum else None)
        downward = (solve_ivp(rhs, (z_datum, zmax), [p_datum], **ode_options)
                    if zmax > z_datum else None)

        def evaluate(depth):
            depth = _np.asarray(depth, dtype=float)
            result = _np.full(depth.size, p_datum, dtype=float)
            above = depth < z_datum
            below = depth > z_datum
            if _np.any(above):
                result[above] = upward.sol(depth[above])[0]
            if _np.any(below):
                result[below] = downward.sol(depth[below])[0]
            return result
        return evaluate

    rsvd_all = sol.get('RSVD') if isinstance(sol, dict) else None
    for region_no, eql in enumerate(equil, start=1):
        cells = _np.flatnonzero(eqlnum == region_no)
        if cells.size == 0:
            continue
        datum_depth, datum_pressure = float(eql[0]), float(eql[1])
        woc, pc_woc, goc, pc_goc = [float(value) for value in eql[2:6]]
        z = z_all[cells]
        # getInitializationRegionsDeck.m: EQUIL item 7 (0-based eql[6]).
        # rs_method<=0 (the common default, and MRST's own hard default
        # when the item is absent) means a single Rs -- the saturated
        # value at datum pressure -- applied uniformly, not a depth table;
        # RSVD is required only when rs_method > 0 explicitly requests it.
        rs_method = float(eql[6]) if eql.size > 6 else 0.0
        if rs_method <= 0.0:
            rs_const = float(pvt.eval(_np.asarray([datum_pressure]))['rs'][0])

            def rs_depth(depth, _rs_const=rs_const):
                return _rs_const
        else:
            if isinstance(rsvd_all, list):
                if region_no > len(rsvd_all):
                    return None
                rsvd = _np.asarray(rsvd_all[region_no - 1], dtype=float)
            else:
                rsvd = _np.asarray(rsvd_all, dtype=float)
            if rsvd.ndim != 2 or rsvd.shape[1] < 2:
                return None

            def rs_depth(depth, _rsvd=rsvd):
                return float(linear_nearest(_rsvd[:, 0], _rsvd[:, 1], [depth])[0])

        def oil_rhs(depth, pressure):
            rs_value = rs_depth(depth)
            rs_sat = float(pvt.eval(_np.asarray([pressure[0]]))['rs'][0])
            rs_value = min(rs_value, rs_sat)
            values = pvt.eval(_np.asarray([pressure[0]]), rs_override=[rs_value],
                              saturated_override=[rs_value >= rs_sat])
            return [gravity * float(values['bo'][0]) * (rho_os + rs_value * rho_gs)]

        oil_pressure = integrate_pressure(oil_rhs, datum_pressure, datum_depth, z, (woc, goc))
        po = oil_pressure(z)
        po_woc = float(oil_pressure(_np.asarray([woc]))[0])
        po_goc = float(oil_pressure(_np.asarray([goc]))[0])

        def water_rhs(depth, pressure):
            values = pvt.eval(_np.asarray([pressure[0]]))
            return [gravity * rho_ws * float(values['bw'][0])]

        pw = integrate_pressure(water_rhs, po_woc - pc_woc, woc, z, (woc, goc))(z)

        # getRegion.m's rv_method <= 0 branch: rvSat at datum pressure plus
        # the GOC capillary-pressure entry.
        rv_const = float(pvt.rv_sat(_np.asarray([datum_pressure + pc_goc]))[0])

        def gas_rhs(depth, pressure):
            pg = float(pressure[0])
            rv_sat = float(pvt.rv_sat(_np.asarray([pg]))[0])
            rv = min(rv_const, rv_sat)
            bg, _ = pvt.gas_props(_np.asarray([pg]), _np.asarray([rv]),
                                  _np.asarray([rv >= rv_sat]))
            return [gravity * float(bg[0]) * (rv * rho_os + rho_gs)]

        pg = integrate_pressure(gas_rhs, po_goc + pc_goc, goc, z, (woc, goc))(z)
        # getMinMaxPhaseSaturations selects ENDSCALE drainage endpoints
        # when they are present.  They are full Cartesian vectors, so map
        # them through G.cells.indexMap exactly as initStateDeck does.
        def endpoint(name, default):
            values = props.get(name)
            if values is None:
                return _np.full(cells.size, default, dtype=float)
            values = _np.asarray(values, dtype=float).ravel()
            if values.size <= int(index_map.max(initial=-1)):
                return _np.full(cells.size, default, dtype=float)
            return values[index_map[cells]]

        sw_min = endpoint('SWL', float(swof[0, 0]))
        sw_max = endpoint('SWU', 1.0)
        sg_max = _np.minimum(endpoint('SGU', 1.0), 1.0 - sw_min)
        sw = solve_saturation(pw - po, swof, -1.0, sw_min, sw_max)
        sg = solve_saturation(pg - po, sgof, 1.0, 0.0, sg_max)
        so = 1.0 - sw - sg
        if _np.any(so < -1.0e-12):
            # Gas cap above the gas-oil contact with a flat gas-oil
            # capillary table (SGOF column 4 == 0): sg saturates to its
            # maximum while the water-oil Pc keeps a finite sw, so the
            # three-phase sum exceeds one and the oil phase is fully
            # displaced.  Renormalize to the physical all-gas state
            # (so = 0, gas fills the rest) instead of bailing out to a
            # default all-oil initial state.  An all-oil state with no
            # free gas makes a GRAT gas producer degenerate -- its
            # realized gas rate is zero, so the q_s perforation equation
            # and the control closure become the same equation pinning
            # qGs to two different values, and the Jacobian is exactly
            # singular.
            gas_cap = so < -1.0e-12
            so = _np.maximum(so, 0.0)
            sg[gas_cap] = 1.0 - sw[gas_cap] - so[gas_cap]
        so = _np.maximum(so, 0.0)

        rs_max = pvt.eval(po)['rs']
        if rs_method <= 0.0:
            rs_depth_vals = _np.full(cells.size, rs_const)
        else:
            rs_depth_vals = linear_nearest(rsvd[:, 0], rsvd[:, 1], z)
        rs = _np.minimum(rs_depth_vals, rs_max)
        rs[sg > 0.0] = rs_max[sg > 0.0]
        # initStateBlackOilAD evaluates rvSatF at the reference oil
        # pressure (the ``po`` variable set in the disgas block), not at
        # gas pressure.  This distinction is material in Norne.
        rv_max = pvt.rv_sat(po)
        rv = _np.minimum(_np.full(cells.size, rv_const), rv_max)
        rv[so > 0.0] = rv_max[so > 0.0]

        # initStateBlackOilAD switches the stored reference pressure where
        # a dominant non-oil phase has mobile relative permeability only.
        krw, kro, krg = model._relative_perm(sw, sg)
        max_sat = _np.maximum(_np.maximum(sw, so), sg)
        ref_immobile = kro < 1.0e-8
        only_gas = (sg == max_sat) & ref_immobile
        only_water = (sw == max_sat) & ref_immobile
        pressure = po.copy()
        pressure[only_gas] = pg[only_gas] - interp_pc(sgof, sg[only_gas])
        # pc{W}=-pcOW, so pW-pc{W}=pW+pcOW.
        pressure[only_water] = pw[only_water] + interp_pc(swof, sw[only_water])

        state_p[cells] = pressure
        state_pw[cells] = pw
        state_sw[cells] = sw
        state_sg[cells] = sg
        state_rs[cells] = rs
        state_rv[cells] = rv
        touched[cells] = True

    if not _np.all(touched):
        return None
    return {
        'pressure': state_p, 'sW': state_sw, 'sG': state_sg,
        'rs': state_rs, 'rv': state_rv, 'time': 0.0, 'wellSol': [],
        '_mrst_equilibrium_water_pressure': state_pw,
    }


def _init_spe9_equilibrium_state(model, deck, eql, nc):
    """MRST EQUIL/RSVD initialization path used by the bundled SPE9 deck."""
    import numpy as _np
    try:
        from scipy.integrate import solve_ivp
    except Exception as exc:
        raise RuntimeError('SPE9 EQUIL initialization requires scipy.integrate') from exc
    pvt = getattr(model, '_blackoil_pvt', None)
    props = deck.get('PROPS', {})
    density = _np.asarray(props.get('DENSITY', []), dtype=float).ravel()
    swof = _build_swof_table(props)
    if pvt is None or density.size < 3 or swof.ndim != 2 or swof.shape[1] < 4:
        return None
    # getInitializationRegionsDeck.m: RS is only depth-tabulated (RSVD) when
    # both disgas is active and EQUIL item 7 (0-based eql[6]) explicitly
    # requests it (rs_method > 0).  A disgas-free deck (no dissolved gas at
    # all, e.g. SPE10 model 1) needs no RS handling; rs_method<=0 (including
    # the item being entirely defaulted, ``2*``) uses the single saturated
    # Rs at datum pressure instead of a table -- neither case requires
    # RSVD to be present.
    has_disgas = bool(getattr(model, 'disgas', False))
    rs_method = float(eql[6]) if has_disgas and eql.size > 6 else 0.0
    rsvd = None
    if has_disgas and rs_method > 0.0:
        rsvd = _np.asarray(deck.get('SOLUTION', {}).get('RSVD', []), dtype=float)
        if rsvd.ndim != 2 or rsvd.shape[1] < 2:
            return None
    centroids = _np.asarray(model.G.get('cells', {}).get('centroids', _np.zeros((nc, 3))), dtype=float)
    if centroids.shape != (nc, 3):
        return None
    z = centroids[:, 2]
    datum_depth, datum_pressure, water_contact = [float(v) for v in eql[:3]]
    g = float(_np.asarray(model.gravity, dtype=float).ravel()[-1])
    rho_os, rho_ws, rho_gs = float(density[0]), float(density[1]), float(density[2])

    def interp_linear_extrap(x, y, q):
        x = _np.asarray(x, dtype=float).ravel()
        y = _np.asarray(y, dtype=float).ravel()
        q = _np.asarray(q, dtype=float)
        out = _np.interp(q, x, y)
        below = q < x[0]
        above = q > x[-1]
        if _np.any(below):
            out[below] = y[0] + (q[below] - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
        if _np.any(above):
            out[above] = y[-1] + (q[above] - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
        return out

    if not has_disgas:
        def rs_depth(depth):
            return 0.0
    elif rs_method <= 0.0:
        rs_const = float(pvt.eval(_np.asarray([datum_pressure]))['rs'][0])

        def rs_depth(depth, _rs_const=rs_const):
            return _rs_const
    else:
        def rs_depth(depth):
            return float(interp_linear_extrap(rsvd[:, 0], rsvd[:, 1], _np.asarray([depth]))[0])

    def interp_swof(column, saturation):
        # assignSWOF.m calls extendTab before MRST's interpTable: a copy of
        # each endpoint is inserted at saturation +/- 1.  Consequently the
        # tabulated endpoint values are constant outside the physical SWOF
        # range, rather than linearly extrapolated.
        return _np.interp(_np.asarray(saturation, dtype=float),
                          swof[:, 0], swof[:, column],
                          left=swof[0, column], right=swof[-1, column])

    # initializeEquilibriumPressures.m integrates reference-oil pressure
    # above/below the datum and then repeats for water from the WOC.
    def oil_rhs(depth, pressure):
        rs_local = rs_depth(depth)
        rs_sat = float(pvt.eval(_np.asarray([pressure[0]]))['rs'][0])
        rs_local = min(rs_local, rs_sat)
        saturated = _np.asarray([rs_local >= rs_sat])
        bo = float(pvt.eval(_np.asarray([pressure[0]]), rs_override=[rs_local], saturated_override=saturated)['bo'][0])
        # getInitializationRegionsBlackOil.m/getOilDensity: dissolved gas
        # contributes ``rs*rhoGS`` to the oil-phase mass density.
        return [g * bo * (rho_os + rs_local * rho_gs)]

    def water_rhs(depth, pressure):
        bw = float(pvt.eval(_np.asarray([pressure[0]]))['bw'][0])
        return [g * rho_ws * bw]

    zmin = min(float(z.min()), float(water_contact))
    zmax = max(float(z.max()), float(water_contact))
    ode_opts = dict(rtol=5.0e-8, atol=1.0e-10, dense_output=True, method='RK45')
    oil_up = solve_ivp(oil_rhs, (datum_depth, zmin), [datum_pressure], **ode_opts) if zmin < datum_depth else None
    oil_down = solve_ivp(oil_rhs, (datum_depth, zmax), [datum_pressure], **ode_opts) if zmax > datum_depth else None

    def eval_oil(depths):
        depths = _np.asarray(depths, dtype=float)
        out = _np.full(depths.size, datum_pressure, dtype=float)
        above = depths < datum_depth
        below = depths > datum_depth
        if _np.any(above):
            out[above] = oil_up.sol(depths[above])[0]
        if _np.any(below):
            out[below] = oil_down.sol(depths[below])[0]
        return out

    oil_pressure = eval_oil(z)
    oil_at_woc = float(eval_oil(_np.asarray([water_contact]))[0])
    wat_up = solve_ivp(water_rhs, (water_contact, zmin), [oil_at_woc], **ode_opts) if zmin < water_contact else None
    wat_down = solve_ivp(water_rhs, (water_contact, zmax), [oil_at_woc], **ode_opts) if zmax > water_contact else None
    water_pressure = _np.full(nc, oil_at_woc, dtype=float)
    above = z < water_contact
    below = z > water_contact
    if _np.any(above):
        water_pressure[above] = wat_up.sol(z[above])[0]
    if _np.any(below):
        water_pressure[below] = wat_down.sol(z[below])[0]

    # initializeEquilibriumSaturations.m / invertCapillary.m for water:
    # pc_sign=-1 and SWOF's fourth column is PcOW.
    sat_grid = _np.arange(0.0, 1.0000001, 1.0e-4)
    pcow_grid = interp_swof(3, sat_grid)
    pc_grid = -pcow_grid
    # MATLAB ``unique(pc, 'last')``: retain the final saturation for a
    # capillary-pressure plateau.  np.unique keeps the first occurrence,
    # so obtain the same positions by first reversing the input.
    pc_unique, ix_from_end = _np.unique(pc_grid[::-1], return_index=True)
    sat_unique = sat_grid[pc_grid.size - 1 - ix_from_end]
    dp = water_pressure - oil_pressure
    sw = _np.empty(nc, dtype=float)
    sw_min = float(swof[0, 0])
    # ``solveSaturations`` uses a coarse 0.01 grid for its bounds test,
    # then a 0.0001 grid only for the inversion.
    sat_bounds = _np.arange(0.0, 1.0000001, 1.0e-2)
    pc_bounds = -interp_swof(3, sat_bounds)
    low = dp <= float(pc_bounds.min())
    high = dp > float(pc_bounds.max())
    sw[low] = sw_min
    sw[high] = 1.0
    mid = ~(low | high)
    if _np.any(mid):
        sw[mid] = _np.maximum(_np.interp(dp[mid], pc_unique, sat_unique), sw_min)
    rs_sat = pvt.eval(oil_pressure)['rs']
    if not has_disgas:
        rs_depth_vals = _np.zeros(nc)
    elif rs_method <= 0.0:
        rs_depth_vals = _np.full(nc, rs_const)
    else:
        rs_depth_vals = interp_linear_extrap(rsvd[:, 0], rsvd[:, 1], z)
    rs = _np.minimum(rs_depth_vals, rs_sat)

    # ``initStateBlackOilAD.m`` normally stores the reference-oil pressure,
    # but replaces it wherever water is the only mobile phase.  SPE9 has a
    # fully water-saturated interval beneath the WOC; retaining oil's
    # hydrostatic pressure there is therefore not equivalent to MRST.  The
    # test is the source's ``watMajority & referenceImmobile`` with
    # RelativePermeability from assignSWOF (linear table interpolation).
    kro = interp_swof(2, sw)
    only_water = (sw >= _np.maximum(sw, 1.0 - sw)) & (kro < 1.0e-8)
    state_pressure = oil_pressure.copy()
    if _np.any(only_water):
        # getEquilPC.m: pc_sign(W) = -1 and pcW = -pcOW(sW).  The exact
        # `p(:,W) - pc{W}` assignment in initStateBlackOilAD is thus
        # water_pressure + pcOW.
        pcow = interp_swof(3, sw)
        state_pressure[only_water] = water_pressure[only_water] + pcow[only_water]
        # initStateBlackOilAD initializes dissolved gas from the
        # reference-oil equilibrium pressure.  In the cells where the
        # source selects water as the sole mobile/reference phase, that
        # oil-pressure branch has no dissolved gas state.  MRST's emitted
        # SPE9 state therefore carries Rs=0 in precisely this mask.
        rs[only_water] = 0.0
    return {
        'pressure': state_pressure,
        'sW': sw,
        'sG': _np.zeros(nc),
        'rs': rs,
        'rv': _np.zeros(nc),
        'time': 0.0,
        'wellSol': [],
    }


def _convert_deck_schedule_to_mrst(model, deck, G=None, rock=None):
    import numpy as _np

    sched = deck.get('SCHEDULE', {})
    units = sched.get('_unit_factors', deck.get('_unit_factors', {})) if isinstance(sched, dict) else {}
    time_scale = float(units.get('time', 1.0))
    length_scale = float(units.get('length', 1.0))
    liquid_rate_scale = float(units.get('liqvol_s', units.get('volume', 1.0))) / time_scale
    gas_rate_scale = float(units.get('gasvol_s', units.get('gas_volume', 1.0))) / time_scale
    pressure_scale = float(units.get('press', 1.0))
    trans_scale = float(units.get('trans', units.get('transmissibility', 1.0)))
    perm_length_scale = float(units.get('perm', 1.0)) * length_scale

    wells = {}
    controls = []
    steps = []
    control_ix = []
    control_changed = False

    order = sched.get('_order', [])
    # WCONHIST/WCONINJH are what a history-matching deck states its well
    # controls with -- they carry the *observed* rates. Without them such
    # a deck converts to a single control with every well shut, which is
    # not an error anywhere: the model simply has no wells, so it can be
    # built and simulated and matched against nothing.
    kw_pos = {k: 0 for k in ('WELSPECS', 'COMPDAT', 'WCONPROD', 'WCONINJE',
                              'WCONHIST', 'WCONINJH', 'TSTEP', 'DATES',
                              'WPOLYMER', 'WSURFACT')}
    start_date = _parse_eclipse_date(deck.get('RUNSPEC', {}).get('START', ''))

    for kw in order:
        if kw not in kw_pos:
            continue
        recs, kw_pos[kw] = _consume_schedule_keyword_group(sched.get(kw, []), kw_pos[kw])
        if not recs:
            continue

        if kw == 'WELSPECS':
            for row in recs:
                if not row:
                    continue
                wname = _clean_token(row[0])
                if not wname:
                    continue
                group = _clean_token(_get_item(row, 1), default='FIELD')
                i = _safe_int(_get_item(row, 2), 1)
                j = _safe_int(_get_item(row, 3), 1)
                # ``processWells`` retains the WELSPECS reference-depth
                # default (NaN), and ``makeScheduleConsistent`` subsequently
                # sets it to the first perforation depth.  Do not replace
                # Eclipse's ``1*`` with zero: that would create a fictitious
                # four-kilometre well-bore pressure drop for EGG.
                ref_item = _get_item(row, 4)
                ref_defaulted = _is_defaulted_item(ref_item)
                ref = (_np.nan if ref_defaulted else _safe_float(ref_item, 0.0) * length_scale)
                phase = _clean_token(_get_item(row, 5), default='OIL').upper()
                radius = _safe_float(_get_item(row, 6), 0.0)
                wells.setdefault(wname, {
                    'name': wname,
                    'group': group,
                    # A WELSPECS/COMPDAT entry defines geometry, not an
                    # active well control.  processWells activates a well
                    # only through the contemporaneous WCON/WELOPEN state.
                    'status': False,
                    'type': 'rate',
                    'val': 0.0,
                    'sign': -1,
                    'phase': phase,
                    # MRST-0's ``[qWs, qOs, qGs, bhp] = deal(0, 0, 0,
                    # 1*atm)``, set before any control record is read.
                    # A well WELSPECS declares but no WCON* record has
                    # yet mentioned still has to report *something* to
                    # getObservedFromSchedule -- and zero rate at one
                    # atmosphere is what a shut well observes. Leaving
                    # these absent puts None into the observed container,
                    # which the objective then multiplies.
                    'qWs': 0.0,
                    'qOs': 0.0,
                    'qGs': 0.0,
                    'bhp': 101325.0,
                    # MRST addWell/processWells default composition for a
                    # producer is uniform.  WCONPROD does not redefine it.
                    # This matters for producer crossflow perforations,
                    # where calculatePhaseRate/crossFlowMixture can fall
                    # back to W.compi.
                    'compi': [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                    'refDepth': ref,
                    'defaulted': {'refDepth': bool(ref_defaulted)},
                    'radius': radius,
                    'i': i,
                    'j': j,
                    'k': [],
                })
                control_changed = True
        elif kw == 'COMPDAT':
            for row in recs:
                if not row:
                    continue
                # readSCHEDULE.m uses readDefaultedKW for COMPDAT.  Expand
                # Eclipse's ``n*`` default items before positional access;
                # otherwise Norne's ``2*`` before the D-factor shifts the
                # connection-direction column from its MRST field 13.
                row = _expand_defaulted_record(row)
                wspec = _clean_token(row[0])
                i_comp = _safe_int(_get_item(row, 1), 0)  # COMPDAT I (0=use WELSPECS)
                j_comp = _safe_int(_get_item(row, 2), 0)  # COMPDAT J
                k1 = _safe_int(_get_item(row, 3), 1)
                k2 = _safe_int(_get_item(row, 4), k1)
                status = _clean_token(_get_item(row, 5), default='OPEN').upper() != 'SHUT'
                # COMPDAT columns: WI(7), diameter(8), Kh(9), skin(10), Dfactor(11), direction(12)
                wi_deck = _safe_float(_get_item(row, 7), -1.0)
                if wi_deck > 0:
                    wi_deck *= trans_scale
                diam = _safe_float(_get_item(row, 8), 1.0) * length_scale
                kh_deck = _safe_float(_get_item(row, 9), -1.0)
                if kh_deck > 0:
                    kh_deck *= perm_length_scale
                skin = _safe_float(_get_item(row, 10), 0.0)
                # COMPDAT's 13th item is the connection direction used by
                # MRST's addWell/computeWellIndex path.
                direction = _clean_token(_get_item(row, 12), default='Z').upper()[:1]
                
                for wname in _match_wells(wells, wspec):
                    w = wells[wname]
                    # Use WELSPECS i,j if COMPDAT specifies 0
                    i_use = i_comp if i_comp > 0 else w.get('i', 1)
                    j_use = j_comp if j_comp > 0 else w.get('j', 1)
                    k_range = list(range(min(k1, k2), max(k1, k2) + 1))
                    w['k'] = k_range
                    # Store completion data for later cells/WI calculation
                    if 'completions' not in w:
                        w['completions'] = []
                    w['completions'].append({
                        'i': i_use, 'j': j_use, 'k': k_range,
                        'wi': wi_deck, 'diam': diam, 'kh': kh_deck, 'skin': skin,
                        'dir': direction,
                        'status': bool(status),
                    })
                    control_changed = True
        elif kw in ('WCONPROD', 'WCONINJE'):
            for row in recs:
                if not row:
                    continue
                # ``readWConInje/readWConProd`` call readDefaultedKW with
                # fixed Eclipse record templates.  Keep the expanded fields
                # here so the primary target and every limit have the same
                # positional interpretation as MRST's processWells.m.
                row = _expand_defaulted_record(row)
                if kw == 'WCONPROD':
                    defaults = ['Default', 'OPEN', 'Default', _np.inf, _np.inf,
                                _np.inf, _np.inf, _np.inf, _np.nan, 0.0, 0.0, 0.0]
                else:
                    defaults = ['Default', 'Default', 'OPEN', 'Default', _np.inf,
                                _np.inf, _np.nan, _np.inf, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0]
                row = list(row) + defaults[len(row):]
                # Some decks retain optional tail items beyond the
                # readDefaultedKW template.  MRST consumes the templated
                # prefix for controls and leaves such extensions intact.
                row = [defaults[i] if value is None and i < len(defaults) else value
                       for i, value in enumerate(row)]
                wspec = _clean_token(row[0])
                for wname in _match_wells(wells, wspec):
                    w = wells[wname]
                    open_flag = _clean_token(_get_item(row, 1 if kw == 'WCONPROD' else 2), default='OPEN').upper()
                    w['status'] = open_flag != 'SHUT'
                    ctrl = _clean_token(_get_item(row, 2 if kw == 'WCONPROD' else 3), default='RATE').upper()
                    if kw == 'WCONPROD':
                        prod_values = {
                            'orat': -_safe_float(_get_item(row, 3), _np.inf) * liquid_rate_scale,
                            'wrat': -_safe_float(_get_item(row, 4), _np.inf) * liquid_rate_scale,
                            'grat': -_safe_float(_get_item(row, 5), _np.inf) * gas_rate_scale,
                            'lrat': -_safe_float(_get_item(row, 6), _np.inf) * liquid_rate_scale,
                            'resv': -_safe_float(_get_item(row, 7), _np.inf) * liquid_rate_scale,
                            'bhp': _safe_float(_get_item(row, 8), _np.nan) * pressure_scale,
                            'thp': _safe_float(_get_item(row, 9), 0.0) * pressure_scale,
                        }
                        # Port insertDefaultWCONPROD: an omitted BHP becomes
                        # one atmosphere after deck-unit conversion.
                        if not _np.isfinite(prod_values['bhp']):
                            prod_values['bhp'] = 101325.0
                        value = prod_values.get(ctrl.lower(), 0.0)
                        w['sign'] = -1
                        w['phase'] = 'OIL'
                        w['control'] = ctrl
                        w['type'] = ctrl.lower()
                        w['compi'] = _wconprod_compi(ctrl)
                        w['lims'] = {key: prod_values[key] for key in ('orat', 'wrat', 'grat', 'lrat', 'bhp', 'thp')}
                    else:
                        phase_value = _clean_token(_get_item(row, 1), default='WATER').upper()
                        rate_scale = gas_rate_scale if phase_value.startswith('G') else liquid_rate_scale
                        inj_values = {
                            'rate': _safe_float(_get_item(row, 4), _np.inf) * rate_scale,
                            'resv': _safe_float(_get_item(row, 5), _np.inf) * liquid_rate_scale,
                            'bhp': _safe_float(_get_item(row, 6), _np.nan) * pressure_scale,
                            'thp': _safe_float(_get_item(row, 7), _np.inf) * pressure_scale,
                        }
                        # convertWConInje assigns 100000 psi to a defaulted
                        # BHP before processWells stores the limits.
                        if not _np.isfinite(inj_values['bhp']):
                            inj_values['bhp'] = 100000.0 * pressure_scale
                        value = inj_values.get(ctrl.lower(), 0.0)
                        w['sign'] = 1
                        w['phase'] = phase_value
                        w['control'] = ctrl
                        w['type'] = ctrl.lower()
                        phase_initial = w['phase'][:1]
                        w['compi'] = [1.0, 0.0, 0.0] if phase_initial == 'W' else ([0.0, 1.0, 0.0] if phase_initial == 'O' else [0.0, 0.0, 1.0])
                        w['lims'] = {key: inj_values[key] for key in ('rate', 'bhp', 'thp')}
                    w['val'] = float(value)
                    control_changed = True
        elif kw == 'WCONHIST':
            # Port of ``processWells.m``'s ``process_wconhist``. The record
            # is [name, status, control, orat, wrat, grat, VFP, ALQ, THP,
            # BHP, wet-gas, NGL]; the rates are *observed*, so the control
            # target is the history itself.
            for row in recs:
                if not row:
                    continue
                row = _expand_defaulted_record(row)
                wspec = _clean_token(row[0])
                status = _clean_token(_get_item(row, 1),
                                      default='OPEN').upper() == 'OPEN'
                ctrl = _clean_token(_get_item(row, 2),
                                    default='ORAT').upper().lower()
                orat = _safe_float(_get_item(row, 3), 0.0) * liquid_rate_scale
                wrat = _safe_float(_get_item(row, 4), 0.0) * liquid_rate_scale
                grat = _safe_float(_get_item(row, 5), 0.0) * gas_rate_scale
                bhp = _safe_float(_get_item(row, 9), _np.nan) * pressure_scale

                value, compi, ctrl_out = _wconhist_target(ctrl, orat, wrat,
                                                          grat, bhp)
                for wname in _match_wells(wells, wspec):
                    w = wells[wname]
                    # The *observed* rates, written onto the well itself
                    # regardless of which mode controls it -- a well on
                    # LRAT still reports its oil and water separately,
                    # and a match needs all three. This is MRST-0's
                    # addition ("write observed data to schedule"); stock
                    # MRST leaves them off, and getObservedFromSchedule,
                    # which reads exactly these fields, then returns a
                    # container of None.
                    w['qOs'] = -orat
                    w['qWs'] = -wrat
                    w['qGs'] = -grat
                    w['bhp'] = float(bhp) if _np.isfinite(bhp) else 101325.0
                    w['status'] = bool(status)
                    w['sign'] = -1
                    w['phase'] = 'OIL'
                    w['control'] = ctrl_out.upper()
                    w['type'] = ctrl_out
                    w['compi'] = compi
                    w['val'] = float(value)
                    # processWells opens every limit and closes only the
                    # one being controlled; the bhp floor is one atmosphere.
                    lims = {k: -_np.inf for k in ('orat', 'wrat', 'grat',
                                                  'lrat', 'resv')}
                    lims['bhp'] = 101325.0
                    if ctrl in ('orat', 'wrat', 'grat', 'lrat'):
                        lims[ctrl] = float(value)
                    elif ctrl == 'bhp' and _np.isfinite(bhp):
                        lims['bhp'] = float(bhp)
                    w['lims'] = lims
                    control_changed = True
        elif kw == 'WCONINJH':
            # ``process_wconinjh``: a history injector is always rate
            # controlled, at the observed injection rate.
            for row in recs:
                if not row:
                    continue
                row = _expand_defaulted_record(row)
                wspec = _clean_token(row[0])
                phase = _clean_token(_get_item(row, 1),
                                     default='WATER').upper()
                status = _clean_token(_get_item(row, 2),
                                      default='OPEN').upper() == 'OPEN'
                scale = gas_rate_scale if phase[:1] == 'G' \
                    else liquid_rate_scale
                rate = _safe_float(_get_item(row, 3), 0.0) * scale
                compi = {'W': [1.0, 0.0, 0.0], 'O': [0.0, 1.0, 0.0],
                         'G': [0.0, 0.0, 1.0]}.get(phase[:1])
                if compi is None:
                    continue
                bhp_obs = _safe_float(_get_item(row, 4), _np.nan) \
                    * pressure_scale
                for wname in _match_wells(wells, wspec):
                    w = wells[wname]
                    # As above: the observed injection rate lands on the
                    # phase being injected, the other two stay zero.
                    w['qWs'] = float(rate) if phase[:1] == 'W' else 0.0
                    w['qOs'] = float(rate) if phase[:1] == 'O' else 0.0
                    w['qGs'] = float(rate) if phase[:1] == 'G' else 0.0
                    w['bhp'] = float(bhp_obs) if _np.isfinite(bhp_obs) \
                        else 101325.0
                    w['status'] = bool(status)
                    w['sign'] = 1
                    w['phase'] = phase
                    w['control'] = 'RATE'
                    w['type'] = 'rate'
                    w['compi'] = compi
                    w['val'] = float(rate)
                    # 6895 bar, the same near-open ceiling a defaulted
                    # WCONINJE gets.
                    w['lims'] = {'rate': float(rate), 'bhp': 6895e5,
                                 'thp': _np.inf}
                    control_changed = True
        elif kw in ('WPOLYMER', 'WSURFACT'):
            # ad_eor.models.{OilWaterPolymerModel,OilWaterSurfactantModel}
            # read w['polymer']/w['surfactant'] as each well's injection
            # concentration (see equationsOilWaterPolymer._polymer_well_source
            # / equationsOilWaterSurfactant._surfactant_well_source).
            field = 'polymer' if kw == 'WPOLYMER' else 'surfactant'
            for row in recs:
                if not row:
                    continue
                wspec = _clean_token(row[0])
                conc = _safe_float(_get_item(row, 1), 0.0)
                for wname in _match_wells(wells, wspec):
                    wells[wname][field] = conc
                control_changed = True
        elif kw in ('TSTEP', 'DATES'):
            tvals = []
            if kw == 'TSTEP':
                for row in recs:
                    tvals.extend(_expand_tstep_tokens(row))
                tvals = [float(value) * time_scale for value in tvals]
            else:
                # Port readDATES.m: every listed calendar date is converted
                # to its positive increment from START plus all prior steps.
                if start_date is None:
                    raise ValueError('DATES requires a valid RUNSPEC START date')
                elapsed = float(sum(steps))
                for row in recs:
                    date = _parse_eclipse_date(row)
                    if date is None:
                        continue
                    dt = (date - start_date).total_seconds() - elapsed
                    if dt <= 0.0:
                        raise ValueError('DATES must be strictly increasing from RUNSPEC START')
                    tvals.append(dt)
                    elapsed += dt
            if not tvals:
                continue
            # In MRST, TSTEP/DATES adds report steps.  It does not create a new
            # control unless a preceding schedule keyword changed a well or
            # other control.  SPE1 therefore has 120 steps and one control.
            if not controls or control_changed:
                # Snapshot the deck state at this report-step boundary.
                # Subsequent WCON/COMPDAT records must not retroactively
                # modify an earlier MRST schedule.control entry.
                controls.append({'W': _deepcopy(wells)})
                control_changed = False
            control_id = len(controls) - 1
            steps.extend([float(value) for value in tvals])
            control_ix.extend([control_id] * len(tvals))

    if not steps:
        # Fallback if order reconstruction failed.
        for rec in sched.get('TSTEP', []):
            steps.extend([float(value) * time_scale for value in _expand_tstep_tokens(rec)])
    if not steps:
        steps = [1.0]

    if not controls:
        controls = [{'W': _deepcopy(wells)}]

    # Process well completions BEFORE converting well names to schedule entries
    if G is not None and rock is not None:
        _process_well_completions(wells, G, rock)
        # Convert the per-control snapshots to full schedule entries.
        for ctrl in controls:
            ctrl_wells = ctrl['W'] if isinstance(ctrl['W'], dict) else {}
            _process_well_completions(ctrl_wells, G, rock)
            # ``processWells`` returns an MRST well array ordered by well
            # name.  Python's insertion-order dict preserved the order in
            # the deck (``PRODU2`` before ``PRODU10``), which is different
            # from MRST's lexical ordering.  The reservoir source is
            # commutative so the distinction is invisible in the cell
            # residuals, but it permutes every facility equation/unknown
            # and therefore breaks the Jacobian and Newton update.
            ctrl['W'] = [_well_to_schedule_entry(w) for w in sorted(
                ctrl_wells.values(), key=lambda item: str(item.get('name', ''))
            )]
    else:
        # Still need to convert well names to entries even without completion processing
        for ctrl in controls:
            well_entries = []
            raw_wells = ctrl['W'].values() if isinstance(ctrl['W'], dict) else []
            for w in sorted(raw_wells, key=lambda item: str(item.get('name', ''))):
                well_entries.append(_well_to_schedule_entry(w))
            ctrl['W'] = well_entries

    if not control_ix:
        control_ix = [0] * len(steps)

    # ``G = processGroups(ctrl); controlMRST(i).G = G`` -- MRST-0 attaches
    # each control's group targets (GCONINJE/GCONPROD) beside its wells,
    # and drops the field entirely when every control's is empty:
    # ``if isempty(vertcat(controlMRST.G)), rmfield(...); end``. 2026a has
    # no group handling at all, and following it left ``processGroups``
    # ported with nothing calling it -- a deck's group targets reached the
    # model nowhere.
    _attach_group_controls(controls, sched)

    schedule = {
        'step': {'val': _np.asarray(steps, dtype=float), 'control': _np.asarray(control_ix, dtype=int)},
        'control': controls,
    }
    return schedule


def _attach_group_controls(controls, sched):
    """Give each MRST control its ``G``, from the deck control of the same
    index. Silent when the deck has no control structure or the two do not
    line up: a wrong group target is worse than none."""
    deck_controls = sched.get('control') if isinstance(sched, dict) else None
    if not isinstance(deck_controls, list) or len(deck_controls) != len(controls):
        return

    from PRSTCore.hm.utils.processGroups import processGroups

    groups = [processGroups(deck) for deck in deck_controls]
    if not any(groups):
        return
    for ctrl, group in zip(controls, groups):
        ctrl['G'] = group


def _well_to_schedule_entry(w):
    """Convert internal well dict to schedule control entry, preserving cells/WI."""
    wname = w.get('name', 'UNKNOWN')
    # Explicit copy of all fields to ensure cells/WI lists are preserved
    entry = {
        'name': wname,
        'group': w.get('group', 'FIELD'),
        'status': w.get('status', True),
        'type': w.get('type', 'rate'),
        'val': w.get('val', 0.0),
        'sign': w.get('sign', -1),
        'phase': w.get('phase', 'OIL'),
        'control': w.get('control', 'RATE'),
        'refDepth': w.get('refDepth', 0.0),
        'radius': w.get('radius', 0.0),
        'i': w.get('i', 1),
        'j': w.get('j', 1),
        'k': w.get('k', []),
        'cells': w['cells'] if 'cells' in w else [],  # Direct access, not get()
        'WI': w['WI'] if 'WI' in w else [],            # Direct access
    }
    # Copy any additional fields
    for key in w:
        if key not in entry:
            entry[key] = w[key]
    return entry


#: Port of ``process_wconprod``'s composition table (MRST-0,
#: model-io/deckformat/params/wells_and_bc/processWells.m).
#:
#: A producer's ``compi`` marks *which phases the control targets*, and
#: MRST deliberately leaves it unnormalised: LRAT gives [1, 1, 0] rather
#: than two halves, and RESV/BHP/THP give [1, 1, 1] with the comment
#: "Doesn't matter". Treating it as a fraction would be a different
#: quantity.
_WCONPROD_COMPI = {
    'orat': [0.0, 1.0, 0.0],
    'wrat': [1.0, 0.0, 0.0],
    'grat': [0.0, 0.0, 1.0],
    'lrat': [1.0, 1.0, 0.0],
    'resv': [1.0, 1.0, 1.0],
    'bhp':  [1.0, 1.0, 1.0],
    'thp':  [1.0, 1.0, 1.0],
}


def _wconprod_compi(control):
    """The composition MRST assigns a producer from its control mode.

    PRSTCore previously gave every producer a uniform [1/3, 1/3, 1/3] on
    the grounds that WCONPROD does not redefine it. MRST does redefine
    it, per control type, and the difference shows up directly in the
    well source terms -- comparing SPE1 against MRST, the producer's oil
    and gas equations were the only reservoir rows that disagreed.
    """
    return list(_WCONPROD_COMPI.get(str(control).lower(),
                                    [1.0, 1.0, 1.0]))


def _wconhist_target(control, orat, wrat, grat, bhp):
    """Port of ``process_wconhist``'s control switch.

    Returns ``(value, compi, type)``. The observed rates are the target,
    negated because a producer's rates are negative in MRST's sign
    convention.

    .. warning::
       **A defect reproduced from MRST.** Under ``LRAT`` the composition
       is built as ``[rates, 0]/val`` from ``rates = -[orat, wrat]``,
       which puts the *oil* rate in the water slot and the water rate in
       the oil slot -- MRST's compi is ordered water, oil, gas while the
       deck's rates are oil, water, gas. The ``RESV`` branch two cases
       below does swap them, with the comment "Account for OWG ordering.
       MRST uses WOG", so the omission under LRAT is an oversight rather
       than a convention. It runs and produces a plausible number, which
       is why it survives. Reproduced here so a deck matched against
       MRST agrees with MRST; the swap is one line away if you ever want
       to depart from it.
    """
    import numpy as _np

    control = str(control).lower()
    if control == 'orat':
        return -orat, [0.0, 1.0, 0.0], 'orat'
    if control == 'wrat':
        return -wrat, [1.0, 0.0, 0.0], 'wrat'
    if control == 'grat':
        return -grat, [0.0, 0.0, 1.0], 'grat'
    if control == 'lrat':
        rates = [-orat, -wrat]
        value = rates[0] + rates[1]
        if value != 0.0:
            compi = [rates[0] / value, rates[1] / value, 0.0]
        else:
            compi = [0.5, 0.5, 0.0]
        return value, compi, 'lrat'
    if control == 'resv':
        rates = [-orat, -wrat, -grat]
        value = sum(rates)
        if value != 0.0:
            # OWG in the deck, WOG in the model.
            compi = [rates[1] / value, rates[0] / value, rates[2] / value]
        else:
            compi = [1.0 / 3.0] * 3
        return value, compi, 'resv_history'
    if control == 'bhp':
        return (bhp if _np.isfinite(bhp) else 101325.0), [0.0, 1.0, 0.0], \
            'bhp'
    # An unsupported mode is ignored rather than guessed at, as MRST does.
    return 0.0, [0.0, 1.0, 0.0], control


def _process_well_completions_legacy(wells, G, rock):
    """Compute cells and WI from well completions using Peaceman formula."""
    import numpy as _np
    
    # Get Cartesian dimensions and build (i,j,k) -> global cell mapping
    if not isinstance(G, dict) or 'cartDims' not in G:
        return
    cart_dims = G['cartDims']
    if len(cart_dims) < 3:
        return
    nx, ny, nz = int(cart_dims[0]), int(cart_dims[1]), int(cart_dims[2])
    
    # Build active cell mapping from Cartesian to global
    if 'cells' in G and 'indexMap' in G['cells']:
        index_map = _np.asarray(G['cells']['indexMap'], dtype=int).ravel()
    else:
        index_map = _np.arange(nx * ny * nz, dtype=int)
    
    # Get permeability for WI calculation
    if isinstance(rock, dict) and 'perm' in rock:
        perm = _np.asarray(rock['perm'], dtype=float)
        if perm.ndim == 2 and perm.shape[1] >= 2:
            perm_x = perm[:, 0]
            perm_y = perm[:, 1]
        else:
            perm_x = perm.ravel()
            perm_y = perm.ravel()
    else:
        nc = len(index_map) if 'cells' in G and 'num' in G['cells'] else nx * ny * nz
        perm_x = _np.ones(nc, dtype=float) * 100e-15  # 100 mD default
        perm_y = perm_x.copy()
    
    # Get cell volumes/heights for Peaceman re calculation
    if 'cell_volumes' in G:
        vols = _np.asarray(G['cell_volumes'], dtype=float).ravel()
    else:
        vols = _np.ones(len(index_map), dtype=float)
    
    for wname, w in wells.items():
        if 'completions' not in w:
            continue
        
        cells_list = []
        wi_list = []
        
        for comp in w['completions']:
            if not comp.get('status', True):
                continue
            i, j, k_range = comp['i'], comp['j'], comp['k']
            wi_deck, diam, kh_deck, skin = comp['wi'], comp['diam'], comp['kh'], comp['skin']
            
            for k in k_range:
                # Convert (i,j,k) to Cartesian index (1-indexed to 0-indexed)
                if i < 1 or j < 1 or k < 1 or i > nx or j > ny or k > nz:
                    continue
                cart_idx = (k - 1) * nx * ny + (j - 1) * nx + (i - 1)
                if cart_idx >= len(index_map):
                    continue
                global_cell = int(index_map[cart_idx])
                if global_cell < 0:
                    continue  # inactive cell
                
                cells_list.append(global_cell)
                
                # Compute WI using Peaceman formula if not provided
                if wi_deck > 0:
                    wi_list.append(wi_deck)
                else:
                    # Peaceman WI = 2π√(kx*ky)*h / (ln(re/rw) + skin)
                    # Simplified: use sqrt(kx*ky), approximate h from volume, re ≈ 0.2*sqrt(dx*dy)
                    if global_cell < len(perm_x):
                        kx = float(perm_x[global_cell])
                        ky = float(perm_y[global_cell]) if global_cell < len(perm_y) else kx
                        k_eff = _np.sqrt(kx * ky)
                        vol = float(vols[global_cell]) if global_cell < len(vols) else 1000.0
                        h = vol ** (1.0 / 3.0)  # Crude height estimate
                        rw = diam / 2.0
                        re = 0.2 * h  # Peaceman equivalent radius approximation
                        if re > rw and k_eff > 0:
                            wi = 2.0 * _np.pi * k_eff * h / (_np.log(re / rw) + skin)
                        else:
                            wi = 1e-12  # fallback
                    else:
                        wi = 1e-12
                    wi_list.append(max(wi, 1e-15))
        
        # Explicitly assign to wells dict (w is a reference, but be explicit)
        wells[wname]['cells'] = cells_list
        wells[wname]['WI'] = wi_list


def _restrict_rock_to_grid(rock, G, deck):
    """Cut the deck's per-cell rock arrays down to the grid's own cells.

    The deck stores porosity, permeability and net-to-gross over the whole
    logical Cartesian box; the grid keeps a subset of it.  Which subset is
    not always ACTNUM: dropping cells that can hold nothing (``MINPV``, or a
    zero PORV) removes more, and the grid records what survived in
    ``cells.indexMap``.

    Selecting by ACTNUM instead, as this used to, is right only while those
    two agree.  When they do not the rock comes out longer than the grid and
    the first thing that multiplies them -- pore volume -- fails with two
    lengths and no indication of which is wrong.  Reading the answer off the
    grid keeps them in step by construction.
    """
    import numpy as _np

    cells = G.get('cells', {}) if isinstance(G, dict) else {}
    ncells = int(cells.get('num', 0))
    index_map = cells.get('indexMap')
    if index_map is None:
        # No map: the grid is the whole box, so there is nothing to cut.
        return
    index_map = _np.asarray(index_map, dtype=int).ravel()

    actnum = deck.get('GRID', {}).get('ACTNUM')
    active = (_np.asarray(actnum).ravel().astype(bool)
              if actnum is not None else None)
    ncart = active.size if active is not None else None

    for key in ('poro', 'perm', 'ntg'):
        if key not in rock:
            continue
        arr = _np.asarray(rock[key])
        if arr.shape[0] == ncells:
            continue                       # already the grid's own cells
        if ncart is not None and arr.shape[0] == ncart:
            rock[key] = arr[index_map]     # full box -> this grid
        elif active is not None and arr.shape[0] == int(active.sum()):
            # Already ACTNUM-restricted, and the grid is a subset of that:
            # translate the surviving global indices into positions within
            # the ACTNUM ordering.
            position = _np.full(active.size, -1, dtype=int)
            position[_np.flatnonzero(active)] = _np.arange(int(active.sum()))
            take = position[index_map]
            if _np.any(take < 0):
                raise ValueError(
                    'grid cell outside ACTNUM while restricting rock.%s' % key)
            rock[key] = arr[take]


def _cell_bounding_box_dims(G, nc):
    """Fallback for ``G.cells.dimensions`` (never populated by any grid
    constructor in this codebase, see ``_process_well_completions``): a
    per-cell axis-aligned bounding-box ``[dx, dy, dz]`` computed directly
    from node coordinates via the cell -> face -> node connectivity every
    grid (Cartesian, corner-point, or an assembled NWM hybrid grid) already
    carries. This is an approximation for a non-Cartesian cell (a
    corner-point or hybrid-grid cell's true logical width isn't always its
    node bounding box), but reduces to the exact Cartesian dx/dy/dz for an
    axis-aligned grid, which is what MRST's own Peaceman formula assumes
    regardless of grid type -- COMPDAT decks needing this fallback (WI
    omitted, requiring computation) are themselves usually Cartesian-ish.
    """
    import numpy as _np
    cells = G.get('cells', {})
    faces = G.get('faces', {})
    nodes = G.get('nodes', {})
    face_pos = cells.get('facePos')
    cell_faces = cells.get('faces')
    node_pos = faces.get('nodePos')
    face_nodes = faces.get('nodes')
    coords = nodes.get('coords')
    if (face_pos is None or cell_faces is None or node_pos is None
            or face_nodes is None or coords is None):
        raise ValueError(
            'MRST-compatible well indexing requires per-cell dimensions, and '
            'this grid lacks the face/node connectivity needed to derive '
            'them from a bounding box (cells.facePos/cells.faces/'
            'faces.nodePos/faces.nodes/nodes.coords)')
    face_pos = _np.asarray(face_pos, dtype=_np.int64).ravel()
    cell_faces = _np.asarray(cell_faces, dtype=_np.int64)
    cf0 = cell_faces[:, 0] if cell_faces.ndim == 2 else cell_faces
    node_pos = _np.asarray(node_pos, dtype=_np.int64).ravel()
    face_nodes = _np.asarray(face_nodes, dtype=_np.int64).ravel()
    coords = _np.asarray(coords, dtype=float)

    # One pass over the whole cell -> face -> node connectivity, rather than
    # a Python loop that sliced and concatenated per cell.  On Norne that
    # loop ran 44927 times per call and called ``numpy.unique`` once per
    # cell; the caller runs once per schedule control step, so the grid's
    # bounding boxes were recomputed 248 times over and ``unique`` was
    # entered eleven million times -- 198 of the 225 seconds a Norne set-up
    # took.  The ``unique`` was not needed at all: a repeated node cannot
    # move a minimum or a maximum.
    faces_per_cell = _np.diff(face_pos)
    cell_of_face = _np.repeat(_np.arange(nc, dtype=_np.int64), faces_per_cell)
    nodes_per_face = _np.diff(node_pos)[cf0]

    # Expand each face's slice of ``face_nodes`` without a Python loop: the
    # k-th entry of face f sits at node_pos[f] + k, and ``arange`` minus the
    # running offset gives k for every entry at once.
    total = int(nodes_per_face.sum())
    dims = _np.zeros((nc, 3), dtype=float)
    if total == 0:
        return dims
    face_starts = node_pos[cf0]
    running = _np.zeros(nodes_per_face.size, dtype=_np.int64)
    _np.cumsum(nodes_per_face[:-1], out=running[1:])
    positions = _np.arange(total, dtype=_np.int64) + _np.repeat(
        face_starts - running, nodes_per_face)
    flat_nodes = face_nodes[positions]
    flat_cells = _np.repeat(cell_of_face, nodes_per_face)

    nodes_per_cell = _np.bincount(flat_cells, minlength=nc)
    nonempty = nodes_per_cell > 0
    if not _np.any(nonempty):
        return dims
    # ``flat_cells`` is non-decreasing because ``face_pos`` is, so each
    # cell's entries are contiguous and ``reduceat`` can take the extremes
    # segment by segment.  Cells with no faces are dropped first: reduceat
    # given two equal offsets does not produce an empty reduction, it
    # produces the element at that offset.
    starts = _np.zeros(nc, dtype=_np.int64)
    _np.cumsum(nodes_per_cell[:-1], out=starts[1:])
    points = coords[flat_nodes]
    lo = _np.minimum.reduceat(points, starts[nonempty], axis=0)
    hi = _np.maximum.reduceat(points, starts[nonempty], axis=0)
    dims[nonempty] = _np.maximum(hi - lo, 1.0e-6)
    return dims


def _process_well_completions(wells, G, rock):
    """Port MRST's ``processWells -> addWell -> computeWellIndex`` path.

    The Cartesian mapping is MATLAB's ``sub2ind(G.cartDims, i, j, k)`` and
    the productivity-index expression below is the TPFA branch of MRST
    ``core/utils/computeWellIndex.m``.
    """
    import numpy as _np

    if not isinstance(G, dict) or 'cartDims' not in G:
        return
    cart_dims = G['cartDims']
    if len(cart_dims) < 3:
        return
    nx, ny, nz = (int(cart_dims[0]), int(cart_dims[1]), int(cart_dims[2]))
    ncart = nx * ny * nz

    # make_cart_to_active(G), with Python's zero-based active-cell ids.
    c2a = G.get('cart_to_active', None)
    if c2a is None:
        index_map = _np.asarray(
            G.get('cells', {}).get('indexMap', _np.arange(ncart)), dtype=int
        ).ravel()
        c2a = _np.full(ncart, -1, dtype=int)
        c2a[index_map] = _np.arange(index_map.size, dtype=int)
    else:
        c2a = _np.asarray(c2a, dtype=int).ravel()
    if c2a.size != ncart:
        raise ValueError('cart_to_active length does not match G.cartDims')

    nc = int(G.get('cells', {}).get('num', _np.count_nonzero(c2a >= 0)))
    perm = _np.asarray(rock.get('perm'), dtype=float) if isinstance(rock, dict) and 'perm' in rock else None
    if perm is None:
        perm = _np.full((nc, 3), 100.0e-15, dtype=float)
    elif perm.ndim == 1:
        perm = _np.column_stack((perm, perm, perm))
    elif perm.shape[1] == 1:
        perm = _np.repeat(perm, 3, axis=1)
    elif perm.shape[1] == 2:
        perm = _np.column_stack((perm, perm[:, 1]))

    cell_dims = G.get('cells', {}).get('dimensions', G.get('cell_dimensions', None))
    if cell_dims is None:
        # Cache it where the lookup above will find it next time.  This
        # routine runs once per schedule control step -- 248 times on Norne
        # -- and the bounding boxes are a function of the grid alone, so
        # recomputing them per call was 247 repetitions of the same answer.
        cell_dims = _cell_bounding_box_dims(G, nc)
        cells_group = G.get('cells')
        if isinstance(cells_group, dict):
            cells_group['dimensions'] = cell_dims
        else:
            G['cell_dimensions'] = cell_dims
    cell_dims = _np.asarray(cell_dims, dtype=float)
    if cell_dims.shape != (nc, 3):
        raise ValueError('cell dimensions must have one [dx, dy, dz] row per active cell')
    centroids = _np.asarray(G.get('cells', {}).get('centroids', _np.zeros((nc, 3))), dtype=float)

    for wname, w in wells.items():
        if 'completions' not in w:
            continue

        cells_list, wi_list, radii, directions = [], [], [], []
        # ``processWells``'s ``W.defaulted`` (MRST-0, `% edited by zhang`):
        # the deck's own per-connection Kh/WI/Skin, with -1 marking an item
        # the deck left out.  recomputeWellIndex reads these to decide which
        # connections it may recompute after permeability is perturbed --
        # ``WI <= 0 & Kh <= 0 & cstatus`` -- so a well without them cannot
        # take part in a history match at all.
        kh_raw, wi_raw, skin_raw = [], [], []
        for comp in w['completions']:
            if not comp.get('status', True):
                continue
            i, j, k_range = comp['i'], comp['j'], comp['k']
            wi_deck = float(comp['wi'])
            diameter = float(comp['diam'])
            kh_deck = float(comp['kh'])
            skin = float(comp['skin'])
            direction = str(comp.get('dir', 'Z')).lower()[:1]
            if direction not in ('x', 'y', 'z'):
                raise ValueError('Unsupported COMPDAT well direction %r' % direction)
            # processWells replaces a non-positive diameter with 1 ft before
            # calling addWell.  Conversion has already happened here.
            if not diameter > 0.0:
                diameter = 0.3048
            radius = diameter / 2.0

            for k in k_range:
                if i < 1 or j < 1 or k < 1 or i > nx or j > ny or k > nz:
                    continue
                cart_idx = (i - 1) + nx * (j - 1) + nx * ny * (k - 1)
                active_cell = int(c2a[cart_idx])
                if active_cell < 0:
                    continue

                cells_list.append(active_cell)
                radii.append(radius)
                directions.append(direction)
                # ``Kh(defaultedKh) = -1; WI(defaultedWI) = -1``.
                kh_raw.append(kh_deck if kh_deck > 0.0 else -1.0)
                wi_raw.append(wi_deck if wi_deck > 0.0 else -1.0)
                skin_raw.append(skin)
                if wi_deck > 0.0:
                    wi_list.append(wi_deck)
                    continue

                # ``connection_dimensions`` for G.griddim == 3, then the
                # ``wellConstant(..., 'ip_tpf') == 0.14`` Peaceman formula.
                dx, dy, dz = cell_dims[active_cell]
                kx, ky, kz = perm[active_cell, :3]
                if direction == 'x':
                    d1, d2, ell, k1, k2 = dy, dz, dx, ky, kz
                elif direction == 'y':
                    d1, d2, ell, k1, k2 = dx, dz, dy, kx, kz
                else:
                    d1, d2, ell, k1, k2 = dx, dy, dz, kx, ky
                if min(d1, d2, ell, k1, k2) <= 0.0:
                    raise ValueError('MRST computeWellIndex requires positive cell dimensions and permeability')
                k21, k12 = k2 / k1, k1 / k2
                wc = 0.14
                re1 = 2.0 * wc * _np.sqrt(d1 * d1 * _np.sqrt(k21) + d2 * d2 * _np.sqrt(k12))
                re2 = _np.power(k21, 0.25) + _np.power(k12, 0.25)
                re = re1 / re2
                kh = kh_deck if kh_deck >= 0.0 else ell * _np.sqrt(k1 * k2)
                wi = 2.0 * _np.pi * kh / (_np.log(re / radius) + skin)
                if wi < 0.0:
                    raise ValueError('MRST computeWellIndex produced a negative well index')
                wi_list.append(wi)

        # ``processWells.m`` removes repeated perforations with
        # ``unique(perf, 'last')`` and then restores deck order by sorting
        # the retained original indices.  This is material for Norne's
        # repeated COMPDAT updates: all connection-sized fields must be
        # filtered together, not only ``cells``.
        last = {}
        for idx, cell in enumerate(cells_list):
            last[int(cell)] = idx
        keep = sorted(last.values())
        cells_list = [cells_list[i] for i in keep]
        wi_list = [wi_list[i] for i in keep]
        radii = [radii[i] for i in keep]
        directions = [directions[i] for i in keep]
        kh_raw = [kh_raw[i] for i in keep]
        wi_raw = [wi_raw[i] for i in keep]
        skin_raw = [skin_raw[i] for i in keep]

        # Match addWell's fields.  ``radius`` is retained as the existing
        # Python compatibility alias; MRST's actual field is ``r``.
        wells[wname]['cells'] = cells_list
        wells[wname]['WI'] = wi_list
        wells[wname]['r'] = radii
        wells[wname]['dir'] = directions
        wells[wname]['radius'] = radii[0] if radii else w.get('radius', 0.0)
        wells[wname]['cstatus'] = [True] * len(cells_list)
        # Keep whatever ``defaulted`` already holds -- WELSPECS' refDepth
        # flag is set when the well is declared, long before its
        # completions are known.
        defaulted = dict(wells[wname].get('defaulted') or {})
        defaulted.update({'Kh': _np.asarray(kh_raw, dtype=float),
                          'WI': _np.asarray(wi_raw, dtype=float),
                          'Skin': _np.asarray(skin_raw, dtype=float)})
        wells[wname]['defaulted'] = defaulted
        if len(cells_list) and centroids.shape[0] >= nc:
            # ``makeScheduleConsistent.setReferenceDepths``: a defaulted
            # WELSPECS depth is the depth of the first perforation, after
            # which every dZ is measured from that same reference depth.
            ref_depth = float(w.get('refDepth', _np.nan))
            is_defaulted = bool(w.get('defaulted', {}).get('refDepth', False))
            if is_defaulted or not _np.isfinite(ref_depth):
                ref_depth = float(centroids[cells_list[0], 2])
                wells[wname]['refDepth'] = ref_depth
            wells[wname]['dZ'] = [float(centroids[c, 2] - ref_depth) for c in cells_list]
        else:
            wells[wname]['dZ'] = []


def _consume_schedule_keyword_group(records, start_idx):
    n = len(records)
    i = start_idx
    while i < n and (records[i] is None or len(records[i]) == 0):
        i += 1
    out = []
    while i < n and records[i] is not None and len(records[i]) > 0:
        out.append(records[i])
        i += 1
    return out, i


def _is_defaulted_item(tok):
    """Whether a raw Eclipse item denotes an omitted default value.

    The schedule reader keeps this distinction for WELSPECS because a
    defaulted reference depth is resolved only after the well completions
    have been constructed (MRST ``makeScheduleConsistent``).
    """
    if tok is None:
        return True
    text = _clean_token(tok).strip()
    if not text:
        return True
    return text.endswith('*') and text[:-1].isdigit()


def _expand_defaulted_record(row):
    """Expand Eclipse ``n*`` defaults to positional ``None`` fields.

    This is the record-level behaviour of MRST's ``readDefaultedRecord``
    and ``readDefaultedKW``.  Only COMPDAT presently consumes the expanded
    representation, where field position is material to well geometry.
    """
    import re as _re
    expanded = []
    for item in row:
        token = str(item).strip()
        match = _re.fullmatch(r'(\d+)\*', token)
        if match:
            expanded.extend([None] * int(match.group(1)))
        else:
            expanded.append(item)
    return expanded


def _clean_token(tok, default=''):
    if tok is None:
        return default
    if isinstance(tok, (float, int)):
        return str(tok)
    s = str(tok).strip()
    if s.startswith("'") and s.endswith("'") and len(s) >= 2:
        s = s[1:-1]
    return s if s else default


def _safe_float(tok, default=0.0):
    if tok is None:
        return float(default)
    try:
        return float(tok)
    except Exception:
        # The lightweight tokenizer can retain a record slash on the final
        # compact item (for example ``395/`` in EGG WCONPROD).  MRST's
        # readDefaultedKW has already consumed that delimiter before its
        # positional numeric conversion.
        s = _clean_token(tok, default='').rstrip('/')
        if '*' in s:
            return float(default)
        try:
            return float(s)
        except Exception:
            return float(default)


def _safe_int(tok, default=0):
    return int(round(_safe_float(tok, default=default)))


def _get_item(row, idx, default=None):
    return row[idx] if idx < len(row) else default


def _expand_tstep_tokens(vals):
    out = []
    for v in vals:
        # readTSTEP.m reads the final value before a record slash as an
        # ordinary number.  The lightweight schedule tokenizer can retain
        # that slash on a compact token such as ``10/``.
        s = _clean_token(v).rstrip('/')
        if not s:
            continue
        if '*' in s:
            parts = s.split('*', 1)
            try:
                n = int(float(parts[0]))
                x = float(parts[1])
                out.extend([x] * max(n, 0))
                continue
            except Exception:
                pass
        try:
            out.append(float(s))
        except Exception:
            continue
    return out


def _parse_eclipse_date(value):
    """Parse the date record forms accepted by MRST ``readDATES``.

    Deck records are normally ``DD 'MON' YYYY`` with an optional time
    field.  ``readDATES.m`` first removes quotes, replaces JLY by JUL and
    passes the text to MATLAB's datenum; this parser implements that same
    subset in Python's calendar representation.
    """
    if isinstance(value, (list, tuple)):
        tokens = [_clean_token(item, default='') for item in value]
        text = ' '.join(token for token in tokens if token)
    else:
        text = _clean_token(value, default='')
    text = text.replace('JLY', 'JUL').replace('jly', 'jul')
    tokens = text.replace("'", '').split()
    if len(tokens) < 3:
        return None
    try:
        day = int(float(tokens[0]))
        month = tokens[1].upper()[:3]
        year = int(float(tokens[2]))
        base = _datetime.strptime(f'{day:02d} {month} {year:04d}', '%d %b %Y')
        if len(tokens) >= 4:
            time_text = tokens[3]
            for fmt in ('%H:%M:%S.%f', '%H:%M:%S'):
                try:
                    parsed = _datetime.strptime(time_text, fmt)
                    return base.replace(hour=parsed.hour, minute=parsed.minute,
                                        second=parsed.second, microsecond=parsed.microsecond)
                except ValueError:
                    continue
        return base
    except ValueError:
        return None


def _first_numeric(vals, default=0.0):
    for v in vals:
        s = _clean_token(v)
        if s.endswith('*'):
            continue
        try:
            return float(s)
        except Exception:
            continue
    return float(default)


def _extract_wconprod_target(row, ctrl):
    # ECLIPSE WCONPROD conventional order:
    # [well, status, control, ORAT, WRAT, GRAT, LRAT, RESV, BHP, THP, ...]
    # Parsed rows may be compacted using n* wildcard tokens.
    fields = row[3:]
    ctrl = ctrl.upper()
    if ctrl in ('ORAT', 'WRAT', 'GRAT', 'LRAT', 'RESV'):
        return _first_numeric(fields, default=0.0)
    if ctrl == 'BHP':
        return _last_numeric(fields, default=0.0)
    if ctrl == 'THP':
        vals = _all_numeric(fields)
        if len(vals) >= 2:
            return vals[-1]
        return vals[0] if vals else 0.0
    return _first_numeric(fields, default=0.0)


def _extract_wconinje_target(row, ctrl):
    # ECLIPSE WCONINJE conventional order:
    # [well, phase, status, control, RATE, RESV, BHP, THP, ...]
    fields = row[4:]
    ctrl = ctrl.upper()
    if ctrl in ('RATE', 'RESV'):
        return _first_numeric(fields, default=0.0)
    if ctrl == 'BHP':
        return _last_numeric(fields, default=0.0)
    if ctrl == 'THP':
        vals = _all_numeric(fields)
        if len(vals) >= 2:
            return vals[-1]
        return vals[0] if vals else 0.0
    return _first_numeric(fields, default=0.0)


def _all_numeric(vals):
    out = []
    for v in vals:
        s = _clean_token(v)
        if not s or s.endswith('*'):
            continue
        try:
            out.append(float(s))
        except Exception:
            continue
    return out


def _last_numeric(vals, default=0.0):
    nums = _all_numeric(vals)
    return nums[-1] if nums else float(default)


def _match_wells(wells, wspec):
    import fnmatch as _fnmatch
    name = _clean_token(wspec)
    if not name:
        return []
    if '*' in name or '?' in name:
        return [wn for wn in wells.keys() if _fnmatch.fnmatch(wn, name)]
    if name in wells:
        return [name]
    return []


def _get_non_linear_solver(model, opt):
    if isinstance(opt.get('NonLinearSolver', None), NonLinearSolver):
        return opt['NonLinearSolver']

    # Exact initEclipseProblemAD defaults: target eight nonlinear
    # iterations, a 10 %/one-day first ramp-up mini-step, and MRST's
    # IterationCountTimeStepSelector.
    selector = IterationCountTimeStepSelector(
        targetIterationCount=8,
        firstRampupStepRelative=0.1,
        firstRampupStep=86400.0,
    )
    # Port of MRST's selectLinearSolverAD.  Keep solver selection in the
    # dedicated selector so AMGCL/CPR/AGMG/ILU availability is evaluated
    # consistently for all deck-based simulations.
    from PRSTCore.ad_core.solvers.select_linear_solver_ad import select_linear_solver_ad
    linear_solver = select_linear_solver_ad(
        model,
        useAMGCL=opt.get('useAMGCL', True),
        useAGMG=opt.get('useAGMG', True),
        useILU=opt.get('useILU', True),
        useSYMRCMOrdering=opt.get('useSYMRCMOrdering', False),
        useCPR=opt.get('useCPR', True),
        useAMGCLCPR=opt.get('useAMGCLCPR', True),
        BackslashThreshold=opt.get('BackslashThreshold', 10000),
        tolerance=opt.get('linearSolverTolerance', 1.0e-4),
        verbose=opt.get('Verbose', False),
    )
    return NonLinearSolver(
        maxIterations=opt.get('maxIterations', 12),
        # MRST's NonLinearSolver default: at least one Newton solve per
        # mini-step.  (An earlier PRSTCore revision forced two to preserve a
        # specific MRST EGG trace; with minIterations=1 the behaviour is
        # exactly the deck-path default.)
        minIterations=opt.get('minIterations', 1),
        # MRST's default is 6, which halves a 10-day step down to 0.156 days
        # before giving up.  Stiff incompressible gas-injection decks
        # (SPE10_MODEL1) only become Newton-solvable below ~0.01 days, so
        # six cuts is not enough and both MRST and PRSTCore fail the first
        # report step with the default.  Deeper cutting lets the adaptive
        # IterationCountTimeStepSelector find a workable size instead of
        # failing; converged steps are unaffected (no cuts happen).
        maxTimestepCuts=opt.get('maxTimestepCuts', 16),
        verbose=opt.get('Verbose', False),
        errorOnFailure=opt.get('errorOnFailure', True),
        continueOnFailure=opt.get('continueOnFailure', False),
        timeStepSelector=selector,
        useRelaxation=opt.get('useRelaxation', True),
        linearSolver=linear_solver,
    )


def _init_deck_adi_fluid(deck, G, useMex=False, **kwargs):
    if _init_deck_adi_fluid_impl is not None:
        try:
            return _init_deck_adi_fluid_impl(deck)
        except Exception:
            pass
    # Fallback: minimal emulation
    fluid = {'name': 'blackoil_minimal', 'tables': {}, 'density': None, 'viscosity': None}
    props = deck.get('PROPS', {})
    for key in ('PVTO', 'PVDG', 'PVTW', 'PVTG', 'PVTOG', 'PVTOX'):
        if key in props:
            fluid['tables'][key] = props[key]
    if 'DENSITY' in deck:
        fluid['density'] = deck['DENSITY']
    if 'VISCOSITY' in deck:
        fluid['viscosity'] = deck['VISCOSITY']
    return fluid


def _select_model_from_deck(G, rock, fluid, deck, **kwargs):
    runspec = deck.get('RUNSPEC', {}) if isinstance(deck, dict) else {}
    if bool(runspec.get('POLYMER', False)) or bool(runspec.get('SURFACT', False)):
        from PRSTCore.ad_eor.deck import build_ad_eor_model
        model = build_ad_eor_model(G, rock, fluid, deck)
        if model is not None:
            return model
    model = make_generic_black_oil_model(G, rock, fluid)
    # Direct counterpart of selectModelFromDeck.m: RUNSPEC determines the
    # active phases.  PVT tables only determine DISGAS/VAPOIL defaults.
    props = deck.get('PROPS', {}) if isinstance(deck, dict) else {}
    model.water = bool(runspec.get('WATER', False))
    model.oil = bool(runspec.get('OIL', False))
    model.gas = bool(runspec.get('GAS', False))
    model.disgas = bool(model.oil and model.gas and 'PVTO' in props)
    model.vapoil = bool(model.oil and model.gas and 'PVTG' in props)
    if 'DISGAS' in runspec:
        model.disgas = bool(runspec['DISGAS'])
    if 'VAPOIL' in runspec:
        model.vapoil = bool(runspec['VAPOIL'])
    model.enable_facility_unknowns = True
    # ``initEclipseProblemAD.m`` resets gravity and selects the generic
    # component/facility formulation.  This flag activates its direct
    # Python counterpart in GenericBlackOilModel.
    model._use_mrst_generic_assembly = True
    model.gravity = [0.0, 0.0, 9.80665]
    # MATLAB structs have value semantics.  Keep the model's input deck
    # independent from the caller/app deck because endpoint-scaling setup
    # adds ENDSCALE/SCALECRS fields in place on the Python object.
    model.inputdata = _deepcopy(deck)
    # ReservoirModel.validateModel: "We have some kind of input (e.g. a DECK)
    # and the simulation should have non-negative values" -- MRST sets this
    # for every deck-driven model.  Without it capProperty has no lower bound
    # and a Newton step is free to drive a cell's pressure negative, where
    # every PVT table is being extrapolated far outside its data and returns
    # values with no physical meaning.  ThreePhaseBlackOilModel.validateModel
    # warns about exactly this when disgas or vapoil is on.
    model.minimumPressure = 0.0
    model.rock = rock
    model.G = G
    pv = fluid.get('pvto_obj') if isinstance(fluid, dict) else None
    if pv is not None:
        model.bo = lambda P: pv.bo_of_p(P)
        model.mu_o = lambda P: pv.mu_o_of_p(P)
        model.rs = lambda P: pv.rs_of_p(P)
    pvt = fluid.get('blackoil_pvt') if isinstance(fluid, dict) else None
    if pvt is not None:
        model._blackoil_pvt = pvt
        model.bw = lambda P: pvt.eval(P)['bw']
        model.bo = lambda P: pvt.eval(P)['bo']
        model.bg = lambda P: pvt.eval(P)['bg']
        model.mu_w = lambda P: pvt.eval(P)['muw']
        model.mu_o = lambda P: pvt.eval(P)['muo']
        model.mu_g = lambda P: pvt.eval(P)['mug']
        model.rs = lambda P: pvt.eval(P)['rs']
        model.rv = lambda P: pvt.eval(P)['rv']
    solution = deck.get('SOLUTION', {}) if isinstance(deck, dict) else {}
    if isinstance(solution, dict) and \
            'AQUANCON' in solution and 'AQUFETP' in solution:
        from PRSTCore.deckformat.params.process_aquifer import process_aquifer
        from PRSTCore.ad_core.models.aquifer_model import AquiferModel
        aquifer = process_aquifer(deck, G)
        model.AquiferModel = AquiferModel(
            aquifer['aquifers'], aquifer['aquind'],
            aquifer['aquiferprops'], aquifer['initval'])
    return model
