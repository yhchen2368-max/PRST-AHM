"""Deck-driven wiring for the ``ad_eor`` models: turns PROPS/REGIONS keyword
arrays already parsed generically by ``deckformat/deckinput/read_props.py``
and ``read_regions.py`` into the fluid-dict callables/tables the
``ad_eor.properties``/``ad_eor.utils.equations*`` functions expect, and
builds the AD-compatible PVT closures (``fluid['bW']``/``fluid['muW']``/...)
those functions call directly (PRSTCore's deck pipeline previously only
built numpy-only ``model.bw``/``model.mu_w`` attributes, see
``init_eclipse_problem_ad.py::_select_model_from_deck``).

Not a port of any single ``.m`` file: MRST gets this wiring "for free" from
``initDeckADIFluid.m``'s per-region ``assignPLYVISC.m``/``assignPLYROCK.m``/
``assignPLYADS.m``/``assignSURFST.m``/etc. plus the StateFunction graph's
SATNUM/SURFNUM dispatch; this module is the corresponding PRSTCore glue for
the procedural ``ad_eor`` equations.
"""

import numpy as _np

from PRSTCore.ad_core.adi import SparseADI as _SparseADI
from PRSTCore.ad_core.adi import ad_interp_linear as _ad_interp_linear


def table_interp_closure(x, y):
    """AD-safe piecewise-linear 1D table lookup: dispatches to
    :func:`PRSTCore.ad_core.adi.ad_interp_linear` for a ``SparseADI`` query
    (matching this codebase's established ADI-table-lookup convention) or
    plain ``numpy.interp`` otherwise -- used for every 2-column ECLIPSE
    EOR table keyword (PLYVISC, PLYADS, SURFST, SURFVISC, SURFADS,
    SURFCAPD) which MRST wraps as a ``fluid.X = @(c) interpTable(...)``
    closure."""
    x = _np.asarray(x, dtype=float).ravel()
    y = _np.asarray(y, dtype=float).ravel()
    order = _np.argsort(x)
    xs, ys = x[order], y[order]

    def fn(q):
        if isinstance(q, _SparseADI):
            return _ad_interp_linear(xs, ys, q)
        qv = _np.asarray(q, dtype=float)
        return _np.interp(qv, xs, ys, left=ys[0], right=ys[-1])
    return fn


def ad_pvt_closure(pvt, key):
    """``fluid['bW'] = ad_pvt_closure(pvt, 'bw')`` etc: dispatches a
    ``DeckBlackOilPVT`` evaluation to ``.eval_adi()`` for a ``SparseADI``
    pressure or ``.eval()`` for a plain array -- ``getFluxAndPropsWaterPolymer_BO``/
    ``equationsOilWaterSurfactant`` call ``fluid['bW'](p)`` with both kinds
    of pressure (current-iterate ADI and previous-timestep plain array)."""
    def fn(p):
        if isinstance(p, _SparseADI):
            return pvt.eval_adi(p)[key]
        return pvt.eval(p)[key]
    return fn


def _table(props, name, ncol):
    """Reshape a flat PROPS array into an ``(n, ncol)`` table, or ``None``
    if the keyword is absent."""
    raw = props.get(name)
    if raw is None:
        return None
    arr = _np.asarray(raw, dtype=float).ravel()
    if arr.size == 0 or arr.size % ncol != 0:
        return None
    return arr.reshape(-1, ncol)


def split_regions_by_column_reset(table):
    """Split a table whose rows are several concatenated NTSFUN/NSURFNUM
    blocks (column 0 -- the saturation/concentration axis -- resets to a
    lower value at each new block, since ECLIPSE table blocks are
    independently ascending) into a list of per-region 2D tables. Reuses
    the same reset-detection heuristic
    ``GenericBlackOilModel._get_relperm_tables`` already applies (there,
    only the first region is kept; here every region is kept)."""
    col0 = table[:, 0]
    breaks = _np.flatnonzero(_np.diff(col0) < 0.0) + 1
    if breaks.size == 0:
        return [table]
    bounds = [0] + breaks.tolist() + [table.shape[0]]
    return [table[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]


def parse_polymer_fluid(props):
    """Port of the relevant fields ``initDeckADIFluid.m`` assigns from
    PLYVISC/PLYROCK/PLYADS/TLMIXPAR/PLYMAX/PLYSHEAR/PLYSHLOG.

    Returns a dict with keys matching what
    ``ad_eor.utils.equationsOilWaterPolymer``/
    ``equationsThreePhaseBlackOilPolymer``/``PolymerAdsorption`` etc. read
    off ``fluid``: ``muWMult``, ``ads``, ``dps``, ``rrf``, ``rhoR``,
    ``adsInx``, ``adsMax``, ``mixPar``, ``cpmax``, and (if present)
    ``plyshearMult``/``plyshlog``.
    """
    plyvisc = _table(props, 'PLYVISC', 2)
    plyads = _table(props, 'PLYADS', 2)
    plyrock = _np.asarray(props.get('PLYROCK', []), dtype=float).ravel()
    # readPROPS.m reads TLMIXPAR and PLMIXPAR through the same branch, and
    # assignPLMIXPAR/assignTLMIXPAR both take fluid.mixPar from item 1.
    # PLMIXPAR is the one a polymer deck should carry -- ECLIPSE 2013.2
    # warns that TLMIXPAR "is no longer compatible with the polymer flood
    # model", and MRST's own 1D_FLOODING.DATA uses PLMIXPAR -- so accept
    # either, preferring PLMIXPAR when both appear.
    mixpar = _np.asarray(props.get('PLMIXPAR', props.get('TLMIXPAR', [])),
                         dtype=float).ravel()
    plymax = _np.asarray(props.get('PLYMAX', []), dtype=float).ravel()
    if plyvisc is None or plyads is None or plyrock.size < 5 or mixpar.size < 1 or plymax.size < 1:
        missing = [name for name, ok in (
            ('PLYVISC', plyvisc is not None), ('PLYADS', plyads is not None),
            ('PLYROCK', plyrock.size >= 5), ('PLMIXPAR/TLMIXPAR', mixpar.size >= 1),
            ('PLYMAX', plymax.size >= 1)) if not ok]
        raise ValueError('POLYMER active but %s missing or incomplete'
                         % ', '.join(missing))

    fluid = {
        'muWMult': table_interp_closure(plyvisc[:, 0], plyvisc[:, 1]),
        'ads': table_interp_closure(plyads[:, 0], plyads[:, 1]),
        'dps': float(plyrock[0]),
        'rrf': float(plyrock[1]),
        'rhoR': float(plyrock[2]),
        'adsInx': int(plyrock[3]),
        'adsMax': float(plyrock[4]),
        'mixPar': float(mixpar[0]),
        'cpmax': float(plymax[0]),
    }
    plyshear = _table(props, 'PLYSHEAR', 2)
    if plyshear is not None:
        fluid['plyshearMult'] = table_interp_closure(plyshear[:, 0], plyshear[:, 1])
    plyshlog = _table(props, 'PLYSHLOG', 2)
    if plyshlog is not None:
        # PLYSHLOG's header record (SHRATE/reference-concentration) is a
        # separate free-text-ish record MRST reads via assignPLYSHLOG.m;
        # PRSTCore's generic array parser flattens it in among the table
        # rows, which this port does not attempt to disentangle -- callers
        # needing PLYSHLOG must build fluid['plyshlog'] by hand (as done
        # for POLYMER.DATA/1D_FLOODING.DATA, neither of which use it).
        pass
    return fluid


def parse_surfactant_fluid(props):
    """Port of the relevant fields ``initDeckADIFluid.m`` assigns from
    SURFST/SURFVISC/SURFADS/SURFROCK/SURFCAPD.

    ``SURFADS``/``SURFROCK``/``SURFCAPD`` may carry more than one
    NTSFUN/NSURFNUM-region block (multiple ``/``-terminated sub-tables);
    only the first block is used here, matching this codebase's existing
    "first region" scope for relperm tables -- correct whenever, as in the
    bundled decks, regions share identical adsorption/capillary-desaturation
    physics and differ only in the *relative-permeability* table (handled
    separately, see ``parse_surfactant_relperm_regions``).
    """
    surfst = _table(props, 'SURFST', 2)
    surfvisc = _table(props, 'SURFVISC', 2)
    surfads_all = _table(props, 'SURFADS', 2)
    surfrock = _np.asarray(props.get('SURFROCK', []), dtype=float).ravel()
    surfcapd_all = _table(props, 'SURFCAPD', 2)
    if surfst is None or surfvisc is None or surfads_all is None or surfrock.size < 2 or surfcapd_all is None:
        raise ValueError('SURFACT active but SURFST/SURFVISC/SURFADS/SURFROCK/SURFCAPD incomplete')

    surfads = split_regions_by_column_reset(surfads_all)[0]
    surfcapd = split_regions_by_column_reset(surfcapd_all)[0]

    return {
        'ift': table_interp_closure(surfst[:, 0], surfst[:, 1]),
        'muWSft': table_interp_closure(surfvisc[:, 0], surfvisc[:, 1]),
        'muWr': float(surfvisc[0, 1]),
        'surfads': table_interp_closure(surfads[:, 0], surfads[:, 1]),
        'adsInxSft': int(surfrock[0]),
        'rhoRSft': float(surfrock[1]),
        'miscfact': table_interp_closure(surfcapd[:, 0], surfcapd[:, 1]),
    }


def parse_surfactant_relperm_regions(props, regions):
    """Resolve the SATNUM ("base"/no-surfactant) and SURFNUM
    ("fully-desaturated"/miscible) SWOF table blocks and residual-saturation
    endpoints ``SurfactantRelativePermeability`` needs.

    Only a single, grid-uniform SATNUM/SURFNUM assignment is supported (the
    first cell's region value is used for the whole grid) -- see
    ``SurfactantRelativePermeability``'s module docstring.

    Returns ``(krPts_base, krPts_surf, fluid_base, fluid_surf)``.
    """
    swof_all = _table(props, 'SWOF', 4)
    if swof_all is None:
        raise ValueError('SURFACT active but SWOF table missing')
    blocks = split_regions_by_column_reset(swof_all)

    satnum = _np.asarray(regions.get('SATNUM', [1]), dtype=int).ravel()
    surfnum = _np.asarray(regions.get('SURFNUM', [2 if len(blocks) > 1 else 1]), dtype=int).ravel()
    base_idx = int(satnum[0]) - 1 if satnum.size else 0
    surf_idx = int(surfnum[0]) - 1 if surfnum.size else min(1, len(blocks) - 1)
    base_idx = min(max(base_idx, 0), len(blocks) - 1)
    surf_idx = min(max(surf_idx, 0), len(blocks) - 1)

    swof_base, swof_surf = blocks[base_idx], blocks[surf_idx]

    def _endpoints(swof):
        # Connate water = first (lowest) Sw row; residual oil = 1 - last
        # (highest) Sw row, since krow reaches 0 there in a well-formed
        # SWOF table (assignSWOF.m's convention).
        return {'w': float(swof[0, 0]), 'ow': float(1.0 - swof[-1, 0])}

    def _fluid(swof):
        return {
            'krW': table_interp_closure(swof[:, 0], swof[:, 1]),
            'krOW': table_interp_closure(1.0 - swof[::-1, 0], swof[::-1, 2]),
        }

    return _endpoints(swof_base), _endpoints(swof_surf), _fluid(swof_base), _fluid(swof_surf)


def cartesian_sq_veloc(model):
    """Substitute for ``ad_eor.utils.computeSqVelocTPFA`` on a tensor
    (axis-aligned Cartesian) grid built by
    ``deckformat/grid/init_eclipse_grid.py``, which does not build the
    generic half-face connectivity (``cells.facePos``/``cells.faces``)
    ``computeSqVelocTPFA`` needs.

    Rather than depending on ``ad_core.operators.setup_operators``'s
    internal x/y/z face-ordering convention (private, and not worth
    coupling to), this derives each internal connection's axis directly
    from its two cell centroids (exactly one coordinate differs on an
    axis-aligned grid) and its cross-sectional area from that axis's
    per-cell width/height (``dx``/``dy``/``dz`` from ``G['xfaces']`` etc.,
    arithmetic-averaged between the two neighbor cells, matching
    ``ad_core.operators._directional_interfaces``'s area convention). Each
    face's squared velocity ``(q/area)^2`` is then distributed evenly to
    its two neighbor cells and averaged per cell -- a simpler, isotropic
    stand-in for ``computeSqVelocTPFA``'s distance-weighted vector
    reconstruction, adequate for the smoothly-varying capillary-number
    interpolation this feeds.

    Requires ``model.operators['N']``/``['T']`` and
    ``model.G['cells']['centroids']``/``['xfaces']``/``['yfaces']``/
    ``['zfaces']``/``['cartDims']``; raises clearly for a corner-point or
    unstructured grid, which does not expose the latter three.
    """
    G = model.G
    if 'xfaces' not in G or 'yfaces' not in G or 'zfaces' not in G:
        raise NotImplementedError(
            'cartesian_sq_veloc requires a tensor grid with xfaces/yfaces/zfaces '
            '(corner-point/unstructured grids are not supported by this substitute)')
    nx, ny, nz = (int(v) for v in G['cartDims'])
    nc = nx * ny * nz
    ops = model.operators or {}
    N = _np.asarray(ops.get('N', _np.zeros((0, 2))), dtype=int)
    if N.ndim != 2 or N.shape[1] < 2 or N.size == 0:
        c1 = c2 = _np.zeros(0, dtype=int)
    elif _np.min(N) >= 1:
        c1, c2 = N[:, 0] - 1, N[:, 1] - 1
    else:
        c1, c2 = N[:, 0], N[:, 1]

    centroids = _np.asarray(G['cells']['centroids'], dtype=float)
    diff = centroids[c2] - centroids[c1] if c1.size else _np.zeros((0, 3))
    axis = _np.argmax(_np.abs(diff), axis=1) if c1.size else _np.zeros(0, dtype=int)

    dx = _np.diff(_np.asarray(G['xfaces'], dtype=float))
    dy = _np.diff(_np.asarray(G['yfaces'], dtype=float))
    dz = _np.diff(_np.asarray(G['zfaces'], dtype=float))
    # Fortran-order (x-fastest) flattening maps cell (i,j,k) -> global id
    # i + j*nx + k*nx*ny, matching ad_core.operators._tensor_dimensions'
    # cell numbering, so DXc/DYc/DZc below are indexed by global cell id.
    DXc = _np.broadcast_to(dx[:, None, None], (nx, ny, nz)).ravel(order='F')
    DYc = _np.broadcast_to(dy[None, :, None], (nx, ny, nz)).ravel(order='F')
    DZc = _np.broadcast_to(dz[None, None, :], (nx, ny, nz)).ravel(order='F')
    area_x = DYc * DZc
    area_y = DXc * DZc
    area_z = DXc * DYc

    if c1.size:
        area_face = _np.where(axis == 0, 0.5 * (area_x[c1] + area_x[c2]),
                     _np.where(axis == 1, 0.5 * (area_y[c1] + area_y[c2]),
                                          0.5 * (area_z[c1] + area_z[c2])))
        area_face = _np.maximum(area_face, 1.0e-30)
    else:
        area_face = _np.zeros(0)

    def sqVeloc(q):
        q = _np.asarray(q, dtype=float).ravel()
        v2 = (q / area_face) ** 2 if area_face.size else _np.zeros(0)
        out = _np.zeros(nc)
        count = _np.zeros(nc)
        if c1.size:
            _np.add.at(out, c1, v2)
            _np.add.at(count, c1, 1.0)
            _np.add.at(out, c2, v2)
            _np.add.at(count, c2, 1.0)
        count[count == 0.0] = 1.0
        return out / count

    return sqVeloc


def _ad_pcow_closure(props):
    """AD-safe ``fluid['pcOW'](sW)`` closure sourced from the (first-region)
    SWOF table's Pcow column, self-consistently interpolated the same way
    as every other ``ad_eor`` table (``table_interp_closure``) rather than
    reusing ``GenericBlackOilModel._interp_relperm_table`` (numpy-only, no
    ADI variant)."""
    swof_all = _table(props, 'SWOF', 4)
    if swof_all is None:
        return None
    swof = split_regions_by_column_reset(swof_all)[0]
    if not _np.any(swof[:, 3] != 0.0):
        return None
    return table_interp_closure(swof[:, 0], swof[:, 3])


def build_ad_eor_model(G, rock, fluid, deck):
    """Construct an ``ad_eor`` model (``OilWaterPolymerModel`` or
    ``OilWaterSurfactantModel``) from a parsed deck, or return ``None`` if
    the deck's phase/EOR combination isn't one of the two currently wired
    (gas-active and combined polymer+surfactant decks fall through to the
    plain ``GenericBlackOilModel`` -- ``ThreePhaseBlackOilPolymerModel``
    exists and could be wired the same way, but ``ThreePhaseSurfactantPolymerModel``'s
    equations are not yet ported, see ``ad_eor`` package docstring).
    """
    runspec = deck.get('RUNSPEC', {})
    props = deck.get('PROPS', {})
    regions = deck.get('REGIONS', {})
    has_gas = bool(runspec.get('GAS', False))
    has_polymer = bool(runspec.get('POLYMER', False))
    has_surfact = bool(runspec.get('SURFACT', False))

    # A deck that activates POLYMER or SURFACT and then silently gets a
    # plain GenericBlackOilModel is not a graceful degradation: the EOR
    # component is simply absent from the physics, and the run completes
    # and reports numbers as though it had been simulated. Say which model
    # is missing instead.
    if has_polymer and has_surfact:
        raise NotImplementedError(
            'Deck activates both POLYMER and SURFACT; the port of MRST\'s '
            'ThreePhaseSurfactantPolymerModel/equationsThreePhaseSurfactantPolymer '
            'is not available yet.')
    if has_gas and has_surfact:
        raise NotImplementedError(
            'Deck activates SURFACT with GAS; the port of MRST\'s '
            'ThreePhaseBlackOilSurfactantModel/'
            'equationsThreePhaseBlackOilSurfactant is not available yet.')

    pvt = fluid.get('blackoil_pvt') if isinstance(fluid, dict) else None
    if pvt is None:
        raise ValueError('EOR deck has no black-oil PVT tables to build from')
    density = _np.asarray(props.get('DENSITY', []), dtype=float).ravel()
    rhoOS = float(density[0]) if density.size >= 1 else 800.0
    rhoWS = float(density[1]) if density.size >= 2 else 1000.0

    base_fluid = {
        'bW': ad_pvt_closure(pvt, 'bw'), 'muW': ad_pvt_closure(pvt, 'muw'),
        'bO': ad_pvt_closure(pvt, 'bo'), 'muO': ad_pvt_closure(pvt, 'muo'),
        'pcOW': _ad_pcow_closure(props),
        'rhoWS': rhoWS, 'rhoOS': rhoOS,
    }

    if has_polymer and has_gas:
        from .models.ThreePhaseBlackOilPolymerModel import ThreePhaseBlackOilPolymerModel
        fluid_out = dict(base_fluid)
        fluid_out.update(parse_polymer_fluid(props))
        fluid_out['bG'] = ad_pvt_closure(pvt, 'bg')
        fluid_out['muG'] = ad_pvt_closure(pvt, 'mug')
        density = _np.asarray(props.get('DENSITY', []), dtype=float).ravel()
        fluid_out['rhoGS'] = float(density[2]) if density.size >= 3 else 1.0
        model = ThreePhaseBlackOilPolymerModel(
            G=G, rock=rock, fluid=fluid_out,
            water=True, oil=True, gas=True,
            disgas=bool(runspec.get('DISGAS', False)),
            vapoil=bool(runspec.get('VAPOIL', False)))
    elif has_polymer:
        from .models.OilWaterPolymerModel import OilWaterPolymerModel
        fluid_out = dict(base_fluid)
        fluid_out.update(parse_polymer_fluid(props))
        model = OilWaterPolymerModel(G=G, rock=rock, fluid=fluid_out)
    else:
        from .models.OilWaterSurfactantModel import OilWaterSurfactantModel
        fluid_out = dict(base_fluid)
        fluid_out.update(parse_surfactant_fluid(props))
        krPts_base, krPts_surf, fluid_base, fluid_surf = parse_surfactant_relperm_regions(props, regions)
        fluid_out['krPts_base'] = krPts_base
        fluid_out['krPts_surf'] = krPts_surf
        fluid_out['fluid_base'] = fluid_base
        fluid_out['fluid_surf'] = fluid_surf
        model = OilWaterSurfactantModel(G=G, rock=rock, fluid=fluid_out)

    model._blackoil_pvt = pvt
    model.inputdata = deck
    model.rock = rock
    model.G = G
    model.gravity = [0.0, 0.0, 9.80665]
    model.enable_facility_unknowns = True
    if not has_gas:
        # A gas-free deck (OIL+WATER[+POLYMER/SURFACT]) has no PVTO table;
        # GenericBlackOilModel defaults disgas=True, which is only
        # meaningful with an active gas phase.
        model.disgas = False
        model.vapoil = False
    return model
