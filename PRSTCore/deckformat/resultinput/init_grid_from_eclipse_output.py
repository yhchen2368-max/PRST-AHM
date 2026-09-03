"""Initialize grid and rock from ECLIPSE result files (INIT + EGRID/GRID).

Port of MRST ``initGridFromEclipseOutput.m``
(model-io/deckformat/resultinput), following MRST-0's version.

The point of building the model this way rather than from the deck is
that the simulator has already done the work: the INIT file carries the
transmissibilities it actually used, region by region, after every
multiplier and pinch-out. Deriving them again from the deck would give a
model that differs from the one the simulator integrated -- which is the
whole reason ``HistoryMatching`` runs a NOSIM pass first.

So the neighbour list and transmissibilities here come from INIT's
TRANX/TRANY/TRANZ (plus TRANNNC for non-neighbour connections), not from
geometry. A connection is kept only where the transmissibility is
positive and both cells are active.

Three things MRST-0 reads that 2026a's version does not, and that the
previous PRSTCore stub did not either:

* **regions** -- PVTNUM, SATNUM, IMBNUM, FIPNUM, EQLNUM, ROCKNUM,
  SURFNUM. Without them a multi-region deck is silently treated as
  single-region.
* **multipliers** -- MULTX/Y/Z and their ``_`` (negative-direction)
  forms.
* **eMap** -- when processing the corner-point geometry drops cells (a
  disconnected region, say), the grid no longer lines up with the INIT
  arrays one-to-one. eMap maps between them, and every per-cell array
  read from INIT is indexed through it.
"""

import warnings as _warnings

import numpy as np

#: Region keywords, and the rock.regions field each becomes.
_REGIONS = (('PVTNUM', 'pvt'), ('SATNUM', 'saturation'),
            ('IMBNUM', 'imbibition'), ('FIPNUM', 'fluid'),
            ('EQLNUM', 'equilibration'), ('ROCKNUM', 'rock'),
            ('SURFNUM', 'surfactant'))

#: These are taken as present whenever the array exists, even if it is
#: all ones -- unlike the others, which need more than one region to
#: matter.
_REGIONS_ALWAYS = ('ROCKNUM', 'SURFNUM')

_MULTIPLIERS = ('MULTX', 'MULTY', 'MULTZ', 'MULTX_', 'MULTY_', 'MULTZ_')


def init_grid_from_eclipse_output(init, grid, output_sim_grid=False):
    """Initialize grid and rock structures from ECLIPSE INIT and EGRID files.

    Parameters
    ----------
    init : dict
        INIT file data (from read_eclipse_output_file_unfmt).
    grid : dict
        EGRID (or GRID) file data.
    output_sim_grid : bool
        If True, also return the TPFA simulation grid alongside G.

    Returns
    -------
    G : dict
        MRST-compatible grid, carrying cells.eMap/eMapInv.
    rock : dict
        Rock structure with perm, poro, ntg, regions and multipliers.
    N : ndarray
        Neighbour list, one row per connection, 0-based.
    T : ndarray
        Transmissibility for each connection, in SI.
    """
    units_list = ["metric", "field", "lab"]
    unit_idx = init["INTEHEAD"]["values"][2] - 1 \
        if init["INTEHEAD"]["values"][2] > 0 else 0
    u = _get_unit_system(units_list[min(unit_idx, 2)])

    # ---- active cells ---------------------------------------------------
    if "cartDims" in grid:
        cart_dims = tuple(int(d) for d in grid["cartDims"])
        act_num = grid.get("ACTNUM", np.ones(int(np.prod(cart_dims)), bool))
        coord = grid["COORD"]
        zcorn = grid["ZCORN"]
    else:
        cart_dims = tuple(int(d) for d in grid["GRIDHEAD"]["values"][1:4])
        n = int(np.prod(cart_dims))
        act_num = grid.get("ACTNUM", {}).get("values", np.ones(n))
        coord = grid.get("COORD", {}).get("values",
                                          np.zeros(6 * int(np.prod(cart_dims[:2]))))
        zcorn = grid.get("ZCORN", {}).get("values", np.zeros(8 * n))

    porv = _values(init, "PORV", np.ones(int(np.prod(cart_dims))))
    act_num = np.asarray(act_num, dtype=bool).ravel() & (np.asarray(porv).ravel() > 0)
    na = int(np.count_nonzero(act_num))

    # ---- geometry -------------------------------------------------------
    G = _build_grid(cart_dims, coord, zcorn, act_num, u)

    # ---- eMap: the grid may hold fewer cells than ACTNUM marks ----------
    e_map, e_map_inv, consistent = _build_emap(G, act_num, na)
    G["cells"]["eMap"] = e_map
    G["cells"]["eMapInv"] = e_map_inv

    index_map = np.asarray(G["cells"]["indexMap"], dtype=int)
    if "DEPTH" in init:
        G["cells"]["centroids"][:, 2] = \
            _pick(_values(init, "DEPTH"), e_map) * u["length"]
    if "PORV" in init:
        G["cells"]["PORV"] = np.asarray(porv).ravel()[index_map] * u["resvolume"]
    for name in ("DX", "DY", "DZ"):
        if name in init:
            G["cells"][name] = _pick(_values(init, name), e_map) * u["length"]

    # ---- rock -----------------------------------------------------------
    pick = (slice(None) if consistent and output_sim_grid else e_map)
    rock = {
        "poro": _pick(_values(init, "PORO"), pick),
        "perm": np.column_stack([
            _pick(_values(init, "PERMX"), pick),
            _pick(_values(init, "PERMY"), pick),
            _pick(_values(init, "PERMZ"), pick),
        ]) * u["perm"],
    }
    if "NTG" in init:
        rock["ntg"] = _pick(_values(init, "NTG"), pick)
    rock = _assign_multipliers(rock, init, G)
    rock = _get_regions(rock, init, pick)

    # ---- connections and transmissibilities -----------------------------
    N, T, nnc = _get_active_neighbors(init, grid, act_num, cart_dims)
    T = T * u["trans"]
    if nnc["cells"].size:
        nnc = {"cells": nnc["cells"], "trans": nnc["trans"] * u["trans"]}
        G["nnc"] = nnc

    return G, rock, N, T


# --------------------------------------------------------------- grid --

def _build_grid(cart_dims, coord, zcorn, act_num, u):
    """Process the corner-point geometry, falling back to a cell list.

    ``processGRDECL`` needs COORD and ZCORN; a GRID file that carries
    neither (or a caller that only wants rock and transmissibilities)
    still gets a grid with the right cell count and index map, just
    without faces.
    """
    from PRSTCore.gridprocessing.compute_geometry import compute_geometry
    from PRSTCore.gridprocessing.process_grdecl import process_grdecl

    coord = np.asarray(coord, dtype=float).ravel()
    zcorn = np.asarray(zcorn, dtype=float).ravel()

    if coord.size and zcorn.size and np.any(zcorn):
        grdecl = {"cartDims": np.asarray(cart_dims, dtype=int),
                  "COORD": coord * u["length"],
                  "ZCORN": zcorn * u["length"],
                  "ACTNUM": act_num.astype(np.int32)}
        try:
            # MRST passes SplitDisconnected=false and PreserveCpNodes=true;
            # PRSTCore's process_grdecl takes neither and does not split,
            # which is the behaviour those options ask for.
            G = process_grdecl(grdecl)
            return compute_geometry(G)
        except Exception as exc:
            _warnings.warn('Could not process the corner-point geometry '
                           '(%s); continuing with a cell list, so faces '
                           'and geometry are unavailable.' % exc,
                           RuntimeWarning)

    na = int(np.count_nonzero(act_num))
    return {
        "cells": {"num": na,
                  "indexMap": np.flatnonzero(act_num),
                  "volumes": np.ones(na),
                  "centroids": np.zeros((na, 3))},
        "faces": {"num": 0, "neighbors": np.zeros((0, 2), dtype=int)},
        "cartDims": np.asarray(cart_dims, dtype=int),
        "griddim": 3,
    }


def _build_emap(G, act_num, na):
    """Port of the eMap block.

    ``eMap`` takes a grid cell to its row in the INIT arrays; ``eMapInv``
    goes the other way. When the two agree cell for cell both are a plain
    slice, which is what the MATLAB's ``':'`` means.
    """
    if int(G["cells"]["num"]) == na:
        return slice(None), slice(None), True

    tmp = np.zeros(act_num.size, dtype=int)
    tmp[act_num] = np.arange(na)
    e_map = tmp[np.asarray(G["cells"]["indexMap"], dtype=int)]
    e_map_inv = np.zeros(na, dtype=int)
    e_map_inv[e_map] = np.arange(int(G["cells"]["num"]))
    return e_map, e_map_inv, False


# --------------------------------------------------------------- rock --

def _get_regions(rock, init, e_map):
    """Port of ``getRegions``.

    A region array with a single region is not recorded: it carries no
    information and MRST leaves it out so downstream code can test for
    the field rather than for its contents.
    """
    regions = {}
    for keyword, field in _REGIONS:
        if keyword not in init:
            continue
        values = _values(init, keyword)
        if keyword not in _REGIONS_ALWAYS and \
                (values.size == 0 or int(np.max(values)) <= 1):
            continue
        regions[field] = _pick(values, e_map).astype(int)

    # Relative permeability with surfactant is given as saturation
    # tables, so SATNUM is needed even when it holds a single region.
    if 'surfactant' in regions and 'saturation' not in regions \
            and 'SATNUM' in init:
        regions['saturation'] = _pick(_values(init, 'SATNUM'),
                                      e_map).astype(int)

    if regions:
        rock["regions"] = regions
    return rock


def _assign_multipliers(rock, init, G):
    """Port of ``assign_multipliers``.

    A multiplier that is 1 everywhere is dropped -- it changes nothing
    and only costs a per-cell array downstream.
    """
    multipliers = {}
    for keyword in _MULTIPLIERS:
        if keyword not in init:
            continue
        values = _values(init, keyword)
        nc = int(G["cells"]["num"])
        if values.size != nc:
            if values.size == int(np.prod(G["cartDims"])):
                values = values[np.asarray(G["cells"]["indexMap"], dtype=int)]
            else:
                continue
        bad = int(np.count_nonzero(~np.isfinite(values)))
        if bad:
            _warnings.warn("%d non-finite values in multiplier '%s'"
                           % (bad, keyword), RuntimeWarning)
        if not np.all(values == 1):
            # MULTX -> x, MULTY_ -> y_
            multipliers[keyword[4:].lower()] = values
    if multipliers:
        rock["multipliers"] = multipliers
    return rock


# -------------------------------------------------------- connections --

def _get_active_neighbors(init, grid, act_num, cart_dims):
    """Port of ``getActiveNeighbors``.

    Builds the I, J and K neighbour pairs by shifting an index volume one
    cell in each direction, then keeps the connections INIT gave a
    positive transmissibility and whose two cells are both active.
    """
    cart_dims = tuple(int(d) for d in cart_dims)
    n_cells = int(np.count_nonzero(act_num))

    act_index = np.zeros(int(np.prod(cart_dims)), dtype=int)
    act_index[act_num] = np.arange(1, n_cells + 1)      # 1-based, 0 = inactive

    # An index volume padded by one in each direction, so a shift cannot
    # wrap around into the opposite face.
    M = np.zeros(tuple(d + 1 for d in cart_dims), dtype=int)
    M[:-1, :-1, :-1] = act_index.reshape(cart_dims, order='F')

    NX = M[1:, :-1, :-1].reshape(-1, order='F')[act_num]
    NY = M[:-1, 1:, :-1].reshape(-1, order='F')[act_num]
    NZ = M[:-1, :-1, 1:].reshape(-1, order='F')[act_num]

    nnc1, nnc2, trannnc = _nnc(init, grid, act_index)

    own = np.arange(1, n_cells + 1)
    N = np.vstack([np.column_stack([own, NX]),
                   np.column_stack([own, NY]),
                   np.column_stack([own, NZ])])
    T = np.concatenate([_values(init, "TRANX", np.zeros(n_cells)),
                        _values(init, "TRANY", np.zeros(n_cells)),
                        _values(init, "TRANZ", np.zeros(n_cells))])
    if nnc1.size:
        N = np.vstack([N, np.column_stack([nnc1, nnc2])])
        T = np.concatenate([T, trannnc])

    keep = (T > 0) & (np.prod(N, axis=1) > 0)
    N, T = N[keep], T[keep]

    if nnc1.size:
        keep = (trannnc > 0) & (nnc1 > 0) & (nnc2 > 0)
        nnc = {"cells": np.column_stack([nnc1[keep], nnc2[keep]]) - 1,
               "trans": trannnc[keep]}
    else:
        nnc = {"cells": np.zeros((0, 2), dtype=int), "trans": np.zeros(0)}

    # Back to 0-based, which is what everything downstream expects.
    return N - 1, T, nnc


def _nnc(init, grid, act_index):
    """Non-neighbour connections, if the files carry any."""
    source = grid if "NNC1" in grid else init
    if "NNC1" not in source:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0)

    nnc1 = act_index[_values(source, "NNC1").astype(int) - 1]
    nnc2 = act_index[_values(source, "NNC2").astype(int) - 1]

    if "TRANNNC" in init:
        trannnc = _values(init, "TRANNNC")
    else:
        # MRST warns and uses inf, which then survives the T > 0 filter
        # and lands in the model as an infinite transmissibility.
        _warnings.warn('Transmissibilities for NNCs not given, values are '
                       'set to inf!', RuntimeWarning)
        trannnc = np.full(nnc1.size, np.inf)
    return nnc1, nnc2, trannnc


# ------------------------------------------------------------ support --

def _values(container, keyword, default=None):
    """The values array of an INIT/EGRID keyword."""
    entry = container.get(keyword)
    if entry is None:
        if default is None:
            raise KeyError('%s not present in the output file' % keyword)
        return np.asarray(default, dtype=float).ravel()
    if isinstance(entry, dict):
        entry = entry.get("values")
    return np.asarray(entry, dtype=float).ravel()


def _pick(values, index):
    """Index an INIT array through eMap, which may be a plain slice."""
    values = np.asarray(values).ravel()
    if isinstance(index, slice):
        return values[index]
    return values[np.asarray(index, dtype=int)]


def _get_unit_system(unit):
    """Port of ``getUnitSystem``, delegating to the shared factor table.

    Transmissibility is not permeability times length: ECLIPSE's unit is
    ``cP * rm^3 / (day * bar)`` in METRIC, so the factor is 1.157e-13,
    not the 1e-25 that perm*length would give. Reusing
    ``unit_conversion_factors`` keeps that from being got wrong twice.
    """
    from PRSTCore.deckformat.unit_conversion_factors import \
        unit_conversion_factors

    u = unit_conversion_factors(unit.upper(), "SI")
    return {"perm": u["perm"], "length": u["length"],
            "resvolume": u["resvolume"], "trans": u["trans"],
            "poro_factor": 1.0}
