"""Convert ECLIPSE restart data to PRSTCore/MRST-style states."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from ..unit_conversion_factors import unit_conversion_factors

def convert_restart_to_states(prefix, G, restart_info=None, steps=None,
                               include_well_sols=True, include_fluxes=True,
                               include_aquifers=False, neighbors=None,
                               well_sols_from_restart=True,
                               consistent_well_sols=True,
                               split_wells_on_sign_change=False,
                               remove_closed_wells=True,
                               remove_crossflow=True,
                               set_to_closed_tol=0.0,
                               add_trajectory=True,
                               include_components=False,
                               unit_system=None):
    """Convert an ECLIPSE unified restart file to state dictionaries.

    Follows MRST-0's ``convertRestartToStates``. Each state carries
    pressure, saturations, Rs/Rv and time, and -- when the restart has
    well records -- a full well solution per step: name, open/shut, sign,
    control mode, perforated cells, rates, bhp, reservoir rate, and
    per-connection fluxes and phase rates.

    ``split_wells_on_sign_change``, ``remove_closed_wells``,
    ``remove_crossflow`` and ``set_to_closed_tol`` are applied by
    :func:`process_well_states`.
    """
    # Reservoir-face fluxes and the summary-based well-solution alternative
    # are not needed by FAHM's ``includeWellSols=false`` state0 import.  The
    # options remain explicit so callers do not silently change call shape;
    # aquifers and compositional cell fields, which FAHM does request, are
    # handled below.
    del (include_fluxes, neighbors, well_sols_from_restart,
         consistent_well_sols, add_trajectory)

    prefix = _restart_prefix(prefix)
    restart_file = _resolve_restart_file(prefix)

    from .read_eclipse_output_file_unfmt import read_eclipse_output_file_unfmt
    from .process_eclipse_restart_spec import process_eclipse_restart_spec

    rsspec = restart_info
    if rsspec is None:
        try:
            rsspec, _ = process_eclipse_restart_spec(prefix, "all")
        except (FileNotFoundError, KeyError, ValueError, OSError):
            rsspec = None

    raw_restart = read_eclipse_output_file_unfmt(str(restart_file))
    restart_blocks = _restart_blocks_from_records(raw_restart.get("__records__", []))
    if steps is not None:
        restart_blocks = [
            restart_blocks[int(step)]
            for step in steps
            if 0 <= int(step) < len(restart_blocks)
        ]

    unit_name = _restart_unit_name(unit_system, rsspec, raw_restart)
    units = unit_conversion_factors(unit_name)
    states = [
        _restart_block_to_state(
            block, G, units, index, rsspec, include_well_sols,
            include_aquifers=include_aquifers,
            include_components=include_components)
        for index, block in enumerate(restart_blocks)
        if "PRESSURE" in block
    ]
    if include_well_sols and states and states[0].get("wellSol"):
        states = process_well_states(
            states,
            split_wells_on_sign_change=split_wells_on_sign_change,
            remove_closed_wells=remove_closed_wells,
            remove_crossflow=remove_crossflow,
            set_to_closed_tol=set_to_closed_tol,
        )
    return states, restart_blocks


def process_well_states(states, split_wells_on_sign_change=False,
                        remove_closed_wells=True, remove_crossflow=True,
                        set_to_closed_tol=0.0):
    """Port of MRST-0's ``processWellStates``.

    Three passes over the well solutions, each optional:

    * a well that changes sign over the run is **split in two** -- one
      ``(inj)`` and one ``(prod)`` -- because a single well cannot be
      matched against a target that flips direction, and plotting one
      curve through the flip is meaningless;
    * connections flowing **against** the well's own sign are crossflow;
      their flux is zeroed rather than counted as production;
    * a well whose reservoir rate is at or below ``set_to_closed_tol``
      is **treated as shut** for that step, and one shut in every step is
      dropped entirely.
    """
    if not states:
        return states

    nphase = 3
    nw = len(states[0].get("wellSol") or [])
    if nw == 0:
        return states

    if split_wells_on_sign_change:
        states = _split_on_sign_change(states, nphase)
        nw = len(states[0]["wellSol"])

    always_closed = np.ones(nw, dtype=bool)
    for state in states:
        for well in state.get("wellSol") or []:
            if not well.get("status"):
                continue
            sign = float(well.get("sign", 0.0))
            if remove_crossflow and well.get("flux") is not None:
                flux = np.asarray(well["flux"], dtype=float)
                flux[flux * sign < 0] = 0.0
                well["flux"] = flux
            if float(well.get("resv", 0.0)) * sign <= set_to_closed_tol:
                _shut(well, nphase)
        status = np.array([bool(w.get("status"))
                           for w in state.get("wellSol") or []])
        if status.size == always_closed.size:
            always_closed &= ~status

    if remove_closed_wells and np.any(always_closed):
        keep = ~always_closed
        for state in states:
            state["wellSol"] = [w for w, k in zip(state["wellSol"], keep) if k]
    return states


def _shut(well, nphase):
    """Blank a well's flow for this step, keeping its shape."""
    ncon = np.size(well.get("cells", []))
    well["status"] = False
    well["cstatus"] = np.zeros(ncon, dtype=bool)
    well["resv"] = 0.0
    well["flux"] = np.zeros(ncon)
    well["cqs"] = np.zeros((ncon, nphase))


def _split_on_sign_change(states, nphase):
    """Port of the splitWellsOnSignChange branch.

    A well that both injects and produces over the run becomes two, named
    ``<name> (inj)`` and ``<name> (prod)``. Each is open only in the
    steps where it had that sign; in the others it carries zero flow.
    """
    nw = len(states[0]["wellSol"])
    signs = np.array([[float(w.get("sign", 0.0)) for w in s["wellSol"]]
                      for s in states])

    for k in range(nw):
        if np.all(signs[:, k] == signs[0, k]):
            continue
        first_sign = signs[0, k]
        first_name = '%s (inj)' % states[0]["wellSol"][k].get("name", "")
        second_name = '%s (prod)' % states[0]["wellSol"][k].get("name", "")
        if first_sign < 0:
            first_name, second_name = second_name, first_name

        for state in states:
            original = state["wellSol"][k]
            twin = dict(original)
            state["wellSol"].append(twin)

            original["name"] = first_name
            original["status"] = bool(original.get("status")) and \
                float(original.get("sign", 0.0)) == first_sign
            original["sign"] = first_sign
            if not original["status"]:
                _shut(original, nphase)

            twin["name"] = second_name
            twin["status"] = bool(twin.get("status")) and \
                float(twin.get("sign", 0.0)) != first_sign
            twin["sign"] = -first_sign
            if not twin["status"]:
                _shut(twin, nphase)
    return states


def _restart_prefix(prefix) -> str:
    pth, nm = os.path.split(str(prefix))
    name, ext = os.path.splitext(nm)
    if ext.upper() in {".UNRST", ".FUNRST"}:
        return os.path.join(pth, name)
    return str(prefix)


def _resolve_restart_file(prefix: str) -> Path:
    base = Path(prefix)
    for suffix in (".UNRST", ".FUNRST"):
        candidate = base.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find unified restart file for prefix {prefix!r}")


def _restart_blocks_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    current: dict[str, dict[str, Any]] = {}

    for record in records:
        keyword = str(record.get("name", "")).strip().upper()
        if not keyword:
            continue
        if keyword == "SEQNUM":
            if current:
                blocks.append(current)
            current = {}
        _put_keyword(current, keyword, {
            "values": record.get("values"),
            "type": record.get("type", ""),
        })

    if current:
        blocks.append(current)
    return blocks


def _put_keyword(block: dict[str, Any], keyword: str, item: dict[str, Any]) -> None:
    if keyword not in block:
        block[keyword] = item
        return

    old = block[keyword]["values"]
    new = item["values"]
    if isinstance(old, np.ndarray) and isinstance(new, np.ndarray):
        block[keyword]["values"] = np.concatenate([old, new])
    elif isinstance(old, list) and isinstance(new, list):
        block[keyword]["values"] = old + new
    elif isinstance(old, str) and isinstance(new, str):
        block[keyword]["values"] = old + new
    else:
        block[keyword]["values"] = [old, new]


def _restart_unit_name(unit_system, rsspec, raw_restart) -> str:
    if unit_system:
        return str(unit_system).upper()

    if isinstance(rsspec, dict) and rsspec.get("unit"):
        return str(rsspec["unit"]).upper()

    ih = raw_restart.get("INTEHEAD", {}).get("values", np.zeros(0))
    ih = np.asarray(ih).ravel()
    if ih.size > 2:
        units = ["METRIC", "FIELD", "LAB"]
        return units[max(0, min(int(ih[2]) - 1, len(units) - 1))]
    return "METRIC"


def _restart_block_to_state(block, G, units, index, rsspec,
                            include_well_sols, *, include_aquifers=False,
                            include_components=False):
    nc = int(G["cells"]["num"])
    pressure = _mapped_restart_vector(
        block, "PRESSURE", G, np.full(nc, 200.0))
    pressure = np.asarray(pressure, dtype=float) * float(units["press"])

    phase_names = _active_restart_phases(block)
    saturation_keywords = {'WAT': 'SWAT', 'OIL': 'SOIL', 'GAS': 'SGAS'}
    saturation = []
    missing = []
    for phase in phase_names:
        keyword = saturation_keywords[phase]
        if keyword in block:
            saturation.append(_mapped_restart_vector(block, keyword, G))
        else:
            saturation.append(None)
            missing.append(len(saturation) - 1)
    if len(missing) > 1:
        raise ValueError('Saturation output found for less than nPh-1 phases')
    if missing:
        present = [np.asarray(v, dtype=float) for v in saturation
                   if v is not None]
        saturation[missing[0]] = 1.0 - np.sum(present, axis=0)
    s = np.column_stack(saturation)
    sw = (s[:, phase_names.index('WAT')]
          if 'WAT' in phase_names else np.zeros(nc))
    sg = (s[:, phase_names.index('GAS')]
          if 'GAS' in phase_names else np.zeros(nc))

    state: dict[str, Any] = {
        "pressure": pressure,
        "s": np.asarray(s, dtype=float),
        # PRST's GenericBlackOilModel consumes these aliases.  Keep them as
        # independent arrays: MATLAB state structs have value semantics and
        # changing an alias must not mutate the canonical restart matrix.
        "sW": np.array(sw, dtype=float, copy=True),
        "sG": np.array(sg, dtype=float, copy=True),
        "wellSol": [],
    }

    if "RS" in block:
        state["rs"] = _mapped_restart_vector(block, "RS", G) * (
            float(units["gasvol_s"]) / float(units["liqvol_s"]))
    else:
        # FAHM's edited convertRestartToStates explicitly stores scalar 0.
        state["rs"] = 0.0
    if "RV" in block:
        state["rv"] = _mapped_restart_vector(block, "RV", G) * (
            float(units["liqvol_s"]) / float(units["gasvol_s"]))
    else:
        state["rv"] = 0.0
    if "SEQNUM" in block:
        seq = np.asarray(block["SEQNUM"]["values"]).ravel()
        if seq.size:
            state["seqnum"] = int(seq[0])
            state["index"] = int(seq[0])

    state["time"] = _restart_time(block, index, rsspec, units)
    _add_restart_cell_fields(state, block, G, units)
    if include_components:
        _add_restart_components(state, block, G,
                                is_eclipse=_is_eclipse_restart(block))

    if include_well_sols and "IWEL" in block:
        state["wellSol"] = _parse_well_solutions(block, G, units)
    if include_aquifers:
        aquifers = _parse_aquifer_solutions(block, G, units)
        if aquifers:
            state["aquiferSol"] = aquifers

    return state


def _restart_vector(block, keyword: str, default):
    values = block.get(keyword, {}).get("values", default)
    return np.asarray(values, dtype=float).ravel()


def _mapped_restart_vector(block, keyword: str, G, default=None):
    """Read one simulator-active vector and apply ``G.cells.eMap``.

    ``initGridFromEclipseOutput`` can remove zero-volume/disconnected cells
    after ECLIPSE has numbered its active rows.  MRST reduces pressure,
    saturation, Rs and Rv through ``eMap`` after constructing every state;
    taking the first ``G.cells.num`` entries is not equivalent.
    """
    nc = int(G["cells"]["num"])
    if default is None:
        default = np.zeros(nc)
    values = _restart_vector(block, keyword, default)
    if values.size == 1:
        return np.full(nc, float(values[0]))
    emap = G.get("cells", {}).get("eMap", slice(None))
    if not isinstance(emap, slice):
        indices = np.asarray(emap, dtype=int).ravel()
        if indices.size == nc and indices.size and indices.max() < values.size:
            return np.asarray(values[indices], dtype=float)
    if values.size < nc:
        raise ValueError('%s has %d rows for a %d-cell grid'
                         % (keyword, values.size, nc))
    return np.asarray(values[:nc], dtype=float)


def _active_restart_phases(block):
    """Port ``checkAndProcessInput``'s INTEHEAD phase map."""
    ih = np.asarray(block.get("INTEHEAD", {}).get("values", []),
                    dtype=int).ravel()
    indicator = int(ih[14]) if ih.size > 14 else 7
    mapping = {
        1: ('OIL',), 2: ('WAT',), 3: ('WAT', 'OIL'), 4: ('GAS',),
        5: ('WAT', 'GAS'), 6: ('OIL', 'GAS'),
        7: ('WAT', 'OIL', 'GAS'),
    }
    return mapping.get(indicator, mapping[7])


def _is_eclipse_restart(block):
    ih = np.asarray(block.get("INTEHEAD", {}).get("values", []),
                    dtype=int).ravel()
    return ih.size > 94 and int(ih[94]) in (100, 300, 500, 700)


def _add_restart_cell_fields(state, block, G, units):
    """Cell fields read by FAHM's exact ``convertRestartToStates`` path."""
    direct = {
        'POLYMER': ('cp', 1.0), 'SURFACT': ('cs', 1.0),
        'BW': ('bW', 1.0), 'BO': ('bO', 1.0),
        'BG': ('bG', float(units['liqvol_s']) / float(units['gasvol_s'])),
        'WATKR': ('krw', 1.0), 'OILKR': ('kro', 1.0),
        'GASKR': ('krg', 1.0),
        'VWAT': ('muW', float(units['viscosity'])),
        'VOIL': ('muO', float(units['viscosity'])),
        'VGAS': ('muG', float(units['viscosity'])),
        'DENW': ('rhoW', float(units['density'])),
        'DENO': ('rhoO', float(units['density'])),
        'DENG': ('rhoG', float(units['density'])),
        'FFACTO': ('FFACTO', 1.0), 'FFACTG': ('FFACTG', 1.0),
    }
    # Alternate keyword names used by some simulator releases.
    aliases = {
        'KRW': ('krw', 1.0), 'KRO': ('kro', 1.0),
        'KRG': ('krg', 1.0),
        'WAT_VISC': ('muW', float(units['viscosity'])),
        'OIL_VISC': ('muO', float(units['viscosity'])),
        'GAS_VISC': ('muG', float(units['viscosity'])),
    }
    assigned = set()
    for keyword, (field, factor) in tuple(direct.items()) + tuple(aliases.items()):
        if keyword in block and field not in assigned:
            state[field] = _mapped_restart_vector(block, keyword, G) * factor
            assigned.add(field)

    is_eclipse = _is_eclipse_restart(block)
    for keyword, field in (('PCOW', 'pcow'), ('PCOG', 'pcog'),
                           ('PPCW', 'ppcw')):
        if keyword not in block:
            continue
        values = _mapped_restart_vector(block, keyword, G) * float(units['press'])
        if keyword != 'PPCW' and not is_eclipse:
            values = -values
        state[field] = values

    ih = np.asarray(block.get('INTEHEAD', {}).get('values', [])).ravel()
    if ih.size >= 67:
        state['date'] = np.asarray(ih[64:67], dtype=int).copy()


def _add_restart_components(state, block, G, *, is_eclipse):
    prefixes = (('XMF', 'x'), ('YMF', 'y'), ('ZMF', 'components')) \
        if is_eclipse else (('XMF_', 'x'), ('YMF_', 'y'),
                            ('ZMF_', 'components'))
    for prefix, field in prefixes:
        columns = []
        for component in range(1, 1001):
            keyword = '%s%d' % (prefix, component)
            if keyword not in block:
                break
            columns.append(_mapped_restart_vector(block, keyword, G))
        if columns:
            state[field] = np.column_stack(columns)
    liquid_keyword = 'VMF' if is_eclipse else 'RS'
    if liquid_keyword in block:
        state['L'] = 1.0 - _mapped_restart_vector(
            block, liquid_keyword, G)
        pure_vapor = state['L'] == 0
        pure_liquid = state['L'] == 1
        pure_vapor[pure_liquid] = False
        state['flag'] = pure_liquid.astype(int) + 2 * pure_vapor.astype(int)


def _parse_aquifer_solutions(block, G, units):
    """Port ``getRestartAquiInfo`` + ``createAquiSol`` (Fetkovich)."""
    ih = np.asarray(block.get('INTEHEAD', {}).get('values', []),
                    dtype=int).ravel()
    if ih.size <= 47:
        return []
    naq = int(ih[40])
    if naq <= 0 or 'XAAQ' not in block or not any(
            name in block for name in ('ACAQ', 'ACAQ_1')):
        return []
    niaaq, nsaaq, nxaaq = int(ih[42]), int(ih[43]), int(ih[44])
    nicaq, nacaq = int(ih[45]), int(ih[47])
    iaaq = _restart_vector(block, 'IAAQ', [])
    saaq = _restart_vector(block, 'SAAQ', [])
    xaaq = _restart_vector(block, 'XAAQ', [])
    if min(niaaq, nsaaq, nxaaq, nicaq, nacaq) <= 0 or \
            xaaq.size != naq * nxaaq:
        return []
    type_indices = np.concatenate([
        np.asarray([9, 10], dtype=int) + k * niaaq for k in range(naq)])
    if type_indices.max(initial=-1) >= iaaq.size or \
            np.any(iaaq[type_indices] != 0):
        return []

    lookup = _ijk_to_active(G, np.asarray(G['cartDims'], dtype=int))
    qfactor = float(units['resvolume']) / float(units['time'])
    aquifers = []
    for k in range(naq):
        count_index = k * niaaq
        nconn = int(iaaq[count_index]) if count_index < iaaq.size else 0
        suffix = '' if naq == 1 else '_%d' % (k + 1)
        icaq = _restart_vector(block, 'ICAQ' + suffix, [])
        acaq = _restart_vector(block, 'ACAQ' + suffix, [])
        numbers = _restart_vector(block, 'ACAQNUM' + suffix, [k + 1])
        offsets_i = np.arange(nconn, dtype=int) * nicaq
        if nconn and offsets_i[-1] + 2 < icaq.size:
            cijk = np.column_stack([icaq[offsets_i + j]
                                    for j in range(3)]).astype(int)
            cells = _connection_cells(cijk, np.asarray(G['cartDims']), lookup)
        else:
            cells = np.zeros(0, dtype=int)
        offsets_a = np.arange(nconn, dtype=int) * nacaq
        flux = (acaq[offsets_a] * qfactor
                if nconn and offsets_a[-1] < acaq.size else np.zeros(nconn))
        xoff, soff = k * nxaaq, k * nsaaq
        pressure = xaaq[xoff + 1] * float(units['press'])
        q_w = xaaq[xoff] * qfactor
        volume = (saaq[soff + 1] - xaaq[xoff + 2]) * float(units['resvolume'])
        aquifers.append({
            'cells': np.asarray(cells, dtype=int),
            'pressure': float(pressure), 'qW': float(q_w),
            'flux': np.asarray(flux, dtype=float), 'volume': float(volume),
            'num': int(numbers[0]) if numbers.size else k + 1,
        })
    return aquifers


def _restart_time(block, index: int, rsspec, units) -> float:
    if isinstance(rsspec, dict):
        times = np.asarray(rsspec.get("time", []), dtype=float).ravel()
        if index < times.size:
            return float(times[index] * units["time"])

    if "DOUBHEAD" in block:
        values = np.asarray(block["DOUBHEAD"]["values"], dtype=float).ravel()
        if values.size:
            return float(values[0] * units["time"])
    return 0.0


#: IWEL's well-type code. 1 is a producer; the rest are injectors of
#: one phase or another, and all inject, so their sign is +1.
_PRODUCER = 1


def _parse_well_solutions(block, G, units):
    """Well solutions for one restart step, from its ZWEL/IWEL/... records.

    Connections are mapped from their (i, j, k) back to active cell
    numbers through the grid's index map, so ``cells`` lines up with the
    rest of the state.
    """
    from .get_restart_well_info import getRestartWellInfo

    records = {name: block[name]["values"] for name in
               ("INTEHEAD", "ZWEL", "IWEL", "SWEL", "XWEL", "ICON", "SCON",
                "XCON") if name in block}
    info, ih = getRestartWellInfo(records)
    u = _rate_units(units)

    cart_dims = np.asarray(G.get("cartDims",
                                 (ih["nx"], ih["ny"], ih["nz"])), dtype=int)
    ijk_to_active = _ijk_to_active(G, cart_dims)

    wells = []
    for k, w in enumerate(info):
        sign = -1.0 if w.get("type") == _PRODUCER else 1.0
        cells = _connection_cells(w.get("cijk"), cart_dims, ijk_to_active)
        cqs = np.asarray(w.get("cqs", np.zeros((0, 3))), dtype=float)
        cqr = np.asarray(w.get("cqr", np.zeros(0)), dtype=float)
        wells.append({
            "name": w.get("name") or ("W%d" % (k + 1)),
            "status": bool(w.get("stat")),
            "sign": sign,
            "type": "bhp" if w.get("cntr") == 0 else "rate",
            "cells": cells,
            "cstatus": np.asarray(w.get("cstat", np.zeros(0)),
                                  dtype=int) > 0,
            "qOs": float(w.get("qOs") or 0.0) * u["ql"],
            "qWs": float(w.get("qWs") or 0.0) * u["ql"],
            "qGs": float(w.get("qGs") or 0.0) * u["qg"],
            "bhp": float(w.get("bhp") or 0.0) * units["press"],
            "resv": float(w.get("qr") or 0.0) * u["qr"],
            "flux": cqr * u["qr"],
            "cqs": cqs * u["ql"],
            "depth": (float(w["depth"]) * units["length"]
                      if w.get("depth") is not None else None),
        })
    return wells


def _rate_units(units):
    """Port of ``getUnits``' rate entries.

    A rate is a volume per unit time, so these are not in the shared
    factor table directly: ql is sm3/day in METRIC and stb/day in FIELD,
    qg is sm3/day and Mscf/day, and qr is a reservoir volume per day.
    """
    return {"ql": units["liqvol_s"] / units["time"],
            "qg": units["gasvol_s"] / units["time"],
            "qr": units["resvolume"] / units["time"]}


def _ijk_to_active(G, cart_dims):
    """A lookup from cartesian index to active cell number, or None.

    Without an index map every cell is taken as active, which is what a
    grid built straight from the restart looks like.
    """
    n = int(np.prod(cart_dims))
    lookup = np.full(n, -1, dtype=int)
    index_map = G.get("cells", {}).get("indexMap")
    if index_map is None:
        lookup[:] = np.arange(n)
    else:
        index_map = np.asarray(index_map, dtype=int).ravel()
        lookup[index_map] = np.arange(index_map.size)
    return lookup


def _connection_cells(cijk, cart_dims, lookup):
    """Map connection (i, j, k) -- 1-based, as ECLIPSE writes them -- to
    active cell numbers."""
    if cijk is None or np.size(cijk) == 0:
        return np.zeros(0, dtype=int)
    cijk = np.atleast_2d(np.asarray(cijk, dtype=int)) - 1
    nx, ny = int(cart_dims[0]), int(cart_dims[1])
    linear = cijk[:, 0] + nx * (cijk[:, 1] + ny * cijk[:, 2])
    linear = np.clip(linear, 0, lookup.size - 1)
    return lookup[linear]
