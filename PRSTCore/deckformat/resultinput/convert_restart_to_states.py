"""Convert ECLIPSE restart data to PRSTCore/MRST-style states."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from .restart_contract import RestartContractError
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
                               include_mobilities=False,
                               unit_system=None):
    """Convert an ECLIPSE unified restart file to state dictionaries.

    Follows the local FAHM ``convertRestartToStates``. Each state carries
    pressure, saturations, Rs/Rv and time, and -- when the restart has
    well records -- a full well solution per step: name, open/shut, sign,
    control mode, perforated cells, rates, bhp, reservoir rate, and
    per-connection fluxes and phase rates.

    ``split_wells_on_sign_change``, ``remove_closed_wells``,
    ``remove_crossflow`` and ``set_to_closed_tol`` are applied by
    :func:`process_well_states`.
    """
    # FAHM explicitly disables flux/mobility reconstruction. Refuse requested
    # but unimplemented payloads, rather than pretending that they were read.
    del neighbors, add_trajectory
    prefix = _restart_prefix(prefix)
    from .read_eclipse_output_file_unfmt import read_eclipse_output_file_unfmt
    from .process_eclipse_restart_spec import process_eclipse_restart_spec
    from .restart_contract import RestartContractError, report_indices, time_vector
    from .restart_well_solutions import make_well_sols_consistent, merge_summary
    from .restart_summary import read_restart_summary

    rsspec = restart_info
    if isinstance(rsspec, tuple):
        rsspec = rsspec[0]
    if rsspec is None and Path(prefix + '.RSSPEC').exists():
        rsspec, _ = process_eclipse_restart_spec(prefix, "all")
    if rsspec is not None and rsspec.get('type') == 'multiple':
        files = rsspec['fnames']
        if any(not name for name in files):
            raise RestartContractError('Missing multiple restart file')
        restart_blocks = []
        for name in files:
            data = read_eclipse_output_file_unfmt(name)
            blocks = _restart_blocks_from_records(data.get('__records__', []))
            if len(blocks) != 1:
                raise RestartContractError('Multiple restart file must contain exactly one state')
            restart_blocks.extend(blocks)
    else:
        restart_file = _resolve_restart_file(prefix)
        data = read_eclipse_output_file_unfmt(str(restart_file))
        restart_blocks = _restart_blocks_from_records(data.get('__records__', []))
    if not restart_blocks:
        raise RestartContractError('Restart contains no states')
    if rsspec is not None and len(restart_blocks) != len(rsspec['time']):
        raise RestartContractError('RSSPEC/restart state count mismatch')
    if steps is not None and np.size(steps):
        selected = np.asarray(steps).ravel(order='F')
        if np.any(selected != selected.astype(int)) or np.any(selected < 0) or np.any(selected >= len(restart_blocks)):
            raise RestartContractError('Restart step index out of range')
        restart_blocks = [restart_blocks[i] for i in selected.astype(int)]
    first = restart_blocks[0]
    unit_name = _restart_unit_name(unit_system, None, first)
    if unit_name not in ('METRIC', 'FIELD'):
        raise RestartContractError('FAHM restart supports METRIC/FIELD units only')
    units = unit_conversion_factors(unit_name)
    has_flux = any(k.startswith(('FLR', 'FLO')) for k in first)
    if has_flux and include_fluxes:
        raise NotImplementedError('Reservoir flux reconstruction is outside FAHM includeFluxes=false; request false explicitly')
    if include_mobilities and include_fluxes and has_flux:
        raise NotImplementedError('Mobility reconstruction is not implemented')
    states = []
    for index, block in enumerate(restart_blocks):
        for required in ('INTEHEAD', 'DOUBHEAD', 'PRESSURE'):
            if required not in block or not np.size(block[required]['values']):
                raise RestartContractError(f'Restart state {index}: missing {required}')
        if not np.array_equal(np.asarray(block['INTEHEAD']['values'])[[2, 8, 9, 10, 11, 14]],
                              np.asarray(first['INTEHEAD']['values'])[[2, 8, 9, 10, 11, 14]]):
            raise RestartContractError('Restart grid/unit/phase header changed across states')
        state = _restart_block_to_state(
            block, G, units, index, None, include_well_sols and well_sols_from_restart,
            include_aquifers=include_aquifers, include_components=include_components)
        states.append(state)
    time_vector([s['time'] for s in states], 'restart DOUBHEAD')
    if include_well_sols:
        summary, tm = read_restart_summary(prefix, unit_name)
        if summary:
            if states[0]['time'] == 0 and tm[0] != 0:
                tm = np.r_[0., tm]
                summary = [[]] + summary
            selection = report_indices(tm, [s['time'] for s in states], allow_trailing=True)
            summary = [summary[i] for i in selection]
        if not well_sols_from_restart:
            if not summary:
                raise RestartContractError('Requested summary well solutions are missing')
            for state, wells in zip(states, summary):
                state['wellSol'] = deepcopy(wells)
        elif consistent_well_sols:
            if summary:
                merge_summary(states, summary, is_eclipse=_is_eclipse_restart(first),
                              program=int(first['INTEHEAD']['values'][94]),
                              include_components=include_components, connection_quantities=False)
            states = make_well_sols_consistent(states)
            if summary:
                merge_summary(states, summary, is_eclipse=_is_eclipse_restart(first),
                              program=int(first['INTEHEAD']['values'][94]),
                              include_components=False, well_quantities=False)
            states = process_well_states(
                states, split_wells_on_sign_change=split_wells_on_sign_change,
                remove_closed_wells=remove_closed_wells,
                remove_crossflow=remove_crossflow, set_to_closed_tol=set_to_closed_tol)
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

    states = deepcopy(states)
    nphase = states[0]['s'].shape[1] if 's' in states[0] else 3
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
        if status.size != always_closed.size:
            raise RestartContractError('Inconsistent well count in processWellStates')
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
            twin = deepcopy(original)
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
        candidate = Path(str(base) + suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find unified restart file for prefix {prefix!r}")


def _restart_blocks_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    current: dict[str, dict[str, Any]] = {}
    counts = {}
    next_suffix = None

    for record in records:
        keyword = str(record.get("name", "")).strip().upper()
        if not keyword:
            continue
        if keyword == "SEQNUM":
            if current:
                blocks.append(current)
            current = {}
            counts = {}
        if keyword == 'LGR':
            raise NotImplementedError('Restart LGR cannot be merged into the active grid')
        if keyword in ('ICAQNUM', 'SCAQNUM', 'ACAQNUM'):
            counts[keyword] = counts.get(keyword, 0) + 1
            next_suffix = '_' + str(counts[keyword])
            keyword += next_suffix
        elif next_suffix is not None:
            keyword += next_suffix
            next_suffix = None
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
    if ih.size <= 2 or int(ih[2]) not in (1, 2, 3):
        raise RestartContractError('Missing/unknown restart unit indicator')
    return ["METRIC", "FIELD", "LAB"][int(ih[2]) - 1]


def _restart_block_to_state(block, G, units, index, rsspec,
                            include_well_sols, *, include_aquifers=False,
                            include_components=False):
    nc = int(G["cells"]["num"])
    pressure = _mapped_restart_vector(block, "PRESSURE", G)
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
        saturation[missing[0]] = 1.0 - (np.sum(present, axis=0) if present else np.zeros(nc))
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
        "flux": np.array([]),
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

    if include_well_sols:
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
    if keyword not in block:
        raise RestartContractError('Missing restart field ' + keyword)
    values = _restart_vector(block, keyword, [])
    na = int(np.asarray(block['INTEHEAD']['values'])[11])
    if values.size != na:
        raise RestartContractError(f'{keyword} has {values.size} rows; INTEHEAD requires {na}')
    emap = G.get("cells", {}).get("eMap", slice(None))
    if not isinstance(emap, slice):
        indices = np.asarray(emap, dtype=int).ravel(order='F')
        if indices.size != nc or np.unique(indices).size != nc or np.any(indices < 0) or np.any(indices >= na):
            raise RestartContractError('Invalid G.cells.eMap')
        return values[indices].copy()
    if values.size != nc:
        raise RestartContractError(f'{keyword} has {values.size} rows for a {nc}-cell grid')
    return values.copy()


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
    if indicator not in mapping:
        raise RestartContractError('Unknown restart phase indicator')
    return mapping[indicator]


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

    for keyword, field in (('GAS_PRES', 'PGAS'), ('WAT_PRES', 'PWAT')):
        if keyword in block:
            state[field] = _mapped_restart_vector(block, keyword, G) * units['press']
    if 'TEMP' in block:
        state['T'] = (_mapped_restart_vector(block, 'TEMP', G) + units['tempoffset']) * units['temp']
    molar = 1000.0 if units['length'] == 1.0 else 453.59237 / units['liqvol_s']
    for keyword, field in (('BWAT', 'rhoWM'), ('BOIL', 'rhoOM'), ('BGAS', 'rhoGM')):
        if keyword in block:
            state[field] = _mapped_restart_vector(block, keyword, G) * molar
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
        ids = sorted(int(match.group(1)) for key in block
                     if (match := re.fullmatch(re.escape(prefix) + r'(\d+)', key)))
        if ids and (ids != list(range(1, max(ids) + 1)) or max(ids) > 1000):
            raise RestartContractError(f'{prefix}: component keywords must be contiguous from 1')
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
    if naq <= 0:
        return []
    payload = any(name.startswith(('IAAQ', 'SAAQ', 'XAAQ', 'ICAQ', 'ACAQ')) for name in block)
    if not payload:
        return []  # Source path: the model reconstructs wholly absent output.
    if 'XAAQ' not in block or not any(name in block for name in ('ACAQ', 'ACAQ_1')):
        raise RestartContractError('Incomplete aquifer output')
    niaaq, nsaaq, nxaaq = int(ih[42]), int(ih[43]), int(ih[44])
    nicaq, nacaq = int(ih[45]), int(ih[47])
    iaaq = _restart_vector(block, 'IAAQ', [])
    saaq = _restart_vector(block, 'SAAQ', [])
    xaaq = _restart_vector(block, 'XAAQ', [])
    if niaaq < 11 or nsaaq < 2 or nxaaq < 3 or nicaq < 3 or nacaq < 1 or \
            xaaq.size != naq * nxaaq or iaaq.size != naq * niaaq or saaq.size != naq * nsaaq:
        raise RestartContractError('Invalid aquifer record dimensions')
    type_indices = np.concatenate([
        np.asarray([9, 10], dtype=int) + k * niaaq for k in range(naq)])
    if type_indices.max(initial=-1) >= iaaq.size or \
            np.any(iaaq[type_indices] != 0):
        raise RestartContractError('Only Fetkovich aquifer restart records are supported')

    lookup = _ijk_to_active(G, np.asarray(G['cartDims'], dtype=int))
    qfactor = float(units['resvolume']) / float(units['time'])
    aquifers = []
    for k in range(naq):
        count_index = k * niaaq
        nconn = int(iaaq[count_index])
        if nconn < 0:
            raise RestartContractError('Negative aquifer connection count')
        suffix = '' if naq == 1 and 'ICAQ' in block else '_%d' % (k + 1)
        icaq = _restart_vector(block, 'ICAQ' + suffix, [])
        acaq = _restart_vector(block, 'ACAQ' + suffix, [])
        numbers = _restart_vector(block, 'ACAQNUM' + suffix, [])
        if numbers.size != 1 or numbers[0] < 1 or numbers[0] != int(numbers[0]):
            raise RestartContractError('Missing/invalid aquifer number')
        offsets_i = np.arange(nconn, dtype=int) * nicaq
        if icaq.size != nconn * nicaq or acaq.size != nconn * nacaq:
            raise RestartContractError('Aquifer connection array count mismatch')
        if nconn:
            cijk = np.column_stack([icaq[offsets_i + j]
                                    for j in range(3)]).astype(int)
            cells = _connection_cells(cijk, np.asarray(G['cartDims']), lookup)
        else:
            cells = np.zeros(0, dtype=int)
        offsets_a = np.arange(nconn, dtype=int) * nacaq
        flux = acaq[offsets_a] * qfactor
        xoff, soff = k * nxaaq, k * nsaaq
        pressure = xaaq[xoff + 1] * float(units['press'])
        q_w = xaaq[xoff] * qfactor
        volume = (saaq[soff + 1] - xaaq[xoff + 2]) * float(units['resvolume'])
        aquifers.append({
            'cells': np.asarray(cells, dtype=int),
            'pressure': float(pressure), 'qW': float(q_w),
            'flux': np.asarray(flux, dtype=float), 'volume': float(volume),
            'num': int(numbers[0]),
        })
    if len({aq['num'] for aq in aquifers}) != len(aquifers):
        raise RestartContractError('Duplicate aquifer numbers')
    return aquifers


def _restart_time(block, index: int, rsspec, units) -> float:
    values = np.asarray(block.get("DOUBHEAD", {}).get("values", []), dtype=float).ravel(order='F')
    if not values.size or not np.isfinite(values[0]):
        raise RestartContractError('Missing/nonfinite restart DOUBHEAD time')
    return float(values[0] * units["time"])


#: IWEL's well-type code. 1 is a producer; the rest are injectors of
#: one phase or another, and all inject, so their sign is +1.
_PRODUCER = 1


def _parse_well_solutions(block, G, units):
    from .restart_well_solutions import create_well_solutions
    return create_well_solutions(block, G, units)


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
        if index_map.size != int(G['cells']['num']) or np.unique(index_map).size != index_map.size or np.any(index_map < 0) or np.any(index_map >= n):
            raise RestartContractError('Invalid active-cell indexMap')
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
    if np.any(cijk < 0) or np.any(cijk >= np.asarray(cart_dims)):
        raise RestartContractError('Well/aquifer IJK outside cartesian grid')
    if np.any(lookup[linear] < 0):
        raise RestartContractError('Well/aquifer connection refers to inactive cell')
    return lookup[linear].copy()
