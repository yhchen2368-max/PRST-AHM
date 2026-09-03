"""Convert deck units to MRST/SI conventions.

1:1 Python translation of MRST model-io/deckformat/deckinput/convertDeckUnits.m
"""

import numpy as np
import re


def convert_deck_units(deck, output_unit="SI", verbose=False):
    """Convert ECLIPSE deck units to MRST SI conventions.

    Parameters
    ----------
    deck : dict
        Deck structure from read_eclipse_deck.
    output_unit : str
        Output unit system ('SI' default).
    verbose : bool

    Returns
    -------
    dict
        Unit-converted deck.
    """
    assert "RUNSPEC" in deck, "Input does not appear to be a valid deck"

    from ..unit_conversion_factors import unit_conversion_factors

    input_unit = _deck_unit_name(deck["RUNSPEC"])
    u = unit_conversion_factors(input_unit, output_unit)

    if "PCUNIT" not in deck:
        deck["PCUNIT"] = u["press"]

    for sect in list(deck.keys()):
        if sect in ("RUNSPEC", "GRID", "PROPS", "SOLUTION", "SCHEDULE"):
            converter = _converters.get(sect)
            if converter:
                deck[sect] = converter(deck[sect], u)
        elif sect in ("UnhandledKeywords", "PCUNIT", "SUMMARY", "REGIONS"):
            continue
        else:
            if verbose:
                print(f"No converter needed in section '{sect}'.")

    # ``convertDeckUnits`` has already converted the unit system marker to
    # SI at this point.  Keep the factors as private parsing metadata so
    # schedule conversion can apply the same positional conversions as
    # MRST's convertSCHEDULE/convertControl routines.
    deck["_unit_factors"] = dict(u)

    return deck


def _deck_unit_name(rspec):
    """Determine deck unit system."""
    valid = ["METRIC", "FIELD", "LAB", "PVT_M", "SI"]
    for v in valid:
        if v in rspec:
            return v
    return "METRIC"


def _convert_runspec(rspec, u):
    """Convert RUNSPEC unit system to output unit."""
    usys = ["METRIC", "FIELD", "LAB", "PVT_M", "SI"]
    for us in usys:
        rspec.pop(us, None)
    rspec[u["unit_out"].upper()] = True
    return rspec


def _convert_grid(grid, u):
    """Convert GRID section units."""
    if not grid:
        return grid
    perm_kws = ["PERMX", "PERMY", "PERMZ", "PERMXX", "PERMYY", "PERMZZ",
                "PERMXY", "PERMXZ", "PERMYX", "PERMYZ", "PERMZX", "PERMZY"]
    for kw in list(grid.keys()):
        kw_upper = kw.upper() if isinstance(kw, str) else str(kw)
        if kw_upper in perm_kws:
            if isinstance(grid[kw], (int, float, list, np.ndarray)):
                arr = _to_numeric_array(grid[kw])
                if arr is not None:
                    grid[kw] = arr * u["perm"]
        elif kw_upper in ("DXV", "DYV", "DZV", "DEPTHZ", "DX", "DY", "DZ", "TOPS", "COORD", "ZCORN"):
            if isinstance(grid[kw], (int, float, list, np.ndarray)):
                arr = _to_numeric_array(grid[kw])
                if arr is not None:
                    grid[kw] = arr * u["length"]
        elif kw_upper in ("PORV", "MINPV", "MINPVV"):
            # convertDeckUnits.m: these three are reservoir volumes and
            # share one factor.  Left unconverted they are silently in the
            # deck's own units, which is a no-op under METRIC and a factor
            # of 6.29 out under FIELD -- and pore volume feeds the
            # accumulation term of every equation.
            if isinstance(grid[kw], (int, float, list, np.ndarray)):
                arr = _to_numeric_array(grid[kw])
                if arr is not None:
                    factor = float(u.get("liqvol_r", u.get("resvolume", 1.0)))
                    grid[kw] = arr * factor
    return grid


def _expand_repeat_token(tok):
    s = str(tok).strip()
    m = re.fullmatch(r'([+-]?\d+)\*([+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eEdD][+-]?\d+)?)', s)
    if not m:
        return None
    n = int(m.group(1))
    v = float(m.group(2).replace('D', 'E').replace('d', 'e'))
    if n < 0:
        return None
    return [v] * n


def _to_numeric_array(val):
    if isinstance(val, (int, float, np.floating, np.integer)):
        return np.asarray([float(val)], dtype=float)

    try:
        arr = np.asarray(val)
    except Exception:
        return None

    # Fast path: already numeric.
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(float)

    # Object/string path: expand repetition tokens and parse scalar numbers.
    out = []
    flat = arr.ravel().tolist()
    for item in flat:
        rep = _expand_repeat_token(item)
        if rep is not None:
            out.extend(rep)
            continue
        s = str(item).strip()
        if not s:
            continue
        try:
            out.append(float(s.replace('D', 'E').replace('d', 'e')))
        except Exception:
            return None

    return np.asarray(out, dtype=float)


def _convert_props(props, u):
    """Convert PROPS using the same column units as MRST ``convertPROPS``."""
    if not props:
        return props
    pscale = float(u.get('press', 1.0))
    lvol_s = float(u.get('liqvol_s', u.get('volume', 1.0)))
    lvol_r = float(u.get('liqvol_r', u.get('resvolume', lvol_s)))
    gvol_s = float(u.get('gasvol_s', u.get('gas_volume', 1.0)))
    gvol_r = lvol_r
    vfac = float(u.get('viscosity', 1.0))
    cfac = float(u.get('compressibility', 1.0))
    dfac = float(u.get('density', 1.0))
    rsfac = gvol_s / lvol_s
    bgfac = gvol_r / gvol_s
    bofac = lvol_r / lvol_s

    # ``readMisciblePVTTable`` represents PVTO/PVTG as a per-region list
    # of key records.  Keep the same structure beside the legacy flattened
    # arrays and convert it before table interpolation needs to infer a
    # boundary from units.  See convertDeckUnits.m cases PVTG and PVTO.
    miscible = props.get('_miscible_pvt_records')
    if isinstance(miscible, dict):
        for keyword, regions in miscible.items():
            for region in regions:
                for record in region:
                    if keyword == 'PVTG':
                        if (record.size - 1) % 3:
                            raise ValueError('PVTG record has invalid column count')
                        record[0] *= pscale
                        data = record[1:].reshape((-1, 3))
                        data[:, 0] *= lvol_s / gvol_s
                        data[:, 1] *= bgfac
                        data[:, 2] *= vfac
                    elif keyword == 'PVTO':
                        if (record.size - 1) % 3:
                            raise ValueError('PVTO record has invalid column count')
                        record[0] *= rsfac
                        data = record[1:].reshape((-1, 3))
                        data[:, 0] *= pscale
                        data[:, 1] *= bofac
                        data[:, 2] *= vfac

    def convert_pvto(value):
        # Eclipse PVTO: rs, pb, bo, mu, then continuation rows (p, bo, mu)
        arr = _to_numeric_array(value)
        if arr is None or arr.ndim != 1:
            return arr
        i = 0
        while i + 3 < arr.size:
            # Preserve the raw Rs value for structural detection. Comparing
            # converted Rs against raw continuation pressures corrupts table
            # boundaries for FIELD decks.
            raw_rs = float(arr[i])
            arr[i] *= rsfac
            arr[i + 1] *= pscale
            arr[i + 3] *= vfac
            i += 4
            while i + 2 < arr.size and arr[i] > max(5.0, 5.0 * abs(raw_rs)):
                arr[i] *= pscale
                arr[i + 2] *= vfac
                i += 3
        return arr

    for kw in list(props.keys()):
        kwu = str(kw).upper()
        value = props[kw]
        if value is None:
            continue
        arr = _to_numeric_array(value)
        if arr is None:
            continue
        if kwu in ('PVDG', 'PVDS'):
            # MRST: [pressure, gasvol_r/gasvol_s, viscosity]
            if arr.ndim == 1:
                n = (arr.size // 3) * 3
                tab = arr[:n].reshape(-1, 3)
                tab[:, 0] *= pscale; tab[:, 1] *= bgfac; tab[:, 2] *= vfac
                arr[:n] = tab.ravel()
            else:
                arr[:, 0] *= pscale; arr[:, 1] *= bgfac; arr[:, 2] *= vfac
        elif kwu == 'PVDO':
            # model-io/deckformat/deckinput/convertDeckUnits.m, case PVDO:
            # [pressure, liqvol_r/liqvol_s, viscosity].  Same shape as
            # PVDG, but the FVF column is a *liquid* reservoir/surface
            # ratio.  Without this a FIELD deck keeps psi and cP, which
            # puts every table lookup past the last row and leaves the oil
            # a thousand times too viscous -- what stopped both SPE10
            # decks (PVDO) from converging while SPE9 (PVTO) was fine.
            if arr.ndim == 1:
                n = (arr.size // 3) * 3
                tab = arr[:n].reshape(-1, 3)
                tab[:, 0] *= pscale; tab[:, 1] *= bofac; tab[:, 2] *= vfac
                arr[:n] = tab.ravel()
            else:
                arr[:, 0] *= pscale; arr[:, 1] *= bofac; arr[:, 2] *= vfac
        elif kwu == 'PVTW':
            # MRST: [pressure, liquidvol_r/liquidvol_s, compressibility,
            #        viscosity, compressibility]
            if arr.ndim == 1:
                n = (arr.size // 5) * 5
                tab = arr[:n].reshape((-1, 5))
                tab[:, 0] *= pscale; tab[:, 1] *= bofac
                tab[:, 2] *= cfac; tab[:, 3] *= vfac; tab[:, 4] *= cfac
                arr[:n] = tab.ravel()
            else:
                arr[:, 0] *= pscale; arr[:, 1] *= bofac
                arr[:, 2] *= cfac; arr[:, 3] *= vfac
                if arr.shape[1] >= 5: arr[:, 4] *= cfac
        elif kwu == 'PVCDO':
            # model-io/deckformat/deckinput/convertDeckUnits.m, case
            # PVCDO: [reference pressure, reference FVF,
            # compressibility, reference viscosity, viscosibility].
            if arr.ndim == 1:
                n = (arr.size // 5) * 5
                tab = arr[:n].reshape(-1, 5)
                tab[:, 0] *= pscale; tab[:, 2] *= cfac
                tab[:, 3] *= vfac; tab[:, 4] *= cfac
                arr[:n] = tab.ravel()
            else:
                arr[:, 0] *= pscale; arr[:, 2] *= cfac
                arr[:, 3] *= vfac; arr[:, 4] *= cfac
        elif kwu == 'PVTO':
            arr = convert_pvto(value)
        elif kwu in ('SGFN', 'SWFN', 'GSF'):
            # model-io/deckformat/deckinput/convertDeckUnits.m:
            # [saturation, relative permeability, capillary pressure].
            # The parser keeps Eclipse table rows in one numeric array, so
            # scale just column three for every saturation region.
            arr = _scale_columns(arr, [1.0, 1.0, pscale])
        elif kwu in ('SGOF', 'SWOF', 'SGWFN', 'SLGOF'):
            # model-io/deckformat/deckinput/convertDeckUnits.m, lines
            # 396--403: only the fourth (capillary-pressure) column has
            # pressure units.
            arr = _scale_columns(arr, [1.0, 1.0, 1.0, pscale])
        elif kwu == 'DENSITY':
            # Eclipse DENSITY is [oil, water, gas] surface density.
            arr *= dfac
        elif kwu == 'ROCK':
            if arr.size >= 1: arr[0] *= pscale
            if arr.size >= 2: arr[1] *= cfac
            if arr.size >= 3: arr[2] *= cfac
            if arr.size >= 4: arr[3] *= cfac
        props[kw] = arr
    return props


def _convert_solution(solution, u):
    """Convert the SOLUTION keywords used by MRST ``initStateDeck``.

    This follows ``convertSOLUTION`` in MRST's convertDeckUnits.m.  The
    parser has already expanded ECLIPSE repeat syntax, so vector fields can
    be scaled directly.
    """
    if not solution:
        return solution

    pressure_keys = {'PBUB', 'PRESSURE'}
    saturation_keys = {'SGAS', 'SOIL', 'SWAT', 'XMF', 'YMF', 'ZMF'}
    for key, value in list(solution.items()):
        upper = str(key).upper()
        if upper in pressure_keys:
            solution[key] = _scale_numeric(value, u['press'])
        elif upper == 'RS':
            solution[key] = _scale_numeric(value, u.get('gasvol_s', u['gas_volume']) / u.get('liqvol_s', u['volume']))
        elif upper == 'RV':
            solution[key] = _scale_numeric(value, u.get('liqvol_s', u['volume']) / u.get('gasvol_s', u['gas_volume']))
        elif upper == 'EQUIL':
            solution[key] = _scale_columns(value, [u['length'], u['press'], u['length'], u['press'], u['length'], u['press']])
        elif upper in {'PBVD', 'PDVD'}:
            solution[key] = _scale_columns(value, [u['length'], u['press']])
        elif upper == 'RSVD':
            solution[key] = _scale_columns(value, [u['length'], u.get('gasvol_s', u['gas_volume']) / u.get('liqvol_s', u['volume'])])
        elif upper == 'RVVD':
            solution[key] = _scale_columns(value, [u['length'], u.get('liqvol_s', u['volume']) / u.get('gasvol_s', u['gas_volume'])])
        elif upper in saturation_keys or upper in {'RPTSOL', 'RPTRST', 'OUTSOL'}:
            continue
    return solution


def _convert_schedule(schedule, u):
    # read_schedule retains textual default markers (e.g. ``4*``) because
    # their position is significant.  Conversion of individual controls is
    # therefore performed in _convert_deck_schedule_to_mrst, where the
    # control keyword and selected target are known.  Timestep values can
    # be converted eagerly without losing that information.
    if not schedule:
        return schedule
    schedule['_time_factor'] = float(u['time'])
    schedule['_unit_factors'] = dict(u)
    return schedule


def _scale_numeric(value, factor):
    try:
        arr = _to_numeric_array(value)
    except Exception:
        arr = None
    if arr is None:
        return value
    return arr * float(factor)


def _scale_columns(value, factors):
    if isinstance(value, list):
        # MRST keeps RSVD/RVVD/PBVD tables in a cell array, one entry per
        # equilibrium region.  Preserve that representation while applying
        # the same column factors to every cell.
        return [_scale_columns(entry, factors) for entry in value]
    try:
        arr = _to_numeric_array(value)
    except Exception:
        return value
    if arr is None:
        return value
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        if not factors:
            return arr
        width = len(factors)
        if arr.size % width == 0:
            arr = arr.reshape((-1, width))
        else:
            out = arr.copy()
            for index, factor in enumerate(factors):
                out[index::width] *= float(factor)
            return out
    out = arr.copy()
    for index, factor in enumerate(factors[:out.shape[1]]):
        out[:, index] *= float(factor)
    return out


_converters = {
    "RUNSPEC": _convert_runspec,
    "GRID": _convert_grid,
    "PROPS": _convert_props,
    "SOLUTION": _convert_solution,
    "SCHEDULE": _convert_schedule,
}
