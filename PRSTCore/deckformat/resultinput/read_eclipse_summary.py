"""Python port of MRST's ``readEclipseSummaryUnFmt.m`` and
``convertSummaryToWellSols.m`` (mrst-2026a/model-io/deckformat/resultinput):
reads ECLIPSE unified summary files (``.SMSPEC`` + ``.UNSMRY``) and converts
them into per-report-step well-solution histories, e.g. for history matching
or plotting simulated results against a reference run.

Scope: unformatted (binary), *unified* summary files only (``.UNSMRY``) --
not the formatted ``.FSMSPEC``/``.FSMSPEC`` variant or the older split
``.S0001, .S0002, ...`` per-report-step layout. Both are rare in modern
ECLIPSE/OPM output and are not ported.
"""

from __future__ import annotations

from pathlib import Path

import numpy as _np

from .read_eclipse_output_file_unfmt import read_eclipse_output_file_unfmt

_FIELD_NAME = ":+:+:+:+"
_INTEHEAD_UNIT_NAMES = ("metric", "field", "lab", "pvt-m")

# Physical constants (mrst-2026a/core/utils/units/*.m), used to convert
# summary vectors from their on-disk unit system into SI.
_METER = 1.0
_DAY = 86400.0
_HOUR = 3600.0
_BARSA = 1.0e5
_ATM = 101325.0
_PSIA = (0.45359237 * 9.80665) / (0.0254 ** 2)
_STB = 42.0 * 231.0 * (0.0254 ** 3)
_FT3 = (0.3048) ** 3


def _get_units(unit: str) -> dict:
    unit = unit.lower()
    if unit == "metric":
        return {"p": _BARSA, "ql": _METER ** 3 / _DAY, "qg": _METER ** 3 / _DAY, "t": _DAY}
    if unit == "field":
        return {"p": _PSIA, "ql": _STB / _DAY, "qg": 1000.0 * _FT3 / _DAY, "t": _DAY}
    if unit == "lab":
        cm3 = (0.01) ** 3
        return {"p": _ATM, "ql": cm3 / _HOUR, "qg": cm3 / _HOUR, "t": _HOUR}
    if unit in ("pvt-m", "pvt_m"):
        return {"p": _ATM, "ql": _METER ** 3 / _DAY, "qg": _METER ** 3 / _DAY, "t": _DAY}
    raise ValueError(f"Unit {unit!r} not supported")


def read_eclipse_summary(prefix, keywords=None) -> dict:
    """Port of ``readEclipseSummaryUnFmt.m``.

    Parameters
    ----------
    prefix : str or Path
        Case-name prefix such that ``prefix + '.SMSPEC'`` /
        ``prefix + '.UNSMRY'`` name the specification/result files (a path
        ending in ``.SMSPEC``/``.UNSMRY`` is also accepted and stripped).
    keywords : list[str] or None
        If given, only summary vectors whose ``KEYWORDS`` entry is in this
        list are loaded (matches MRST's optional ``keyWords`` argument).

    Returns
    -------
    dict
        ``{'WGNAMES', 'KEYWORDS', 'UNITS', 'NUMS', 'data' ((n_selected,
        n_steps) array), 'STARTDAT', 'cartDims', 'intehead_unit', 'get',
        'get_names', 'get_keywords', 'get_unit'}``. ``get``/``get_names``/
        ``get_keywords``/``get_unit`` are convenience closures mirroring
        MRST's ``smry.get``/``smry.getNms``/``smry.getKws``/``smry.getUnit``
        accessor function handles.
    """
    prefix = str(prefix)
    for suffix in (".SMSPEC", ".UNSMRY"):
        if prefix.upper().endswith(suffix):
            prefix = prefix[: -len(suffix)]
            break

    smspec = read_eclipse_output_file_unfmt(prefix + ".SMSPEC")
    name_key = "WGNAMES" if "WGNAMES" in smspec else "NAMES"
    names = [n if n else _FIELD_NAME for n in smspec[name_key]["values"]]
    kwrds = [k if k else "empty" for k in smspec["KEYWORDS"]["values"]]
    units = [u.strip() for u in smspec["UNITS"]["values"]]
    nlist = len(names)
    nums = (
        _np.asarray(smspec["NUMS"]["values"], dtype=_np.int64)
        if "NUMS" in smspec
        else _np.zeros(nlist, dtype=_np.int64)
    )

    if keywords is not None:
        keyword_set = set(keywords)
        row_mask = _np.array([k in keyword_set for k in kwrds], dtype=bool)
    else:
        row_mask = _np.ones(nlist, dtype=bool)

    unsmry = read_eclipse_output_file_unfmt(prefix + ".UNSMRY")
    params_chunks = [rec["values"] for rec in unsmry["__records__"] if rec["name"] == "PARAMS"]
    data_full = _np.column_stack(params_chunks).astype(float) if params_chunks else _np.zeros((nlist, 0))

    row_names = _np.asarray(names, dtype=object)[row_mask]
    row_kwrds = _np.asarray(kwrds, dtype=object)[row_mask]
    row_units = _np.asarray(units, dtype=object)[row_mask]

    smry: dict = {
        "WGNAMES": row_names,
        "KEYWORDS": row_kwrds,
        "UNITS": row_units,
        "NUMS": nums[row_mask],
        "data": data_full[row_mask, :],
    }
    if "STARTDAT" in smspec:
        sd = _np.asarray(smspec["STARTDAT"]["values"], dtype=_np.int64)
        smry["STARTDAT"] = _np.array([sd[2], sd[1], sd[0]])  # [year, month, day]
    if "DIMENS" in smspec:
        smry["cartDims"] = tuple(int(x) for x in smspec["DIMENS"]["values"][1:4])
    smry["intehead_unit"] = (
        int(smspec["INTEHEAD"]["values"][0]) if "INTEHEAD" in smspec else None
    )

    def get(name, keyword):
        mask = (row_names == name) & (row_kwrds == keyword)
        rows = _np.where(mask)[0]
        if rows.size == 0:
            return None
        return smry["data"][rows[0], :]

    def get_names(keyword):
        mask = row_kwrds == keyword
        seen: list = []
        for n in row_names[mask]:
            if n not in seen:
                seen.append(n)
        return seen

    def get_keywords(name):
        mask = row_names == name
        seen: list = []
        for k in row_kwrds[mask]:
            if k not in seen:
                seen.append(k)
        return seen

    def get_unit(name, keyword):
        mask = (row_names == name) & (row_kwrds == keyword)
        rows = _np.where(mask)[0]
        return str(row_units[rows[0]]) if rows.size else None

    smry["get"] = get
    smry["get_names"] = get_names
    smry["get_keywords"] = get_keywords
    smry["get_unit"] = get_unit
    return smry


def _get_well_names(smry) -> list:
    wkw = {k for k in smry["KEYWORDS"] if str(k).startswith("W")}
    names: set = set()
    for kw in wkw:
        names.update(smry["get_names"](kw))
    names.discard(_FIELD_NAME)
    return sorted(names)


def convert_summary_to_well_sols(fn, unit=None):
    """Port of ``convertSummaryToWellSols.m``: builds a per-report-step
    well-solution history (``qOs``/``qWs``/``qGs``/``bhp``/``sign``, one
    dict per well) from an ECLIPSE summary file.

    Parameters
    ----------
    fn : str, Path, or dict
        Case-name prefix (see :func:`read_eclipse_summary`), or an
        already-loaded summary dict from that function.
    unit : {'metric', 'field', 'lab', 'pvt-m'} or None
        Unit system of the summary vectors. If omitted, inferred from the
        SMSPEC ``INTEHEAD`` record (falls back to ``'metric'`` with a
        warning if that is unavailable, matching MRST).

    Returns
    -------
    (well_sols, time)
        ``well_sols`` is a list (one entry per report step) of lists (one
        dict per well) with keys ``name``, ``bhp``, ``qOs``, ``qWs``,
        ``qGs``, ``sign`` -- all in SI units (Pa, m^3/s, s). ``time`` is a
        ``(n_steps,)`` array of cumulative time in seconds.
    """
    smry = fn if isinstance(fn, dict) else read_eclipse_summary(fn)

    if isinstance(unit, str):
        u = _get_units(unit)
    elif smry.get("intehead_unit") is not None:
        u = _get_units(_INTEHEAD_UNIT_NAMES[smry["intehead_unit"] - 1])
    else:
        u = _get_units("metric")

    wns = _get_well_names(smry)
    t = smry["get"](_FIELD_NAME, "TIME")
    time = _np.asarray(t, dtype=float) * u["t"] if t is not None else _np.zeros(0)

    nt = time.size
    nw = len(wns)
    qOs = _np.zeros((nt, nw))
    qWs = _np.zeros((nt, nw))
    qGs = _np.zeros((nt, nw))
    bhp = _np.zeros((nt, nw))

    for k, wn in enumerate(wns):
        akw = set(smry["get_keywords"](wn))

        if "WBHP" in akw:
            bhp[:, k] = smry["get"](wn, "WBHP") * u["p"]

        if "WOPR" in akw:
            qOs[:, k] = -smry["get"](wn, "WOPR") * u["ql"]

        if "WGPR" in akw:
            qGs[:, k] = -smry["get"](wn, "WGPR") * u["qg"]
        elif "WGOR" in akw:
            qGs[:, k] = qOs[:, k] * smry["get"](wn, "WGOR")

        if "WWPR" in akw:
            qWs[:, k] = -smry["get"](wn, "WWPR") * u["ql"]
        elif "WWCT" in akw:
            wcut = smry["get"](wn, "WWCT")
            with _np.errstate(divide="ignore", invalid="ignore"):
                qWs[:, k] = wcut * qOs[:, k] / (1.0 - wcut)

        if "WWIR" in akw:
            qWs[:, k] = qWs[:, k] + smry["get"](wn, "WWIR") * u["ql"]

        if "WGIR" in akw:
            qGs[:, k] = qGs[:, k] + smry["get"](wn, "WGIR") * u["qg"]

    well_sols = []
    for kt in range(nt):
        step = []
        for kw, wn in enumerate(wns):
            step.append({
                "name": wn,
                "bhp": float(bhp[kt, kw]),
                "qOs": float(qOs[kt, kw]),
                "qWs": float(qWs[kt, kw]),
                "qGs": float(qGs[kt, kw]),
                "sign": float(_np.sign(qWs[kt, kw] + qOs[kt, kw] + qGs[kt, kw])),
            })
        well_sols.append(step)
    return well_sols, time
