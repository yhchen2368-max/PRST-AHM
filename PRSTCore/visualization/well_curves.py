"""Port of MRST's ``plotWellSols.m`` -- well-solution curves over time.

MRST's interactive well plotter (``mrst-2026a/autodiff/ad-core/plotting/
plotWellSols.m``) draws one well property against time for a chosen set of
wells, with a control panel for the field, the unit system, the time scale,
and a row of toggles (grid, log-x, log-y, markers, legend, cumulative sum,
absolute value, zoom, stairs, control changes, use-timesteps).  This module
is the matplotlib equivalent: the same field extraction
(``getNamesFromWS``/``getData``), the same unit conversion
(``getWellUnit``), the same status masking and cumulative-sum weighting --
without the MATLAB GUI, because the plot itself is the part worth keeping.

Two data paths:

* :func:`plot_well_sols` takes well solutions in MRST's shape: a list of
  report steps, each a list of per-well dictionaries carrying ``bhp``,
  ``qOs``/``qWs``/``qGs``, ``status``, ``type``, ... -- exactly what
  ``simulateScheduleAD`` returns and what PRSTCore's runs produce per step.
* :func:`plot_well_rates` is the convenience wrapper for this repo's runs:
  it loads ``results/<case>_full/well_rates.csv`` (one row per well per
  report step) into that shape and plots it, so a finished simulation can be
  looked at with one call.

Nothing here needs VTK or Qt; matplotlib is the only dependency.
"""

from __future__ import annotations

import csv
import os

import numpy as np

try:
    import matplotlib.pyplot as _plt
except Exception:  # pragma: no cover - matplotlib is optional for the solver
    _plt = None

__all__ = [
    "load_well_rates", "well_sol_field_names", "plot_well_sols",
    "plot_well_rates", "well_field_label",
    "load_history", "plot_history", "plot_block_data",
]

#: CSV columns (PRSTCore's run_t142_full.py) -> well-solution dictionary keys.
_CSV_TO_WELLSOL = {
    "qO_sm3d": "qOs", "qW_sm3d": "qWs", "qG_sm3d": "qGs",
    "bhp_bar": "bhp", "status": "status",
}


def load_well_rates(csv_path):
    """Read ``well_rates.csv`` into MRST-shaped well solutions.

    Returns ``(wellsols, timesteps)`` where ``wellsols[s][w]`` is a dict of
    well-solution fields (``bhp`` in bar, ``qOs/qWs/qGs`` in Sm^3/day,
    ``status``) and ``timesteps`` is the per-step cumulative time in days.
    """
    rows = []
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    if not rows:
        return [], []
    steps = {}
    times = {}
    for row in rows:
        step = int(row["step"])
        well = row["well"]
        ws = {"name": well}
        for column, field in _CSV_TO_WELLSOL.items():
            try:
                ws[field] = float(row[column])
            except (KeyError, ValueError):
                continue
        steps.setdefault(step, []).append(ws)
        times[step] = float(row["time_days"])
    order = sorted(steps)
    wellsols = [steps[s] for s in order]
    timesteps = [times[s] for s in order]
    return wellsols, timesteps


def well_sol_field_names(ws):
    """Scalar numeric fields of a well solution -- port of ``getNamesFromWS``.

    A well solution dictionary mixes rates, pressures, status and control
    bookkeeping; only the one-scalar-per-well numeric entries are plottable.
    """
    return [name for name, value in ws.items()
            if isinstance(value, (int, float, np.number))
            and np.asarray(value).ndim == 0][::-1]


def well_field_label(field, unit_system="metric", cumulative=False):
    """The y-axis label and unit for a well-solution field -- port of ``getWellUnit``.

    Returns ``(label, unit_name, scale)`` where ``scale`` converts a value in
    PRSTCore's internal units (Sm^3/day for rates, bar for pressure) into
    ``unit_system``.  Mirrors MRST's rate/pressure/ratio branches.
    """
    key = str(field).lower()
    us = str(unit_system).lower()

    # Time unit of the rates' denominator.
    if us == "si":
        t_str, t_scale = "s", 1.0 / 86400.0        # m^3/s from m^3/day
    elif us == "lab":
        t_str, t_scale = "h", 1.0 / 24.0           # cm^3/h from m^3/day
    else:  # metric, field -- per day
        t_str, t_scale = "d", 1.0

    if key in ("qos", "qor", "qws", "qwr", "qgs", "qgr", "qts", "qtr",
               "rate", "qosft", "qtsft", "qwsft"):
        phase = {"qo": "oil", "qw": "water", "qg": "gas", "qt": "total"}.get(
            key[:2], "total")
        if key.endswith("r"):
            place = "reservoir"
        else:
            place = "surface"
        if cumulative:
            title = "%s: cumulative %s %s production (%s)" % (
                field, place, "rate", phase)
        else:
            title = "%s: well %s rate (%s)" % (field, place, phase)
        # m^3/day (internal) -> chosen unit.
        if us == "field":
            if phase == "gas" and not key.endswith("r"):
                # Surface gas in ft^3/day (scf/day).
                return title, "scf/d", 35.314666721489
            return title, "stb/d", 6.289810770432105
        if us == "lab":
            return title, "cm^3/h", 1e6 * t_scale
        if us == "si":
            return title, "m^3/s", t_scale
        return title, "m^3/d", 1.0

    if key in ("bhp", "pressure"):
        title = "%s: bottom hole pressure" % field
        if us == "field":
            return title, "psia", 14.503773773
        if us == "si":
            return title, "Pa", 1e5
        if us == "lab":
            return title, "atm", 1.0 / 1.01325
        return title, "bar", 1.0

    if key in ("gor", "wgr", "ogr"):
        title = "%s: ratio at surface conditions" % field
        if us == "field":
            return title, "Mscf/stb", 1.0 / 5.614583333
        return title, "Sm^3/Sm^3", 1.0

    if key in ("status", "sign", "val", "type"):
        return field, "", 1.0
    return field, "", 1.0


def _line_styles(dataset_index):
    return ("-", "--", "-.", ":")[dataset_index % 4]


def _marker(dataset_index):
    return ("o", "x", "d", "s", "^", "v", ">", "<", "*")[dataset_index % 9]


def plot_well_sols(wellsols, time=None, field="bhp", wells=None,
                   datasets=None, dataset_names=None, unit_system="metric",
                   time_scale="days", logx=False, logy=False, grid=True,
                   legend=True, markers=False, cumsum=False, abs_value=True,
                   stairs=False, show_control_changes=False, use_timesteps=True,
                   linewidth=2, ax=None, figsize=(10, 6)):
    """Plot well-solution curves -- port of MRST ``plotWellSols``.

    Parameters
    ----------
    wellsols : list[list[dict]] or list[list[list[dict]]]
        A single dataset is a list of report steps, each a list of per-well
        solution dicts (``bhp``, ``qOs``, ...).  Pass a list of such datasets
        to compare several runs; then ``time`` and ``dataset_names`` should
        match.
    time : list[float], optional
        Cumulative time per step, in days (defaults to step number).
    field : str
        Which well property to plot (default ``'bhp'``).
    wells : list[str], optional
        Wells to include (default: all).
    unit_system : {'metric', 'field', 'si', 'lab'}
    time_scale : {'days', 'years', 'hours', 'minutes', 'seconds'}
    logx, logy, grid, legend, markers, cumsum, abs_value, stairs :
        MRST plotWellSols toggles.
    show_control_changes : bool
        Annotate where a well's control ``type`` changes between steps.
    use_timesteps : bool
        Space x by the actual time rather than by step number.
    """
    if _plt is None:
        raise ImportError("matplotlib is required to plot well solutions")
    if not wellsols:
        raise ValueError("no well solutions to plot")

    if datasets is None:
        datasets = [wellsols]
    else:
        datasets = [wellsols] + list(datasets)
    ndata = len(datasets)

    if time is None:
        has_timesteps = False
        time = [list(range(len(ds))) for ds in datasets]
    else:
        has_timesteps = True
        if not isinstance(time[0], (list, tuple)):
            time = [time]
        time = [np.asarray(t, dtype=float) for t in time]
        for t in time:
            if np.any(np.diff(t) <= 0):
                t[:] = np.cumsum(t)

    if dataset_names is None:
        dataset_names = ["data%d" % (i + 1) for i in range(ndata)]

    # Well names, in order of first appearance across all steps (a well that
    # only opens later in the schedule is still selectable).
    well_names = []
    for step in datasets[0]:
        for w in step:
            name = ("%s" % (w.get("name", "well%d" % (len(well_names) + 1)))
                    if isinstance(w, dict)
                    else "well%d" % (len(well_names) + 1))
            if name not in well_names:
                well_names.append(name)
    if wells is None:
        selected = list(well_names)
    else:
        wanted = [str(w) for w in wells]
        selected = [w for w in well_names if w in wanted]
        if not selected:
            raise ValueError("none of %r are wells in the data (have %r)"
                             % (wells, well_names))

    # Time scale factor: days -> requested unit.
    scale = {"days": 1.0, "years": 1.0 / 365.25, "hours": 24.0,
             "minutes": 1440.0, "seconds": 86400.0}
    t_scale = scale.get(str(time_scale).lower(), 1.0)
    t_label = str(time_scale).lower()

    if ax is None:
        _, ax = _plt.subplots(figsize=figsize)

    xunit = 1.0
    plotted = []
    cmap = _plt.get_cmap("tab10")
    for i, ds in enumerate(datasets):
        t = time[i] / t_scale if has_timesteps else np.arange(len(ds))
        line = _line_styles(i)
        marker = _marker(i) if markers else None
        for j, wname in enumerate(selected):
            values = np.full(len(ds), np.nan, dtype=float)
            status = np.ones(len(ds), dtype=float)
            types = []
            for k, step in enumerate(ds):
                ws = next((w for w in step if w.get("name") == wname), None)
                if ws is None:
                    continue
                try:
                    values[k] = float(ws[field])
                except (KeyError, TypeError, ValueError):
                    continue
                if "status" in ws:
                    status[k] = float(ws["status"])
                if "type" in ws:
                    types.append(str(ws["type"]))
                else:
                    types.append("")

            active = status > 0
            d = values.copy()
            if cumsum and has_timesteps:
                # MRST: sum dt * rate over active steps (xunit keeps the
                # unit consistent when x is rescaled).
                d = np.cumsum(xunit * np.diff(np.concatenate([[0.0], t]))
                              * active * d)
            d[~active] = np.nan
            if abs_value:
                d = np.abs(d)

            label = wname
            if ndata > 1:
                label = "%s (%s)" % (wname, dataset_names[i])
            plotter = ax.step if stairs else ax.plot
            plotter(t, d, linestyle=line, marker=marker, linewidth=linewidth,
                    color=cmap((i * len(selected) + j) % 10) if ndata == 1
                    else cmap(i % 10), label=label)
            plotted.append(label)

            if show_control_changes and types:
                changed = [0] + [k + 1 for k in range(len(types) - 1)
                                 if types[k] != types[k + 1]]
                for k in changed:
                    if np.isfinite(d[k]):
                        txt = types[k] if k == 0 or types[k] != types[k - 1] \
                            else types[k]
                        ax.annotate(txt, (t[k], d[k]),
                                    textcoords="offset points", xytext=(0, 8),
                                    ha="center", fontsize=8,
                                    bbox=dict(boxstyle="round,pad=0.2",
                                              fc="0.85"))

    title, ylabel, yscale = well_field_label(field, unit_system,
                                             cumulative=cumsum)
    ax.set_title(title)
    ax.set_xlabel("Time [%s]" % t_label if has_timesteps else "Step #")
    ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, which="both", alpha=0.3)
    ax.set_xscale("log" if logx else "linear")
    ax.set_yscale("log" if logy else "linear")

    # Include the zero axis unless "zoom to data".
    if not logy:
        vals = [line.get_ydata() for line in ax.lines]
        if vals:
            flat = np.concatenate([v[np.isfinite(v)] for v in vals])
            if flat.size:
                lo, hi = float(flat.min()), float(flat.max())
                if lo > 0:
                    ax.set_ylim(0, hi)
                elif hi < 0:
                    ax.set_ylim(lo, 0)
                else:
                    ax.set_ylim(min(lo, 0.0), max(hi, 0.0))
    if legend and plotted:
        ax.legend(loc="best")
    return ax


def plot_well_rates(csv_path, field="bhp", **kwargs):
    """Load ``well_rates.csv`` and plot it -- one call after a run."""
    wellsols, timesteps = load_well_rates(csv_path)
    if not wellsols:
        raise ValueError("no well-rate rows in %r" % csv_path)
    return plot_well_sols(wellsols, time=timesteps, field=field, **kwargs)


#: History-CSV column aliases -> canonical well-solution field names.
_HISTORY_FIELD_ALIASES = {
    "qos": "qOs", "qo_sm3d": "qOs", "oil_rate": "qOs", "qo": "qOs",
    "qws": "qWs", "qw_sm3d": "qWs", "water_rate": "qWs", "qw": "qWs",
    "qgs": "qGs", "qg_sm3d": "qGs", "gas_rate": "qGs", "qg": "qGs",
    "bhp": "bhp", "bhp_bar": "bhp", "wbhp": "bhp", "pressure": "bhp",
}
_HISTORY_TIME_ALIASES = {"time_days", "time", "days", "t", "day"}
_HISTORY_WELL_ALIASES = {"well", "name", "wellname"}


#: Block/field-level series available in the 2D panel: (key, Chinese label).
_BLOCK_FIELDS = [
    ("qO", "产油量"), ("qL", "产液量"), ("qW", "产水量"), ("wcut", "含水"),
    ("cumO", "累积产油"), ("cumL", "累积产液"), ("cumW", "累积产水"),
    ("pavg", "压力"),
]
_BLOCK_KEY = {label: key for key, label in _BLOCK_FIELDS}
_BLOCK_VOLUME_UNIT = {"metric": "m^3", "field": "stb", "si": "m^3",
                      "lab": "cm^3"}


def load_history(csv_path):
    """Read a history CSV into ``{well: {field: (time_days, values)}}``.

    Accepts the same columns as ``well_rates.csv`` (``time_days``, ``well``,
    ``qO_sm3d``/``qW_sm3d``/``qG_sm3d``/``bhp_bar``) plus common aliases.
    Rates are taken as Sm^3/day and pressures as bar -- i.e. already in the
    internal units the well-solution plots assume.
    """
    import csv as _csv

    rows = []
    with open(csv_path, newline="") as handle:
        for row in _csv.DictReader(handle):
            rows.append(row)
    if not rows:
        return {}
    columns = {str(key).strip().lower(): key for key in rows[0].keys()}

    time_col = next((columns[a] for a in _HISTORY_TIME_ALIASES
                     if a in columns), None)
    well_col = next((columns[a] for a in _HISTORY_WELL_ALIASES
                     if a in columns), None)
    field_cols = {columns[a]: field for a, field in _HISTORY_FIELD_ALIASES.items()
                  if a in columns}
    if time_col is None or well_col is None:
        raise ValueError(
            "history CSV needs a time column (%s) and a well column (%s)"
            % (sorted(_HISTORY_TIME_ALIASES), sorted(_HISTORY_WELL_ALIASES)))

    history = {}
    for row in rows:
        try:
            t = float(row[time_col])
        except (TypeError, ValueError):
            continue
        well = str(row[well_col]).strip()
        for column, field in field_cols.items():
            try:
                v = float(row[column])
            except (TypeError, ValueError):
                continue
            history.setdefault(well, {}).setdefault(field, ([], []))
            ts, vs = history[well][field]
            ts.append(t)
            vs.append(v)
    for well, fields in history.items():
        for field, (ts, vs) in fields.items():
            order = np.argsort(ts)
            history[well][field] = (np.asarray(ts)[order],
                                    np.asarray(vs)[order])
    return history


def plot_history(ax, history, field, wells=None, unit_system="metric",
                 time_scale="days", markersize=6, label_suffix=" (hist)"):
    """Overlay history points on an axes already holding simulated curves.

    ``history`` is the dict produced by :func:`load_history`.  The same
    unit/time scaling as the simulated curves is applied, so history-match
    points land exactly on top of the model output.  Per-well colors follow
    the order in ``wells`` (default: insertion order of ``history``).
    """
    if _plt is None or not history:
        return ax
    scale = {"days": 1.0, "years": 1.0 / 365.25, "hours": 24.0,
             "minutes": 1440.0, "seconds": 86400.0}
    t_scale = scale.get(str(time_scale).lower(), 1.0)
    _, _, yscale = well_field_label(field, unit_system)
    order = list(wells) if wells is not None else list(history)
    cmap = _plt.get_cmap("tab10")
    for idx, well in enumerate(order):
        if well not in history or field not in history[well]:
            continue
        t, v = history[well][field]
        ax.plot(t / t_scale, np.asarray(v, float) * yscale,
                linestyle="none", marker="o", markersize=markersize,
                color=cmap(idx % 10), label="%s%s" % (well, label_suffix))
    return ax


def _enable_cjk_font():
    """Pick an installed CJK font so Chinese axis labels render."""
    from matplotlib import font_manager, rcParams
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
                  "PingFang SC", "WenQuanYi Zen Hei", "Source Han Sans SC"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            rcParams["font.sans-serif"] = [name] + list(
                rcParams.get("font.sans-serif", []))
            rcParams["axes.unicode_minus"] = False
            return name
    return None


def plot_block_data(wellsols, times, pavg, field, *, ax=None,
                    unit_system="metric", time_scale="days", linewidth=2,
                    figsize=(10, 6)):
    """Plot one block/field-level quantity aggregated over all wells.

    ``field`` is a Chinese label (e.g. ``'含水'`` for water cut or
    ``'累积产油'`` for cumulative oil).  Rates are summed from the per-well
    solutions (Sm^3/day; only the *producing* part counts, so injectors do
    not drag field totals negative), cumulative quantities integrate them
    over the report-step time steps, and ``pavg`` (per-step mean reservoir
    pressure in bar, from the state) backs the ``'压力'`` series.
    """
    if _plt is None or not wellsols:
        raise ValueError("no well solutions to aggregate")
    if ax is None:
        _, ax = _plt.subplots(figsize=figsize)
    _enable_cjk_font()

    key = _BLOCK_KEY.get(str(field))
    if key is None:
        raise ValueError("unknown block field %r (have %s)"
                         % (field, sorted(_BLOCK_KEY)))

    t = np.asarray(times, dtype=float)
    dt = np.diff(np.concatenate([[0.0], t]))
    # wellSol rates carry the reservoir sign convention: producers are
    # negative, injectors positive.  Field *production* totals therefore
    # count -q only when q<0, so injectors never drag them down.
    qO = np.array([sum(max(-float(w.get("qOs", 0.0) or 0.0), 0.0)
                       for w in step) for step in wellsols])
    qW = np.array([sum(max(-float(w.get("qWs", 0.0) or 0.0), 0.0)
                       for w in step) for step in wellsols])
    qL = qO + qW
    wcut = np.where(qL > 0.0, qW / np.maximum(qL, 1e-12), 0.0)
    series = {
        "qO": qO, "qL": qL, "qW": qW, "wcut": wcut,
        "cumO": np.cumsum(qO * dt),
        "cumL": np.cumsum(qL * dt),
        "cumW": np.cumsum(qW * dt),
    }
    if pavg is not None:
        series["pavg"] = np.asarray(pavg, dtype=float)
    values = series.get(key)
    if values is None or np.asarray(values).size == 0:
        raise ValueError("block field %r has no data" % field)

    scale = {"days": 1.0, "years": 1.0 / 365.25, "hours": 24.0,
             "minutes": 1440.0, "seconds": 86400.0}
    t_scale = scale.get(str(time_scale).lower(), 1.0)
    us = str(unit_system).lower()

    if key in ("qO", "qL", "qW"):
        _, unit, yscale = well_field_label("qOs", us)
    elif key in ("cumO", "cumL", "cumW"):
        yscale = well_field_label("qOs", us)[2]
        unit = _BLOCK_VOLUME_UNIT.get(us, "m^3")
    elif key == "wcut":
        yscale, unit = 1.0, "-"
    else:  # pavg
        _, unit, yscale = well_field_label("bhp", us)

    ylabel = "%s [%s]" % (field, unit)
    ax.plot(t / t_scale, np.asarray(values, float) * yscale,
            linestyle="-", linewidth=linewidth, color="tab:blue")
    ax.set_title(ylabel)
    ax.set_xlabel("Time [%s]" % str(time_scale).lower())
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    vals = np.asarray(values, float) * yscale
    vals = vals[np.isfinite(vals)]
    if vals.size:
        lo, hi = float(vals.min()), float(vals.max())
        if lo > 0.0:
            ax.set_ylim(0.0, hi)
        elif hi < 0.0:
            ax.set_ylim(lo, 0.0)
        else:
            ax.set_ylim(min(lo, 0.0), max(hi, 0.0))
    return ax


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot a PRSTCore well_rates.csv (port of MRST plotWellSols)")
    parser.add_argument("csv", help="path to well_rates.csv")
    parser.add_argument("--field", default="bhp",
                        help="well property to plot (default: bhp)")
    parser.add_argument("--wells", nargs="*", default=None,
                        help="wells to include (default: all)")
    parser.add_argument("--units", default="metric",
                        choices=("metric", "field", "si", "lab"))
    parser.add_argument("--timescale", default="days",
                        choices=("days", "years", "hours", "minutes", "seconds"))
    parser.add_argument("--cumsum", action="store_true",
                        help="plot the cumulative sum (rates only)")
    parser.add_argument("--abs", action="store_false", dest="abs_value",
                        help="do not take absolute values")
    parser.add_argument("--logy", action="store_true")
    parser.add_argument("--logx", action="store_true")
    parser.add_argument("--stairs", action="store_true")
    parser.add_argument("--save", default=None, help="save figure to PATH")
    args = parser.parse_args()

    ax = plot_well_rates(args.csv, field=args.field, wells=args.wells,
                         unit_system=args.units, time_scale=args.timescale,
                         cumsum=args.cumsum, abs_value=args.abs_value,
                         logx=args.logx, logy=args.logy, stairs=args.stairs)
    if args.save:
        ax.figure.savefig(args.save, dpi=150)
        print("saved ->", args.save)
    else:
        _plt.show()
