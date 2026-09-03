"""PRSTCore's own unified HDF5 result format (no GeoView / GeoCode).

The GUI (``simulator_gui``) writes every finished run into this on-disk
schema and can read any directory produced this way -- including the
``<deck>_run_prst`` folders already computed under ``examples/SpE1``:

    <case_dir>/
        states.h5          /pressure, /swat, /soil, /sgas, /dates_iso8601
        wells.h5           /wells/<name>   (rows = report steps; attrs)
        cell_indices.h5    /active_to_natural
        manifest.json      grid / states / wells metadata

Array layout (the same schema the ResSimWorkbench established for both the
PRST and JutulDarcy engines):

* ``states.h5`` arrays are ``(n_active, n_steps + 1)`` -- column 0 is state0.
* ``wells.h5`` per-well matrices are ``(7, n_steps)`` -- one column per report
  step, with the well columns ``time_days, WBHP, WOPR, WWPR, WGPR, WWIR,
  WGIR`` and production/injection split into non-negative columns.
* ``dates_iso8601`` has ``n_steps + 1`` ISO-8601 strings (t = 0 first).

Numpy + h5py only -- no VTK, no Qt -- so results can be written from the
solver environment and loaded by any viewer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

#: Well-matrix columns and units -- identical to GeoCode ``jutul_run.jl``.
WELLS_COLUMNS = ["time_days", "WBHP", "WOPR", "WWPR", "WGPR", "WWIR", "WGIR"]
WELLS_UNITS = {
    "time_days": "d", "WBHP": "bar",
    "WOPR": "sm3/d", "WWPR": "sm3/d", "WGPR": "sm3/d",
    "WWIR": "sm3/d", "WGIR": "sm3/d",
}
_STATE_FILES = ("pressure", "swat", "soil", "sgas")
_SAT_DATASETS = ("swat", "soil", "sgas")

__all__ = [
    "WELLS_COLUMNS", "WELLS_UNITS", "write", "load", "H5Results",
    "to_field_dict", "build_well_matrices", "build_state_arrays",
]


def _as_0based(index_map, n_natural):
    """Normalise an active->natural index map to 0-based column-major."""
    idx = np.asarray(index_map, dtype=np.int64).ravel()
    if idx.size and int(idx.max()) >= n_natural:
        idx = idx - 1  # tolerate 1-based maps
    return idx


def _phase_array(state, key, ncells):
    """Active-cell float array for one state entry (or None)."""
    raw = (state or {}).get(key)
    if raw is None:
        return None
    arr = np.atleast_1d(np.asarray(raw, dtype=float))
    return arr[:ncells] if arr.size else np.zeros(ncells)


def build_well_matrices(well_names, steps):
    """Per-well ``(7, n_steps)`` matrices from PRST per-step wellSol.

    PRST wellSol follows the ECLIPSE sign convention: rates are negative for
    production and positive for injection.  Split into non-negative
    production / injection columns, exactly like ``jutul_run.jl``.
    """
    out = {name: np.zeros((len(WELLS_COLUMNS), len(steps))) for name in well_names}
    for j, info in enumerate(steps):
        sol = {w.get("name"): w for w in (info.get("wellSol") or [])
               if w.get("name")}
        time_days = float(info.get("time_days", 0.0))
        for name in well_names:
            w = sol.get(name, {})
            bhp = w.get("bhp")
            bhp_val = float(np.atleast_1d(np.asarray(bhp, dtype=float))[0]) \
                if bhp is not None else np.nan
            qs = {}
            for k in ("qOs", "qWs", "qGs"):
                raw = w.get(k)
                if raw is not None:
                    arr = np.atleast_1d(np.asarray(raw, dtype=float))
                    qs[k] = float(arr[0]) if arr.size else 0.0
                else:
                    qs[k] = 0.0
            qos = qs["qOs"] * 86400.0
            qws = qs["qWs"] * 86400.0
            qgs = qs["qGs"] * 86400.0
            out[name][0, j] = time_days
            out[name][1, j] = bhp_val * 1e-5          # Pa -> bar
            out[name][2, j] = max(-qos, 0.0)          # WOPR
            out[name][3, j] = max(-qws, 0.0)          # WWPR
            out[name][4, j] = max(-qgs, 0.0)          # WGPR
            out[name][5, j] = max(qws, 0.0)           # WWIR
            out[name][6, j] = max(qgs, 0.0)           # WGIR
    return out


def build_state_arrays(model, state0, steps, start_date):
    """Column-stack state0 + report steps into HDF5-ready arrays.

    Returns ``(pressure, saturations, dates)`` with ``pressure`` and each
    saturation ``(n_active, n_steps + 1)`` and ``dates`` a list of ISO-8601
    strings of length ``n_steps + 1`` (t = 0 first).
    """
    ncells = int(getattr(model, "cells", None) or len(state0["pressure"]))
    pressure = np.column_stack(
        [np.asarray(_phase_array(state0, "pressure", ncells), dtype=float)]
        + [np.asarray(_phase_array(info.get("state"), "pressure", ncells),
                      dtype=float) for info in steps])

    sats = {}
    for prst_key, h5_name in (("sW", "swat"), ("sG", "sgas")):
        first = _phase_array(state0, prst_key, ncells)
        if first is None:
            continue
        cols = [np.asarray(first, dtype=float)]
        for info in steps:
            arr = _phase_array(info.get("state"), prst_key, ncells)
            cols.append(np.asarray(arr if arr is not None
                                   else np.zeros(ncells), dtype=float))
        sats[h5_name] = np.column_stack(cols)

    # Oil saturation is 1 - sW - sG when the model does not carry sO itself.
    if "swat" in sats and "soil" not in sats:
        soil = 1.0 - sats["swat"]
        if "sgas" in sats:
            soil = soil - sats["sgas"]
        sats["soil"] = soil

    start_dt = datetime.combine(start_date, datetime.min.time())
    dates = [start_dt.isoformat()]
    for info in steps:
        dates.append((start_dt + timedelta(
            days=float(info["time_days"]))).isoformat())
    return pressure, sats, dates


# --------------------------------------------------------------------------
# writer
# --------------------------------------------------------------------------
def write(outdir, *, simulator, case, grid_dims, n_active, active_to_natural,
          pressure, saturations=None, dates_iso8601=None, wells=None,
          extra=None):
    """Write the unified HDF5 result set into ``outdir``.

    Parameters
    ----------
    pressure : (n_active, n_steps + 1) array (column 0 = state0)
    saturations : dict[str, (n_active, n_steps + 1) array] (swat/soil/sgas)
    dates_iso8601 : list[str], length n_steps + 1, ISO-8601 (t = 0 first)
    wells : dict[str, (7, n_steps) array]  (rows = WELLS_COLUMNS)
    extra : dict  -- extra keys merged into the manifest
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pressure = np.asarray(pressure, dtype=float)
    n_steps_plus_1 = pressure.shape[1]

    # ---- states.h5 -----------------------------------------------------
    for attempt in (1, 2):
        _write_states(outdir, pressure, saturations, dates_iso8601,
                      n_steps_plus_1)
        if _verify_states(outdir):
            break
        if attempt == 2:
            raise IOError("states.h5 failed to verify after rewrite")

    # ---- wells.h5 ------------------------------------------------------
    with _h5(outdir / "wells.h5") as f:
        g = f.create_group("/wells")
        for name, mat in (wells or {}).items():
            mat = np.asarray(mat, dtype=float)
            ds = g.create_dataset(str(name), data=mat)
            ds.attrs["columns"] = ",".join(WELLS_COLUMNS)
            ds.attrs["units"] = ",".join(WELLS_UNITS[c] for c in WELLS_COLUMNS)

    # ---- cell_indices.h5 -----------------------------------------------
    with _h5(outdir / "cell_indices.h5") as f:
        idx = _as_0based(active_to_natural, int(np.prod(grid_dims)))
        f.create_dataset("/active_to_natural", data=idx.astype(np.int64))

    # ---- manifest.json -------------------------------------------------
    grid_dims = [int(v) for v in grid_dims]
    sat_names = sorted(k for k in _SAT_DATASETS if k in (saturations or {}))
    manifest = {
        "simulator": simulator,
        "case": str(case),
        "grid": {"nx": grid_dims[0], "ny": grid_dims[1], "nz": grid_dims[2],
                 "n_active": int(n_active)},
        "states": {"file": "states.h5", "datasets": sat_names,
                   "n_steps": int(n_steps_plus_1 - 1)},
        "wells": {"file": "wells.h5",
                  "names": sorted(wells or {}),
                  "columns": WELLS_COLUMNS, "units": WELLS_UNITS},
        "restart": "none",
        "completed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    if extra:
        manifest.update(extra)
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return outdir


def _h5(path, mode="w"):
    """Open an HDF5 file; ``mode`` defaults to write/truncate for writers."""
    import h5py
    return h5py.File(path, mode)


def h5_string_dtype():
    """HDF5 dtype for fixed-length strings (compatible with h5py read)."""
    import h5py
    return h5py.string_dtype(encoding="utf-8")


def _default_dates(n):
    return [datetime(1999, 9, 1, 0, 0, 0).isoformat()] * int(n)


def _write_states(outdir, pressure, saturations, dates_iso8601, n_steps_plus_1):
    """Write states.h5 (pressure, saturations, dates)."""
    with _h5(outdir / "states.h5") as f:
        f.create_dataset("/pressure", data=pressure)
        for name, arr in (saturations or {}).items():
            f.create_dataset("/%s" % name, data=np.asarray(arr, dtype=float))
        dates = list(dates_iso8601) if dates_iso8601 is not None else \
            _default_dates(n_steps_plus_1)
        f.create_dataset("/dates_iso8601",
                         data=np.asarray(dates, dtype=h5_string_dtype()))


def _verify_states(outdir):
    """Reopen states.h5 (read-only) and confirm /pressure survived on disk."""
    try:
        with _h5(outdir / "states.h5", "r") as f:
            return "/pressure" in f and f["/pressure"].shape[1] > 0
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------
# reader
# --------------------------------------------------------------------------
@dataclass
class H5Results:
    """Loaded result set (pressure is transposed to (steps + 1, n_active))."""

    manifest: dict
    pressure: np.ndarray                 # (n_steps + 1, n_active)
    active_to_natural: np.ndarray        # (n_active,)
    swat: np.ndarray | None = None       # (n_steps + 1, n_active)
    soil: np.ndarray | None = None
    sgas: np.ndarray | None = None
    wells: dict = field(default_factory=dict)   # {name: DataFrame}
    dates: list = field(default_factory=list)

    @property
    def grid_dims(self):
        g = self.manifest.get("grid", {})
        return [int(g.get("nx", 0)), int(g.get("ny", 0)), int(g.get("nz", 0))]

    @property
    def n_steps(self):
        return max(0, int(self.manifest.get("states", {}).get("n_steps", 0)))


def load(case_dir):
    """Load a unified HDF5 result directory (PRST or JutulDarcy)."""
    case_dir = Path(case_dir)
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))

    with _h5(case_dir / "states.h5", "r") as f:
        pressure = np.asarray(f["/pressure"][:], dtype=float).T   # -> (steps+1, active)
        sats = {}
        for name in _SAT_DATASETS:
            if "/%s" % name in f:
                sats[name] = np.asarray(f["/%s" % name][:], dtype=float).T
        raw_dates = np.asarray(f["/dates_iso8601"][:]).ravel()
        dates = [d.decode("utf-8") if isinstance(d, bytes) else str(d)
                 for d in raw_dates]

    with _h5(case_dir / "cell_indices.h5", "r") as f:
        active_to_natural = np.asarray(f["/active_to_natural"][:], dtype=np.int64)

    wells = {}
    well_file = case_dir / "wells.h5"
    if well_file.exists():
        with _h5(well_file, "r") as f:
            if "/wells" in f:
                g = f["/wells"]
                for name in g:
                    mat = np.asarray(g[name][:], dtype=float).T      # -> (steps, 7)
                    cols = g[name].attrs.get("columns")
                    cols = cols.split(",") if cols else WELLS_COLUMNS
                    wells[str(name)] = pd.DataFrame(mat, columns=cols)

    return H5Results(
        manifest=manifest, pressure=pressure, active_to_natural=active_to_natural,
        swat=sats.get("swat"), soil=sats.get("soil"), sgas=sats.get("sgas"),
        wells=wells, dates=dates)


def to_field_dict(jr):
    """Convert an :class:`H5Results` into per-cell field arrays.

    Returns a dict with ``pressure`` (n_steps + 1, n_natural),
    ``saturations`` {SWAT/SOIL/SGAS: (n_steps + 1, n_natural)},
    ``wellnames``, ``welldata`` (pandas DataFrame) and ``dates`` -- the
    GeoView-compatible results layout, produced here without GeoView.
    """
    import pandas as pd

    grid = jr.manifest["grid"]
    n_natural = int(grid["nx"]) * int(grid["ny"]) * int(grid["nz"])

    def scatter(arr):
        nat = np.full((arr.shape[0], n_natural), np.nan, dtype=float)
        nat[:, jr.active_to_natural] = arr
        return nat

    output = {"pressure": scatter(jr.pressure)}
    sats = {}
    for out_name, attr in (("SWAT", "swat"), ("SOIL", "soil"), ("SGAS", "sgas")):
        arr = getattr(jr, attr)
        if arr is not None:
            sats[out_name] = scatter(arr)
    output["saturations"] = sats

    welldata_frames = []
    for name, df in jr.wells.items():
        wf = df.drop(columns="time_days") if "time_days" in df.columns else df
        record0 = pd.DataFrame([{c: 0.0 for c in wf.columns}])
        wf = pd.concat([record0, wf], ignore_index=True)
        wf.insert(0, "WELL", name)
        wf.insert(1, "DATE", pd.to_datetime(jr.dates[: len(wf)]))
        welldata_frames.append(wf)
    output["welldata"] = (pd.concat(welldata_frames, ignore_index=True)
                          if welldata_frames else pd.DataFrame())
    output["wellnames"] = list(jr.wells.keys())
    output["dates"] = jr.dates
    return output
