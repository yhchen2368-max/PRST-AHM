"""Save a run's per-step states so they can be looked at later.

A simulation script that writes only well rates throws away everything that
is per-cell, which is exactly what a 3D view needs.  This is the other half:
one file holding the states, and -- the part that matters -- **the grid the
run actually used**.

Storing the grid rather than the deck path is the whole point.  Rebuilding a
grid from the deck afterwards does not reliably reproduce the one that was
solved on: ``RemoveZeroPoreVolume`` and similar options drop cells, so a deck
re-read can hand back a different active set (Norne: 44,927 cells from the
deck against 44,431 in a run with zero-pore-volume cells removed).  Colouring
one cell ordering with another's values draws a plausible picture of the
wrong cells, and nothing about the result looks wrong.  The saved grid and
the saved states come out of the same run and cannot drift apart.

The per-cell entries are *discovered*, not listed.  PRSTCore keeps
saturations as separate ``sW``/``sG`` vectors where MRST's ad-blackoil keeps
one ``s`` matrix, and a hard-coded key list quietly saves a file with no
saturations in it at all.  Anything shaped like one value per cell is kept,
whatever it is called.

Numpy only -- no VTK, no Qt -- so the solver environment can write these
files even though it cannot open them.
"""

from __future__ import annotations

import numpy as np


__all__ = ["save_states", "load_states", "per_cell_entries", "view_saved"]


#: State entries that are per-cell in shape but are not field data.
_NOT_FIELDS = frozenset({"time", "wellSol", "facility_wells"})


def per_cell_entries(state, ncells, copy=True):
    """The entries of ``state`` that hold one value (or one row) per cell.

    This is what makes the saved file independent of what the state happens
    to be called this month: ``pressure``, ``sW``, ``sG``, ``rs``, ``rv`` and
    anything else the model carries are picked up by shape.

    ``copy`` defaults to true because the usual caller keeps the result until
    the end of a run.  Handing back views would make that history hostage to
    whether the solver happens to allocate fresh arrays each step or update
    them in place -- and if it ever does the latter, every saved step
    silently becomes the last one, which looks like a converged-to-steady
    result rather than a bug.  Pass ``copy=False`` only to inspect a state
    you are not going to keep.
    """
    out = {}
    for key, value in state.items():
        if key in _NOT_FIELDS:
            continue

        if isinstance(value, np.ndarray):
            array = value
        else:
            # A state also carries per-well bookkeeping -- lists of dicts and
            # other ragged things.  numpy raises on those rather than handing
            # back an object array, so conversion failure is simply "not a
            # field", not an error worth stopping a run for.
            try:
                array = np.asarray(value)
            except (ValueError, TypeError):
                continue

        if array.dtype.kind not in "fiu" or array.ndim not in (1, 2):
            continue
        if array.shape[0] != ncells:
            continue
        out[key] = np.array(array, copy=True) if copy else array
    return out


def _phase_labels(model=None, nphase=None):
    """Column names for a saturation *matrix*.

    Only needed for the MRST-style ``s`` matrix; PRSTCore's own ``sW``/``sG``
    vectors carry their phase in the key already.
    """
    if model is not None:
        names = [label for flag, label in
                 (("water", "SW"), ("oil", "SO"), ("gas", "SG"))
                 if getattr(model, flag, False)]
        if names and (nphase is None or len(names) == nphase):
            return names
    return {1: ["S"], 2: ["SW", "SO"], 3: ["SW", "SO", "SG"]}.get(
        nphase, ["S%d" % (i + 1) for i in range(nphase or 0)])


def _model_phases(model):
    if model is None:
        return []
    return [name for name in ("water", "oil", "gas")
            if getattr(model, name, False)]


def save_states(path, states, G, W=None, times=None, dates=None,
                model=None, dtype=np.float32, meta=None):
    """Write states, grid and wells to a single ``.npz``.

    Parameters
    ----------
    path : str
        Destination.  ``.npz`` is appended by numpy if missing.
    states : sequence of dict
        One state per saved step.  Include the initial state as step 0 to
        have the run's starting point in the view.
    G : dict
        The grid the run solved on -- not one rebuilt from the deck.
    W : list[dict], optional
        Wells, for drawing tracks.  Only ``name`` and ``cells`` are kept; the
        rest is solver state that means nothing to a viewer.
    times : sequence of float, optional
        Cumulative time in days at each saved step.
    dates : sequence, optional
        Calendar date per step; stored as ISO strings.
    model : object, optional
        Read for its phase flags, which name the columns of a saturation
        matrix and let :func:`load_states` know whether an oil saturation can
        be derived.
    dtype : numpy dtype
        Storage precision for the field arrays.  Single precision halves the
        file and is well past what a colour map can resolve; pass
        ``np.float64`` to keep the solver's own values.
    meta : dict, optional
        Extra scalars recorded alongside (deck path, options, ...).
    """
    states = list(states)
    if not states:
        raise ValueError("no states to save")

    ncells = int(G["cells"]["num"])
    keys = list(per_cell_entries(states[0], ncells))
    if not keys:
        raise ValueError(
            "no per-cell arrays found in the first state (looked for entries "
            "of length %d); nothing to save" % ncells)

    record = dict(meta or {})
    record["phases"] = _model_phases(model)

    payload = {
        "__grid__": np.array(G, dtype=object),
        "__wells__": np.array(
            [{"name": str(w.get("name", "")),
              "cells": np.asarray(w["cells"], dtype=np.int64).ravel()}
             for w in (W or [])], dtype=object),
        "__meta__": np.array(record, dtype=object),
        "n_steps": np.int64(len(states)),
    }

    for key in keys:
        stack = np.asarray([np.asarray(s[key]) for s in states], dtype=dtype)
        if stack.ndim == 3:
            # An MRST-style saturation matrix: one column per phase.
            for column, label in enumerate(_phase_labels(model, stack.shape[2])):
                payload["field_" + label] = np.ascontiguousarray(stack[:, :, column])
        else:
            payload["field_" + key.upper()] = np.ascontiguousarray(stack)

    if times is not None:
        payload["times_days"] = np.asarray(times, dtype=float)
    if dates is not None:
        payload["dates"] = np.asarray([str(d) for d in dates])

    np.savez_compressed(path, **payload)
    return path


def load_states(path, derive_so=True):
    """Read back what :func:`save_states` wrote.

    Returns a dict with ``G``, ``W``, ``fields`` (``{name: (nsteps, ncells)}``),
    and whichever of ``times_days``, ``dates`` and ``meta`` were saved.

    ``derive_so`` fills in the oil saturation, which PRSTCore never stores:
    it solves for water and gas and leaves oil as the closure
    ``1 - sW - sG``.  Only done when the run's phases say there *is* an oil
    phase, so a water/gas model is not given a fictitious one.
    """
    with np.load(path, allow_pickle=True) as handle:
        out = {
            "G": handle["__grid__"].item(),
            "W": list(handle["__wells__"]) if "__wells__" in handle else [],
            "meta": handle["__meta__"].item() if "__meta__" in handle else {},
            "fields": {name[len("field_"):]: handle[name]
                       for name in handle.files if name.startswith("field_")},
        }
        for optional in ("times_days", "dates"):
            if optional in handle:
                out[optional] = handle[optional]

    fields = out["fields"]
    phases = out["meta"].get("phases") or []
    if derive_so and "SO" not in fields and "SW" in fields and "oil" in phases:
        so = 1.0 - fields["SW"]
        if "SG" in fields:
            so = so - fields["SG"]
        fields["SO"] = so

    return out


def view_saved(path, **kwargs):
    """Open the 3D viewer on a saved run.

    Imported lazily: this needs VTK and Qt, which the environment that
    *writes* these files does not have.
    """
    from .qt_viewer import view_reservoir

    saved = load_states(path)
    return view_reservoir(saved["G"], W=saved["W"], fields=saved["fields"],
                          **kwargs)
