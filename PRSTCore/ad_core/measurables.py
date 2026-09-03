"""Field and well measurables, in the ECLIPSE mnemonics people read.

A run leaves behind ``well_sols`` (per well, per step) and ``states`` (per
cell, per step).  Neither is what a reservoir engineer looks at first: that
is ``FOPR``, ``FPR``, ``FOIP`` and the rest -- field totals and averages that
have to be aggregated out of the two.  JutulDarcy has this layer
(``plot_reservoir_measurables``, ``plot_summary``); PRSTCore did not, which is
why this module exists.

Two conventions worth stating, because both are easy to get backwards:

**Sign.**  PRSTCore follows MRST: a well rate is positive *into* the
reservoir.  A producer's ``qOs`` is therefore negative.  ECLIPSE's ``FOPR``
is a positive production rate, so production is ``-min(q, 0)`` summed and
injection is ``max(q, 0)`` summed.  Reading a producer's rate straight into
``FOPR`` gives a negative production curve, which is the usual first bug.

**Units.**  PRSTCore computes in SI (m^3/s, Pa); the mnemonics are METRIC
(sm^3/day, sm^3, bar).  The conversion happens here, once, rather than in
every plot.  Pass ``units='si'`` to keep the solver's own numbers.

Numpy only -- no pandas, which the 3.14 solver environment does not have.
"""

from __future__ import annotations

import numpy as np


__all__ = ["well_measurables", "field_measurables", "UNIT_SYSTEMS"]


#: Multipliers taking SI to the named system, per quantity kind.
UNIT_SYSTEMS = {
    "metric": {"rate": 86400.0, "volume": 1.0, "pressure": 1.0e-5},
    "si": {"rate": 1.0, "volume": 1.0, "pressure": 1.0},
}

#: Labels for the unit of each kind, for axis titles.
UNIT_LABELS = {
    "metric": {"rate": "sm3/day", "volume": "sm3", "pressure": "bar"},
    "si": {"rate": "m3/s", "volume": "m3", "pressure": "Pa"},
}

#: Phase letter -> the wellSol key holding its surface rate.
_RATE_KEYS = {"O": "qOs", "W": "qWs", "G": "qGs"}


def _scalar(well, key):
    """One number out of a wellSol entry, which may hold an array."""
    raw = well.get(key)
    if raw is None:
        return 0.0
    array = np.atleast_1d(np.asarray(raw, dtype=float))
    return float(array[0]) if array.size else 0.0


def _factors(units):
    try:
        return UNIT_SYSTEMS[units]
    except KeyError:
        raise ValueError("unknown unit system %r; expected one of %s"
                         % (units, sorted(UNIT_SYSTEMS))) from None


def _time_axis(dt):
    """Cumulative days at the end of each step, and the step lengths."""
    dt = np.asarray(dt, dtype=float).ravel()
    return np.cumsum(dt) / 86400.0, dt


def well_measurables(well_sols, dt, units="metric"):
    """Per-well rates, pressures and cumulatives.

    Parameters
    ----------
    well_sols : sequence
        ``well_sols[step]`` is the list of well dicts for that report step,
        as :func:`~PRSTCore.ad_core.simulators.simulate_schedule_ad` returns.
    dt : sequence of float
        Report step lengths in seconds (``schedule['step']['val']``).
    units : {'metric', 'si'}

    Returns
    -------
    dict
        ``{'time_days', 'names', 'unit_labels', 'wells': {name: {...}}}``
        where each well carries ``WOPR``/``WWPR``/``WGPR`` (production),
        ``WOIR``/``WWIR``/``WGIR`` (injection), ``WBHP``, and the ``*T``
        cumulative of each rate.

    Wells are keyed by name and padded with zeros over steps in which they
    do not appear, because a schedule opens wells as it goes and a plain
    positional index would attribute one well's rates to another the moment
    the well list changes length.
    """
    factor = _factors(units)
    time_days, steps = _time_axis(dt)
    nsteps = len(steps)

    names = []
    for step in well_sols[:nsteps]:
        for well in step or []:
            name = str(well.get("name", ""))
            if name not in names:
                names.append(name)

    keys = (["W%sPR" % p for p in "OWG"] + ["W%sIR" % p for p in "OWG"])
    wells = {name: {key: np.zeros(nsteps) for key in keys} for name in names}
    for name in names:
        wells[name]["WBHP"] = np.zeros(nsteps)

    for index, step in enumerate(well_sols[:nsteps]):
        for well in step or []:
            name = str(well.get("name", ""))
            record = wells[name]
            for phase, key in _RATE_KEYS.items():
                rate = _scalar(well, key)
                # Positive is into the reservoir; split rather than abs() so
                # a well that switches duty is reported on the right curve.
                record["W%sPR" % phase][index] = max(-rate, 0.0) * factor["rate"]
                record["W%sIR" % phase][index] = max(rate, 0.0) * factor["rate"]
            record["WBHP"][index] = _scalar(well, "bhp") * factor["pressure"]

    # Cumulatives integrate the SI rate over the step, so they do not inherit
    # the rate's per-day scaling.
    for record in wells.values():
        for key in keys:
            rate_si = record[key] / factor["rate"]
            record[key[:-1] + "T"] = np.cumsum(rate_si * steps) * factor["volume"]

    return {"time_days": time_days, "names": names, "wells": wells,
            "unit_labels": UNIT_LABELS[units]}


def _saturations(state, nc):
    """``(sW, sO, sG)`` for a state, with oil as the closure.

    PRSTCore solves for water and gas and leaves oil implicit; a state that
    happens to carry ``sO`` is believed over the closure.
    """
    sw = np.asarray(state.get("sW", np.zeros(nc)), dtype=float).ravel()
    sg = np.asarray(state.get("sG", np.zeros(nc)), dtype=float).ravel()
    so = state.get("sO")
    so = (1.0 - sw - sg) if so is None else np.asarray(so, dtype=float).ravel()
    return sw, so, sg


def _in_place(state, pv_ref, model):
    """Surface-condition volumes in place, ``(FOIP, FWIP, FGIP)``.

    ``b`` is the inverse formation volume factor, so reservoir volume times
    ``b`` is surface volume.  Free gas and gas dissolved in the oil are both
    counted into ``FGIP``, which is what the ECLIPSE mnemonic means; without
    the dissolved part a solution-gas-drive case appears to lose gas that has
    only changed phase.

    The PVT is evaluated the way the model evaluates it for its own
    residual -- at the *phase* pressures, with the saturation-state flags
    passed through, and against the pressure-dependent pore volume.  Taking
    the shortcut of one pressure and no flags gets ``bo`` wrong in saturated
    cells and ignores rock compressibility, which shows up as a few percent
    of oil that material balance cannot account for.
    """
    nc = pv_ref.size
    pressure = np.asarray(state["pressure"], dtype=float).ravel()
    sw, so, sg = _saturations(state, nc)

    if model is None or not hasattr(model, "_phase_pvt"):
        return (float(np.sum(pv_ref * so)), float(np.sum(pv_ref * sw)),
                float(np.sum(pv_ref * sg)))

    rs = np.asarray(state.get("rs", np.zeros(nc)), dtype=float).ravel()
    rv = np.asarray(state.get("rv", np.zeros(nc)), dtype=float).ravel()

    pv = pv_ref
    if hasattr(model, "_mrst_pore_volume"):
        pv = np.asarray(model._mrst_pore_volume(pressure), dtype=float).ravel()

    if hasattr(model, "_phase_pressures") and hasattr(
            model, "_phase_pvt_from_phase_pressures"):
        pW, pO, pG = model._phase_pressures(pressure, sw, sg)
        pvt = model._phase_pvt_from_phase_pressures(
            pW, pO, pG, rs_override=rs, rv_override=rv, sG_override=sg,
            oil_saturated_override=(sg > 0.0),
            gas_saturated_override=(so > 0.0))
    else:
        pvt = model._phase_pvt(pressure, rs_override=rs, rv_override=rv)

    bw = np.asarray(pvt["bw"], dtype=float).ravel()
    bo = np.asarray(pvt["bo"], dtype=float).ravel()
    bg = np.asarray(pvt["bg"], dtype=float).ravel()

    oil = pv * so * bo
    water = pv * sw * bw
    gas = pv * sg * bg + pv * so * bo * rs
    if np.any(rv):
        oil = oil + pv * sg * bg * rv

    return float(np.sum(oil)), float(np.sum(water)), float(np.sum(gas))


def field_measurables(well_sols, dt, states=None, G=None, rock=None,
                      model=None, units="metric"):
    """Field totals per report step, under the ECLIPSE mnemonics.

    Parameters
    ----------
    well_sols, dt
        As for :func:`well_measurables`.
    states : sequence of dict, optional
        One state per report step.  Without them the in-place and average
        pressure curves are simply absent rather than approximated.
    G, rock : dict, optional
        Needed for pore volume, and so for ``FPR`` and the ``F*IP`` curves.
    model : optional
        Supplies the PVT.  Without it the in-place figures are *reservoir*
        volumes, not surface volumes, and are labelled as such by being
        omitted -- pass the model to get the mnemonics as ECLIPSE means them.
    units : {'metric', 'si'}

    Returns
    -------
    dict
        ``time_days`` plus whichever of ``FOPR FWPR FGPR FOIR FWIR FGIR``,
        their ``*T`` cumulatives, ``FPR``, ``FOIP FWIP FGIP`` could be
        computed, and ``unit_labels``.
    """
    factor = _factors(units)
    time_days, steps = _time_axis(dt)
    nsteps = len(steps)

    out = {"time_days": time_days, "unit_labels": UNIT_LABELS[units]}

    per_well = well_measurables(well_sols, dt, units=units)["wells"]
    for phase in "OWG":
        for duty in ("PR", "IR"):
            key = "F%s%s" % (phase, duty)
            total = np.zeros(nsteps)
            for record in per_well.values():
                total += record["W%s%s" % (phase, duty)]
            out[key] = total
            out[key[:-1] + "T"] = np.cumsum(total / factor["rate"] * steps) * factor["volume"]

    if states is None or G is None or rock is None:
        return out

    from PRSTCore.ad_core.operators_tpfa import pore_volume

    pv = np.asarray(pore_volume(G, rock), dtype=float).ravel()
    states = list(states)[:nsteps]
    if not states:
        return out

    pressure = np.empty(len(states))
    oil = np.empty(len(states))
    water = np.empty(len(states))
    gas = np.empty(len(states))
    total_pv = float(np.sum(pv)) or 1.0
    for index, state in enumerate(states):
        p = np.asarray(state["pressure"], dtype=float).ravel()
        # Pore-volume weighted, which is what FPR means: a plain mean would
        # let a swarm of tiny cells outvote the bulk of the reservoir.
        pressure[index] = float(np.sum(pv * p)) / total_pv
        oil[index], water[index], gas[index] = _in_place(state, pv, model)

    out["FPR"] = pressure * factor["pressure"]
    if model is not None and hasattr(model, "_phase_pvt"):
        out["FOIP"] = oil * factor["volume"]
        out["FWIP"] = water * factor["volume"]
        out["FGIP"] = gas * factor["volume"]

    return out
