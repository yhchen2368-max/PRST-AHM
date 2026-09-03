"""Grouped summary plots, in the style of JutulDarcy's ``plot_summary``.

Takes what :mod:`PRSTCore.ad_core.measurables` computes and lays it out the
way a summary viewer does: one panel per group, curves that share a unit
drawn together.  A group is written as a comma-separated string of
mnemonics, so ``"FOPR,FWPR,FGPR"`` is one panel with three curves and
``"FPR"`` is a panel on its own.

Deliberately thin -- matplotlib, no interactivity.  The computation lives in
``measurables`` precisely so a different front end can be put in front of it
without touching how the numbers are made.
"""

from __future__ import annotations

import numpy as np


__all__ = ["plot_summary", "plot_measurables"]


#: Which unit kind a mnemonic carries, by its last letter(s).  Used only for
#: axis labels, so an unrecognised name simply goes unlabelled.
def _unit_kind(name):
    # Pressure first: ``FPR`` also ends in ``PR``, so a rate test placed
    # ahead of this one claims it and the average-pressure panel comes out
    # labelled sm3/day.
    if name == "FPR" or name.endswith("BHP"):
        return "pressure"
    if name.endswith("PR") or name.endswith("IR"):
        return "rate"
    if name.endswith("PT") or name.endswith("IT"):
        return "volume"
    if name.endswith("IP"):
        return "volume"
    return None


def plot_summary(measurables, plots=None, cols=2, figsize=None, title=None):
    """Draw grouped curves from a measurables dict.

    Parameters
    ----------
    measurables : dict
        As returned by
        :func:`~PRSTCore.ad_core.measurables.field_measurables`.
    plots : sequence of str, optional
        One entry per panel; each is a comma-separated list of mnemonics.
        Defaults to the rates, the cumulatives, the average pressure and the
        oil in place -- the four a summary is usually opened for.
    cols : int
        Panels per row.
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure

    Mnemonics that are not present are skipped rather than raised on, so a
    default group still draws when a run has no gas phase.
    """
    import matplotlib.pyplot as plt

    if plots is None:
        plots = ["FOPR,FWPR,FGPR", "FOPT,FWPT,FGPT", "FPR", "FOIP"]

    time = np.asarray(measurables["time_days"], dtype=float)
    labels = measurables.get("unit_labels", {})

    groups = []
    for group in plots:
        present = [name.strip() for name in group.split(",")
                   if name.strip() in measurables]
        if present:
            groups.append(present)
    if not groups:
        raise ValueError("none of the requested mnemonics are present; "
                         "available: %s" % sorted(_mnemonics(measurables)))

    cols = max(1, min(int(cols), len(groups)))
    rows = int(np.ceil(len(groups) / cols))
    if figsize is None:
        figsize = (6.0 * cols, 3.2 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    for index, names in enumerate(groups):
        ax = axes[index // cols][index % cols]
        peaks = []
        for name in names:
            values = np.asarray(measurables[name], dtype=float)
            ax.plot(time, values, label=name)
            peaks.append(float(np.nanmax(np.abs(values))) if values.size else 0.0)
        ax.set_xlabel("time (days)")
        kind = _unit_kind(names[0])
        if kind and kind in labels:
            ax.set_ylabel(labels[kind])
        ax.grid(True, alpha=0.3)

        # Curves grouped by unit can still differ by orders of magnitude --
        # SPE9's gas rate is ~250x its oil rate, and on a linear axis the oil
        # and water curves lie flat on zero and cannot be read at all.  A log
        # axis is the only way the panel shows all three; it is used only
        # when the spread makes that necessary, and only when every curve is
        # positive, since a log axis silently drops zeros and negatives.
        if len(peaks) > 1 and min(peaks) > 0 and max(peaks) / min(peaks) > 50:
            ax.set_yscale("log")
        if len(names) > 1:
            ax.legend(fontsize=9)
        else:
            ax.set_title(names[0], fontsize=10)

    for index in range(len(groups), rows * cols):
        axes[index // cols][index % cols].axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def _mnemonics(measurables):
    return [k for k, v in measurables.items()
            if isinstance(v, np.ndarray) and k != "time_days"]


def plot_measurables(measurables, left="FGPR", right="FPR", figsize=(8, 4.5),
                     title=None):
    """Two mnemonics on a shared time axis with independent y scales.

    The equivalent of JutulDarcy's
    ``plot_reservoir_measurables(..., left = :fgpr, right = :pres)``: the
    point is to read one curve against another whose magnitude is nothing
    like it, so the two get their own axes rather than being squeezed onto
    one.
    """
    import matplotlib.pyplot as plt

    for name in (left, right):
        if name not in measurables:
            raise KeyError("%r is not in the measurables; available: %s"
                           % (name, sorted(_mnemonics(measurables))))

    time = np.asarray(measurables["time_days"], dtype=float)
    labels = measurables.get("unit_labels", {})

    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = ax_left.twinx()

    line_l, = ax_left.plot(time, measurables[left], color="tab:blue", label=left)
    line_r, = ax_right.plot(time, measurables[right], color="tab:red", label=right)

    ax_left.set_xlabel("time (days)")
    for ax, name, line in ((ax_left, left, line_l), (ax_right, right, line_r)):
        kind = _unit_kind(name)
        unit = labels.get(kind, "") if kind else ""
        ax.set_ylabel("%s  [%s]" % (name, unit) if unit else name,
                      color=line.get_color())
        ax.tick_params(axis="y", labelcolor=line.get_color())

    ax_left.grid(True, alpha=0.3)
    if title:
        ax_left.set_title(title)
    fig.tight_layout()
    return fig
