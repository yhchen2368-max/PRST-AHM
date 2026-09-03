"""Per-well rate and pressure panels.

The counterpart of :func:`~PRSTCore.ad_core.plotting.plot_summary` for the
well side: one panel per response, one curve per well, off what
:func:`~PRSTCore.ad_core.measurables.well_measurables` computes.

Distinct from the older :func:`plot_well_sols`, which compares *datasets*
(a PRSTCore run against a reference) one response at a time and works
straight off raw ``wellSol`` dicts in SI.  This one takes the aggregated,
unit-converted structure and shows a whole field at once, which is what
JutulDarcy's ``plot_well_results`` does.
"""

from __future__ import annotations

import numpy as np

from .plot_summary import _unit_kind


__all__ = ["plot_well_responses"]


#: Production rates and the bottom-hole pressure -- the four a producer is
#: normally judged on.  Injection responses are ``WOIR``/``WWIR``/``WGIR``.
DEFAULT_KEYS = ("WOPR", "WWPR", "WGPR", "WBHP")


def plot_well_responses(well_data, keys=DEFAULT_KEYS, wells=None,
                        title=None, figsize=None, max_legend=8):
    """Draw each response in ``keys`` as its own panel.

    Parameters
    ----------
    well_data : dict
        As returned by
        :func:`~PRSTCore.ad_core.measurables.well_measurables`.
    keys : sequence of str
        Responses to draw, one panel each.
    wells : sequence of str, optional
        Which wells to include; every well by default.
    max_legend : int
        Above this many curves the legend is dropped rather than allowed to
        cover the axes -- SPE9 has twenty-six wells, and a legend for all of
        them is larger than the plot.

    A well whose response is zero at every step is skipped: an injector has
    no oil production rate, and drawing that as a flat line at zero adds a
    curve and a legend entry that say nothing.
    """
    import matplotlib.pyplot as plt

    time = np.asarray(well_data["time_days"], dtype=float)
    unit_labels = well_data.get("unit_labels", {})
    records = well_data["wells"]
    names = list(wells) if wells is not None else list(well_data["names"])

    rows = len(keys)
    figsize = figsize or (10.0, 2.5 * rows)
    fig, axes = plt.subplots(rows, 1, figsize=figsize, squeeze=False, sharex=True)

    drawn_names = []
    for row, key in enumerate(keys):
        ax = axes[row][0]
        for name in names:
            record = records.get(name)
            if record is None or key not in record:
                continue
            values = np.asarray(record[key], dtype=float)
            if not np.any(values):
                continue
            ax.plot(time[:len(values)], values, linewidth=1.1, label=name)
            if name not in drawn_names:
                drawn_names.append(name)

        kind = _unit_kind(key)
        unit = unit_labels.get(kind, "") if kind else ""
        ax.set_ylabel("%s\n[%s]" % (key, unit) if unit else key)
        ax.grid(alpha=0.3)

    if drawn_names and len(drawn_names) <= max_legend:
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, fontsize=8, loc="upper center",
                       ncol=min(len(labels), 6))

    axes[-1][0].set_xlabel("time [days]")
    if title:
        fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig
