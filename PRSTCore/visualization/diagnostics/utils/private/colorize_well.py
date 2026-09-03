"""MRST private ``colorizeWell.m`` counterpart."""

import numpy as np

from ..helpers import get_field


def colorize_well(type, index, D):
    kind = str(type).lower()
    if kind == "prod":
        n = max(len(np.atleast_1d(get_field(D, "prod", []))), 1)
        return _cmap(index, n, "jet")
    if kind == "inj":
        n = max(len(np.atleast_1d(get_field(D, "inj", []))), 1)
        gray = float(index) / max(n - 1, 1)
        return np.asarray([gray, gray, gray])
    if kind == "global":
        prod = np.atleast_1d(get_field(D, "prod", []))
        inj = np.atleast_1d(get_field(D, "inj", []))
        if index in prod:
            return colorize_well("prod", int(np.flatnonzero(prod == index)[0]), D)
        return colorize_well("inj", int(np.flatnonzero(inj == index)[0]), D)
    return np.asarray([0.0, 0.0, 0.0])


def _cmap(index, n, name):
    try:
        import matplotlib.pyplot as plt

        return np.asarray(plt.get_cmap(name)(float(index) / max(n - 1, 1))[:3])
    except Exception:
        hue = float(index) / max(n, 1)
        return np.asarray([hue, 1.0 - hue, 0.5])


colorizeWell = colorize_well

