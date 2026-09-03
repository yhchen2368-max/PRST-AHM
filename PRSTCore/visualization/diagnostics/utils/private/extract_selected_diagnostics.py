"""MRST private ``extractSelectedDiagnostics.m`` counterpart."""

from ..helpers import get_field


def extract_selected_diagnostics(d, prop, tsel=None, wsel=None):
    data = get_field(d, "Data", get_field(d, "data", d))
    vals = get_field(data, prop, None)
    flag = vals is not None
    lims = None
    return d, vals, lims, flag


extractSelectedDiagnostics = extract_selected_diagnostics

