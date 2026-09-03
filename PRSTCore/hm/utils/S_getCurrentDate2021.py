"""Port of MRST ``S_getCurrentDate2021.m`` (mrst-2026a/hm/utils)."""

from datetime import date as _date


def S_getCurrentDate2021():
    """Today's date as ``yyyy-MM-dd``."""
    return _date.today().strftime('%Y-%m-%d')
