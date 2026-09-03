"""Test _props_tables PVT/relperm parsing on every .DATA under examples/SpE1.

For each deck: parse the PROPS tables, then verify every multi-column table
actually plots (>= 1 line) through the PVT page's rendering path.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6 import QtWidgets  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

from PRSTCore.visualization.simulator_gui import (_props_tables,  # noqa: E402
                                                  _WellCurvesPanel,
                                                  _SlicePanel, PlotsPage)

SPE1 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "examples", "SpE1")
DECK_FILES = sorted(
    glob.glob(os.path.join(SPE1, "*.DATA"))
    + [os.path.join(SPE1, "SPE1.GRDECL")])

all_ok = True
for deck in DECK_FILES:
    name = os.path.basename(deck)
    try:
        tables = _props_tables(deck)
    except Exception as exc:  # noqa: BLE001
        print("%-34s ERROR parsing: %s: %s" % (name, type(exc).__name__, exc))
        all_ok = False
        continue
    if not tables:
        print("%-34s no PROPS tables found" % name)
        continue
    # render each table through the real PVT-page path (set_pvt_tables
    # populates both the table and the x-axis combos)
    pp = PlotsPage(_WellCurvesPanel(), _SlicePanel())
    pp.set_pvt_tables(deck)
    results = []
    for tname in sorted(pp._pvt_tables):
        arr = pp._pvt_tables[tname]
        if arr is None or arr.ndim != 2:
            results.append("%s:(bad)" % tname)
            all_ok = False
            continue
        shape = "x".join(str(int(v)) for v in arr.shape)
        if arr.shape[1] < 2 or arr.shape[0] < 1:
            results.append("%s[%s]:(too-small)" % (tname, shape))
            continue
        # plot it
        pp.pvt_table.setCurrentText(tname)
        pp._pvt_sync_x()
        pp._pvt_redraw()
        lines = len(pp.pvt_figure.axes[0].lines) if pp.pvt_figure.axes else 0
        ok = lines >= 1
        all_ok = all_ok and ok
        results.append("%s[%s]:(%d lines%s)" %
                       (tname, shape, lines, "" if ok else " BAD"))
    print("%-34s %s" % (name, "  ".join(results)))

print("\nALL PVT TESTS", "PASSED" if all_ok else "FAILED")
