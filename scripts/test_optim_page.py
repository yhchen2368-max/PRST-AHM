"""Headless test of the Optimization page (PRST conntrans, max_it=1)."""
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6 import QtCore, QtWidgets  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

from PRSTCore.visualization.simulator_gui import SimulatorWindow  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECK_SRC = os.path.join(ROOT, "examples", "SpE1", "SPE1CASE1.DATA")
# Run from a TEMP copy of the deck so the optim output never touches the
# tracked examples/SpE1/SPE1CASE1_optim_prst folder.
import shutil
TMP = os.path.join(os.environ.get("TEMP", os.path.expanduser("~")),
                   "smoke_rsim_optim")
os.makedirs(TMP, exist_ok=True)
DECK = os.path.join(TMP, "SPE1CASE1.DATA")
shutil.copyfile(DECK_SRC, DECK)
OUTDIR = os.path.join(TMP, "SPE1CASE1_optim_prst")


def wait_until(cond, timeout_ms=1800000):
    loop = QtCore.QEventLoop()
    timer = QtCore.QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(timeout_ms)
    while not cond() and timer.isActive():
        app.processEvents()
    timer.stop()
    return cond()


win = SimulatorWindow()
win._load_deck(DECK)
ok = wait_until(lambda: win.model is not None)
print("deck loaded:", ok, flush=True)
assert ok

win.optim_page.max_it.setValue(1)          # fast: 1 optimisation iteration
win.optim_page.oil_price.setValue(60.0)
win.optim_page.water_cost.setValue(5.0)
win.optim_page._run()
done = wait_until(lambda: not win.optim_page._thread.isRunning(),
                  timeout_ms=1800000)
print("optim finished:", done, flush=True)

import json
summary_path = os.path.join(OUTDIR, "summary.json")
if os.path.exists(summary_path):
    with open(summary_path, encoding="utf-8") as fh:
        summary = json.load(fh)
    print("summary:", {k: summary.get(k) for k in
                       ("base_npv", "opt_npv", "improvement_pct",
                        "iterations", "converged", "n_variables",
                        "wall_time_s")}, flush=True)
    print("production.csv exists:",
          os.path.exists(os.path.join(OUTDIR, "production.csv")), flush=True)
    status = summary.get("status", "")
    print("OPT TEST PASSED" if summary.get("base_npv") is not None
          and not status else "OPT TEST FAILED", flush=True)
else:
    print("OPT TEST FAILED: no summary.json", flush=True)
