"""Instrumented headless GUI repro that prints the FULL traceback of any
GUI-thread exception (the stock diag only prints the type)."""
import os
import sys
import traceback

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6 import QtCore, QtWidgets  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

from PRSTCore.visualization.simulator_gui import SimulatorWindow  # noqa: E402

DECK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "examples", "SpE1", "SPE1CASE1.DATA")


def excepthook(exc_type, exc, tb):
    print("GUI-THREAD EXCEPTION:", exc_type.__name__, exc, flush=True)
    traceback.print_exception(exc_type, exc, tb)
    sys.exit(3)


sys.excepthook = excepthook


def wait_until(cond, timeout_ms=300000):
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
print("window built:", win.windowTitle(), flush=True)

win._load_deck(DECK)
ok = wait_until(lambda: win.model is not None)
print("deck loaded:", ok, flush=True)
win._run()
finished = wait_until(lambda: (win._sim_worker is None)
                      or win._panel3d._states
                      or any("run FAILED" in l
                             for l in win.log_view.toPlainText().splitlines()),
                      timeout_ms=300000)
for _ in range(300):
    app.processEvents()
    if win._panel3d._states:
        break
    import time
    time.sleep(0.1)

log = win.log_view.toPlainText()
print("\n===== FULL LOG =====", flush=True)
for line in log.splitlines():
    print("  |", line, flush=True)
print("=====================", flush=True)
print("\n2D wellsols steps:", len(win._panel2d._wellsols), flush=True)
print("3D states:", len(win._panel3d._states), flush=True)
print("done", flush=True)
