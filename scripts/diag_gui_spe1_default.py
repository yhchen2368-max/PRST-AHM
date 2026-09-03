"""Headless repro of the user's exact GUI scenario: load SPE1, keep the default
steps_box (100), run AMGCL CPR, and report exactly how many steps complete and
any exception in the worker or the GUI thread."""
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

gui_errors = []


def excepthook(exc_type, exc, tb):
    gui_errors.append("".join(traceback.format_exception(exc_type, exc, tb)))
    print("GUI-THREAD EXCEPTION:", exc_type.__name__, exc, flush=True)


sys.excepthook = excepthook


def wait_until(cond, timeout_ms=600000):
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
print("steps_box value:", win.steps_box.value(),
      "max_steps param would be:", win.steps_box.value() if
      win.steps_box.value() < len(win.schedule["step"]["val"]) else None,
      flush=True)
print("solver method:", win.solver_box.currentText(),
      "amgcl_strategy:", win.amgcl_strategy_box.currentText(),
      "amgcl_decoupling:", win.amgcl_decoupling_box.currentText(), flush=True)

win._run()
finished = wait_until(lambda: (win._sim_worker is None)
                      or win._panel3d._states
                      or any("run FAILED" in l for l in win.log_view.toPlainText().splitlines()),
                      timeout_ms=600000)

# wait a little more for the run_finished to land
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
print("run FAILED lines:", [l for l in log.splitlines() if "FAILED" in l], flush=True)
print("GUI-THREAD ERRORS:", gui_errors if gui_errors else "none", flush=True)
