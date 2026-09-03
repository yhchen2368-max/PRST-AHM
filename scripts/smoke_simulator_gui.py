"""Headless smoke test for simulator_gui: load SPE1, run 2 steps, check 2D/3D/log."""
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6 import QtCore, QtWidgets  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

from PRSTCore.visualization.simulator_gui import SimulatorWindow  # noqa: E402

DECK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "examples", "SpE1", "SPE1CASE1.DATA")


def wait_until(cond, timeout_ms=180000):
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
print("window built:", win.windowTitle())

# 1. load deck
win._load_deck(DECK)
ok = wait_until(lambda: win.model is not None)
print("deck loaded:", ok, "|", win.model_label.text().replace("\n", " | ")[:90])

# 2. run 2 steps
win.steps_box.setValue(2)
win._run()
done = wait_until(lambda: win._panel3d._states, timeout_ms=300000)
print("run finished (3D states ready):", done)

# 3. checks
log = win.log_view.toPlainText()
print("log lines:", len(log.splitlines()))
for line in log.splitlines()[:6]:
    print("   |", line)

print("2D wellsols steps:", len(win._panel2d._wellsols),
      "wells:", win._panel2d._all_wells[:5])
print("3D states:", len(win._panel3d._states),
      "fields:", win._panel3d._field_box.count())
print("time slider enabled:", win.time_slider.isEnabled(),
      "max:", win.time_slider.maximum())
print("3D draw ok:", win._panel3d.figure.axes is not None)
print("SMOKE TEST", "PASSED" if (ok and done and win._panel3d._states
                                 and win._panel2d._wellsols) else "FAILED")
