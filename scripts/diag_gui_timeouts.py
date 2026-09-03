"""Diagnose the 6 GUI-test timeouts: is deck load slow, or does the run hang?

Runs each failing deck in its own subprocess (parallel, one CPU-heavy model
per worker) and prints PHASE markers so we can see where time goes.
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAILING = [
    "examples/HM/QIEDIE.DATA",
    "examples/sleipner/SLEIPNER_ORG.DATA",
    "examples/spe3/SPE3CASE1.DATA",
    "examples/spe10model1/SPE10_MODEL1.DATA",
    "examples/spe10model2/SPE10_MODEL2.DATA",
]

CHILD = r'''
import json, os, sys, time
out_path = sys.argv[1]
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, r"__ROOT__")
from PySide6 import QtCore, QtWidgets
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
from PRSTCore.visualization.simulator_gui import SimulatorWindow

def wait_until(cond, timeout_ms):
    loop = QtCore.QEventLoop(); timer = QtCore.QTimer()
    timer.setSingleShot(True); timer.timeout.connect(loop.quit)
    timer.start(timeout_ms)
    while not cond() and timer.isActive():
        app.processEvents()
    timer.stop(); return cond()

def done(out):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out))

out = {"deck": r"__DECK__", "pass": False}
win = SimulatorWindow()
t0 = time.time(); print("PHASE load-start", flush=True)
win._load_deck(r"__DECK__")
if not wait_until(lambda: win.model is not None, 900000):
    out["detail"] = "load-15min-timeout"; done(out); sys.exit(0)
print("PHASE load-done %.1fs cells=%d" % (time.time()-t0, len(win.state0["pressure"])), flush=True)
nc = len(win.state0["pressure"]); nsteps = len(win.schedule["step"]["val"])
run_steps = 1 if (nc >= 100000 or nsteps >= 200) else 2
win.steps_box.setValue(run_steps)
t1 = time.time(); print("PHASE run-start steps=%d" % run_steps, flush=True)
win._run()
# Wait until either the 3D states are ready (success) or the worker thread
# has finished (success *or* failure).  A failure keeps _states empty, so
# waiting on _states alone would spin for the full timeout.
def worker_done():
    w = getattr(win, "_sim_worker", None)
    return w is None or not w.isRunning()
ok = wait_until(lambda: win._panel3d._states or worker_done(), 900000)
ran = len(win._panel3d._states)
worker_alive = (getattr(win, "_sim_worker", None) is not None
                and win._sim_worker.isRunning())
print("PHASE run-done %.1fs ran=%d worker_alive=%s"
      % (time.time()-t1, ran, worker_alive), flush=True)
out.update({"pass": bool(ok and ran), "cells": nc, "schedule_steps": nsteps,
            "ran": ran,
            "worker_alive": bool(worker_alive),
            "two_d": len(win._panel2d._wellsols),
            "fields": win._panel3d._field_box.count(),
            "log": len(win.log_view.toPlainText().splitlines()),
            "detail": "ok" if (ok and ran) else
            ("worker-still-running" if worker_alive else "failed-fast")})
print(json.dumps(out), flush=True)
done(out)
'''

def run_one(rel):
    import tempfile
    deck = os.path.join(ROOT, rel).replace("\\", "/")
    stem = os.path.basename(os.path.dirname(rel)) + "/" + os.path.splitext(os.path.basename(rel))[0]
    code = (CHILD.replace("__ROOT__", ROOT.replace("\\", "/"))
            .replace("__DECK__", deck))
    fd, path = tempfile.mkstemp(suffix=".py", prefix="gui_diag_", dir=ROOT)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(code)
    fd2, res_path = tempfile.mkstemp(suffix=".json", prefix="gui_res_", dir=ROOT)
    os.close(fd2)
    started = time.time()
    try:
        proc = subprocess.run([sys.executable, path, res_path],
                              timeout=1500, cwd=ROOT)  # streamed output
        info = {}
        if os.path.exists(res_path) and os.path.getsize(res_path):
            with open(res_path, "r", encoding="utf-8") as f:
                info = json.load(f)
        info["returncode"] = proc.returncode
    except subprocess.TimeoutExpired:
        info = {"pass": False, "detail": "25min-timeout"}
    finally:
        os.remove(path)
        if os.path.exists(res_path):
            os.remove(res_path)
    wall = time.time() - started
    status = "PASS" if info.get("pass") else "FAIL"
    print("\n=== %-16s %s (%.0fs, rc=%s) ===" % (stem[:16], status, wall,
                                                 info.get("returncode")))
    print("   detail:", info.get("detail", ""), "| cells", info.get("cells"),
          "| ran", info.get("ran"), "| 2D", info.get("two_d"),
          "| 3D", info.get("fields"), "| log", info.get("log"))
    return stem, status, info

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(run_one, FAILING))
