"""Batch-test the simulator GUI pipeline against every example model.

Each model runs in its own subprocess (headless Qt) with a timeout, so one
hung deck cannot stall the rest.  A model is PASS if the deck loads, the
report step(s) run (converged or reported), the 2D well-curve panel receives
data, the 3D panel builds per-cell states and the iteration log has content.

Usage::

    python scripts/test_gui_examples.py                # all models
    python scripts/test_gui_examples.py --deck <path>  # single model (child)
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: One representative deck per example folder.
DECK_LIST = [
    "examples/EGG/Egg_Model_ECL.DATA",
    "examples/HM/QIEDIE.DATA",
    "examples/sleipner/SLEIPNER_ORG.DATA",
    "examples/SpE1/SPE1CASE1.DATA",
    "examples/SpE1/SPE1CASE2_2P.DATA",
    "examples/SPE9/SPE9.DATA",
    "examples/spe3/SPE3CASE1.DATA",
    "examples/spe5/SPE5CASE1.DATA",
    "examples/T142/T142_E100.DATA",
    "examples/Norne/Norne_simplified/NORNE_ATW2013.DATA",
    "examples/spe10model1/SPE10_MODEL1.DATA",
    "examples/spe10model2/SPE10_MODEL2.DATA",
]

#: Per-model wall-clock budget (seconds); big models get one step only.
#: Includes deck load + one report step + run pipeline.
TIMEOUTS = {
    "T142": 300, "NORNE": 240, "SLEIPNER": 480, "QIEDIE": 480,
    "SPE10_MODEL2": 360, "SPE10_MODEL1": 120, "SPE9": 60,
}
DEFAULT_TIMEOUT = 120


def child_main(deck):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    sys.path.insert(0, ROOT)

    from PySide6 import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    from PRSTCore.visualization.simulator_gui import SimulatorWindow

    def wait_until(cond, timeout_ms):
        loop = QtCore.QEventLoop()
        timer = QtCore.QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        timer.start(timeout_ms)
        while not cond() and timer.isActive():
            app.processEvents()
        timer.stop()
        return cond()

    result = {"deck": deck, "pass": False, "detail": ""}
    win = SimulatorWindow()
    t0 = time.time()
    win._load_deck(deck)
    # Completion is signalled by the log lines the load worker's signals
    # append (main-thread, race-free).  isRunning() is False both before a
    # thread starts and after it finishes, so it cannot tell "done" from
    # "not started yet".
    def load_complete():
        log = win.log_view.toPlainText()
        return ("deck loaded:" in log) or ("deck load FAILED" in log)
    if not wait_until(load_complete, 300000):
        result["detail"] = "deck load worker still running"
        print(json.dumps(result))
        return
    if win.model is None:
        result["detail"] = "deck load FAILED (see log)"
        print(json.dumps(result))
        return
    nc = len(win.state0["pressure"])
    nsteps = len(win.schedule["step"]["val"])
    result["cells"] = nc
    result["wells"] = len(win._union_wells())
    result["schedule_steps"] = nsteps
    result["load_s"] = round(time.time() - t0, 1)

    # big models: one report step; small ones: two.
    run_steps = 1 if (nc >= 100000 or nsteps >= 200) else min(2, nsteps)
    win.steps_box.setValue(run_steps)
    win._run()
    # Run completion is also log-driven ("=== run finished" on success,
    # "run FAILED" on failure), avoiding the isRunning() race above.
    def run_complete():
        log = win.log_view.toPlainText()
        return ("=== run finished" in log) or ("run FAILED" in log)
    ok = wait_until(run_complete, 600000)
    result["ran_steps"] = len(win._panel3d._states)
    result["log_lines"] = len(win.log_view.toPlainText().splitlines())
    result["2d_steps"] = len(win._panel2d._wellsols)
    result["3d_fields"] = win._panel3d._field_box.count()
    result["time_slider"] = win.time_slider.isEnabled()
    ok = ok and bool(win._panel3d._states) and bool(win._panel2d._wellsols) \
        and result["log_lines"] >= 3
    result["pass"] = bool(ok)
    if ok:
        result["detail"] = ("ran %d/%d steps, 2D %d, 3D %d fields"
                            % (result["ran_steps"], run_steps,
                               result["2d_steps"], result["3d_fields"]))
    else:
        log = win.log_view.toPlainText()
        if "run FAILED" in log:
            result["detail"] = "run failed fast (check log)"
        elif ok is False:
            result["detail"] = "run did not finish"
    print(json.dumps(result))


def parent_main():
    print("%-14s %8s %8s %6s %6s %8s %7s  %s"
          % ("model", "load_s", "cells", "wells", "steps", "ran", "2d/3d",
             "result"))
    results = []
    for rel in DECK_LIST:
        deck = os.path.join(ROOT, rel)
        stem = os.path.basename(os.path.dirname(rel)) + "/" + \
               os.path.splitext(os.path.basename(rel))[0]
        key = os.path.basename(rel)
        timeout = next((v for k, v in TIMEOUTS.items() if k in key),
                       DEFAULT_TIMEOUT)
        started = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--deck", deck],
                capture_output=True, text=True, timeout=timeout,
                cwd=ROOT)
            line = (proc.stdout.strip().splitlines() or [""])[-1]
            info = json.loads(line) if line.startswith("{") else {}
        except subprocess.TimeoutExpired:
            info = {"pass": False, "detail": "TIMEOUT"}
        except Exception as exc:  # pragma: no cover
            info = {"pass": False, "detail": type(exc).__name__}
        wall = time.time() - started
        status = "PASS" if info.get("pass") else "FAIL"
        print("%-14s %8s %8s %6s %6s %8s %7s  %s (%.0fs)"
              % (stem[:14], info.get("load_s", "-"), info.get("cells", "-"),
                 info.get("wells", "-"), info.get("schedule_steps", "-"),
                 "%s/%s" % (info.get("ran_steps", 0),
                            info.get("ran_steps", 0)),
                 "%s/%s" % (info.get("2d_steps", 0), info.get("3d_fields", 0)),
                 info.get("detail", status), wall))
        results.append((stem, status, info.get("detail", "")))
    npass = sum(1 for _, s, _ in results if s == "PASS")
    print("\n%d/%d models passed" % (npass, len(results)))
    for stem, status, detail in results:
        if status != "PASS":
            print("  FAIL %-16s %s" % (stem, detail))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default=None)
    args = ap.parse_args()
    if args.deck:
        child_main(os.path.abspath(args.deck))
    else:
        parent_main()
