"""ResSimWorkbench-style reservoir simulation GUI built on PRSTCore only.

1:1 port of the ResSimWorkbench main-window interface (itself a port of the
OPM flow-gui), but with **no GeoView / GeoCode dependency**: every page is
driven by PRSTCore's own modules (``init_eclipse_problem_ad``,
``simulate_schedule``, ``grid_plots.plot_slice``, ``well_curves``,
``optimization``) and the on-disk results are the unified **HDF5** format of
:mod:`PRSTCore.visualization.h5_results` (``states.h5`` / ``wells.h5`` /
``cell_indices.h5`` / ``manifest.json``) -- the same schema the
``<deck>_run_prst`` folders under ``examples/SpE1`` use, so the GUI can load
those computed results directly.

Interface (mirrors ``workbench/ui/main_window.py``):

* menu bar: Project (New / Open... / Save / Save As...), View (switches the
  seven tabs), Help (About);
* a bottom Log dock streaming every job's iteration trace;
* central tabs, in order: Run, Deck Editor, Well Hierarchy, 2D Slice, Plots,
  Optimization, Compare.

Run flow: load a deck (or add decks to the job queue), set the run options,
then Run selected / Run queue.  Each finished job writes its HDF5 results
next to the deck (``<deck>_run_prst``) and the 2D slice / Plots / Compare
views refresh from them automatically.  The 2D slice tab also carries an
"Open 3D..." button that pops the interactive VTK reservoir window
(PRSTCore's own ``scene3d``/``qt_viewer``).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from datetime import date, datetime, timedelta

import numpy as np

# The workspace root holds the ``PRSTCore`` package.  Running this file
# directly by path puts only ``PRSTCore/visualization`` on ``sys.path``, so
# ``import PRSTCore`` below would fail; insert the repo root explicitly.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---- Qt (PySide6) ---------------------------------------------------------
from PySide6 import QtCore, QtWidgets  # noqa: E402

_Signal = QtCore.Signal
from PySide6.QtCore import QEvent, QPoint, Qt  # noqa: E402
from PySide6.QtGui import (  # noqa: E402
    QAction, QColor, QFont, QIcon, QKeySequence, QPainter, QShortcut,
    QSyntaxHighlighter, QTextCharFormat, QTextCursor, QTextDocument)
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDockWidget,
    QDoubleSpinBox, QFileDialog, QFrame, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox,
    QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

# ---- matplotlib Qt canvas -------------------------------------------------
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as _Canvas  # noqa: E402
from matplotlib.figure import Figure as _Figure  # noqa: E402

import PRSTCore  # noqa: F401,E402  -- conda Library/bin on PATH before MKL

from PRSTCore.visualization import h5_results  # noqa: E402
from PRSTCore.visualization.agent_dock import AgentDock  # noqa: E402

__all__ = ["SimulatorWindow", "run_simulator"]

DEFAULT_START = date(1999, 9, 1)          # MRST SPE1 run-start convention


def resolve_outdir(deck: str, mode: int, custom: str,
                   engine: str = "prst") -> str:
    """Where a job's HDF5 results go: next to the deck or a custom dir."""
    if mode == 1 and custom.strip():
        return os.path.abspath(custom.strip())
    base = os.path.splitext(os.path.basename(deck))[0]
    suffix = "_run_jutul" if engine == "jutul" else "_run_prst"
    return os.path.join(os.path.dirname(os.path.abspath(deck)),
                        "%s%s" % (base, suffix))


# ===========================================================================
# job model
# ===========================================================================
QUEUED, RUNNING, DONE, FAILED, STOPPED = \
    "Queued", "Running", "Done", "Failed", "Stopped"

_STATUS_COLOR = {
    QUEUED: "black", RUNNING: "#B8860B", DONE: "#1e7d32",
    FAILED: "#c62828", STOPPED: "#666666",
}


class _Job:
    def __init__(self, deck: str, engine: str = "prst"):
        self.deck = os.path.normpath(deck)
        self.name = os.path.basename(self.deck)
        self.engine = engine
        self.state = QUEUED
        self.outdir = ""
        self.progress = 0.0
        self.report_step = 0
        self.report_total = 0
        self.elapsed_ms = 0
        self.error = ""
        self.started_at = 0.0
        self.worker = None          # the live _RunWorker
        self.handle = None          # pre-loaded (model, state0, schedule, solver)
        self.result = None


def _fmt_duration(ms: int) -> str:
    ms = max(0, int(ms))
    s = ms // 1000
    return "%d:%02d:%02d" % (s // 3600, (s // 60) % 60, s % 60)


# ===========================================================================
# workers
# ===========================================================================
class _DeckLoadWorker(QtCore.QThread):
    """Parse a DATA deck in the background (``init_eclipse_problem_ad``)."""

    loaded = _Signal(object, object, object, object, float)
    failed = _Signal(str)

    def __init__(self, deck_path, minporo=None, parent=None):
        super().__init__(parent)
        self.deck_path = deck_path
        self.minporo = minporo

    def run(self):
        from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import (
            init_eclipse_problem_ad)

        t0 = time.perf_counter()
        try:
            kwargs = dict(RemoveZeroPoreVolume=True)
            if self.minporo:
                kwargs['minporo'] = float(self.minporo)
            state0, model, schedule, solver = init_eclipse_problem_ad(
                self.deck_path, **kwargs)
        except Exception as exc:  # pragma: no cover - error path
            self.failed.emit("%s: %s" % (type(exc).__name__, exc))
            return
        self.loaded.emit(model, state0, schedule, solver,
                         time.perf_counter() - t0)


def _apply_params_to_solver(solver, p: dict):
    """Apply a run-parameter dict to the solver (AMGCL CPR / PETSc CPR)."""
    if p.get("max_iterations"):
        solver.maxIterations = int(p["max_iterations"])
    solver.useLinesearch = bool(p.get("use_linesearch", True))
    solver.enforceResidualDecrease = bool(
        p.get("enforce_residual_decrease", True))
    if p.get("acceptance_factor") is not None:
        solver.acceptanceFactor = float(p["acceptance_factor"])

    method = str(p.get("method", "AMGCL CPR"))
    if method == "AMGCL CPR":
        from PRSTCore.ad_core.solvers import AMGCL_CPRSolverBlockAD
        tolerance = p.get("tolerance")
        solver.LinearSolver = AMGCL_CPRSolverBlockAD(
            tolerance=float(tolerance) if tolerance is not None else 1e-4,
            maxIterations=50,
            strategy=str(p.get("amgcl_strategy", "mrst")),
            decoupling=str(p.get("amgcl_decoupling", "trueIMPES")),
        )
        return

    linear = getattr(solver, "LinearSolver", None)
    if p.get("tolerance") is not None and linear is not None:
        linear.tolerance = float(p["tolerance"])
    if linear is not None:
        if p.get("strategy"):
            linear.strategy = str(p["strategy"])
            linear._ksp = None
        if p.get("pressure_precond"):
            linear.pressure_precond = str(p["pressure_precond"])
            linear._ksp = None
        if p.get("second_stage"):
            linear.second_stage = str(p["second_stage"])
            linear._ksp = None


def _grid_from_model(model, nstate):
    """(G, cartdims, index_map) out of a PRST model."""
    G = getattr(model, "G", None)
    if G is None:
        return None, [int(nstate)], np.arange(int(nstate), dtype=np.int64)
    if isinstance(G, dict):
        cartdims = [int(v) for v in G["cartDims"]]
        index_map = G.get("cells", {}).get("indexMap")
    else:
        cartdims = [int(v) for v in G.cartDims]
        index_map = getattr(getattr(G, "cells", None), "indexMap", None)
    if index_map is None:
        index_map = np.arange(int(nstate), dtype=np.int64)
    else:
        index_map = np.asarray(index_map, dtype=np.int64).ravel()
    return G, cartdims, index_map


class _RunWorker(QtCore.QThread):
    """Load (optionally), run ``simulate_schedule``, write HDF5 results."""

    log_line = _Signal(str)
    progress = _Signal(int, int)     # report_step, report_total
    step_done = _Signal(object)      # info dict (live well curves)
    finished_ok = _Signal(object)    # result dict
    failed = _Signal(str)

    def __init__(self, deck, params, outdir, handle=None, start=DEFAULT_START,
                 parent=None):
        super().__init__(parent)
        self.deck = deck
        self.params = dict(params)
        self.outdir = outdir
        self.handle = handle          # (model, state0, schedule, solver) or None
        self._start = start
        self._stop = False

    def request_stop(self):
        self._stop = True

    def run(self):
        t0 = time.perf_counter()
        from PRSTCore.ad_core.solvers.simulate_schedule import simulate_schedule
        from PRSTCore.visualization.results_io import per_cell_entries

        if self.handle is not None:
            model, state0, schedule, solver = self.handle
        else:
            self.log_line.emit("loading deck %s ..." % self.deck)
            from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import (
                init_eclipse_problem_ad)
            try:
                state0, model, schedule, solver = init_eclipse_problem_ad(
                    self.deck, RemoveZeroPoreVolume=True)
            except Exception as exc:
                self.failed.emit("deck load failed: %s: %s"
                                 % (type(exc).__name__, exc))
                return
            self.log_line.emit("loaded: %d cells" % len(state0["pressure"]))

        _apply_params_to_solver(solver, self.params)
        nsteps_total = len(schedule["step"]["val"])

        def on_solve_start(index, meta):
            when = meta["date"]
            when_s = when.strftime("%d-%b-%Y") if when is not None else "?"
            self.log_line.emit(
                "REPORT STEP %d  TIME=%.1f days (%s)  DT=%.1f d"
                % (index + 1, meta["time_days"], when_s,
                   meta["dt"] / 86400.0))
            self.progress.emit(index + 1, nsteps_total)

        def on_step(index, info):
            status = "converged" if info["converged"] else "FAILED"
            self.log_line.emit("   %s in %d iterations, wall=%.2f s"
                               % (status, info["iterations"], info["wall"]))
            self.step_done.emit(info)

        try:
            result = simulate_schedule(
                model, state0, schedule, solver,
                max_steps=self.params.get("max_steps") or None,
                start=self._start,
                on_solve_start=on_solve_start,
                on_step=on_step,
                should_stop=lambda: self._stop)
        except Exception as exc:
            self.failed.emit("%s: %s" % (type(exc).__name__, exc))
            return
        if self._stop:
            self.log_line.emit("--- run stopped by user ---")
            return

        nc = int(len(state0["pressure"]))
        G, cartdims, index_map = _grid_from_model(model, nc)
        states, times = [per_cell_entries(state0, nc)], [0.0]
        for info in result["steps"]:
            states.append(per_cell_entries(info["state"], nc))
            times.append(float(info["time_days"]))
        wells = _union_wells(schedule)

        # ---- write unified HDF5 results ---------------------------------
        try:
            pressure, sats, iso_dates = h5_results.build_state_arrays(
                model, state0, result["steps"], self._start)
            well_names = sorted({w.get("name") for w in wells})
            well_mats = h5_results.build_well_matrices(well_names,
                                                       result["steps"])
            config = {
                "max_steps": self.params.get("max_steps"),
                "method": self.params.get("method", "AMGCL CPR"),
                "tolerance": self.params.get("tolerance"),
                "use_linesearch": bool(
                    self.params.get("use_linesearch", True)),
                "enforce_residual_decrease": bool(
                    self.params.get("enforce_residual_decrease", True)),
                "acceptance_factor": float(
                    self.params.get("acceptance_factor", 2.0)),
                "amgcl_strategy": self.params.get("amgcl_strategy", "mrst"),
                "amgcl_decoupling": self.params.get("amgcl_decoupling",
                                                    "trueIMPES"),
                "strategy": self.params.get("strategy", "cpr"),
                "pressure_precond": self.params.get("pressure_precond",
                                                    "hypre"),
                "second_stage": self.params.get("second_stage", "ilu"),
                "start": self._start.isoformat(),
            }
            h5_results.write(
                self.outdir, simulator="prst", case=self.deck,
                grid_dims=cartdims, n_active=nc, active_to_natural=index_map,
                pressure=pressure, saturations=sats, dates_iso8601=iso_dates,
                wells=well_mats,
                extra={"method": config["method"], "config": config,
                       "n_steps": len(result["steps"])})
            self.log_line.emit("wrote HDF5 results to %s" % self.outdir)
        except Exception as exc:  # pragma: no cover - report, keep going
            self.log_line.emit("   !! HDF5 write failed: %s: %s"
                               % (type(exc).__name__, exc))

        payload = {
            "deck": self.deck, "result_dir": self.outdir, "G": G,
            "wells": wells, "states": states, "times": times,
            "steps": result["steps"],
            "nsteps": len(result["steps"]), "wall": time.perf_counter() - t0,
            "n_active": nc, "cartdims": cartdims, "index_map": index_map,
            "phases": {"oil": bool(model.oil), "water": bool(model.water),
                       "gas": bool(model.gas)},
        }
        self.log_line.emit("=== run finished: %d steps in %.1f s ==="
                           % (payload["nsteps"], payload["wall"]))
        self.finished_ok.emit(payload)


def _union_wells(schedule):
    """The well set across all controls (wells open over the schedule)."""
    seen = {}
    for control in schedule.get("control", []):
        for w in control.get("W", []):
            name = w.get("name")
            if name is not None and name not in seen:
                seen[name] = w
    return list(seen.values())


class _ValidateWorker(QtCore.QThread):
    done = _Signal(str, bool, str)

    def __init__(self, deck, parent=None):
        super().__init__(parent)
        self.deck = deck

    def run(self):
        from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import (
            init_eclipse_problem_ad)
        try:
            init_eclipse_problem_ad(self.deck, RemoveZeroPoreVolume=True)
            self.done.emit(self.deck, True, "")
        except Exception as exc:
            self.done.emit(self.deck, False,
                           "%s: %s" % (type(exc).__name__, exc))


class _JutulWorker(QtCore.QThread):
    """Run a deck through the JutulDarcy Julia driver (PRSTCore.jutul)."""

    log_line = _Signal(str)
    finished_ok = _Signal(object)    # result dict (result_dir + simulator)
    failed = _Signal(str)

    def __init__(self, deck, outdir, timeout_s=None, parent=None):
        super().__init__(parent)
        self.deck = deck
        self.outdir = outdir
        self.timeout_s = timeout_s

    def run(self):
        from PRSTCore.jutul.driver import JULIA_HINT, run_simulate
        self.log_line.emit("running JutulDarcy on %s ..." % self.deck)
        try:
            out = run_simulate(self.deck, result_dir=self.outdir,
                               timeout_s=self.timeout_s,
                               on_line=self.log_line.emit)
        except Exception as exc:  # noqa: BLE001 - report, don't die
            hint = ""
            text = str(exc).lower()
            if isinstance(exc, FileNotFoundError) or "not found" in text \
                    or "cannot find" in text or "julia" in text:
                hint = "\n  (%s)" % JULIA_HINT
            self.failed.emit("JutulDarcy run failed: %s: %s%s"
                             % (type(exc).__name__, exc, hint))
            return
        self.log_line.emit("wrote JutulDarcy HDF5 results to %s" % out)
        payload = {"deck": self.deck, "result_dir": str(out),
                   "simulator": "jutul", "nsteps": 0}
        try:
            jr = h5_results.load(self.outdir)
            payload["nsteps"] = jr.n_steps
        except Exception:  # noqa: BLE001 - metadata only
            pass
        self.finished_ok.emit(payload)


# ===========================================================================
# job queue (PRST-only, in-process threads)
# ===========================================================================
class _JobQueue(QtCore.QObject):
    queue_changed = _Signal()
    job_changed = _Signal(int)
    run_started = _Signal(int)
    run_finished = _Signal()
    log_line = _Signal(str)
    step_done = _Signal(object)        # info dict (live well curves)
    result_ready = _Signal(object)     # result dict

    def __init__(self, parent=None):
        super().__init__(parent)
        self.jobs: list[_Job] = []
        self.current = -1
        self._abort = False
        self._validators = []
        self.params = {}               # run-option dict (set by the window)
        self.outdir_mode = 0
        self.outdir_custom = ""

    def job(self, idx):
        if 0 <= idx < len(self.jobs):
            return self.jobs[idx]
        return None

    def add_deck(self, deck, engine="prst", handle=None):
        for j in self.jobs:
            if j.deck == os.path.normpath(deck):
                if handle is not None:
                    j.handle = handle
                self.queue_changed.emit()
                return j
        job = _Job(deck, engine)
        job.handle = handle
        self.jobs.append(job)
        self.queue_changed.emit()
        return job

    def remove_jobs(self, rows):
        for row in sorted(set(rows), reverse=True):
            if 0 <= row < len(self.jobs) and self.jobs[row].state != RUNNING:
                del self.jobs[row]
        self.current = min(self.current, len(self.jobs) - 1)
        self.queue_changed.emit()

    def clear(self):
        self.jobs = [j for j in self.jobs if j.state == RUNNING]
        self.current = -1
        self.queue_changed.emit()

    def run_selected(self, rows):
        rows = [r for r in sorted(set(rows))
                if 0 <= r < len(self.jobs) and self.jobs[r].state == QUEUED]
        if not rows or self._busy():
            return
        self.queue_changed.emit()
        self._start_next()

    def run_queue(self):
        if self._busy():
            return
        for j in self.jobs:
            if j.state in (QUEUED, FAILED, STOPPED, DONE):
                j.state = QUEUED
        self.queue_changed.emit()
        self._start_next()

    def _busy(self):
        return any(j.state == RUNNING for j in self.jobs)

    def stop(self):
        for j in self.jobs:
            if j.state == RUNNING and j.worker is not None:
                j.worker.request_stop()
            elif j.state == QUEUED:
                j.state = STOPPED
        self.queue_changed.emit()

    def skip(self):
        job = self.job(self.current)
        if job is not None and job.state == RUNNING and job.worker is not None:
            job.worker.request_stop()
            job.state = STOPPED
            self.job_changed.emit(self.current)
        self._start_next()

    def validate(self, rows):
        for r in rows:
            job = self.job(r)
            if job is None:
                continue
            thread = _ValidateWorker(job.deck)
            self._validators.append(thread)
            thread.done.connect(
                lambda path, ok, err, t=thread: self._on_validated(
                    path, ok, err, lambda t=t: self._validators.remove(t)))
            thread.finished.connect(thread.deleteLater)
            thread.start()

    def _on_validated(self, path, ok, err, cleanup):
        self.log_line.emit("%s : %s" % (os.path.basename(path),
                                        "OK" if ok else err))
        if cleanup is not None:
            cleanup()

    # ------------------------------------------------------------ internals
    def _start_next(self):
        if self._busy():
            return
        for i, job in enumerate(self.jobs):
            if job.state == QUEUED:
                self.current = i
                self._launch(job, i)
                return
        self.current = -1
        self.run_finished.emit()

    def _launch(self, job, idx):
        job.outdir = resolve_outdir(job.deck, self.outdir_mode,
                                    self.outdir_custom, job.engine)
        os.makedirs(job.outdir, exist_ok=True)
        job.state = RUNNING
        job.progress = 0.0
        job.report_step = 0
        job.report_total = 0
        job.error = ""
        job.started_at = time.monotonic()
        job.elapsed_ms = 0
        self.job_changed.emit(idx)
        self.run_started.emit(idx)

        if job.engine == "jutul":
            worker = _JutulWorker(job.deck, job.outdir)
        else:
            worker = _RunWorker(job.deck, dict(self.params), job.outdir,
                                handle=job.handle)
        job.worker = worker
        worker.log_line.connect(self.log_line)
        if hasattr(worker, "progress"):
            worker.progress.connect(
                lambda step, total, idx=idx:
                self._on_progress(idx, step, total))
        if hasattr(worker, "step_done"):
            worker.step_done.connect(
                lambda info, idx=idx: self._on_step_done(idx, info))
        worker.finished_ok.connect(
            lambda result, idx=idx: self._on_finished(idx, result))
        worker.failed.connect(
            lambda msg, idx=idx: self._on_failed(idx, msg))
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _on_progress(self, idx, step, total):
        job = self.job(idx)
        if job is None:
            return
        job.report_step = int(step)
        job.report_total = int(total)
        job.progress = (100.0 * step / total) if total else 0.0
        self.job_changed.emit(idx)

    def _on_step_done(self, idx, info):
        job = self.job(idx)
        if job is None:
            return
        job.report_step = int(info.get("index", 0)) + 1
        if job.report_total:
            job.progress = min(100.0, 100.0 * job.report_step /
                               job.report_total)
        self.job_changed.emit(idx)
        self.step_done.emit(info)

    def _on_finished(self, idx, result):
        job = self.job(idx)
        if job is None:
            return
        job.state = DONE
        job.result = result
        job.progress = 100.0
        job.elapsed_ms = int((time.monotonic() - job.started_at) * 1000)
        job.outdir = result.get("result_dir") or job.outdir
        self.job_changed.emit(idx)
        self.result_ready.emit(result)
        self._start_next()

    def _on_failed(self, idx, msg):
        job = self.job(idx)
        if job is None:
            return
        job.state = FAILED
        job.error = msg
        self.log_line.emit("job failed: %s" % msg)
        self.job_changed.emit(idx)
        self._start_next()


# ===========================================================================
# log panel
# ===========================================================================
class _LogPanel(QWidget):
    """Read-only console with tail-following (flow-gui parity)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(200000)
        font = QFont("Consolas", 9)
        self.log_view.setFont(font)
        layout.addWidget(self.log_view)

    def append(self, text):
        self.log_view.appendPlainText(text)
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear(self):
        self.log_view.clear()


# ===========================================================================
# plotting panels
# ===========================================================================
class _MplPanel(QWidget):
    """A matplotlib figure with optional control rows beneath the canvas."""

    def __init__(self, parent=None, nrows=0):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.figure = _Figure(figsize=(7, 5), tight_layout=True)
        self.canvas = _Canvas(self.figure)
        layout.addWidget(self.canvas, 1)
        if nrows:
            controls = QVBoxLayout()
            for _ in range(nrows):
                row = QHBoxLayout()
                controls.addLayout(row)
                self.addControlRow(row)
            layout.addLayout(controls)

    def addControlRow(self, row):
        raise NotImplementedError

    def redraw(self):
        self.canvas.draw_idle()


class _WellCurvesPanel(_MplPanel):
    """2D well-solution curves (PRST ``well_curves.plot_well_sols``)."""

    def __init__(self, parent=None):
        self._wellsols = []
        self._times = []
        self._all_wells = []
        self._pavg = []
        self._history = {}
        super().__init__(parent, nrows=2)

    # ---- control rows ----
    def addControlRow(self, row):
        if not hasattr(self, "_field_box"):
            row.addWidget(QLabel("Field"))
            self._field_box = QComboBox()
            self._field_box.addItems(["bhp", "qOs", "qWs", "qGs", "status"])
            self._field_box.currentTextChanged.connect(self._refresh)
            row.addWidget(self._field_box)

            row.addWidget(QLabel("Well"))
            self._well_box = QComboBox()
            self._well_box.addItem("All wells")
            self._well_box.currentTextChanged.connect(self._refresh)
            row.addWidget(self._well_box)

            row.addWidget(QLabel("Units"))
            self._units_box = QComboBox()
            self._units_box.addItems(["metric", "field", "si", "lab"])
            self._units_box.currentTextChanged.connect(self._refresh)
            row.addWidget(self._units_box)

            row.addWidget(QLabel("Block"))
            self._block_box = QComboBox()
            self._block_box.addItems(
                ["—", "产油量", "产液量", "产水量", "含水",
                 "累积产油", "累积产液", "累积产水", "压力"])
            self._block_box.currentTextChanged.connect(self._refresh)
            row.addWidget(self._block_box)

            self._history_check = QCheckBox("History")
            self._history_check.toggled.connect(self._refresh)
            row.addWidget(self._history_check)
            self._history_btn = QPushButton("Load…")
            self._history_btn.clicked.connect(self._load_history)
            row.addWidget(self._history_btn)
            row.addStretch(1)
        else:
            for text, attr in (("Cumulative", "_cumsum"), ("Abs value", "_abs"),
                               ("Log y", "_logy"), ("Stairs", "_stairs"),
                               ("Legend", "_legend")):
                box = QCheckBox(text)
                setattr(self, attr, box)
                box.toggled.connect(self._refresh)
                row.addWidget(box)
            row.addStretch(1)

    # ---- data ----
    def append_step(self, info):
        ws = []
        for well in info.get("wellSol", []):
            name = well.get("name", "?")
            entry = {"name": name}
            for key in ("bhp", "qOs", "qWs", "qGs", "status"):
                raw = well.get(key)
                if raw is None:
                    continue
                arr = np.atleast_1d(np.asarray(raw, dtype=float))
                if arr.size:
                    value = float(arr[0])
                    if key in ("qOs", "qWs", "qGs"):
                        value *= 86400.0
                    elif key == "bhp":
                        value *= 1e-5
                    entry[key] = value
            ws.append(entry)
        if not ws:
            return
        self._wellsols.append(ws)
        self._times.append(info["time_days"])
        pressure = info.get("state", {}).get("pressure")
        if pressure is not None:
            arr = np.atleast_1d(np.asarray(pressure, dtype=float))
            self._pavg.append(float(arr.mean()) * 1e-5
                              if arr.size else float("nan"))
        else:
            self._pavg.append(float("nan"))
        for w in ws:
            if w["name"] not in self._all_wells:
                self._all_wells.append(w["name"])
        self._sync_well_box()
        self._refresh()

    def _sync_well_box(self):
        current = self._well_box.currentText()
        self._well_box.blockSignals(True)
        self._well_box.clear()
        self._well_box.addItem("All wells")
        for name in self._all_wells:
            self._well_box.addItem(name)
        index = self._well_box.findText(current)
        self._well_box.setCurrentIndex(index if index >= 0 else 0)
        self._well_box.blockSignals(False)

    def clear(self):
        self._wellsols, self._times, self._all_wells = [], [], []
        self._pavg = []
        self.figure.clear()
        self.redraw()

    def _load_history(self):
        from PRSTCore.visualization.well_curves import load_history
        path, _ = QFileDialog.getOpenFileName(
            self, "Load history CSV", "", "CSV files (*.csv);;All files (*)")
        if not path:
            return
        try:
            self._history = load_history(path)
        except Exception as exc:
            QMessageBox.warning(self, "History load failed",
                                "%s: %s" % (type(exc).__name__, exc))
            return
        self._history_check.setChecked(True)
        self._refresh()

    def _refresh(self, *_):
        from PRSTCore.visualization.well_curves import (
            plot_block_data, plot_history, plot_well_sols)

        if not self._wellsols:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        units = self._units_box.currentText()
        block = self._block_box.currentText()
        if block != "—":
            plot_block_data(self._wellsols, self._times, self._pavg, block,
                            ax=ax, unit_system=units, time_scale="days")
        else:
            well_text = self._well_box.currentText()
            wells = None if well_text in ("All wells", "") else [well_text]
            plot_well_sols(
                self._wellsols, time=self._times,
                field=self._field_box.currentText(),
                unit_system=units,
                cumsum=self._cumsum.isChecked(),
                abs_value=self._abs.isChecked(),
                logy=self._logy.isChecked(),
                stairs=self._stairs.isChecked(),
                legend=self._legend.isChecked(),
                wells=wells, ax=ax)
            if self._history_check.isChecked() and self._history:
                order = wells if wells is not None else list(self._all_wells)
                plot_history(ax, self._history,
                             field=self._field_box.currentText(),
                             wells=order or None,
                             unit_system=units, time_scale="days")
                ax.legend(loc="best")
        self.redraw()


class _SlicePanel(_MplPanel):
    """2D slice of per-step fields (PRST ``grid_plots.plot_slice``) with an
    Open-3D button (PRST VTK viewer) and an HDF5 result loader."""

    _AXIS = {"k": 2, "j": 1, "i": 0}

    def __init__(self, parent=None):
        self._G = None
        self._wells = None
        self._states = []
        self._times = []
        self._slice = "k"
        self._index = 1
        self._step = 0
        self._vtk_windows = []
        super().__init__(parent, nrows=1)

    def addControlRow(self, row):
        row.addWidget(QLabel("Slice"))
        self._slice_box = QComboBox()
        self._slice_box.addItems(["k", "j", "i"])
        self._slice_box.currentTextChanged.connect(self._on_slice)
        row.addWidget(self._slice_box)
        self._index_box = QSpinBox()
        self._index_box.setRange(1, 1)
        self._index_box.setValue(1)
        self._index_box.valueChanged.connect(self._on_index)
        row.addWidget(self._index_box)
        row.addWidget(QLabel("Field"))
        self._field_box = QComboBox()
        self._field_box.currentTextChanged.connect(self._draw)
        row.addWidget(self._field_box)
        row.addWidget(QLabel("Step"))
        self._step_spin = QSpinBox()
        self._step_spin.setRange(0, 0)
        self._step_spin.valueChanged.connect(self.set_step)
        row.addWidget(self._step_spin)
        self._time_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self._time_slider.setEnabled(False)
        self._time_slider.valueChanged.connect(self.set_step)
        row.addWidget(self._time_slider, 1)
        self.time_label = QLabel("step 0/0")
        row.addWidget(self.time_label)
        row.addWidget(QLabel("cmap"))
        self._cmap_box = QComboBox()
        self._cmap_box.addItems(
            ["viridis", "turbo", "plasma", "jet", "coolwarm", "RdBu_r"])
        self._cmap_box.currentTextChanged.connect(self._draw)
        row.addWidget(self._cmap_box)
        self._wells_check = QCheckBox("Wells")
        self._wells_check.setChecked(True)
        self._wells_check.toggled.connect(self._draw)
        row.addWidget(self._wells_check)
        self.load_btn = QPushButton("Load results...")
        self.load_btn.setToolTip("Load an HDF5 result folder (states.h5)")
        self.load_btn.clicked.connect(self._load_results)
        row.addWidget(self.load_btn)
        self._vtk_button = QPushButton("Open 3D...")
        self._vtk_button.setEnabled(False)
        self._vtk_button.clicked.connect(self._open_vtk_viewer)
        row.addWidget(self._vtk_button)
        row.addStretch(1)

    def _on_slice(self, name):
        if name:
            self._slice = name
            self._sync_index_range()
            self._draw()

    def _on_index(self, value):
        self._index = int(value)
        self._draw()

    def _sync_index_range(self):
        if self._G is None:
            return
        dims = np.asarray(self._G.get("cartDims", [1, 1, 1]),
                          dtype=int).ravel()
        maximum = max(1, int(dims[self._AXIS[self._slice]]))
        was = self._index_box.blockSignals(True)
        self._index_box.setRange(1, maximum)
        self._index_box.setValue(min(self._index, maximum))
        self._index_box.blockSignals(was)
        self._index = self._index_box.value()

    # ------------------------------------------------------------ data
    def set_run(self, G, wells, states, times):
        """Install a finished run's grid/wells/per-cell states."""
        self._G, self._wells = G, wells
        self._states, self._times = list(states), list(times)
        self._reload_field_list()
        self._set_step(0)
        self._vtk_button.setEnabled(bool(states))

    def set_h5(self, jr, G=None, wells=None):
        """Show HDF5 results.  With a matching loaded model ``G`` the active
        ordering is reused; otherwise a uniform grid is synthesised and the
        values are scattered onto the natural grid via ``active_to_natural``.
        """
        if jr is None or jr.pressure is None or jr.pressure.shape[0] < 1:
            return False
        dims = jr.grid_dims or [1, 1, 1]
        if G is not None and int(G["cells"]["num"]) == jr.pressure.shape[1]:
            states = self._h5_states_active(jr)
        else:
            G = self._synthetic_grid(dims)
            states = self._h5_states_natural(jr, G)
        self._G, self._wells = G, wells
        self._states = states
        n_steps = len(states)
        times = []
        t0 = jr.dates[0] if jr.dates else None
        for d in jr.dates[:n_steps]:
            try:
                times.append((datetime.fromisoformat(d) -
                              datetime.fromisoformat(t0)).days)
            except Exception:
                times.append(0.0)
        self._times = times
        self._reload_field_list()
        self._set_step(0)
        self._vtk_button.setEnabled(len(states) > 0)
        return True

    @staticmethod
    def _synthetic_grid(dims):
        from PRSTCore.gridprocessing import cart_grid, compute_geometry
        dims = [int(v) for v in dims]
        return compute_geometry(cart_grid(dims, dims))

    def _h5_states_active(self, jr):
        attrs = {"SWAT": "swat", "SOIL": "soil", "SGAS": "sgas"}
        keys = ["pressure"] + [k for k in attrs
                               if getattr(jr, attrs[k]) is not None]
        states = []
        n = jr.pressure.shape[0]
        for s in range(n):
            state = {"pressure": np.asarray(jr.pressure[s], dtype=float)}
            for k in keys[1:]:
                state[k] = np.asarray(getattr(jr, attrs[k])[s], dtype=float)
            states.append(state)
        return states

    def _h5_states_natural(self, jr, G):
        field_dict = h5_results.to_field_dict(jr)
        nat = field_dict["pressure"]
        sats = field_dict["saturations"]
        keys = ["pressure"] + sorted(sats)
        states = []
        for s in range(nat.shape[0]):
            state = {"pressure": nat[s]}
            for k in sats:
                state[k] = sats[k][s]
            states.append({k: state[k] for k in keys})
        return states

    def _reload_field_list(self):
        names = []
        if self._states:
            names = [k for k in self._states[0].keys()
                     if np.asarray(self._states[0][k]).ndim == 1]
        current = self._field_box.currentText()
        self._field_box.blockSignals(True)
        self._field_box.clear()
        self._field_box.addItems(names)
        if current in names:
            self._field_box.setCurrentText(current)
        self._field_box.blockSignals(False)
        self._sync_index_range()

    def _set_step(self, index):
        if not self._states:
            return
        self._step = max(0, min(int(index), len(self._states) - 1))
        self._sync_step_widgets()
        self._draw()

    def set_step(self, index):
        if not self._states:
            return
        index = max(0, min(int(index), len(self._states) - 1))
        if index != self._step:
            self._step = index
            self._sync_step_widgets()
            self._draw()

    def _sync_step_widgets(self):
        n = len(self._states)
        self._step_spin.blockSignals(True)
        self._step_spin.setMaximum(n - 1)
        self._step_spin.setValue(self._step)
        self._step_spin.blockSignals(False)
        self._time_slider.blockSignals(True)
        self._time_slider.setMaximum(n - 1)
        self._time_slider.setValue(self._step)
        self._time_slider.setEnabled(n > 1)
        self._time_slider.blockSignals(False)
        self.time_label.setText("step %d/%d" % (self._step, n - 1))

    def current_step(self):
        return self._step

    def _draw(self, *_):
        from PRSTCore.visualization.grid_plots import plot_slice
        if self._G is None or not self._states:
            return
        field = self._field_box.currentText()
        if not field:
            return
        step = max(0, min(self._step, len(self._states) - 1))
        values = np.asarray(self._states[step][field], dtype=float)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        plot_slice(self._G, values, self._AXIS[self._slice], self._index,
                   wells=self._wells if self._wells_check.isChecked() else None,
                   ax=ax, cmap=self._cmap_box.currentText())
        ax.set_title("%s — %s slice %d (step %d/%d, t=%.1f d)"
                     % (field, self._slice, self._index, step + 1,
                        len(self._states),
                        self._times[step] if step < len(self._times) else 0.0))
        self.redraw()

    # ------------------------------------------------------------ results
    def _load_results(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select HDF5 result folder (states.h5)")
        if not path:
            return
        try:
            jr = h5_results.load(path)
        except Exception as exc:
            QMessageBox.warning(self, "Load results",
                                "Could not read %s: %s: %s"
                                % (path, type(exc).__name__, exc))
            return
        self.set_h5(jr)
        win = self.window()
        if win is not None and hasattr(win, "_log"):
            win._log("Loaded HDF5 results from %s (%d steps)"
                     % (path, jr.n_steps))

    # ------------------------------------------------------------- 3D
    def _open_vtk_viewer(self):
        """Open the interactive VTK 3D viewer (PRST ``scene3d``/``qt_viewer``)."""
        if self._G is None or not self._states:
            return
        try:
            from PRSTCore.visualization.scene3d import ReservoirScene
            from PRSTCore.visualization.qt_viewer import ReservoirWindow
        except Exception as exc:
            QMessageBox.warning(
                self, "VTK viewer unavailable",
                "The VTK/Qt 3D viewer could not be loaded (vtk not "
                "installed?):\n%s: %s" % (type(exc).__name__, exc))
            return
        try:
            scene = ReservoirScene(self._G, W=list(self._wells or []))
            keys = tuple(k for k in ("pressure", "SWAT", "SOIL", "SGAS",
                                     "sW", "sG", "rs", "rv")
                         if k in self._states[0])
            scene.add_states(self._states, keys=keys)
            window = ReservoirWindow(scene,
                                     title="PRSTCore 3D (VTK)").start()
            self._vtk_windows.append(window)   # keep the window alive
        except Exception as exc:
            QMessageBox.warning(self, "VTK viewer failed",
                                "%s: %s" % (type(exc).__name__, exc))


# ===========================================================================
# deck editor (pure Qt; port of the workbench deck_page)
# ===========================================================================
_SECTIONS = ["RUNSPEC", "GRID", "EDIT", "PROPS", "REGIONS", "SOLUTION",
             "SUMMARY", "SCHEDULE"]

#: PROPS keywords that carry control/report records, not PVT/relperm tables.
_SKIP_PROPS_KEYWORDS = frozenset({"RPTPROPS", "RPTRST", "RPTSCHED", "RPTONLY",
                                  "NUPCOL"})

_KEYWORD_RE = re.compile(
    r"^([A-Z][A-Z0-9_-]{0,7})(?:\s*/\s*)?"
    r"(?:\s*(?:--.*|={2,}.*|-{2,}.*))?$")
_HL_KEYWORD_RE = re.compile(r"^\s*([A-Z][A-Z0-9_]{1,7})\b")
_NUM_RE = re.compile(r"\b\d+\*?(?:\d+(?:\.\d*)?(?:[eEdD][+-]?\d+)?)?\b"
                     r"|\b\d*\.\d+(?:[eEdD][+-]?\d+)?\b")
_STR_RE = re.compile(r"'[^']*'")

_ROLE_FILE = 256          # (file_path, line_index)
_ROLE_SECTION = 257       # True for section headers
_ROLE_INCLUDE = 258       # resolved include target path (INCLUDE nodes)


def _include_target(line: str) -> str:
    line = line.split("--")[0]
    m = re.search(r"INCLUDE\s+'([^']+)'", line, re.I)
    if m:
        return m.group(1).strip()
    for tok in re.findall(r"\S+", line):
        t = tok.strip("'\"")
        if not t:
            continue
        if t.upper() == "INCLUDE":
            continue
        if t == "/":
            break
        t = t.rstrip("/")
        if t:
            return t
    return ""


def _resolve_include(inc: str, basedir: str, rootdir: str) -> str:
    inc = inc.strip().strip("'\"").replace("\\", "/")
    for base in (basedir, rootdir):
        cand = os.path.normpath(os.path.join(base, inc))
        if os.path.isfile(cand):
            return cand
    return ""


def _read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


class _EclipseHighlighter(QSyntaxHighlighter):
    """Eclipse deck highlighting: sections, keywords, comments, strings."""

    def __init__(self, document):
        super().__init__(document)
        self._section = QTextCharFormat()
        self._section.setForeground(QColor(0x8a, 0x1f, 0x6e))
        self._section.setFontWeight(QFont.Bold)
        self._keyword = QTextCharFormat()
        self._keyword.setForeground(QColor(0x24, 0x35, 0x8a))
        self._keyword.setFontWeight(QFont.Bold)
        self._comment = QTextCharFormat()
        self._comment.setForeground(QColor(0x4e, 0x7d, 0x3a))
        self._comment.setFontItalic(True)
        self._number = QTextCharFormat()
        self._number.setForeground(QColor(0x0b, 0x66, 0x6b))
        self._string = QTextCharFormat()
        self._string.setForeground(QColor(0x9a, 0x30, 0x0e))
        self._slash = QTextCharFormat()
        self._slash.setForeground(QColor(0x24, 0x35, 0x8a))
        self._slash.setFontWeight(QFont.Bold)

    def highlightBlock(self, text: str) -> None:
        for m in _NUM_RE.finditer(text):
            self.setFormat(m.start(), m.end() - m.start(), self._number)
        for m in _STR_RE.finditer(text):
            self.setFormat(m.start(), m.end() - m.start(), self._string)
        km = _HL_KEYWORD_RE.match(text)
        if km:
            kw = km.group(1)
            fmt = self._section if kw in _SECTIONS else self._keyword
            self.setFormat(km.start(1), len(kw), fmt)
        for i, ch in enumerate(text):
            if ch == "/":
                self.setFormat(i, 1, self._slash)
        idx = text.find("--")
        if idx >= 0:
            self.setFormat(idx, len(text) - idx, self._comment)


class _LineNumberArea(QWidget):
    def __init__(self, editor):
        super().__init__(editor)
        self._editor = editor

    def sizeHint(self):
        from PySide6.QtCore import QSize
        return QSize(self._editor.line_number_area_width(), 0)

    def paintEvent(self, event):
        self._editor.line_number_area_paint(event)


class _DeckTextEdit(QPlainTextEdit):
    """QPlainTextEdit with a line-number margin and an INCLUDE double-click
    signal (position of the click in the document)."""

    doubleClickedAt = _Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._line_area = _LineNumberArea(self)
        self.blockCountChanged.connect(self._update_line_area_width)
        self.updateRequest.connect(self._update_line_area)
        self._update_line_area_width()

    def line_number_area_width(self):
        digits = len(str(max(1, self.blockCount())))
        return 8 + self.fontMetrics().horizontalAdvance("9") * digits

    def _update_line_area_width(self, *_):
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def _update_line_area(self, rect, dy):
        if dy:
            self._line_area.scroll(0, dy)
        else:
            self._line_area.update(0, rect.y(), self._line_area.width(),
                                   rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_line_area_width()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        from PySide6.QtCore import QRect
        cr = self.contentsRect()
        self._line_area.setGeometry(QRect(cr.left(), cr.top(),
                                          self.line_number_area_width(),
                                          cr.height()))

    def line_number_area_paint(self, event):
        painter = QPainter(self._line_area)
        painter.fillRect(event.rect(), QColor("#f2f2f2"))
        block = self.firstVisibleBlock()
        number = block.blockNumber()
        top = round(self.blockBoundingGeometry(block)
                    .translated(self.contentOffset()).top())
        bottom = top + round(self.blockBoundingRect(block).height())
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                painter.setPen(QColor("#808080"))
                painter.drawText(0, top, self._line_area.width() - 4,
                                 self.fontMetrics().height(), Qt.AlignRight,
                                 str(number + 1))
            block = block.next()
            top = bottom
            bottom = top + round(self.blockBoundingRect(block).height())

    def mouseDoubleClickEvent(self, event):
        self.doubleClickedAt.emit(
            self.cursorForPosition(event.position()).position())
        super().mouseDoubleClickEvent(event)


class _FileTab(QWidget):
    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.editor = _DeckTextEdit()
        self.highlighter = _EclipseHighlighter(self.editor.document())
        layout.addWidget(self.editor)


class DeckEditorPage(QWidget):
    """Left: filterable section/keyword tree; right: tabbed deck editors
    with syntax highlighting, find/replace, undo/redo and save-back."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.path = ""
        self._mtimes: dict = {}
        self._deck_files: list = []

        split = QSplitter()
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.tree_filter = QLineEdit()
        self.tree_filter.setPlaceholderText("Filter keywords / INCLUDEs...")
        self.tree_filter.textChanged.connect(self._filter_tree)
        lv.addWidget(self.tree_filter)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Deck structure"])
        self.tree.itemClicked.connect(self._on_tree_click)
        lv.addWidget(self.tree, 1)
        split.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(2)

        bar = QHBoxLayout()
        self.open_btn = QPushButton("Open...")
        self.open_btn.clicked.connect(self._open_dialog)
        bar.addWidget(self.open_btn)
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self._save)
        bar.addWidget(self.save_btn)
        self.save_all_btn = QPushButton("Save all")
        self.save_all_btn.clicked.connect(self._save_all)
        bar.addWidget(self.save_all_btn)
        bar.addSpacing(12)
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self._undo)
        bar.addWidget(self.undo_btn)
        self.redo_btn = QPushButton("Redo")
        self.redo_btn.clicked.connect(self._redo)
        bar.addWidget(self.redo_btn)
        bar.addStretch(1)
        self.status_label = QLabel("No deck open — click Open or Ctrl+O")
        self.status_label.setStyleSheet("color: #555555;")
        bar.addWidget(self.status_label)
        rv.addLayout(bar)

        fbar = QHBoxLayout()
        fbar.addWidget(QLabel("Find:"))
        self.find_edit = QLineEdit()
        self.find_edit.setPlaceholderText("Find...")
        self.find_edit.returnPressed.connect(self._find)
        fbar.addWidget(self.find_edit, 1)
        b_find = QPushButton("Find")
        b_find.clicked.connect(self._find)
        fbar.addWidget(b_find)
        self.case_chk = QCheckBox("Case")
        self.case_chk.setChecked(True)
        fbar.addWidget(self.case_chk)
        self.deck_chk = QCheckBox("All files")
        fbar.addWidget(self.deck_chk)
        self.find_info = QLabel("")
        fbar.addWidget(self.find_info)
        fbar.addWidget(QLabel("Replace:"))
        self.replace_edit = QLineEdit()
        self.replace_edit.setPlaceholderText("Replace...")
        fbar.addWidget(self.replace_edit, 1)
        b_rep = QPushButton("Replace")
        b_rep.clicked.connect(self._replace)
        fbar.addWidget(b_rep)
        b_rep_all = QPushButton("Replace all")
        b_rep_all.clicked.connect(self._replace_all)
        fbar.addWidget(b_rep_all)
        rv.addLayout(fbar)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.currentChanged.connect(self._update_status)
        rv.addWidget(self.tabs, 1)
        split.addWidget(right)
        split.setSizes([300, 720])

        layout = QVBoxLayout(self)
        layout.addWidget(split)

        QShortcut(QKeySequence("Ctrl+O"), self, activated=self._open_dialog)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._save)
        QShortcut(QKeySequence("Ctrl+Shift+S"), self, activated=self._save_all)
        QShortcut(QKeySequence("Ctrl+F"), self, activated=self._focus_find)
        QShortcut(QKeySequence("F3"), self, activated=self._find)

    # ------------------------------------------------------------- opening
    @property
    def editor(self):
        tab = self._current_tab()
        return tab.editor if tab is not None else None

    def _current_tab(self):
        idx = self.tabs.currentIndex()
        if 0 <= idx < self.tabs.count():
            return self.tabs.widget(idx)
        return None

    def _open_dialog(self):
        start = (os.path.dirname(self.path) if self.path
                 else os.path.expanduser("~"))
        path, _ = QFileDialog.getOpenFileName(
            self, "Open deck", start,
            "Eclipse decks (*.DATA);;All files (*)")
        if path:
            self.open_deck(path)

    def open_deck(self, path: str):
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Deck editor",
                                "No such file: %s" % path)
            return
        self.path = os.path.normpath(path)
        self._mtimes = {}
        self._deck_files = []
        self._close_all_tabs()
        self._build_tree()
        self._open_file(self.path)
        self.status_label.setText(
            "Deck: %s  (%d file%s)" % (
                os.path.basename(self.path), len(self._deck_files),
                "" if len(self._deck_files) == 1 else "s"))

    def _close_all_tabs(self):
        while self.tabs.count():
            w = self.tabs.widget(0)
            self.tabs.removeTab(0)
            w.deleteLater()

    def _open_file(self, path: str, line: int = -1):
        if not path or not os.path.isfile(path):
            return
        path = os.path.normpath(path)
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            if tab.path == path:
                self.tabs.setCurrentIndex(i)
                if line >= 0:
                    self._goto_line(tab.editor, line)
                return
        tab = _FileTab(path)
        tab.editor.setPlainText(_read_text(path))
        tab.editor.document().setModified(False)
        tab.editor.textChanged.connect(
            lambda: self._on_tab_changed(tab))
        tab.editor.cursorPositionChanged.connect(self._update_status)
        tab.editor.doubleClickedAt.connect(
            lambda pos: self._on_double_click(tab, pos))
        self.tabs.addTab(tab, os.path.basename(path))
        self.tabs.setCurrentWidget(tab)
        self._mtimes[path] = os.path.getmtime(path)
        if line >= 0:
            self._goto_line(tab.editor, line)
        self._update_status()

    def _goto_line(self, editor, line: int):
        cursor = editor.textCursor()
        cursor.setPosition(0)
        cursor.movePosition(QTextCursor.Down, QTextCursor.MoveAnchor, line)
        editor.setTextCursor(cursor)
        editor.centerCursor()

    # ------------------------------------------------------------- tree
    def _build_tree(self):
        self.tree.clear()

        def scan_file(path: str, parent_item):
            if path not in self._deck_files:
                self._deck_files.append(path)
            lines = _read_text(path).splitlines()
            section_item = None
            for i, line in enumerate(lines):
                m = _KEYWORD_RE.match(line)
                if not m:
                    continue
                kw = m.group(1)
                if kw in _SECTIONS:
                    item = QTreeWidgetItem(
                        ["%s  (line %d)" % (kw, i + 1)])
                    item.setData(0, _ROLE_FILE, (path, i))
                    item.setData(0, _ROLE_SECTION, True)
                    if parent_item is None:
                        self.tree.addTopLevelItem(item)
                    else:
                        parent_item.addChild(item)
                    section_item = item
                elif kw == "INCLUDE":
                    inc = _include_target(line)
                    if not inc:
                        for j in range(i + 1, min(i + 5, len(lines))):
                            inc = _include_target(lines[j])
                            if inc:
                                break
                    target = _resolve_include(inc, os.path.dirname(path),
                                              os.path.dirname(self.path))
                    label = "INCLUDE  %s  (line %d)" % (inc or "?", i + 1)
                    item = QTreeWidgetItem([label])
                    item.setData(0, _ROLE_FILE, (path, i))
                    item.setData(0, _ROLE_SECTION, False)
                    item.setData(0, _ROLE_INCLUDE, target or None)
                    target_parent = (parent_item
                                     if parent_item is not None
                                     else self.tree.invisibleRootItem())
                    target_parent.addChild(item)
                    if target:
                        scan_file(target, item)
                else:
                    item = QTreeWidgetItem(["%s  (line %d)" % (kw, i + 1)])
                    item.setData(0, _ROLE_FILE, (path, i))
                    item.setData(0, _ROLE_SECTION, False)
                    host = section_item
                    if host is None:
                        host = (parent_item if parent_item is not None
                                else self.tree.invisibleRootItem())
                    host.addChild(item)

        scan_file(self.path, None)
        self.tree.expandAll()

    def _filter_tree(self, text: str):
        needle = text.strip().lower()

        def walk(item) -> bool:
            self_match = not needle or needle in item.text(0).lower()
            child_match = False
            for i in range(item.childCount()):
                child_match = walk(item.child(i)) or child_match
            item.setHidden(bool(needle) and not self_match and not child_match)
            if bool(needle) and child_match:
                item.setExpanded(True)
            return self_match or child_match

        walk(self.tree.invisibleRootItem())

    def _on_tree_click(self, item, column):
        _ = column
        inc = item.data(0, _ROLE_INCLUDE)
        if inc:
            self._open_file(inc, 0)
            return
        data = item.data(0, _ROLE_FILE)
        if data is None:
            return
        file_path, line = data
        self._open_file(file_path, line)

    def _on_double_click(self, tab, position):
        cursor = tab.editor.textCursor()
        cursor.setPosition(position)
        block = cursor.block()
        inc = _include_target(block.text())
        if not inc:
            nxt = block.next()
            if nxt.isValid():
                inc = _include_target(nxt.text())
        if not inc:
            return
        target = _resolve_include(inc, os.path.dirname(tab.path),
                                  os.path.dirname(self.path))
        if target:
            self._open_file(target)

    # ------------------------------------------------------ find/replace
    def _focus_find(self):
        self.find_edit.setFocus()
        self.find_edit.selectAll()

    def _search_flags(self):
        flags = QTextDocument.FindFlags()
        if self.case_chk.isChecked():
            flags |= QTextDocument.FindCaseSensitively
        return flags

    def _target_editors(self):
        if self.deck_chk.isChecked():
            return [self.tabs.widget(i).editor
                    for i in range(self.tabs.count())
                    if self.tabs.widget(i) is not None]
        editor = self.editor
        return [editor] if editor is not None else []

    def _find(self):
        text = self.find_edit.text()
        if not text:
            return
        flags = self._search_flags()
        editor = self.editor
        if editor is None:
            return
        found = editor.document().find(text, editor.textCursor(), flags)
        if found.isNull():
            found = editor.document().find(
                text, QTextCursor(editor.document()), flags)
        if not found.isNull():
            editor.setTextCursor(found)
        rx = re.compile(re.escape(text), 0 if self.case_chk.isChecked()
                        else re.IGNORECASE)
        total = sum(len(rx.findall(ed.toPlainText()))
                    for ed in self._target_editors())
        self.find_info.setText("%d match%s" % (total,
                                               "" if total == 1 else "es"))

    def _replace(self):
        old, new = self.find_edit.text(), self.replace_edit.text()
        if not old:
            return
        editor = self.editor
        if editor is None:
            return
        cursor = editor.textCursor()
        if cursor.hasSelection() and cursor.selectedText() == old:
            cursor.insertText(new)
        else:
            flags = self._search_flags()
            found = editor.document().find(old, cursor, flags)
            if not found.isNull():
                found.insertText(new)

    def _replace_all(self):
        old, new = self.find_edit.text(), self.replace_edit.text()
        if not old:
            return
        rx = re.compile(re.escape(old), 0 if self.case_chk.isChecked()
                        else re.IGNORECASE)
        for editor in self._target_editors():
            editor.setPlainText(rx.sub(lambda _m: new,
                                       editor.toPlainText()))

    # ------------------------------------------------------------- save
    def _check_external_change(self, path: str) -> bool:
        old = self._mtimes.get(path)
        return bool(old) and os.path.getmtime(path) != old

    def _save(self):
        tab = self._current_tab()
        if tab is None or not tab.path:
            return
        path = tab.path
        if self._check_external_change(path):
            answer = QMessageBox.question(
                self, "Deck editor",
                "The file changed on disk. Overwrite anyway?")
            if answer != QMessageBox.Yes:
                return
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(tab.editor.toPlainText())
        self._mtimes[path] = os.path.getmtime(path)
        tab.editor.document().setModified(False)
        self._update_status()
        self.status_label.setText("Saved %s" % os.path.basename(path))

    def _save_all(self):
        saved = 0
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            if tab is None or not tab.path:
                continue
            if not tab.editor.document().isModified():
                continue
            if self._check_external_change(tab.path):
                answer = QMessageBox.question(
                    self, "Deck editor",
                    "%s changed on disk. Overwrite anyway?"
                    % os.path.basename(tab.path))
                if answer != QMessageBox.Yes:
                    continue
            with open(tab.path, "w", encoding="utf-8") as fh:
                fh.write(tab.editor.toPlainText())
            self._mtimes[tab.path] = os.path.getmtime(tab.path)
            tab.editor.document().setModified(False)
            saved += 1
        self._update_status()
        if saved:
            self.status_label.setText("Saved %d file%s"
                                      % (saved, "" if saved == 1 else "s"))

    # ------------------------------------------------------- undo/redo
    def _undo(self):
        editor = self.editor
        if editor is not None:
            editor.undo()

    def _redo(self):
        editor = self.editor
        if editor is not None:
            editor.redo()

    # ------------------------------------------------------------- state
    def _on_tab_changed(self, tab):
        idx = self.tabs.indexOf(tab)
        if idx < 0:
            return
        modified = tab.editor.document().isModified()
        self.tabs.setTabText(idx, os.path.basename(tab.path)
                             + (" *" if modified else ""))
        self._update_status()

    def _update_status(self, *_):
        tab = self._current_tab()
        if tab is None:
            self.status_label.setText(
                "No deck open — click Open or Ctrl+O")
            return
        cursor = tab.editor.textCursor()
        line = cursor.blockNumber() + 1
        col = cursor.positionInBlock() + 1
        mod = " *" if tab.editor.document().isModified() else ""
        self.status_label.setText(
            "%s%s  —  Ln %d, Col %d" % (
                os.path.basename(tab.path), mod, line, col))


# ===========================================================================
# Run page (flow-gui / workbench parity)
# ===========================================================================
class RunPage(QWidget):
    def __init__(self, jobs, window, parent=None):
        super().__init__(parent)
        self.jobs = jobs
        self.window = window

        root = QVBoxLayout(self)

        # ---- model load (explicit entry) --------------------------------
        load_row = QHBoxLayout()
        load_row.addWidget(QLabel("Model:"))
        self.load_path = QLineEdit()
        self.load_path.setPlaceholderText(
            "path to a .DATA deck, e.g. F:\\...\\SPE1CASE1.DATA")
        self.load_path.returnPressed.connect(self._load_typed)
        load_row.addWidget(self.load_path, 1)
        b_load = QPushButton("Load")
        b_load.clicked.connect(self._load_typed)
        load_row.addWidget(b_load)
        self.load_status = QLabel("")
        self.load_status.setStyleSheet("color: #1e7d32;")
        load_row.addWidget(self.load_status)
        root.addLayout(load_row)

        # ---- engine picker ----------------------------------------------
        row = QHBoxLayout()
        row.addWidget(QLabel("Simulator:"))
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("PRST", "prst")
        self.engine_combo.addItem("JutulDarcy", "jutul")
        self.engine_combo.setCurrentIndex(0)
        self.engine_combo.currentIndexChanged.connect(self._on_engine)
        row.addWidget(self.engine_combo)
        self.engine_info = QLabel("PRSTCore (AMGCL/PETSc)")
        row.addWidget(self.engine_info)
        row.addStretch(1)
        root.addLayout(row)

        # ---- job queue table --------------------------------------------
        box = QGroupBox("Job queue  (drop *.DATA files anywhere)")
        box_row = QHBoxLayout(box)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Deck", "Status", "Progress", "Elapsed", "ETA"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setMaximumHeight(240)
        self.table.cellDoubleClicked.connect(
            lambda r, c: self._open_result_folder(r))
        box_row.addWidget(self.table, 1)

        col = QVBoxLayout()
        b_add = QPushButton("Add deck...")
        b_edit = QPushButton("View/Edit deck")
        b_rem = QPushButton("Remove")
        b_clr = QPushButton("Clear")
        b_add.clicked.connect(self._add_decks)
        b_edit.clicked.connect(self._edit_deck)
        b_rem.clicked.connect(self._remove)
        b_clr.clicked.connect(self._clear)
        for b in (b_add, b_edit, b_rem, b_clr):
            col.addWidget(b)
        col.addStretch(1)
        box_row.addLayout(col)
        root.addWidget(box, 1)

        # ---- run options (2 rows x 2 columns, compact) ------------------
        opts = QGroupBox("Run options")
        grid = QVBoxLayout(opts)
        grid.setContentsMargins(6, 2, 6, 4)
        grid.setSpacing(2)
        row_top = QHBoxLayout()
        row_top.setSpacing(8)
        row_bottom = QHBoxLayout()
        row_bottom.setSpacing(8)
        grid.addLayout(row_top)
        grid.addLayout(row_bottom)

        def _field(parent, label, widget):
            row = QHBoxLayout()
            row.setSpacing(4)
            row.addWidget(QLabel(label))
            row.addWidget(widget, 1)
            parent.addLayout(row)

        g1 = QVBoxLayout()
        g1.setSpacing(2)
        g1r1 = QHBoxLayout()
        g1r1.setSpacing(4)
        g1r1.addWidget(QLabel("Method"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["AMGCL CPR", "PETSc CPR"])
        self.method_combo.currentTextChanged.connect(self._on_method)
        g1r1.addWidget(self.method_combo, 1)
        g1r1.addWidget(QLabel("Tolerance"))
        self.tol_edit = QLineEdit("")
        self.tol_edit.setPlaceholderText("default")
        g1r1.addWidget(self.tol_edit, 1)
        g1.addLayout(g1r1)
        g1r2 = QHBoxLayout()
        g1r2.setSpacing(4)
        g1r2.addWidget(QLabel("Max steps"))
        self.max_steps = QSpinBox()
        self.max_steps.setRange(0, 100000)
        self.max_steps.setValue(0)
        self.max_steps.setSpecialValueText("all")
        self.max_steps.setToolTip("0 = run all report steps")
        g1r2.addWidget(self.max_steps, 1)
        g1r2.addWidget(QLabel("Acceptance"))
        self.acc = QDoubleSpinBox()
        self.acc.setRange(0.1, 100.0)
        self.acc.setValue(2.0)
        g1r2.addWidget(self.acc, 1)
        g1.addLayout(g1r2)
        row_top.addLayout(g1)

        g2 = QVBoxLayout()
        g2.setSpacing(2)
        self.amgcl_strat = QComboBox()
        self.amgcl_strat.addItems(["mrst", "mrst_drs", "amgcl", "amgcl_drs"])
        self.amgcl_strat.setCurrentText("mrst")
        _field(g2, "AMGCL strategy", self.amgcl_strat)
        self.amgcl_dec = QComboBox()
        self.amgcl_dec.addItems(["trueIMPES", "quasiIMPES", "none"])
        self.amgcl_dec.setCurrentText("trueIMPES")
        _field(g2, "AMGCL decoupling", self.amgcl_dec)
        chk_row = QHBoxLayout()
        chk_row.setSpacing(6)
        self.linesearch_chk = QCheckBox("Line search")
        self.linesearch_chk.setChecked(True)
        chk_row.addWidget(self.linesearch_chk)
        self.enforce_chk = QCheckBox("Enforce residual decrease")
        self.enforce_chk.setChecked(True)
        chk_row.addWidget(self.enforce_chk)
        chk_row.addStretch(1)
        g2.addLayout(chk_row)
        row_top.addLayout(g2)

        g3 = QVBoxLayout()
        g3.setSpacing(2)
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["cpr", "fieldsplit"])
        self.strategy_combo.setCurrentText("cpr")
        _field(g3, "PETSc strategy", self.strategy_combo)
        self.precond_combo = QComboBox()
        self.precond_combo.addItems(["hypre", "gamg", "ilu", "lu"])
        self.precond_combo.setCurrentText("hypre")
        _field(g3, "Pressure precond", self.precond_combo)
        self.second_combo = QComboBox()
        self.second_combo.addItems(["ilu", "sor", "bjacobi", "icc"])
        self.second_combo.setCurrentText("ilu")
        _field(g3, "2nd stage", self.second_combo)
        row_bottom.addLayout(g3)

        g4 = QVBoxLayout()
        g4.setSpacing(2)
        out_row = QHBoxLayout()
        out_row.setSpacing(4)
        out_row.addWidget(QLabel("Output:"))
        self.outdir_mode = QComboBox()
        self.outdir_mode.addItem("next to deck (<deck>_run_prst)")
        self.outdir_mode.addItem("custom directory")
        self.outdir_mode.setCurrentIndex(0)
        out_row.addWidget(self.outdir_mode, 1)
        g4.addLayout(out_row)
        outdir_row = QHBoxLayout()
        outdir_row.setSpacing(4)
        outdir_row.addWidget(QLabel("Out dir:"))
        self.outdir_edit = QLineEdit("")
        self.outdir_edit.setEnabled(False)
        outdir_row.addWidget(self.outdir_edit, 1)
        b_browse = QPushButton("Browse...")
        b_browse.clicked.connect(self._browse_outdir)
        outdir_row.addWidget(b_browse)
        g4.addLayout(outdir_row)
        g4.addStretch(1)
        row_bottom.addLayout(g4)
        root.addWidget(opts)

        # ---- run controls -----------------------------------------------
        row = QHBoxLayout()
        self.run_sel = QPushButton("Run selected")
        self.run_all = QPushButton("Run queue")
        self.stop_btn = QPushButton("Stop queue")
        self.skip_btn = QPushButton("Skip job")
        self.validate_btn = QPushButton("Validate deck")
        b_clear_log = QPushButton("Clear log")
        self.stop_btn.setEnabled(False)
        self.skip_btn.setEnabled(False)
        self.run_sel.clicked.connect(self._run_selected_clicked)
        self.run_all.clicked.connect(self._run_all)
        self.stop_btn.clicked.connect(self.jobs.stop)
        self.skip_btn.clicked.connect(self.jobs.skip)
        self.validate_btn.clicked.connect(
            lambda: self.jobs.validate(self._selected_rows() or
                                       [self.table.currentRow()]))
        b_clear_log.clicked.connect(self.window.log_panel.clear)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(200)
        self.progress.setVisible(False)
        for b in (self.run_sel, self.run_all, self.stop_btn, self.skip_btn,
                  self.validate_btn, b_clear_log):
            row.addWidget(b)
        row.addWidget(self.progress)
        row.addStretch(1)
        root.addLayout(row)

        # ---- log dock target (bottom of window, not this page) ----------
        self.model_label = QLabel("no deck loaded")
        self.model_label.setWordWrap(True)
        root.addWidget(self.model_label)

        self.jobs.queue_changed.connect(self._rebuild)
        self.jobs.job_changed.connect(self._refresh_row)
        self.jobs.run_started.connect(self._on_run_started)
        self.jobs.run_finished.connect(self._on_run_finished)

    # ------------------------------------------------------------- actions
    def _load_typed(self):
        self.window._load_typed()

    def _on_engine(self, index):
        engine = self.engine_combo.itemData(index)
        self.engine_info.setText(
            "PRSTCore (AMGCL/PETSc)" if engine == "prst"
            else "JutulDarcy (Julia driver)")

    def _on_method(self, method):
        petsc_only = method != "AMGCL CPR"
        for widget in (self.precond_combo, self.second_combo,
                       self.strategy_combo):
            widget.setEnabled(petsc_only)
        for widget in (self.amgcl_strat, self.amgcl_dec):
            widget.setEnabled(not petsc_only)

    def _run_selected_clicked(self):
        """Run the selected jobs with the current UI run options."""
        self.window._prepare_jobs()
        self.jobs.run_selected(self._selected_rows())

    def _run_all(self):
        self.window._prepare_jobs()
        if not self.jobs.jobs:
            # Nothing queued: fall back to running the loaded model.
            if self.window.model is not None:
                self.window._run()
            return
        self.jobs.run_queue()

    def _selected_rows(self):
        rows = sorted({item.row() for item in self.table.selectedItems()})
        if not rows and self.table.currentRow() >= 0:
            rows = [self.table.currentRow()]
        return rows

    def _add_decks(self):
        start = (os.path.dirname(self.load_path.text())
                 if self.load_path.text() else os.path.expanduser("~"))
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add decks", start, "Eclipse decks (*.DATA);;All files (*)")
        for f in files:
            self.jobs.add_deck(f, self.engine_combo.currentData())

    def _remove(self):
        self.jobs.remove_jobs(self._selected_rows())

    def _clear(self):
        self.jobs.clear()

    def _browse_outdir(self):
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self.outdir_edit.setText(path)

    def _edit_deck(self):
        job = None
        rows = self._selected_rows()
        if rows:
            job = self.jobs.job(rows[0])
        if job is None and self.jobs.jobs:
            job = self.jobs.jobs[0]
        if job is None:
            return
        self.window._open_deck_in_editor(job.deck)

    def _open_result_folder(self, row):
        job = self.jobs.job(row)
        if job is None or not job.outdir or not os.path.isdir(job.outdir):
            return
        if os.name == "nt":
            os.startfile(job.outdir)  # noqa: S606
        else:
            import subprocess
            subprocess.Popen(["xdg-open", job.outdir])

    # ------------------------------------------------------------ job table
    def _rebuild(self):
        self.table.setRowCount(len(self.jobs.jobs))
        for i in range(len(self.jobs.jobs)):
            self._refresh_row(i)

    def _refresh_row(self, i):
        job = self.jobs.job(i)
        if job is None:
            return
        self.table.setRowCount(len(self.jobs.jobs))
        deck = QTableWidgetItem(job.deck)
        deck.setToolTip(job.deck)
        status = QTableWidgetItem(job.state + (("  " + job.error)
                                               if job.error else ""))
        status.setForeground(QColor(_STATUS_COLOR[job.state]))
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(int(job.progress))
        elapsed = QTableWidgetItem(_fmt_duration(job.elapsed_ms))
        eta = QTableWidgetItem("")
        if job.state == RUNNING and job.progress > 0:
            remaining = job.elapsed_ms * (100.0 - job.progress) / job.progress
            eta = QTableWidgetItem(_fmt_duration(remaining))
        self.table.setItem(i, 0, deck)
        self.table.setItem(i, 1, status)
        self.table.setCellWidget(i, 2, progress)
        self.table.setItem(i, 3, elapsed)
        self.table.setItem(i, 4, eta)

    def _on_run_started(self, _idx):
        self.stop_btn.setEnabled(True)
        self.skip_btn.setEnabled(True)
        self.run_sel.setEnabled(False)
        self.run_all.setEnabled(False)
        self.progress.setVisible(True)

    def _on_run_finished(self):
        self.stop_btn.setEnabled(False)
        self.skip_btn.setEnabled(False)
        self.run_sel.setEnabled(True)
        self.run_all.setEnabled(True)
        self.progress.setVisible(False)


# ===========================================================================
# well hierarchy page
# ===========================================================================
class WellPage(QWidget):
    """FIELD -> GROUP -> WELL tree (flow-gui parity).

    Populated from a PRST schedule (:meth:`set_schedule`) or from HDF5
    results (:meth:`set_h5`).  A ``source`` callable can be installed so the
    page refreshes itself whenever it is shown; the Refresh button forces a
    re-population at any time.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source = None
        root = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.info = QLabel("No wells. Load a model or results to see wells.")
        bar.addWidget(self.info)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh)
        bar.addWidget(self.refresh_btn)
        self.export_btn = QPushButton("Export PNG...")
        self.export_btn.clicked.connect(self._export)
        self.export_btn.setEnabled(False)
        bar.addWidget(self.export_btn)
        bar.addStretch(1)
        root.addLayout(bar)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Type", "IJK"])
        root.addWidget(self.tree, 1)

    # ---------------------------------------------------------------- data
    def set_source(self, source):
        """Install a callable returning well entries, then refresh."""
        self._source = source
        self._refresh()

    def _refresh(self):
        if self._source is None:
            return
        try:
            wells = self._source() or []
        except Exception:  # noqa: BLE001 - never let the tree die
            wells = []
        self.set_well_data(wells)

    def showEvent(self, event):
        "Re-populate on tab switch so the latest wells are always shown."
        super().showEvent(event)
        self._refresh()

    def set_well_data(self, wells):
        """Populate the tree from ``[{name, group, type, ijk}, ...]``."""
        self.tree.clear()
        self.export_btn.setEnabled(False)
        if not wells:
            self.info.setText("No wells. Load a model or results to see "
                              "wells.")
            return
        self.info.setText("%d wells" % len(wells))

        field_item = QTreeWidgetItem(["FIELD", "", ""])
        self.tree.addTopLevelItem(field_item)
        group_items = {}
        for w in wells:
            name = str(w.get("name", "?"))
            group = str(w.get("group", "") or "FIELD")
            kind = str(w.get("type", "PROD"))
            parent = field_item
            if group and group != "FIELD":
                if group not in group_items:
                    item = QTreeWidgetItem([group, "GROUP", ""])
                    field_item.addChild(item)
                    group_items[group] = item
                parent = group_items[group]
            well = QTreeWidgetItem([name, kind, str(w.get("ijk", ""))])
            well.setForeground(1, QColor("#c62828") if kind != "INJ"
                               else QColor("#1565c0"))
            parent.addChild(well)
        field_item.setExpanded(True)
        self.export_btn.setEnabled(True)

    def set_schedule(self, schedule):
        """Build the well list from a PRST schedule and populate."""
        entries = []
        for w in _union_wells(schedule) if schedule is not None else []:
            i, j = int(w.get("i", 1)), int(w.get("j", 1))
            k = int(w.get("k")[0]) if isinstance(w.get("k"), list) \
                and w.get("k") else 1
            sign = float(w.get("sign", -1.0))
            entries.append({
                "name": str(w.get("name", "?")),
                "group": str(w.get("group", "") or "FIELD"),
                "type": "INJ" if sign > 0 else "PROD",
                "ijk": "%d,%d,%d" % (i, j, k),
            })
        self.set_well_data(entries)

    def set_h5(self, jr):
        """Build the well list from HDF5 results (names + inferred type)."""
        entries = []
        for name, df in ((jr.wells or {}).items()
                         if jr is not None else []):
            inj = any(c in df.columns and np.asarray(df[c]).size
                      and float(np.nanmax(np.asarray(df[c], dtype=float)))
                      > 0.0 for c in ("WWIR", "WGIR"))
            prod = any(c in df.columns and np.asarray(df[c]).size
                       and float(np.nanmax(np.asarray(df[c], dtype=float)))
                       > 0.0 for c in ("WOPR", "WWPR", "WGPR"))
            kind = "INJ" if inj and not prod else ("PROD" if prod else "?")
            entries.append({"name": str(name), "group": "FIELD",
                            "type": kind, "ijk": ""})
        self.set_well_data(entries)

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export well hierarchy",
                                              "wells.png", "PNG (*.png)")
        if not path:
            return
        self.tree.grab().save(path)


# ===========================================================================
# plots page: Timeseries / PVT / Summary sub-tabs (matplotlib)
# ===========================================================================
class _ClickableLineEdit(QLineEdit):
    """QLineEdit that also emits clicked() on any mouse press."""

    clicked = _Signal()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.clicked.emit()


class _SummaryCase:
    """One HDF5 result directory; summary vectors read with h5_results."""

    _COL_LABEL = {
        "WOPR": "Oil prod rate", "WWPR": "Water prod rate",
        "WGPR": "Gas prod rate", "WWIR": "Water inj rate",
        "WGIR": "Gas inj rate", "WBHP": "BHP",
    }

    def __init__(self, outdir: str):
        self.outdir = os.path.normpath(outdir)
        self.name = os.path.basename(self.outdir.rstrip("/\\"))
        self._jr = None

    def _load(self):
        if self._jr is None:
            self._jr = h5_results.load(self.outdir)
        return self._jr

    def vector_keys(self):
        try:
            jr = self._load()
        except Exception:
            return []
        keys = ["FIELD:" + c for c in ("WOPR", "WWPR", "WGPR", "WWIR",
                                       "WGIR", "WBHP")]
        for name in sorted(jr.wells):
            for col in ("WBHP", "WOPR", "WWPR", "WGPR", "WWIR", "WGIR"):
                keys.append("WELL:%s:%s" % (name, col))
        return keys

    def series(self, key):
        """(times_days, values) for a FIELD:/WELL: vector, or None."""
        try:
            jr = self._load()
        except Exception:
            return None
        if not jr.wells:
            return None
        prefix, rest = key.split(":", 1)
        first = next(iter(jr.wells.values()))
        times = first["time_days"].values
        if prefix == "FIELD":
            col = rest
            if col == "WBHP":
                num, den = None, None
                for df in jr.wells.values():
                    if "WOPR" not in df or "WBHP" not in df:
                        continue
                    w = df["WOPR"].values
                    num = w if num is None else num + w
                    den = (df["WOPR"].values * df["WBHP"].values
                           if den is None
                           else den + df["WOPR"].values * df["WBHP"].values)
                if den is None:
                    return None
                with np.errstate(divide="ignore", invalid="ignore"):
                    v = np.where(num > 0, den / np.maximum(num, 1e-30),
                                 np.full_like(num, np.nan))
                return times, v
            vals = None
            for df in jr.wells.values():
                if col not in df:
                    continue
                vals = df[col].values if vals is None else vals + \
                    df[col].values
            if vals is None:
                return None
            return times, vals
        # WELL:name:col
        well, col = rest.split(":", 1)
        df = jr.wells.get(well)
        if df is None or col not in df:
            return None
        return df["time_days"].values, df[col].values

    @staticmethod
    def friendly(key):
        try:
            _, rest = key.split(":", 1)
        except ValueError:
            return key
        col = rest.split(":")[-1]
        return _SummaryCase._COL_LABEL.get(col, col)


def _draw_summary_into(figure, cases, keys):
    figure.clear()
    ax = figure.add_subplot(111)
    for case in cases:
        for key in keys:
            try:
                series = case.series(key)
            except Exception:
                continue
            if series is None:
                continue
            t, v = series
            ax.plot(t, v, lw=1.2,
                    label="%s: %s" % (case.name, _SummaryCase.friendly(key)))
    ax.set_xlabel("time, days")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, loc="best")
    figure.tight_layout()


def _read_deck_text(path, _seen=None):
    """Read a deck file, recursively expanding ``INCLUDE`` files.

    Handles the single-line ``INCLUDE 'file' /`` form and the multi-line
    form (``INCLUDE`` on its own line, path on a following line).  Cycles
    are guarded with a visited set.
    """
    _seen = set() if _seen is None else _seen
    path = os.path.normpath(path)
    if path in _seen:
        return ""
    _seen.add(path)
    text = _read_text(path)
    if not text:
        return ""
    lines = text.splitlines()
    out = []
    i = 0
    basedir = os.path.dirname(path)
    while i < len(lines):
        line = lines[i]
        if re.match(r"^\s*INCLUDE\b", line, re.I):
            inc = _include_target(line)
            consumed = 1
            if not inc:
                for j in range(i + 1, min(i + 5, len(lines))):
                    inc = _include_target(lines[j])
                    if inc:
                        consumed = j - i + 1
                        break
            target = _resolve_include(inc, basedir, basedir) if inc else ""
            if target:
                out.append(_read_deck_text(target, _seen))
            i += consumed
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _props_tables(deck_path):
    """Parse the PROPS section's numeric tables out of a deck file
    (INCLUDE files expanded).

    Returns {keyword: ndarray} -- one numeric block per keyword (one row
    per data record, Eclipse ``n*value`` repeats expanded, ragged records
    padded with NaN so tables like PVTO stay plottable).
    """
    text = _read_deck_text(deck_path)
    if not text:
        return {}
    lines = text.splitlines()
    in_props = False
    tables = {}
    current = None
    rows = []

    def finalize():
        nonlocal current, rows
        if current is not None and rows:
            width = max(len(r) for r in rows)
            arr = np.full((len(rows), width), np.nan, dtype=float)
            for i, r in enumerate(rows):
                arr[i, : len(r)] = r
            tables[current] = arr
        current, rows = None, []

    for raw in lines:
        line = raw.split("--")[0].strip()
        if not line:
            continue
        m = _KEYWORD_RE.match(line)
        if m:
            kw = m.group(1)
            if kw == "PROPS":
                in_props, current, rows = True, None, []
                continue
            if kw in _SECTIONS:
                if in_props:
                    finalize()
                in_props = False
                continue
            if in_props:
                finalize()
                # Report/control keywords are not tables: skip their
                # numeric records until the next real PVT/relperm keyword.
                current, rows = ((kw, []) if kw not in _SKIP_PROPS_KEYWORDS
                                 else (None, []))
                continue
        if in_props and current is not None:
            if "/" in line:
                line = line.split("/")[0]
            row = []
            for token in line.split():
                if "*" in token:
                    mult, _, val = token.partition("*")
                    try:
                        value = float(val)
                        n = min(int(mult), 1000) if mult.isdigit() else 1
                        row.extend([value] * n)
                    except ValueError:
                        pass
                else:
                    try:
                        row.append(float(token))
                    except ValueError:
                        pass
            if row:
                rows.append(row)
    if in_props:
        finalize()
    return tables


class PlotsPage(QWidget):
    """Timeseries / PVT / Summary sub-tabs (matplotlib).

    The Timeseries tab plots a grid attribute (per-cell field) over the
    report steps at a chosen cell -- or averaged over an axis -- exactly
    like the reference ``workbench/views/ts_render.py``, plus per-well
    well-solution curves when a well-level field is selected.
    """

    _WELL_FIELDS = frozenset({"bhp", "qOs", "qWs", "qGs", "status"})

    def __init__(self, data, slice_panel, parent=None):
        super().__init__(parent)
        self._data = data            # shared _WellCurvesPanel (well curves)
        self._slice = slice_panel    # _SlicePanel (per-cell field states)
        root = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.ts_tab = self._build_ts_tab()
        self.pvt_tab = self._build_pvt_tab()
        self.summary_tab = self._build_summary_tab()
        self.tabs.addTab(self.ts_tab, "Timeseries")
        self.tabs.addTab(self.pvt_tab, "PVT")
        self.tabs.addTab(self.summary_tab, "Summary")
        root.addWidget(self.tabs)

    def showEvent(self, event):
        super().showEvent(event)
        self._ts_redraw()

    # ------------------------------------------------------------- timeseries
    def _build_ts_tab(self):
        page = QWidget()
        v = QVBoxLayout(page)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Data:"))
        self.ts_attr = QComboBox()
        bar.addWidget(self.ts_attr)
        bar.addWidget(QLabel("Well:"))
        self.ts_well = QComboBox()
        self.ts_well.addItem("None")
        bar.addWidget(self.ts_well)
        for label, combo_name in (("I", "ts_i"), ("J", "ts_j"),
                                  ("K", "ts_k")):
            bar.addWidget(QLabel(label))
            combo = QComboBox()
            setattr(self, combo_name, combo)
            bar.addWidget(combo)
        self.ts_second = QCheckBox("2nd axis")
        self.ts_second.toggled.connect(lambda _on: self._ts_redraw())
        bar.addWidget(self.ts_second)
        self.ts_add = QPushButton("Add line")
        self.ts_clear = QPushButton("Clear")
        self.ts_add.clicked.connect(self._ts_add)
        self.ts_clear.clicked.connect(self._ts_clear)
        bar.addWidget(self.ts_add)
        bar.addWidget(self.ts_clear)
        bar.addStretch(1)
        v.addLayout(bar)
        self.ts_figure = _Figure(figsize=(9, 4.5), tight_layout=True)
        self.ts_canvas = _Canvas(self.ts_figure)
        v.addWidget(self.ts_canvas, 1)
        self._ts_requests = []
        return page

    def _sync_wells(self):
        names = list(self._data._all_wells)
        current = self.ts_well.currentText()
        self.ts_well.blockSignals(True)
        self.ts_well.clear()
        self.ts_well.addItem("None")
        for name in names:
            self.ts_well.addItem(name)
        if current in names:
            self.ts_well.setCurrentText(current)
        self.ts_well.blockSignals(False)

    def _sync_controls(self):
        """Refresh field / well / cell-index selectors from the data."""
        names = []
        if self._slice is not None and self._slice._states:
            names = [k for k in self._slice._states[0].keys()
                     if np.asarray(self._slice._states[0][k]).ndim == 1]
        current = self.ts_attr.currentText()
        self.ts_attr.blockSignals(True)
        self.ts_attr.clear()
        self.ts_attr.addItems(names + [f for f in self._WELL_FIELDS
                                       if f not in names])
        if current:
            self.ts_attr.setCurrentText(current)
        self.ts_attr.blockSignals(False)

        self._sync_wells()

        dims = [1, 1, 1]
        if self._slice is not None and self._slice._G is not None:
            dims = [int(x) for x in np.asarray(
                self._slice._G.get("cartDims", [1, 1, 1])).ravel()]
        for combo_name, n in (("ts_i", dims[0]), ("ts_j", dims[1]),
                              ("ts_k", dims[2])):
            combo = getattr(self, combo_name)
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("Average")
            for value in range(1, n + 1):
                combo.addItem(str(value))
            combo.blockSignals(False)

    def _ts_add(self):
        if not self.ts_attr.currentText():
            return
        self._ts_requests.append(
            (self.ts_attr.currentText(), self.ts_i.currentText(),
             self.ts_j.currentText(), self.ts_k.currentText(),
             self.ts_well.currentText()))
        self._ts_redraw()

    def _ts_clear(self):
        self._ts_requests = []
        self._ts_redraw()

    def _series_for(self, attr, i_sel, j_sel, k_sel, well):
        """(times, values, label) for one request, or None."""
        if attr in self._WELL_FIELDS:
            return self._well_series(attr, well)
        return self._grid_series(attr, i_sel, j_sel, k_sel)

    def _well_series(self, attr, well):
        if not self._data._wellsols:
            return None
        if well in ("None", "", "All wells"):
            well = None
        times = self._data._times
        if well is None:
            values = []
            for ws in self._data._wellsols:
                vals = [w.get(attr, float("nan")) for w in ws
                        if w.get(attr) is not None]
                values.append(float(np.nanmean(vals)) if vals
                              else float("nan"))
            label = "%s (all wells)" % attr
        else:
            values = []
            for ws in self._data._wellsols:
                v = next((w.get(attr, float("nan")) for w in ws
                          if w.get("name") == well), float("nan"))
                values.append(float(v))
            label = "%s/%s" % (well, attr)
        return times, values, label

    def _grid_series(self, attr, i_sel, j_sel, k_sel):
        """Value of a per-cell field at (i, j, k) over the report steps.

        Each selector is ``Average`` (mean over that axis) or a 1-based
        logical index, matching the reference ts_render semantics.
        """
        panel = self._slice
        if panel is None or not panel._states or panel._G is None:
            return None
        if attr not in panel._states[0]:
            return None
        G = panel._G
        dims = [int(x) for x in np.asarray(
            G.get("cartDims", [1, 1, 1])).ravel()]
        n_nat = int(np.prod(dims)) if len(dims) == 3 else 0
        index_map = (G.get("cells", {}).get("indexMap")
                     if isinstance(G, dict) else None)
        sel = [i_sel, j_sel, k_sel]
        avg_axes, idx = [], []
        for axis, s in enumerate(sel):
            if isinstance(s, str) and s.lower() in ("average", "avg", ":"):
                avg_axes.append(axis)
            else:
                try:
                    idx.append(int(s) - 1)
                except (TypeError, ValueError):
                    avg_axes.append(axis)
        times = panel._times
        values = []
        for st in panel._states:
            arr = np.asarray(st[attr], dtype=float).ravel()
            if index_map is not None and arr.size == index_map.size \
                    and n_nat:
                nat = np.full(n_nat, np.nan, dtype=float)
                nat[np.asarray(index_map, dtype=int)] = arr
                cube = nat.reshape(tuple(dims), order="F")
            elif arr.size == n_nat and n_nat:
                cube = arr.reshape(tuple(dims), order="F")
            else:
                cube = arr
            v = cube
            if avg_axes:
                with np.errstate(invalid="ignore"):
                    v = np.nanmean(v, axis=tuple(avg_axes))
            if idx and np.ndim(v) >= len(idx):
                v = v[tuple(idx)]
            values.append(float(np.nanmean(np.atleast_1d(v))))
        label = "%s (%s, %s, %s)" % (attr, i_sel, j_sel, k_sel)
        return times, values, label

    def _ts_redraw(self):
        self._sync_controls()
        self.ts_figure.clear()
        ax = self.ts_figure.add_subplot(111)
        ax2 = ax.twinx() if self.ts_second.isChecked() else None
        ax.grid(True, alpha=0.3)
        plotted = False
        for idx, req in enumerate(self._ts_requests):
            try:
                res = self._series_for(*req)
            except Exception:  # noqa: BLE001 - skip bad requests
                res = None
            if res is None:
                continue
            t, values, name = res
            target = (ax2 if (self.ts_second.isChecked() and idx % 2 == 1)
                      else ax)
            target.plot(np.asarray(t, dtype=float),
                        np.asarray(values, dtype=float), label=name, lw=1.5)
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "select a series to plot", ha="center",
                    va="center", transform=ax.transAxes)
        ax.set_xlabel("time, days")
        if ax2 is not None:
            ax2.set_ylabel("secondary")
        if plotted:
            ax.legend(fontsize=8)
        self.ts_figure.tight_layout()
        self.ts_canvas.draw()

    # ------------------------------------------------------------------ pvt
    _PVT_COLUMNS = {
        "PVTW": ["Pref", "Bw", "Cw", "muw", "dmuw/dp"],
        "PVDO": ["P", "Bo", "muo"],
        "PVTO": ["P", "Bo", "muo"],
        "PVCDO": ["Pref", "Bo", "Co", "muo", "dmuo/dp"],
        "PVDG": ["P", "Bg", "mug"],
        "PVTG": ["P", "Bg", "mug"],
        "ROCK": ["Pref", "Cr"],
        "DENSITY": ["rho_o", "rho_w", "rho_g"],
        "SWOF": ["Sw", "Krw", "Krow", "Pcow"],
        "SGOF": ["Sg", "Krg", "Krog", "Pcog"],
        "SWFN": ["Sw", "Krw", "Pcow"],
        "SGFN": ["Sg", "Krg", "Pcog"],
        "SOF2": ["So", "Kro"],
        "SOF3": ["So", "Krow", "Krog"],
        "SGWFN": ["Sg", "Krg", "Pcog"],
        "SWOF": ["Sw", "Krw", "Krow", "Pcow"],
    }

    def _build_pvt_tab(self):
        page = QWidget()
        v = QVBoxLayout(page)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Table:"))
        self.pvt_table = QComboBox()
        bar.addWidget(self.pvt_table)
        bar.addWidget(QLabel("X axis:"))
        self.pvt_x = QComboBox()
        bar.addWidget(self.pvt_x)
        self.pvt_plot = QPushButton("Plot")
        self.pvt_plot.clicked.connect(self._pvt_redraw)
        bar.addWidget(self.pvt_plot)
        bar.addStretch(1)
        v.addLayout(bar)
        self.pvt_figure = _Figure(figsize=(8, 4.5), tight_layout=True)
        self.pvt_canvas = _Canvas(self.pvt_figure)
        v.addWidget(self.pvt_canvas, 1)
        self._pvt_tables = {}
        return page

    def set_pvt_tables(self, deck_path):
        self._pvt_tables = _props_tables(deck_path) if deck_path else {}
        names = sorted(self._pvt_tables)
        current = self.pvt_table.currentText()
        self.pvt_table.blockSignals(True)
        self.pvt_table.clear()
        self.pvt_table.addItems(names)
        if current in names:
            self.pvt_table.setCurrentText(current)
        self.pvt_table.blockSignals(False)
        self._pvt_sync_x()

    def _pvt_sync_x(self):
        name = self.pvt_table.currentText()
        arr = self._pvt_tables.get(name)
        ncols = arr.shape[1] if arr is not None and arr.ndim == 2 else 0
        self.pvt_x.blockSignals(True)
        self.pvt_x.clear()
        self.pvt_x.addItems([str(i) for i in range(ncols)])
        self.pvt_x.blockSignals(False)

    def _pvt_redraw(self):
        name = self.pvt_table.currentText()
        arr = self._pvt_tables.get(name)
        self.pvt_figure.clear()
        ax = self.pvt_figure.add_subplot(111)
        if arr is None or arr.ndim != 2 or arr.shape[1] < 2 or \
                arr.shape[0] < 1:
            ax.text(0.5, 0.5, "No PROPS table available", ha="center",
                    va="center", transform=ax.transAxes)
            self.pvt_canvas.draw()
            return
        try:
            xcol = int(self.pvt_x.currentText() or 0)
        except ValueError:
            xcol = 0
        xcol = max(0, min(xcol, arr.shape[1] - 1))
        cols = self._PVT_COLUMNS.get(name, [])
        x = arr[:, xcol]
        for j in range(arr.shape[1]):
            if j == xcol:
                continue
            y = arr[:, j]
            label = cols[j] if j < len(cols) else "col %d" % j
            # single-row / few-point tables need markers to stay visible
            marker = "o" if x.size <= 2 else None
            ax.plot(x, y, lw=1.2, marker=marker, markersize=3,
                    label=label)
        ax.set_xlabel(cols[xcol] if xcol < len(cols) else "col %d" % xcol)
        ax.set_title("%s — PROPS table" % name)
        ax.grid(True, alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc="best")
        self.pvt_figure.tight_layout()
        self.pvt_canvas.draw()

    # --------------------------------------------------------------- summary
    def _build_summary_tab(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(2, 2, 2, 2)
        v.setSpacing(3)

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Cases:"))
        self.sm_cases_combo = QComboBox()
        self.sm_cases_combo.setMinimumWidth(220)
        r1.addWidget(self.sm_cases_combo)
        self.sm_add = QPushButton("+ Add result dir...")
        self.sm_remove = QPushButton("- Remove")
        self.sm_add.clicked.connect(self._sm_add)
        self.sm_remove.clicked.connect(self._sm_remove)
        r1.addWidget(self.sm_add)
        r1.addWidget(self.sm_remove)
        r1.addStretch(1)
        v.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Search:"))
        self.sm_search = _ClickableLineEdit()
        self.sm_search.setPlaceholderText("Search vectors (click to open list)")
        self.sm_search.textChanged.connect(self._sm_filter_vectors)
        self.sm_search.clicked.connect(self._sm_toggle_popup)
        sm_expand = QAction("▾", self)
        sm_expand.triggered.connect(self._sm_toggle_popup)
        self.sm_search.addAction(sm_expand, QLineEdit.TrailingPosition)
        r2.addWidget(self.sm_search, 1)
        r2.addWidget(QLabel("Type:"))
        self.sm_type = QComboBox()
        self.sm_type.addItems(["All", "Field", "Well"])
        self.sm_type.currentIndexChanged.connect(self._sm_filter_vectors)
        r2.addWidget(self.sm_type)
        self.sm_plot = QPushButton("Plot")
        self.sm_plot.clicked.connect(self._sm_plot)
        self.sm_plot.clicked.connect(self._sm_popup_hide)
        r2.addWidget(self.sm_plot)
        self.sm_clear_sel = QPushButton("Clear sel.")
        self.sm_clear_sel.clicked.connect(
            lambda: self.sm_vectors.clearSelection())
        r2.addWidget(self.sm_clear_sel)
        self.sm_count = QLabel("")
        r2.addWidget(self.sm_count)
        r2.addStretch(1)
        v.addLayout(r2)

        self.sm_figure = _Figure(figsize=(9, 5), tight_layout=True)
        self.sm_canvas = _Canvas(self.sm_figure)
        v.addWidget(self.sm_canvas, 1)

        self._sm_page = page
        self._sm_popup = QFrame(page)
        self._sm_popup.setObjectName("smPopup")
        self._sm_popup.setStyleSheet(
            "QFrame#smPopup{background:palette(base);"
            "border:1px solid palette(mid);border-radius:4px;}")
        pv = QVBoxLayout(self._sm_popup)
        pv.setContentsMargins(4, 4, 4, 4)
        pv.setSpacing(2)
        self.sm_vectors = QListWidget()
        self.sm_vectors.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.sm_vectors.setMaximumHeight(220)
        self.sm_vectors.itemSelectionChanged.connect(self._sm_update_count)
        pv.addWidget(self.sm_vectors)
        self._sm_popup.hide()

        self._sm_cases = []
        self._all_vectors = []
        QApplication.instance().installEventFilter(self)
        return page

    def _sm_popup_hide(self):
        self._sm_popup.hide()

    def _sm_toggle_popup(self):
        if self._sm_popup.isVisible():
            self._sm_popup.hide()
            return
        top = self.sm_search.mapTo(self._sm_page, QPoint(0, 0)).y()
        w = max(380, int(self._sm_page.width() * 0.7))
        w = min(w, self._sm_page.width() - 8)
        h = min(240, int(self._sm_page.height() * 0.55))
        self._sm_popup.setFixedSize(w, h)
        self._sm_popup.move(4, top + self.sm_search.height() + 2)
        self._sm_popup.show()
        self._sm_popup.raise_()
        self.sm_search.setFocus()

    def eventFilter(self, obj, event):
        if self._sm_popup.isVisible():
            if (event.type() == QEvent.KeyPress
                    and event.key() == Qt.Key_Escape):
                self._sm_popup.hide()
                return True
            if event.type() == QEvent.MouseButtonPress:
                page_pos = self._sm_page.mapFromGlobal(
                    event.globalPosition().toPoint())
                inside = (self._sm_popup.geometry().contains(page_pos)
                          or self.sm_search.geometry().contains(page_pos))
                if not inside:
                    self._sm_popup.hide()
        return super().eventFilter(obj, event)

    def _sm_add(self):
        dirs = QFileDialog.getExistingDirectory(self, "Select result directory")
        if not dirs:
            return
        self._add_summary_case(dirs)

    def _add_summary_case(self, outdir: str):
        for item in self._sm_cases:
            if os.path.normpath(item.outdir) == os.path.normpath(outdir):
                return
        case = _SummaryCase(outdir)
        self._sm_cases.append(case)
        self.sm_cases_combo.addItem(case.name)
        self._sm_reload_vectors()

    def _sm_remove(self):
        idx = self.sm_cases_combo.currentIndex()
        if 0 <= idx < len(self._sm_cases):
            self._sm_cases.pop(idx)
            self.sm_cases_combo.removeItem(idx)
        self._sm_reload_vectors()

    def _sm_reload_vectors(self):
        keys = []
        for case in self._sm_cases:
            for key in case.vector_keys():
                if key not in keys:
                    keys.append(key)
        self._all_vectors = keys
        self._sm_filter_vectors()

    def _sm_filter_vectors(self, *_):
        keys = self._all_vectors
        text = self.sm_search.text().strip().lower()
        kind = self.sm_type.currentText()
        self.sm_vectors.blockSignals(True)
        self.sm_vectors.clear()
        for key in keys:
            if kind != "All" and not key.startswith(kind.upper() + ":"):
                continue
            if text and text not in key.lower():
                continue
            item = QListWidgetItem("%s  (%s)" % (key,
                                                 _SummaryCase.friendly(key)))
            item.setData(256, key)
            self.sm_vectors.addItem(item)
        self.sm_vectors.blockSignals(False)
        self._sm_update_count()

    def _sm_update_count(self):
        n = len(self.sm_vectors.selectedItems())
        total = self.sm_vectors.count()
        self.sm_count.setText("%d/%d" % (n, total) if total else "")

    def _sm_selected_keys(self):
        return [item.data(256) for item in self.sm_vectors.selectedItems()]

    def _sm_plot(self):
        if not self._sm_cases:
            return
        keys = self._sm_selected_keys()
        if not keys:
            keys = [k for k in self._sm_cases[0].vector_keys()
                    if k.startswith("FIELD:")
                    and k.endswith((":WOPR", ":WBHP"))][:4]
            if not keys:
                keys = [k for k in self._sm_cases[0].vector_keys()
                        if k.startswith("FIELD:")][:4]
        _draw_summary_into(self.sm_figure, self._sm_cases, keys)
        self.sm_canvas.draw()


# ===========================================================================
# optimisation page (PRSTCore.optimization -- conntrans multipliers)
# ===========================================================================
def _rate_table(states):
    """Per-report-step per-well production rows (sm3/day) from states."""
    rows = []
    start = datetime(1999, 9, 1)
    for t, state in enumerate(states):
        sol = state.get("wellSol") or []
        for w in sol:
            name = w.get("name", "?")
            date = start + timedelta(days=float(state.get("time", 0.0))
                                     / 86400.0)
            sign = float(w.get("sign", -1.0))
            qos = float(np.atleast_1d(np.asarray(w.get("qOs", 0.0),
                                                 dtype=float))[0])
            qws = float(np.atleast_1d(np.asarray(w.get("qWs", 0.0),
                                                 dtype=float))[0])
            qos_sm3d = qos * 86400.0
            qws_sm3d = qws * 86400.0
            producing = sign < 0 or (qos_sm3d + qws_sm3d) < 0
            rows.append({
                "well": name, "period": t,
                "end_date": date.strftime("%Y-%m-%d"),
                "oil": abs(qos_sm3d) if producing else 0.0,
                "water_prod": abs(qws_sm3d) if producing else 0.0,
                "water_inj": abs(qws_sm3d) if not producing else 0.0,
            })
    return rows


class _OptimWorker(QtCore.QThread):
    """NPV optimisation: one conntrans multiplier per well, tuned with
    ``unit_box_bfgs``; gradients by finite differences (PRSTCore)."""

    log_line = _Signal(str)
    finished_ok = _Signal(str, dict)     # outdir, summary
    failed = _Signal(str)

    def __init__(self, deck, params, handle=None, parent=None):
        super().__init__(parent)
        self.deck = deck
        self.params = dict(params)
        self.handle = handle

    def run(self):
        outdir = self.params["outdir"]
        os.makedirs(outdir, exist_ok=True)

        def log(msg):
            self.log_line.emit(msg)

        try:
            from PRSTCore.optimization import (
                update_setup_from_scaled_parameters, npv_ow,
                _finite_difference_gradient, unit_box_bfgs)
            from PRSTCore.optimization.utils.parameters import add_parameter
            from PRSTCore.ad_core.simulators.simulate_schedule_ad import \
                simulate_schedule_ad

            if self.handle is not None:
                model, state0, schedule, solver = self.handle
            else:
                log("loading deck %s" % self.deck)
                from PRSTCore.ad_core.initialization.init_eclipse_problem_ad \
                    import init_eclipse_problem_ad
                state0, model, schedule, solver = init_eclipse_problem_ad(
                    self.deck, RemoveZeroPoreVolume=True)
            setup = {"state0": state0, "model": model, "schedule": schedule}

            wells = []
            for control in schedule.get("control", []):
                for w in control.get("W", []):
                    if w.get("name") and w["name"] not in wells:
                        wells.append(w["name"])
            n_wells = len(wells)
            if n_wells == 0:
                raise RuntimeError("No wells in deck")
            # One oil-rate target multiplier per well (the same parameter the
            # ResSimWorkbench prst_optimize_worker tunes by default).
            params = add_parameter(
                None, setup, name="orat",
                lumping=np.arange(n_wells),
                relative_limits=[float(self.params["rel_lo"]),
                                 float(self.params["rel_hi"])],
                uniform_limits=False)
            log("%d parameters (oil-rate multiplier per well)" % n_wells)

            discount = float(self.params["discount_rate"]) / 100.0
            oil_price = float(self.params["oil_price"])
            water_cost = float(self.params["water_cost"])

            def npv_of(setup_new):
                _ws, states = simulate_schedule_ad(
                    setup_new["state0"], setup_new["model"],
                    setup_new["schedule"], NonLinearSolver=None,
                    Verbose=False)
                vals = npv_ow(setup_new["model"], states,
                              setup_new["schedule"], oil_price=oil_price,
                              water_production_cost=water_cost,
                              water_injection_cost=water_cost,
                              discount_factor=discount)
                return -float(np.sum(vals)), states

            def objh(u):
                setup_new = update_setup_from_scaled_parameters(
                    setup, params, u)
                v, _ = npv_of(setup_new)

                def scalar(us):
                    sn = update_setup_from_scaled_parameters(setup, params,
                                                             us)
                    vv, _ = npv_of(sn)
                    return vv

                g = _finite_difference_gradient(
                    scalar, np.asarray(u, dtype=float))
                return v, g

            u0 = np.full(n_wells, 0.5)
            log("baseline evaluation (u=0.5)")
            v_base, states_base = npv_of(
                update_setup_from_scaled_parameters(setup, params, u0))
            log("base NPV = %.3f" % v_base)

            log("optimising with unit_box_bfgs (max %d iterations)"
                % int(self.params["max_it"]))
            t0 = time.perf_counter()
            v_opt, u_opt, history = unit_box_bfgs(
                u0, objh, maximize=True, max_it=int(self.params["max_it"]),
                grad_tol=1e-4)
            wall = time.perf_counter() - t0

            setup_opt = update_setup_from_scaled_parameters(setup, params,
                                                            u_opt)
            _ws, states_opt = simulate_schedule_ad(
                setup_opt["state0"], setup_opt["model"],
                setup_opt["schedule"], NonLinearSolver=None, Verbose=False)

            log("opt NPV = %.3f (improvement %.1f%%)"
                % (v_opt, 100.0 * (v_opt - v_base) / max(abs(v_base), 1e-12)))

            # ---- production.csv (Jutul-compatible) --------------------
            rows_base = _rate_table(states_base)
            rows_opt = _rate_table(states_opt)
            opt_by = {(r["well"], r["period"]): r for r in rows_opt}
            import csv
            csv_path = os.path.join(outdir, "production.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["well", "period", "end_date",
                                 "base_oil_rate_m3_day", "opt_oil_rate_m3_day",
                                 "base_water_rate_m3_day",
                                 "opt_water_rate_m3_day",
                                 "base_water_inj_m3_day",
                                 "opt_water_inj_m3_day"])
                for r in rows_base:
                    o = opt_by.get((r["well"], r["period"]), r)
                    writer.writerow([r["well"], r["period"], r["end_date"],
                                     r["oil"], o["oil"],
                                     r["water_prod"], o["water_prod"],
                                     r["water_inj"], o["water_inj"]])

            summary = {
                "simulator": "prst",
                "base_npv": float(v_base),
                "opt_npv": float(v_opt),
                "improvement_pct": float(100.0 * (v_opt - v_base)
                                         / max(abs(v_base), 1e-12)),
                "converged": bool(len(history) < int(self.params["max_it"])),
                "n_variables": int(n_wells),
                "iterations": int(len(history)),
                "max_it": int(self.params["max_it"]),
                "wall_time_s": float(wall),
                "param": "orat",
            }
            with open(os.path.join(outdir, "summary.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(summary, fh, ensure_ascii=False, indent=2)
            log("wrote summary.json and production.csv to %s" % outdir)
            self.finished_ok.emit(outdir, summary)

        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            try:
                with open(os.path.join(outdir, "summary.json"), "w",
                          encoding="utf-8") as fh:
                    json.dump({"simulator": "prst",
                               "status": "%s: %s" % (type(exc).__name__,
                                                     exc)},
                              fh, ensure_ascii=False, indent=2)
            except Exception:  # noqa: BLE001
                pass
            self.failed.emit("%s: %s" % (type(exc).__name__, exc))


class _JutulOptimWorker(QtCore.QThread):
    """NPV optimisation through the JutulDarcy Julia driver."""

    log_line = _Signal(str)
    finished_ok = _Signal(str, dict)     # outdir, summary
    failed = _Signal(str)

    def __init__(self, deck, params, parent=None):
        super().__init__(parent)
        self.deck = deck
        self.params = dict(params)

    def run(self):
        outdir = self.params["outdir"]
        os.makedirs(outdir, exist_ok=True)
        from PRSTCore.jutul.driver import JULIA_HINT, run_optimize
        p = {
            "months": int(self.params["months"]),
            "oil-price": float(self.params["oil_price"]),
            "gas-price": float(self.params["gas_price"]),
            "water-price": float(self.params["water_cost"]),
            "water-cost": float(self.params["water_cost"]),
            "gas-cost": 0.0,
            "discount-rate": float(self.params["discount_rate"]),
            "max-it": int(self.params["max_it"]),
            "bhp-prod-min": float(self.params["bhp_prod_min"]),
            "bhp-prod-max": float(self.params["bhp_prod_max"]),
            "bhp-inj-min": float(self.params["bhp_inj_min"]),
            "bhp-inj-max": float(self.params["bhp_inj_max"]),
        }
        self.log_line.emit("running JutulDarcy optimisation on %s"
                           % self.deck)
        try:
            out = run_optimize(self.deck, out_dir=outdir, params=p,
                               on_line=self.log_line.emit)
        except Exception as exc:  # noqa: BLE001 - report, don't die
            hint = ""
            text = str(exc).lower()
            if isinstance(exc, FileNotFoundError) or "not found" in text \
                    or "cannot find" in text or "julia" in text:
                hint = "\n  (%s)" % JULIA_HINT
            self.failed.emit("JutulDarcy optimisation failed: %s: %s%s"
                             % (type(exc).__name__, exc, hint))
            return
        summary_path = os.path.join(str(out), "summary.json")
        if not os.path.exists(summary_path):
            self.failed.emit("JutulDarcy optimisation produced no "
                             "summary.json in %s" % out)
            return
        with open(summary_path, encoding="utf-8") as fh:
            summary = json.load(fh)
        if summary.get("status"):
            self.failed.emit(summary["status"])
            return
        self.log_line.emit("wrote JutulDarcy optimisation results to %s"
                           % out)
        self.finished_ok.emit(str(out), summary)


class OptimPage(QWidget):
    def __init__(self, log_panel, parent=None):
        super().__init__(parent)
        self.log_panel = log_panel
        root = QVBoxLayout(self)

        form = QGroupBox("NPV optimisation")
        grid = QHBoxLayout(form)
        g1 = QVBoxLayout()
        g1.addWidget(QLabel("Engine"))
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("PRST (oil-rate multipliers)", "prst")
        self.engine_combo.addItem("JutulDarcy (forecast BHP)", "jutul")
        g1.addWidget(self.engine_combo)
        g1.addWidget(QLabel("Oil price ($/sm3)"))
        self.oil_price = self._spin(0.0, 1e6, 60.0)
        g1.addWidget(self.oil_price)
        g1.addWidget(QLabel("Water cost ($/sm3)"))
        self.water_cost = self._spin(0.0, 1e6, 5.0)
        g1.addWidget(self.water_cost)
        g1.addWidget(QLabel("Discount rate (%/yr)"))
        self.discount = self._spin(0.0, 100.0, 0.0)
        g1.addWidget(self.discount)
        grid.addLayout(g1)

        g2 = QVBoxLayout()
        g2.addWidget(QLabel("Gas price ($/sm3)"))
        self.gas_price = self._spin(0.0, 1e6, 0.0)
        g2.addWidget(self.gas_price)
        g2.addWidget(QLabel("Horizon months (Jutul)"))
        self.months = QSpinBox()
        self.months.setRange(1, 1200)
        self.months.setValue(60)
        g2.addWidget(self.months)
        g2.addWidget(QLabel("Max iterations"))
        self.max_it = QSpinBox()
        self.max_it.setRange(1, 50)
        self.max_it.setValue(5)
        g2.addWidget(self.max_it)
        self.run_btn = QPushButton("Optimize")
        self.run_btn.clicked.connect(self._run)
        g2.addWidget(self.run_btn)
        grid.addLayout(g2)

        g3 = QVBoxLayout()
        g3.addWidget(QLabel("Producer BHP min/max (Jutul, bar)"))
        bh = QHBoxLayout()
        self.bhp_prod_min = self._spin(1.0, 1e5, 200.0)
        self.bhp_prod_max = self._spin(1.0, 1e5, 350.0)
        bh.addWidget(self.bhp_prod_min)
        bh.addWidget(self.bhp_prod_max)
        g3.addLayout(bh)
        g3.addWidget(QLabel("Injector BHP min/max (Jutul, bar)"))
        bh = QHBoxLayout()
        self.bhp_inj_min = self._spin(1.0, 1e5, 300.0)
        self.bhp_inj_max = self._spin(1.0, 1e5, 450.0)
        bh.addWidget(self.bhp_inj_min)
        bh.addWidget(self.bhp_inj_max)
        g3.addLayout(bh)
        g3.addWidget(QLabel("PRST rate-multiplier bounds (rel.)"))
        bh = QHBoxLayout()
        self.rel_lo = self._spin(0.0, 10.0, 0.5)
        self.rel_hi = self._spin(0.0, 10.0, 1.5)
        bh.addWidget(self.rel_lo)
        bh.addWidget(self.rel_hi)
        g3.addLayout(bh)
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(90)
        g3.addWidget(self.summary)
        grid.addLayout(g3)
        root.addWidget(form)

        self.figure = _Figure(figsize=(9, 4.5), tight_layout=True)
        self.canvas = _Canvas(self.figure)
        root.addWidget(self.canvas, 1)

        self._thread = None

    @staticmethod
    def _spin(lo, hi, val):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(4)
        s.setValue(val)
        return s

    def _params(self):
        return {
            "oil_price": self.oil_price.value(),
            "gas_price": self.gas_price.value(),
            "water_cost": self.water_cost.value(),
            "discount_rate": self.discount.value(),
            "months": self.months.value(),
            "max_it": self.max_it.value(),
            "rel_lo": self.rel_lo.value(),
            "rel_hi": self.rel_hi.value(),
            "bhp_prod_min": self.bhp_prod_min.value(),
            "bhp_prod_max": self.bhp_prod_max.value(),
            "bhp_inj_min": self.bhp_inj_min.value(),
            "bhp_inj_max": self.bhp_inj_max.value(),
        }

    def _run(self):
        win = self.window()
        deck = win._loaded_deck_path if win is not None else ""
        if not deck:
            QMessageBox.information(self, "Workbench",
                                    "Load a model first.")
            return
        if self._thread is not None and self._thread.isRunning():
            return
        engine = self.engine_combo.currentData()
        if engine == "prst" and (win is None or win.model is None):
            QMessageBox.information(self, "Workbench",
                                    "Load a PRST model first.")
            return
        base = os.path.splitext(os.path.basename(deck))[0]
        suffix = "_optim_jutul" if engine == "jutul" else "_optim_prst"
        outdir = os.path.join(os.path.dirname(os.path.abspath(deck)),
                              "%s%s" % (base, suffix))
        params = self._params()
        params["outdir"] = outdir
        self.run_btn.setEnabled(False)
        self.log_panel.append("Optimisation started (%s)" % engine)
        if engine == "jutul":
            self._thread = _JutulOptimWorker(deck, params)
        else:
            self._thread = _OptimWorker(deck, params,
                                        handle=win._loaded_handle)
        self._thread.log_line.connect(self.log_panel.append)
        self._thread.finished_ok.connect(self._on_done)
        self._thread.failed.connect(self._on_failed)
        self._thread.finished.connect(
            lambda: self.run_btn.setEnabled(True))
        self._thread.start()

    def _on_done(self, outdir, summary):
        self.log_panel.append("Optimisation done -> %s" % outdir)
        self.summary.setPlainText(
            "base NPV %.3f  |  opt NPV %.3f  |  improvement %.1f%%\n"
            "variables %d, iterations %d, converged %s, wall %.1f s\n%s"
            % (summary.get("base_npv", 0), summary.get("opt_npv", 0),
               summary.get("improvement_pct", 0),
               summary.get("n_variables", 0), summary.get("iterations", 0),
               summary.get("converged", False),
               summary.get("wall_time_s", 0), outdir))
        self._plot(outdir)

    def _on_failed(self, err):
        self.log_panel.append("Optimisation failed: %s" % err)
        self.summary.setPlainText("Failed: %s" % err)

    def _plot(self, outdir):
        import pandas as pd
        csv_path = os.path.join(outdir, "production.csv")
        if not os.path.exists(csv_path):
            return
        df = pd.read_csv(csv_path)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if "well" in df:
            df = df.groupby("period", as_index=False).sum()
        x = df["period"]
        for col, label, ls in (
                ("base_oil_rate_m3_day", "Oil (base)", "-"),
                ("opt_oil_rate_m3_day", "Oil (opt)", "--"),
                ("base_water_rate_m3_day", "Water prod (base)", ":"),
                ("opt_water_rate_m3_day", "Water prod (opt)", "-."),
        ):
            if col in df.columns:
                ax.plot(x, df[col], label=label, ls=ls, lw=1.5)
        ax.set_xlabel("period")
        ax.set_ylabel("rate, sm3/d")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        self.figure.tight_layout()
        self.canvas.draw()


# ===========================================================================
# compare page (two HDF5 result dirs)
# ===========================================================================
class _CompareResult:
    def __init__(self, dir_a: str, dir_b: str, abs_tol: float, rel_tol: float):
        self.dir_a, self.dir_b = dir_a, dir_b
        self.abs_tol, self.rel_tol = float(abs_tol), float(rel_tol)
        self.jr_a = h5_results.load(dir_a)
        self.jr_b = h5_results.load(dir_b)
        self.fields = []
        for name, attr in (("pressure", "pressure"),
                           ("SWAT", "swat"), ("SOIL", "soil"),
                           ("SGAS", "sgas")):
            if getattr(self.jr_a, attr) is not None and \
                    getattr(self.jr_b, attr) is not None:
                self.fields.append(name)
        self._a_nat = h5_results.to_field_dict(self.jr_a)
        self._b_nat = h5_results.to_field_dict(self.jr_b)
        self._dims = self.jr_a.grid_dims or [1, 1, 1]

    def _arrays(self, field):
        a = self._a_nat["pressure"] if field == "pressure" else \
            self._a_nat["saturations"].get(field)
        b = self._b_nat["pressure"] if field == "pressure" else \
            self._b_nat["saturations"].get(field)
        if a is None or b is None:
            return None, None
        n = min(a.shape[0], b.shape[0])
        return a[:n], b[:n]

    def draw_overview_into(self, figure, field="pressure"):
        figure.clear()
        ax = figure.add_subplot(111)
        a, b = self._arrays(field)
        if a is None:
            return
        nx, ny, nz = self._dims
        diff = a[-1] - b[-1]
        grid = diff.reshape((nx, ny, nz), order="F")
        layer = nz // 2
        im = ax.imshow(grid[:, :, layer].T, origin="lower",
                       aspect="auto", cmap="RdBu_r")
        ax.set_title("%s difference (A-B), step %d, layer %d"
                     % (field, a.shape[0] - 1, layer + 1))
        ax.set_xlabel("i")
        ax.set_ylabel("j")
        figure.colorbar(im, ax=ax, shrink=0.8)
        figure.tight_layout()

    def draw_history_into(self, figure, field, cell):
        figure.clear()
        ax = figure.add_subplot(111)
        a, b = self._arrays(field)
        if a is None:
            return
        cell = max(0, min(int(cell), a.shape[1] - 1))
        x = np.arange(a.shape[0])
        ax.plot(x, a[:, cell], label="A", lw=1.5)
        ax.plot(x, b[:, cell], label="B", lw=1.5, ls="--")
        ax.set_xlabel("step")
        ax.set_ylabel(field)
        ax.set_title("%s at natural cell %d" % (field, cell))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        figure.tight_layout()

    def summary_text(self) -> str:
        a, b = self._arrays("pressure")
        lines = []
        if a is not None:
            diff = np.abs(a[-1] - b[-1])
            denom = np.maximum(np.abs(b[-1]), 1e-12)
            rel = diff / denom
            n_bad = int(np.sum((diff > self.abs_tol) &
                               (rel > self.rel_tol)))
            lines.append(
                "pressure @ last step: max|A-B| = %.4g, cells beyond tol = "
                "%d / %d" % (float(np.nanmax(diff)) if diff.size else 0.0,
                             n_bad, diff.size))
        lines.append("A: %s" % os.path.basename(self.dir_a))
        lines.append("B: %s" % os.path.basename(self.dir_b))
        return "\n".join(lines)


class ComparePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)

        picker = QHBoxLayout()
        picker.addWidget(QLabel("Run A:"))
        self.dir_a = QLineEdit()
        picker.addWidget(self.dir_a, 1)
        b_a = QPushButton("Browse...")
        b_a.clicked.connect(lambda: self._browse("dir_a"))
        picker.addWidget(b_a)
        picker.addWidget(QLabel("Run B:"))
        self.dir_b = QLineEdit()
        picker.addWidget(self.dir_b, 1)
        b_b = QPushButton("Browse...")
        b_b.clicked.connect(lambda: self._browse("dir_b"))
        picker.addWidget(b_b)
        picker.addWidget(QLabel("abs tol:"))
        self.abs_tol = QDoubleSpinBox()
        self.abs_tol.setRange(0.0, 1e9)
        self.abs_tol.setDecimals(6)
        self.abs_tol.setValue(0.05)
        picker.addWidget(self.abs_tol)
        picker.addWidget(QLabel("rel tol:"))
        self.rel_tol = QDoubleSpinBox()
        self.rel_tol.setRange(0.0, 1.0)
        self.rel_tol.setDecimals(6)
        self.rel_tol.setValue(0.05)
        picker.addWidget(self.rel_tol)
        b_go = QPushButton("Compare")
        b_go.clicked.connect(self._compare)
        picker.addWidget(b_go)
        root.addLayout(picker)

        split = QSplitter()
        right = QWidget()
        rv = QVBoxLayout(right)
        rbar = QHBoxLayout()
        rbar.addWidget(QLabel("Field:"))
        self.field_combo = QComboBox()
        self.field_combo.currentIndexChanged.connect(self._overview)
        rbar.addWidget(self.field_combo)
        rbar.addWidget(QLabel("Cell:"))
        self.cell_spin = QSpinBox()
        self.cell_spin.setRange(0, 1000000)
        rbar.addWidget(self.cell_spin)
        b_hist = QPushButton("Cell history")
        b_hist.clicked.connect(self._history)
        rbar.addWidget(b_hist)
        rbar.addStretch(1)
        rv.addLayout(rbar)
        self.history_figure = _Figure(figsize=(9, 4), tight_layout=True)
        self.history_canvas = _Canvas(self.history_figure)
        rv.addWidget(self.history_canvas, 1)
        split.addWidget(right)

        self.overview_figure = _Figure(figsize=(10, 6), tight_layout=True)
        self.overview_canvas = _Canvas(self.overview_figure)
        split.addWidget(self.overview_canvas)
        split.setSizes([500, 600])
        root.addWidget(split, 1)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(90)
        root.addWidget(self.summary_text)

        self._result = None

    def _browse(self, which):
        path = QFileDialog.getExistingDirectory(self, "Select result directory")
        if path:
            getattr(self, which).setText(path)

    def set_pair(self, dir_a: str, dir_b: str):
        self.dir_a.setText(dir_a)
        self.dir_b.setText(dir_b)
        self._compare()

    def _compare(self):
        if not self.dir_a.text() or not self.dir_b.text():
            return
        try:
            self._result = _CompareResult(self.dir_a.text(),
                                          self.dir_b.text(),
                                          self.abs_tol.value(),
                                          self.rel_tol.value())
        except Exception as err:
            self.summary_text.setPlainText("Compare failed: %s" % err)
            return
        self.field_combo.blockSignals(True)
        self.field_combo.clear()
        self.field_combo.addItems(self._result.fields)
        self.field_combo.blockSignals(False)
        self.summary_text.setPlainText(self._result.summary_text())
        self._overview()

    def _overview(self):
        if self._result is None or not self.field_combo.currentText():
            return
        self._result.draw_overview_into(self.overview_figure,
                                        self.field_combo.currentText())
        self.overview_canvas.draw()

    def _history(self):
        if self._result is None or not self.field_combo.currentText():
            return
        field = self.field_combo.currentText()
        cell = self.cell_spin.value()
        self._result.draw_history_into(self.history_figure, field, cell)
        self.history_canvas.draw()


# ===========================================================================
# main window
# ===========================================================================
class SimulatorWindow(QMainWindow):
    """The ResSimWorkbench-style main window (menu, log dock, 7 tabs)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ResFine — Reservoir Simulation Office")
        _icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "assets", "app_icon.png")
        if os.path.isfile(_icon_path):
            self.setWindowIcon(QIcon(_icon_path))
        self._fit_to_screen()

        # ---- core -------------------------------------------------------
        self.jobs = _JobQueue(self)
        self.model = self.state0 = self.schedule = self.solver = None
        self._loaded_deck_path = ""
        self._loaded_handle = None
        self._deck_worker = None
        self._sim_worker = None
        self._loading_deck = ""

        # ---- log dock ---------------------------------------------------
        self.log_panel = _LogPanel(self)
        log_dock = QDockWidget("Log", self)
        log_dock.setObjectName("LogDock")
        log_dock.setWidget(self.log_panel)
        self.addDockWidget(Qt.BottomDockWidgetArea, log_dock)

        # ---- DeepSeek agent dock (right side, hidden until launched) ----
        self.agent_dock = AgentDock(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.agent_dock)
        self.agent_dock.hide()

        # ---- pages ------------------------------------------------------
        self.run_page = RunPage(self.jobs, self)
        self.deck_page = DeckEditorPage(self)
        self.well_page = WellPage(self)
        self.slice_page = _SlicePanel(self)
        self.plots_page = PlotsPage(self._make_wells_panel(), self.slice_page,
                                    self)
        self.optim_page = OptimPage(self.log_panel, self)
        self.compare_page = ComparePage(self)
        self._last_h5 = None
        self.well_page.set_source(self._well_entries)

        self.run_page._open_deck_in_editor = self._open_deck_in_editor

        self.tabs = QTabWidget()
        self.tabs.addTab(self.run_page, "Run")
        self.tabs.addTab(self.deck_page, "Deck Editor")
        self.tabs.addTab(self.well_page, "Well Hierarchy")
        self.tabs.addTab(self.slice_page, "2D Slice")
        self.tabs.addTab(self.plots_page, "Plots")
        self.tabs.addTab(self.optim_page, "Optimization")
        self.tabs.addTab(self.compare_page, "Compare")
        self.setCentralWidget(self.tabs)

        # ---- compatibility aliases (kept for the existing tests/scripts)
        self._slice_panel = self.slice_page
        self._panel3d = self.slice_page
        self.time_slider = self.slice_page._time_slider
        self.time_label = self.slice_page.time_label
        self.steps_box = self.run_page.max_steps
        self.solver_box = self.run_page.method_combo
        self.amgcl_strategy_box = self.run_page.amgcl_strat
        self.amgcl_decoupling_box = self.run_page.amgcl_dec
        self.linesearch_check = self.run_page.linesearch_chk
        self.erd_check = self.run_page.enforce_chk
        self.accept_box = self.run_page.acc
        self.progress = self.run_page.progress
        self.model_label = self.run_page.model_label
        self.log_view = self.log_panel.log_view

        self._build_menu()

        # ---- wiring -----------------------------------------------------
        self.jobs.log_line.connect(self.log_panel.append)
        self.jobs.step_done.connect(self._panel2d.append_step)
        self.jobs.result_ready.connect(self._on_result_ready)
        self.jobs.run_started.connect(self._on_queue_run_started)
        self.jobs.run_finished.connect(self._on_queue_run_finished)

        self.setAcceptDrops(True)
        self.tabs.setCurrentIndex(0)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._clamp_to_screen)
        self._log("ResSimWorkbench-style PRSTCore GUI started. Load a .DATA "
                  "deck or drop one on the window to queue it.")

    def _make_wells_panel(self):
        """The shared live well-curves data store (kept on the window)."""
        if not hasattr(self, "_panel2d"):
            self._panel2d = _WellCurvesPanel(self)
        return self._panel2d

    # ---------------------------------------------------------------- sizing
    def _fit_to_screen(self):
        from PySide6.QtGui import QGuiApplication
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(1280, 800)
            return
        geo = screen.availableGeometry()
        self.setMaximumSize(geo.width(), geo.height())
        self.setMinimumSize(min(1000, geo.width()), min(620, geo.height()))
        width = max(min(1280, int(geo.width() * 0.94)), 1000)
        height = max(min(800, int(geo.height() * 0.94)), 620)
        self.resize(width, height)
        self._center_on_screen()

    def _center_on_screen(self):
        from PySide6.QtGui import QGuiApplication
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        self.move(geo.x() + (geo.width() - self.width()) // 2,
                  geo.y() + (geo.height() - self.height()) // 2)

    def _clamp_to_screen(self):
        from PySide6.QtGui import QGuiApplication
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        frame = self.frameGeometry()
        if frame.width() > geo.width() or frame.height() > geo.height():
            self.resize(min(frame.width(), int(geo.width() * 0.92)),
                        min(frame.height(), int(geo.height() * 0.92)))
        self._center_on_screen()

    # ---------------------------------------------------------------- menu
    def _build_menu(self):
        menubar = self.menuBar()

        proj = menubar.addMenu("&Project")
        a_new = QAction("&New", self)
        a_new.setShortcut("Ctrl+N")
        a_open = QAction("&Open...", self)
        a_open.setShortcut("Ctrl+O")
        a_save = QAction("&Save", self)
        a_save.setShortcut("Ctrl+S")
        a_save_as = QAction("Save &As...", self)
        a_new.triggered.connect(self._new_project)
        a_open.triggered.connect(self._open_project)
        a_save.triggered.connect(self._save_project)
        a_save_as.triggered.connect(self._save_project_as)
        proj.addAction(a_new)
        proj.addAction(a_open)
        proj.addAction(a_save)
        proj.addAction(a_save_as)

        view = menubar.addMenu("&View")
        for i, name in enumerate(["Run", "Deck Editor", "Well Hierarchy",
                                  "2D Slice", "Plots", "Optimization",
                                  "Compare"]):
            a = QAction(name, self)
            a.triggered.connect(
                lambda checked=False, idx=i: self.tabs.setCurrentIndex(idx))
            view.addAction(a)

        agent_menu = menubar.addMenu("&Agent")
        self.agent_start_action = QAction("&Start", self)
        self.agent_config_action = QAction("&API Settings...", self)
        a_launch = self.agent_start_action
        a_config = self.agent_config_action
        a_launch.triggered.connect(self._agent_launch)
        a_config.triggered.connect(self._agent_configure)
        agent_menu.addAction(a_launch)
        agent_menu.addAction(a_config)

        # Help always goes last.
        help_menu = menubar.addMenu("&Help")
        a_about = QAction("&About", self)
        a_about.triggered.connect(self._about)
        help_menu.addAction(a_about)

    # -------------------------------------------------------------- project
    def _project_data(self) -> dict:
        return {
            "deck": self._loaded_deck_path or "",
            "outdir_mode": self.run_page.outdir_mode.currentIndex(),
            "outdir_custom": self.run_page.outdir_edit.text(),
            "run": {
                "method": self.run_page.method_combo.currentText(),
                "tolerance": self.run_page.tol_edit.text(),
                "max_steps": self.run_page.max_steps.value(),
                "acceptance": self.run_page.acc.value(),
                "amgcl_strategy": self.run_page.amgcl_strat.currentText(),
                "amgcl_decoupling": self.run_page.amgcl_dec.currentText(),
                "linesearch": self.run_page.linesearch_chk.isChecked(),
                "enforce": self.run_page.enforce_chk.isChecked(),
                "strategy": self.run_page.strategy_combo.currentText(),
                "pressure_precond": self.run_page.precond_combo.currentText(),
                "second_stage": self.run_page.second_combo.currentText(),
            },
            "queue": [{"deck": j.deck, "outdir": j.outdir}
                      for j in self.jobs.jobs],
        }

    def _restore_project(self, data: dict):
        run = data.get("run") or {}
        self.run_page.method_combo.setCurrentText(
            str(run.get("method", "AMGCL CPR")))
        self.run_page.tol_edit.setText(str(run.get("tolerance", "")))
        self.run_page.max_steps.setValue(int(run.get("max_steps", 0)))
        self.run_page.acc.setValue(float(run.get("acceptance", 2.0)))
        self.run_page.amgcl_strat.setCurrentText(
            str(run.get("amgcl_strategy", "mrst")))
        self.run_page.amgcl_dec.setCurrentText(
            str(run.get("amgcl_decoupling", "trueIMPES")))
        self.run_page.linesearch_chk.setChecked(
            bool(run.get("linesearch", True)))
        self.run_page.enforce_chk.setChecked(
            bool(run.get("enforce", True)))
        self.run_page.strategy_combo.setCurrentText(
            str(run.get("strategy", "cpr")))
        self.run_page.precond_combo.setCurrentText(
            str(run.get("pressure_precond", "hypre")))
        self.run_page.second_combo.setCurrentText(
            str(run.get("second_stage", "ilu")))
        self.run_page.outdir_mode.setCurrentIndex(
            int(data.get("outdir_mode", 0)))
        self.run_page.outdir_edit.setText(str(data.get("outdir_custom", "")))
        for entry in data.get("queue", []):
            deck = entry.get("deck")
            if deck and os.path.isfile(deck):
                self.jobs.add_deck(deck)
        deck = data.get("deck")
        if deck and os.path.isfile(deck):
            self._load_deck(deck)

    def _new_project(self):
        self.jobs.jobs = []
        self.jobs.queue_changed.emit()
        self._loaded_deck_path = ""
        self._log("New project (queue cleared)")

    def _open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open project", "",
            "Workbench projects (*.rsimproj);;All files (*)")
        if path:
            try:
                data = json.loads(_read_text(path))
            except Exception as exc:
                QMessageBox.warning(self, "Open project",
                                    "Could not read project: %s" % exc)
                return
            self._restore_project(data)
            self._log("Opened project %s" % path)

    def _save_project(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save project", "study.rsimproj",
            "Workbench projects (*.rsimproj);;All files (*)")
        if not path:
            return
        if not path.lower().endswith(".rsimproj"):
            path += ".rsimproj"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._project_data(), fh, ensure_ascii=False, indent=2)
        self._log("Saved project %s" % path)

    def _save_project_as(self):
        self._save_project()

    def _about(self):
        QMessageBox.about(
            self, "PRSTCore Simulator",
            "Qt desktop GUI for the PRSTCore reservoir simulator.\n"
            "Interface modelled on Agent.  junjian@cup.edu.cn")

    # -------------------------------------------------------------- agent
    def _agent_launch(self):
        """Show the DeepSeek agent dock and start it (Agent menu: 启动)."""
        self.agent_dock.start()
        if not self.agent_dock.config.ready:
            self._log("Agent: No API Key, please fill in.")
            self._agent_configure()

    def _agent_configure(self):
        """Open the agent API-parameter dialog (Agent menu: API Settings...)."""
        self.agent_dock.configure()

    # ----------------------------------------------------------- drag/drop
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(".data"):
                self.jobs.add_deck(path, "prst")

    # ---------------------------------------------------------------- log
    def _log(self, text):
        self.log_panel.append(text)

    # ------------------------------------------------------------- deck
    def _load_typed(self):
        path = self.run_page.load_path.text().strip()
        if not path:
            files, _ = QFileDialog.getOpenFileName(
                self, "Load model", os.path.expanduser("~"),
                "Eclipse decks (*.DATA);;All files (*)")
            if not files:
                return
            path = files
            self.run_page.load_path.setText(path)
        self._load_deck(path)

    def _load_deck(self, path):
        self._log("Loading deck %s ..." % path)
        self._loading_deck = os.path.normpath(path)
        self.run_page.load_status.setText("loading...")
        self.run_page.load_status.setStyleSheet("color: #B8860B;")
        self._deck_worker = _DeckLoadWorker(self._loading_deck)
        self._deck_worker.loaded.connect(self._on_deck_loaded)
        self._deck_worker.failed.connect(self._on_load_failed)
        self._deck_worker.start()

    def _on_deck_loaded(self, model, state0, schedule, solver, seconds):
        self.model, self.state0, self.schedule, self.solver = (
            model, state0, schedule, solver)
        self._loaded_deck_path = self._loading_deck
        self._loaded_handle = (model, state0, schedule, solver)
        self._last_h5 = None
        nc = len(state0["pressure"])
        nsteps = len(schedule["step"]["val"])
        wells = _union_wells(schedule)
        self.run_page.load_status.setText(
            os.path.basename(self._loaded_deck_path))
        self.run_page.load_status.setStyleSheet("color: #1e7d32;")
        self.model_label.setText(
            "cells=%d  phases(o/w/g)=%s/%s/%s  wells=%d  steps=%d  "
            "load: %.1f s"
            % (nc, model.oil, model.water, model.gas, len(wells), nsteps,
               seconds))
        self.run_page.max_steps.setRange(0, nsteps)
        self.well_page.set_schedule(schedule)
        self.plots_page.set_pvt_tables(self._loaded_deck_path)
        self._log("deck loaded: %d cells, %d wells, %d report steps (%.1f s)"
                  % (nc, len(wells), nsteps, seconds))
        # An existing computed folder for this deck is shown right away.
        outdir = resolve_outdir(self._loaded_deck_path,
                                self.run_page.outdir_mode.currentIndex(),
                                self.run_page.outdir_edit.text(),
                                self.run_page.engine_combo.currentData())
        if os.path.isfile(os.path.join(outdir, "states.h5")):
            self._log("found existing HDF5 results: %s" % outdir)
            self._load_h5_into_slice(outdir)

    def _on_load_failed(self, error):
        self.run_page.load_status.setText("failed: %s" % error)
        self.run_page.load_status.setStyleSheet("color: #c62828;")
        self._log("deck load FAILED: %s" % error)

    def _load_h5_into_slice(self, outdir):
        try:
            jr = h5_results.load(outdir)
        except Exception as exc:
            self._log("loading HDF5 results failed: %s: %s"
                      % (type(exc).__name__, exc))
            return
        self._last_h5 = jr
        self.slice_page.set_h5(jr, G=self._grid_of(self.model),
                               wells=_union_wells(self.schedule)
                               if self.schedule is not None else None)
        # A loaded PRST schedule carries richer well info (group + IJK), so
        # only fall back to the HDF5 well names when no schedule is loaded.
        if self.schedule is None:
            self.well_page.set_h5(jr)
        self._log("Loaded HDF5 results from %s (%d steps)"
                  % (outdir, jr.n_steps))

    def _well_entries(self):
        """Well entries for the Well Hierarchy page (schedule or HDF5)."""
        if self.schedule is not None:
            entries = []
            for w in _union_wells(self.schedule):
                i, j = int(w.get("i", 1)), int(w.get("j", 1))
                k = int(w.get("k")[0]) if isinstance(w.get("k"), list) \
                    and w.get("k") else 1
                sign = float(w.get("sign", -1.0))
                entries.append({
                    "name": str(w.get("name", "?")),
                    "group": str(w.get("group", "") or "FIELD"),
                    "type": "INJ" if sign > 0 else "PROD",
                    "ijk": "%d,%d,%d" % (i, j, k)})
            return entries
        if self._last_h5 is not None:
            entries = []
            for name, df in (self._last_h5.wells or {}).items():
                inj = any(c in df.columns and np.asarray(df[c]).size
                          and float(np.nanmax(np.asarray(df[c],
                                                         dtype=float))) > 0.0
                          for c in ("WWIR", "WGIR"))
                prod = any(c in df.columns and np.asarray(df[c]).size
                           and float(np.nanmax(np.asarray(df[c],
                                                          dtype=float))) > 0.0
                           for c in ("WOPR", "WWPR", "WGPR"))
                kind = "INJ" if inj and not prod else ("PROD" if prod
                                                        else "?")
                entries.append({"name": str(name), "group": "FIELD",
                                "type": kind, "ijk": ""})
            return entries
        return []

    @staticmethod
    def _grid_of(model):
        if model is None:
            return None
        return getattr(model, "G", None)

    # ------------------------------------------------------------- run
    def _run_params(self) -> dict:
        tolerance = None
        text = self.run_page.tol_edit.text().strip()
        if text:
            try:
                tolerance = float(text)
            except ValueError:
                tolerance = None
        return dict(
            max_steps=self.run_page.max_steps.value() or None,
            use_linesearch=self.run_page.linesearch_chk.isChecked(),
            enforce_residual_decrease=self.run_page.enforce_chk.isChecked(),
            acceptance_factor=self.run_page.acc.value(),
            method=self.run_page.method_combo.currentText(),
            tolerance=tolerance,
            amgcl_strategy=self.run_page.amgcl_strat.currentText(),
            amgcl_decoupling=self.run_page.amgcl_dec.currentText(),
            strategy=self.run_page.strategy_combo.currentText(),
            pressure_precond=self.run_page.precond_combo.currentText(),
            second_stage=self.run_page.second_combo.currentText(),
        )

    def _prepare_jobs(self):
        """Push the current Run-page options into the job queue so every
        job launched from the UI runs with the shown settings."""
        self.jobs.params = self._run_params()
        self.jobs.outdir_mode = self.run_page.outdir_mode.currentIndex()
        self.jobs.outdir_custom = self.run_page.outdir_edit.text()

    def _run(self):
        """Run the loaded model through the job queue (programmatic API)."""
        if self.model is None:
            return
        self._prepare_jobs()
        job = self.jobs.add_deck(self._loaded_deck_path,
                                 handle=self._loaded_handle)
        job.state = QUEUED
        self.jobs.queue_changed.emit()
        idx = self.jobs.jobs.index(job)
        self.jobs.run_selected([idx])

    def _on_queue_run_started(self, idx):
        job = self.jobs.job(idx)
        self._sim_worker = job.worker if job is not None else None

    def _on_queue_run_finished(self):
        self._sim_worker = None

    def _on_result_ready(self, result):
        outdir = result.get("result_dir")
        if outdir:
            self.plots_page._add_summary_case(outdir)
        G, states = result.get("G"), result.get("states")
        if G is not None and states:
            self._slice_panel.set_run(G, result.get("wells"), states,
                                      result.get("times") or [])
        elif outdir and os.path.isfile(os.path.join(outdir, "states.h5")):
            # JutulDarcy (or any HDF5-only) result: read it straight in.
            self._load_h5_into_slice(outdir)
        done = [j for j in self.jobs.jobs
                if j.state == DONE and j.outdir]
        if len(done) >= 2:
            self.compare_page.set_pair(done[-2].outdir, done[-1].outdir)
        self._log("Results ready: %s (%d steps)"
                  % (outdir or result.get("deck"), result.get("nsteps", 0)))

    # ------------------------------------------------------------- hooks
    def _open_deck_in_editor(self, path):
        self.deck_page.open_deck(path)
        self.tabs.setCurrentWidget(self.deck_page)

    def _open_result_folder(self, outdir):
        if not outdir or not os.path.isdir(outdir):
            return
        if os.name == "nt":
            os.startfile(outdir)  # noqa: S606

    # ------------------------------------------------------------- close
    def closeEvent(self, event):
        for w in getattr(self.slice_page, "_vtk_windows", []):
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass
        for job in self.jobs.jobs:
            if job.state == RUNNING and job.worker is not None:
                job.worker.request_stop()
        worker = getattr(self.agent_dock, "_worker", None)
        if worker is not None:
            worker.request_stop()
        super().closeEvent(event)


def run_simulator():
    """Start the workbench (Qt event loop)."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = SimulatorWindow()
    window.show()
    app.exec()
    return window


if __name__ == "__main__":
    run_simulator()
