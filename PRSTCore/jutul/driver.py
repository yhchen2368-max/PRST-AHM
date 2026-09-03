"""PRSTCore-native JutulDarcy driver (no GeoCode dependency).

Runs the JutulDarcy Julia drivers that ship with PRSTCore (``jutul_run.jl`` /
``jutul_optimize.jl`` under this package) through a direct ``julia``
subprocess -- the same contract GeoCode's ``execute_julia_*`` provided, but
implemented here so nothing outside PRSTCore is needed:

* :func:`run_simulate`  -- simulate a deck, write the unified HDF5 result set
  (``states.h5`` / ``wells.h5`` / ``cell_indices.h5`` / ``manifest.json``);
* :func:`run_optimize`  -- forecast BHP NPV optimisation, writing
  ``optimal_bhp.csv`` / ``production.csv`` / ``summary.json``.

Julia must be installed; the executable is taken from the ``JULIA``
environment variable (``julia`` on PATH otherwise).  The Julia environment
(``Project.toml`` + ``Manifest.toml``) lives next to the drivers, so the
first run may instantiate packages from the manifest.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_BIN = Path(__file__).resolve().parent

__all__ = ["find_julia", "available", "run_simulate", "run_optimize",
           "JULIA_HINT"]


JULIA_HINT = ("is Julia installed? set the JULIA env var to julia.exe, "
              "e.g.  $env:JULIA='C:\\Julia\\bin\\julia.exe'")


def find_julia() -> str:
    """Path to the julia executable (``JULIA`` env var, then PATH)."""
    exe = os.environ.get("JULIA") or "julia"
    # An explicit path (contains a separator / ends in .exe) is used as-is.
    if os.path.sep in exe or (os.name == "nt" and exe.lower().endswith(".exe")):
        if os.path.isfile(exe):
            return exe
        raise FileNotFoundError("JULIA points to a missing executable: %s" % exe)
    found = shutil.which(exe)
    if found:
        return found
    raise FileNotFoundError("julia not found on PATH (%s)" % JULIA_HINT)


def available() -> bool:
    """Whether julia can be launched on this machine."""
    try:
        find_julia()
        return True
    except FileNotFoundError:
        return False


def _stream_julia(argv, logpath, timeout_s, script_name, on_line=None):
    """Run a Julia subprocess, streaming stdout to the console and ``logpath``.

    ``on_line`` (optional) is called with each stripped output line; it runs
    on the calling thread.  A non-zero exit raises, and ``timeout_s`` kills
    the process and raises :class:`TimeoutError`.
    """
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    with open(logpath, "wb") as log:
        proc = subprocess.Popen(  # noqa: S603 - argv is built here
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while True:
            chunk = proc.stdout.read1(4096)
            if not chunk:
                break
            log.write(chunk)
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
            if on_line is not None:
                text = chunk.decode("utf-8", errors="replace")
                for raw in text.splitlines():
                    line = raw.split("\r")[-1].strip()
                    if line:
                        on_line(line)
            if deadline is not None and time.monotonic() > deadline:
                try:
                    proc.kill()
                except OSError:
                    pass
                proc.wait()
                raise TimeoutError("%s exceeded %s s, see %s"
                                   % (script_name, timeout_s, logpath))
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError("%s exited with code %s, see %s"
                           % (script_name, rc, logpath))


def run_simulate(case_path, result_dir, timeout_s=None, on_line=None):
    """Run ``jutul_run.jl``: deck -> unified HDF5 results.

    Returns the ``result_dir`` (a fresh directory: an existing one is
    cleared first, matching the unified-HDF5 convention).
    """
    case_path = Path(case_path)
    result_dir = Path(result_dir)
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True)

    script = _BIN / "jutul_run.jl"
    julia = find_julia()
    argv = [julia, "--threads=auto", "--project=%s" % _BIN, str(script),
            "--case=%s" % case_path, "--out=%s" % result_dir,
            "--restart=none"]
    _stream_julia(argv, result_dir / "julia.log", timeout_s, script.name,
                  on_line)
    return result_dir


def run_optimize(case_path, out_dir, params, timeout_s=None, on_line=None):
    """Run ``jutul_optimize.jl``: forecast BHP NPV optimisation.

    ``params`` must carry the driver keys (``months``, ``oil-price``,
    ``gas-price``, ``water-price``, ``water-cost``, ``gas-cost``,
    ``discount-rate``, ``bhp-prod-min/max``, ``bhp-inj-min/max``, optional
    ``max-it``).  Two are converted to the driver's raw units, matching the
    GUI's user-friendly inputs:

    * ``water-price``: positive water-handling cost in $/m3 -> negated
      (produced water is a cost in ``npv_objective``);
    * ``discount-rate``: percent per year -> divided by 100.

    Returns the ``out_dir``.
    """
    required = ("months", "oil-price", "gas-price", "water-price",
                "water-cost", "gas-cost", "discount-rate",
                "bhp-prod-min", "bhp-prod-max", "bhp-inj-min", "bhp-inj-max")
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError("Missing required params: %s" % ", ".join(missing))

    params = dict(params)
    params["water-price"] = -abs(float(params["water-price"]))
    params["discount-rate"] = float(params["discount-rate"]) / 100.0

    case_path = Path(case_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    script = _BIN / "jutul_optimize.jl"
    julia = find_julia()
    argv = [julia, "--threads=auto", "--project=%s" % _BIN, str(script),
            "--case=%s" % case_path, "--out=%s" % out_dir]
    for key, value in params.items():
        argv.append("--%s=%s" % (key, value))
    _stream_julia(argv, out_dir / "julia.log", timeout_s, script.name,
                  on_line)
    return out_dir
