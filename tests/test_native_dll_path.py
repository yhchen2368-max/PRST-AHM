"""A three-dimensional sparse solve must not take the interpreter down.

conda keeps MKL's OpenMP runtime in ``<env>/Library/bin`` and expects
``conda activate`` to put that directory on ``PATH``.  Launching
``<env>/python.exe`` directly -- an IDE run configuration, a bare ``pytest``
path -- skips activation, and MKL's delay-load of ``libiomp5md.dll`` fails
inside the OS loader.  That raises a structured exception no ``except`` can
catch: the process dies with no traceback, part-way through the first solve
big enough to reach a threaded kernel.

``PRSTCore.ensure_native_dll_path`` (run from ``PRSTCore/__init__.py``) is
what prevents it.  The check runs in a subprocess because the failure mode
under test is a process kill, which would otherwise take the whole pytest
session with it and report nothing.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# 20x20x10 is the smallest size observed to reach SuperLU's supernodal path,
# where the dispatch into MKL's threaded dgemm triggers the delay-load.
_PROBE = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, %r)
    import PRSTCore  # noqa: F401  -- the import under test
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    nx = ny = 20
    nz = 10
    n = nx * ny * nz
    idx = np.arange(n).reshape(nx, ny, nz)
    rows, cols = [], []
    for a, b in ((idx[:-1].ravel(), idx[1:].ravel()),
                 (idx[:, :-1].ravel(), idx[:, 1:].ravel()),
                 (idx[:, :, :-1].ravel(), idx[:, :, 1:].ravel())):
        rows += [a, b]
        cols += [b, a]
    rows = np.concatenate(rows + [np.arange(n)])
    cols = np.concatenate(cols + [np.arange(n)])
    vals = np.r_[-np.ones(rows.size - n), np.full(n, 7.0)]
    A = sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
    b = np.ones(n)

    x = spla.spsolve(A, b)
    residual = float(np.linalg.norm(A @ x - b))
    assert residual < 1e-8, residual
    print("OK", residual)
    """
)


def test_three_dimensional_spsolve_survives_without_conda_activate():
    probe = _PROBE % (str(REPO_ROOT),)
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, (
        "a 4000-unknown 3D sparse solve did not survive:\n"
        f"returncode={completed.returncode}\n"
        f"stdout={completed.stdout}\nstderr={completed.stderr}"
    )
    assert "OK" in completed.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="conda DLL layout is Windows-specific")
def test_library_bin_is_on_path_after_import():
    import os

    from PRSTCore import _dll_directories

    _dll_directories.ensure_native_dll_path()
    library_bin = Path(sys.prefix) / "Library" / "bin"
    if not library_bin.is_dir():
        pytest.skip("not a conda-style installation")
    entries = {
        os.path.normcase(os.path.normpath(p))
        for p in os.environ.get("PATH", "").split(os.pathsep)
        if p
    }
    assert os.path.normcase(os.path.normpath(str(library_bin))) in entries


def test_ensure_native_dll_path_is_idempotent():
    import os

    from PRSTCore import _dll_directories

    _dll_directories.ensure_native_dll_path()
    before = os.environ.get("PATH", "")
    assert _dll_directories.ensure_native_dll_path() == []
    assert os.environ.get("PATH", "") == before
