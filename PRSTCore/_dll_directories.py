"""Make a non-activated conda environment's native DLLs loadable on Windows.

conda puts MKL, and everything else that ships as a shared library, in
``<env>/Library/bin`` and relies on ``conda activate`` to prepend that
directory to ``PATH``.  Invoking ``<env>/python.exe`` directly -- which is
what an IDE launch configuration, a ``pytest`` runner or a bare script path
does -- skips activation, and the directory never gets there.

numpy and scipy survive that because their own extension modules are loaded
through Python, which since 3.8 resolves dependencies via
``os.add_dll_directory`` rather than ``PATH``.  MKL does not: once loaded,
``mkl_intel_thread`` resolves ``libiomp5md.dll`` through the *operating
system's* delay-load helper, which searches ``PATH`` and ignores the
directories Python added for itself.  The lookup fails, and the delay-load
helper raises a structured exception (``0xc06d007f``, ERROR_MOD_NOT_FOUND)
that no Python ``except`` can catch -- the interpreter dies mid-call with no
traceback.

Nothing triggers it until MKL actually dispatches to a threaded kernel, so
small problems run fine and the failure looks size-dependent: a tridiagonal
``spsolve`` at 20000 unknowns is fine, a 4000-unknown three-dimensional one
takes SuperLU's supernodal path into ``dgemm`` and kills the process.  That
is the sort of thing that reads as "the solver blew up on the big model".

Prepending the directory to ``os.environ['PATH']`` is what fixes it -- the
delay-load helper reads ``PATH`` at lookup time, not at start-up, so doing
this from Python before MKL is first used works.  ``add_dll_directory`` is
called as well for anything that *does* use the modern search order.
"""

from __future__ import annotations

import os
import sys
import sysconfig

#: Kept alive for the process lifetime: the handles returned by
#: ``os.add_dll_directory`` remove the directory again when garbage collected.
_HANDLES = []


def _candidate_directories():
    """conda's native-library directories for the running interpreter."""
    prefix = sys.prefix
    for relative in (('Library', 'bin'),
                     ('Library', 'mingw-w64', 'bin'),
                     ('Library', 'usr', 'bin'),
                     ('DLLs',)):
        yield os.path.join(prefix, *relative)
    # A venv layered on a conda base resolves its own Library/bin to nothing;
    # fall back to the base installation that actually holds the DLLs.
    base = getattr(sys, 'base_prefix', prefix)
    if base != prefix:
        yield os.path.join(base, 'Library', 'bin')
    scripts = sysconfig.get_paths().get('scripts')
    if scripts:
        yield scripts


def ensure_native_dll_path():
    """Put conda's ``Library/bin`` on ``PATH`` and the DLL search path.

    Safe to call more than once and on non-Windows platforms, where it does
    nothing.  Returns the directories that were added.
    """
    if sys.platform != 'win32':
        return []

    path = os.environ.get('PATH', '')
    entries = [p for p in path.split(os.pathsep) if p]
    known = {os.path.normcase(os.path.normpath(p)) for p in entries}

    added = []
    for directory in _candidate_directories():
        if not os.path.isdir(directory):
            continue
        key = os.path.normcase(os.path.normpath(directory))
        if key in known:
            continue
        known.add(key)
        added.append(directory)
        try:
            _HANDLES.append(os.add_dll_directory(directory))
        except (AttributeError, OSError):
            # add_dll_directory is Windows-only and fails on a path that
            # disappeared between the isdir check and here; PATH still works.
            pass

    if added:
        os.environ['PATH'] = os.pathsep.join(added + entries)
    return added
