"""Build the AMGCL extensions for the interpreter that runs this script.

Two independent extensions live in
``PRSTCore/solvers/linearsolvers/pyamgcl``:

``pyamgcl_ext``
    the upstream pybind11 bindings -- scalar AMG plus Krylov solvers, used
    for the pressure block of ``CPRSolverAD`` and by ``AMGCLSolverAD``.
``pyamgcl_block_cpr_capi_ext``
    a self-contained extension written against the bare Python C API,
    carrying the block-CPR preconditioner ``AMGCL_CPRSolverBlockAD`` calls.

Both are CPython-ABI specific: a ``.pyd`` built for 3.13 will not import on
3.14, and when it fails to import ``selectLinearSolverAD`` quietly falls all
the way back to a direct solve.  On SPE9 that is the difference between
0.025 s and 0.81 s per linear system; at 32000 unknowns, between 0.04 s and
19 s.  Nothing reports it, so the symptom is "the simulator got slow".

Everything needed is vendored -- amgcl and a Boost subset under
``PRSTCore/solvers/linearsolvers/amgcl/dependencies`` -- so this runs
offline.  pybind11 is only required for ``pyamgcl_ext``; if it is absent the
C-API extension is still built, which is enough for block CPR.

Usage::

    <python> scripts/build_pyamgcl.py            # build what is possible
    <python> scripts/build_pyamgcl.py --only capi
    <python> scripts/build_pyamgcl.py --force    # rebuild even if current
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG = REPO_ROOT / 'PRSTCore' / 'solvers' / 'linearsolvers' / 'pyamgcl'
DEPS = REPO_ROOT / 'PRSTCore' / 'solvers' / 'linearsolvers' / 'amgcl' / 'dependencies'

#: One entry per extension: module stem, source file, whether pybind11 is needed.
EXTENSIONS = {
    'capi': ('pyamgcl_block_cpr_capi_ext', 'pyamgcl_block_cpr_capi.cpp', False),
    'pybind': ('pyamgcl_ext', 'pyamgcl.cpp', True),
}

#: Kernels that live outside the pyamgcl package and need numpy's headers
#: rather than amgcl's.  Same compiler, different include set.
KERNELS = {
    'divergence': ('discrete_divergence_ext', 'discrete_divergence.cpp'),
    'faceops': ('face_operators_ext', 'face_operators.cpp'),
}
KERNEL_DIR = REPO_ROOT / 'PRSTCore' / 'ad_core' / 'mex' 


def _amgcl_include():
    """The vendored amgcl checkout, whose directory name carries a commit id."""
    candidates = sorted(DEPS.glob('amgcl-*'))
    if not candidates:
        raise SystemExit('no vendored amgcl under %s' % DEPS)
    return candidates[-1]


def _boost_include():
    candidates = sorted(DEPS.glob('boost-*'))
    if not candidates:
        raise SystemExit('no vendored Boost under %s' % DEPS)
    return candidates[-1]


def _sibling_interpreters():
    """Other Python installations that might carry pybind11's headers."""
    seen = set()
    for name in ('python', 'python3'):
        found = shutil.which(name)
        if found and found not in seen:
            seen.add(found)
            yield Path(found)
    home = Path.home()
    for pattern in ('anaconda3/python.exe', 'miniconda3/python.exe',
                    'AppData/Local/Programs/Python/*/python.exe'):
        for candidate in home.glob(pattern):
            if str(candidate) not in seen:
                seen.add(str(candidate))
                yield candidate


def _pybind11_include():
    """pybind11's headers, from this interpreter or from a sibling one.

    The headers are the whole of pybind11 -- there is nothing to link -- so
    borrowing them from another Python installation still builds a perfectly
    good extension for *this* one.  That is what lets an offline environment
    with no pybind11 installed build anyway.
    """
    try:
        import pybind11
        return Path(pybind11.get_include())
    except ImportError:
        pass
    for other in _sibling_interpreters():
        try:
            out = subprocess.run(
                [str(other), '-c', 'import pybind11;print(pybind11.get_include())'],
                capture_output=True, text=True, timeout=60,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode == 0:
            include = Path(out.stdout.strip())
            if (include / 'pybind11' / 'pybind11.h').is_file():
                return include
    return None


def _vcvars():
    """Locate ``vcvars64.bat`` -- vswhere first, then the usual install roots."""
    program_files_x86 = os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')
    vswhere = Path(program_files_x86) / 'Microsoft Visual Studio' / 'Installer' / 'vswhere.exe'
    if vswhere.is_file():
        out = subprocess.run(
            [str(vswhere), '-latest', '-products', '*',
             '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
             '-property', 'installationPath'],
            capture_output=True, text=True,
        )
        for line in out.stdout.strip().splitlines():
            candidate = Path(line.strip()) / 'VC' / 'Auxiliary' / 'Build' / 'vcvars64.bat'
            if candidate.is_file():
                return candidate
    roots = (Path(program_files_x86),
             Path(os.environ.get('ProgramFiles', r'C:\Program Files')))
    for base in roots:
        for candidate in base.glob('Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvars64.bat'):
            return candidate
    raise SystemExit(
        'vcvars64.bat not found -- install the Visual Studio Build Tools '
        'with the "Desktop development with C++" workload.')


def _python_link_library():
    """``pythonXY.lib`` for the running interpreter."""
    version = '%d%d' % (sys.version_info[0], sys.version_info[1])
    for directory in (Path(sys.prefix) / 'libs', Path(sys.base_prefix) / 'libs'):
        candidate = directory / ('python%s.lib' % version)
        if candidate.is_file():
            return candidate
    raise SystemExit('python%s.lib not found under %s/libs' % (version, sys.prefix))


def _output_name(stem):
    """``<stem>.cp314-win_amd64.pyd`` -- the ABI tag the import system wants."""
    suffix = sysconfig.get_config_var('EXT_SUFFIX') or '.pyd'
    return stem + suffix


def build(kind, force=False):
    stem, source, needs_pybind11 = EXTENSIONS[kind]
    target = PKG / _output_name(stem)
    if target.is_file() and not force:
        print('[skip]  %s already present (use --force to rebuild)' % target.name)
        return target

    includes = [Path(sysconfig.get_paths()['include']), _amgcl_include(), _boost_include()]
    if needs_pybind11:
        pybind11_include = _pybind11_include()
        if pybind11_include is None:
            print('[skip]  %s: pybind11 headers not found on this machine' % stem)
            return None
        includes.append(pybind11_include)

    vcvars = _vcvars()
    python_lib = _python_link_library()

    with tempfile.TemporaryDirectory(prefix='pyamgcl-build-') as tmp:
        objdir = Path(tmp)
        include_flags = ' '.join('/I"%s"' % path for path in includes)
        # A batch file rather than ``cmd /c "call ... && cl ..."``: cmd strips
        # one layer of quoting from its own argument, which mangles the quoted
        # paths in both halves before either program sees them.
        # /bigobj: amgcl's runtime dispatch instantiates more sections than
        # the default object-file limit allows.
        script = objdir / 'build.bat'
        script.write_text(
            '@echo off\r\n'
            'call "%s" >nul\r\n'
            'if errorlevel 1 exit /b 1\r\n'
            # The doubled backslash before each closing quote is required:
            # a lone one escapes the quote in MSVC's own argument parsing, and
            # /Fo swallows the rest of the command line.
            # C4819 fires on every pybind11 header when the console codepage
            # is not UTF-8, which is noise, not a finding.
            'cl /nologo /LD /EHsc /O2 /std:c++17 /bigobj /MD '
            '/wd4267 /wd4244 /wd4996 /wd4819 /DNDEBUG /D_CRT_SECURE_NO_WARNINGS '
            '%s /Fo"%s\\\\" /Fd"%s\\\\" "%s" '
            '/link /OUT:"%s" /IMPLIB:"%s" "%s"\r\n'
            'exit /b %%errorlevel%%\r\n'
            % (vcvars, include_flags, objdir, objdir, PKG / source,
               target, objdir / 'ext.lib', python_lib),
            encoding='ascii',
        )
        print('[build] %s -> %s' % (stem, target.name))
        # errors='replace': MSVC writes diagnostics in the console codepage,
        # which on a non-English Windows is not UTF-8 and not always decodable
        # as the locale encoding either.  A build log that cannot be decoded
        # must not become the failure being reported.
        completed = subprocess.run(['cmd.exe', '/c', str(script)],
                                   capture_output=True, text=True,
                                   errors='replace')
        if completed.returncode != 0 or not target.is_file():
            sys.stderr.write(completed.stdout or '')
            sys.stderr.write(completed.stderr or '')
            raise SystemExit('build of %s failed (exit %d)' % (stem, completed.returncode))

    # cl leaves the import library and export table beside the output.
    for leftover in (target.with_suffix('.lib'), target.with_suffix('.exp')):
        if leftover.is_file():
            leftover.unlink()
    print('[ok]    %s  (%d KiB)' % (target.name, target.stat().st_size // 1024))
    return target


def build_kernel(kind, force=False):
    """Build one of the numpy-based kernels in PRSTCore/ad_core/mex.

    Same compiler and same output naming as the AMGCL extensions; the only
    difference is the include set -- these need numpy's headers rather than
    amgcl's and Boost's.
    """
    import numpy

    stem, source = KERNELS[kind]
    target = KERNEL_DIR / _output_name(stem)
    if target.is_file() and not force:
        print('[skip]  %s already present (use --force to rebuild)' % target.name)
        return target

    includes = [Path(sysconfig.get_paths()['include']), Path(numpy.get_include())]
    vcvars = _vcvars()
    python_lib = _python_link_library()

    with tempfile.TemporaryDirectory(prefix='kernel-build-') as tmp:
        objdir = Path(tmp)
        include_flags = ' '.join('/I"%s"' % path for path in includes)
        script = objdir / 'build.bat'
        script.write_text(
            '@echo off\r\n'
            'call "%s" >nul\r\n'
            'if errorlevel 1 exit /b 1\r\n'
            'cl /nologo /LD /EHsc /O2 /std:c++17 /MD '
            '/wd4267 /wd4244 /wd4996 /wd4819 /DNDEBUG /D_CRT_SECURE_NO_WARNINGS '
            '%s /Fo"%s\\\\" /Fd"%s\\\\" "%s" '
            '/link /OUT:"%s" /IMPLIB:"%s" "%s"\r\n'
            'exit /b %%errorlevel%%\r\n'
            % (vcvars, include_flags, objdir, objdir, KERNEL_DIR / source,
               target, objdir / 'ext.lib', python_lib),
            encoding='ascii',
        )
        print('[build] %s -> %s' % (stem, target.name))
        completed = subprocess.run(['cmd.exe', '/c', str(script)],
                                   capture_output=True, text=True,
                                   errors='replace')
        if completed.returncode != 0 or not target.is_file():
            sys.stderr.write(completed.stdout or '')
            sys.stderr.write(completed.stderr or '')
            raise SystemExit('build of %s failed (exit %d)' % (stem, completed.returncode))

    for leftover in (target.with_suffix('.lib'), target.with_suffix('.exp')):
        if leftover.is_file():
            leftover.unlink()
    print('[ok]    %s  (%d KiB)' % (target.name, target.stat().st_size // 1024))
    return target


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', choices=sorted(EXTENSIONS) + sorted(KERNELS),
                        action='append',
                        help='build just this extension or kernel (repeatable)')
    parser.add_argument('--force', action='store_true',
                        help='rebuild even when the .pyd is already there')
    args = parser.parse_args()

    if sys.platform != 'win32':
        raise SystemExit('this build script drives MSVC; on other platforms use CMake')

    print('CPython %d.%d at %s' % (sys.version_info[0], sys.version_info[1], sys.prefix))
    kinds = args.only or (['capi', 'pybind'] + list(KERNELS))
    built = [build_kernel(kind, force=args.force) if kind in KERNELS
             else build(kind, force=args.force)
             for kind in kinds]

    print()
    sys.path.insert(0, str(REPO_ROOT))
    from PRSTCore.solvers.linearsolvers import pyamgcl
    print('has_pyamgcl_ext   : %s' % pyamgcl.has_pyamgcl_ext())
    print('has_block_cpr_ext : %s' % pyamgcl.has_block_cpr_ext())
    from PRSTCore.ad_core import mex
    print('divergence kernel : %s' % (mex.load_discrete_divergence() is not None))
    return 0 if any(b is not None for b in built) else 1


if __name__ == '__main__':
    raise SystemExit(main())
