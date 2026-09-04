"""Headless backend for MRST's ``FAHM.mlapp`` / extracted ``FAHM.m``.

``FAHM.m`` is 4,404 lines, of which roughly 2,700 are MATLAB App
Designer widget construction and several hundred more are one-line
checkbox/button callbacks. None of that has a 1:1 Python target -- there
is no App Designer, and a widget tree is not a computation.

What *is* portable is the pipeline those widgets configure, and that is
what this module is: the same steps in the same order, with the widget
state replaced by a :class:`FahmConfig`.

The pipeline, mirroring ``ModelProceedButtonPushed`` and the run block at
``FAHM_M.m`` lines 1850-2010:

1. read and condition the deck (``processEclipseDeck``);
2. build the model, schedule and initial state;
3. assemble the observed data -- from monitoring files, or (as here) from
   the deck's own history vectors;
4. declare the tunable parameters and resolve their consistency
   constraints (``checkParameterConsistency``);
5. build the mismatch function;
6. loop: write the deck, run ECLIPSE, read the summary, score it.

Step 6's inner evaluation is :func:`run_forward`, which is what makes the
whole thing testable without an optimiser: it is one full round trip
through the external simulator.
"""

import os as _os
import shutil as _shutil
import subprocess as _subprocess
from copy import deepcopy as _deepcopy
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as _np

DEFAULT_ECLIPSE = r'C:\ecl\2022.4\bin\pc_x86_64\eclipse.exe'

# Tokens appended by ``CreatProjectButtonPushed`` for the tNavigator base
# run.  Keep these as individual argv entries: the MATLAB source builds one
# shell string, but token identity is the Stage 4 cross-language contract.
_TNAVIGATOR_BASE_ARGS = (
    '--no-dump-res', '--ecl-root', '-e', '-i', '-r', '-u', '--no-gui',
    '--ignore-lock', '--use-gpu',
)
_BASE_CASE_OUTPUT_SUFFIXES = ('.INIT', '.EGRID', '.UNRST')


class CreateProjectError(RuntimeError):
    """Base class for failures in FAHM's Create Project preflight."""


class UnsupportedSimulatorError(CreateProjectError):
    """Raised at the same point as FAHM's ``Unsupported simulator``."""


class BaseCaseLaunchError(CreateProjectError):
    """The simulator command could not be started."""

    def __init__(self, argv, base_case, cause):
        self.argv = tuple(argv)
        self.base_case = str(base_case)
        self.cause = cause
        super().__init__(
            'Failed to launch the base-case simulator command %r: %s'
            % (self.argv, cause))


class MissingBaseCaseOutputError(CreateProjectError):
    """The command returned without files FAHM immediately reads next."""

    def __init__(self, result, missing):
        self.result = result
        self.missing = tuple(str(path) for path in missing)
        tail = (result.stderr or result.stdout or '')[-2000:]
        message = (
            'Base-case simulation produced no usable output. Missing: %s; '
            'exit status: %d' % (', '.join(self.missing), result.returncode))
        if tail:
            message += '\nLast simulator output:\n' + tail
        super().__init__(message)


class ProjectStateError(CreateProjectError):
    """INIT/EGRID/UNRST exists but cannot form FAHM's project state."""


@dataclass(frozen=True)
class BaseCaseCommand:
    """One of the three command branches at FAHM.m:1727-1737."""

    simulator_kind: str
    argv: tuple
    nosim: bool
    stop_step: Optional[int] = None


@dataclass(frozen=True)
class BaseCaseResult:
    """Observable result of the Stage 4 Create Project preflight."""

    base_dir: str
    base_case: str
    written_data_file: str
    simulator_kind: str
    argv: tuple
    nosim: bool
    stop_step: Optional[int]
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    output_files: tuple


@dataclass(frozen=True)
class FahmProject:
    """The App properties established by FAHM.m:1753-1824.

    ``deck``, ``G``, ``rock`` and ``fluid`` are App-owned snapshots.  The
    model owns independent copies, matching MATLAB's struct value
    semantics rather than Python reference aliasing.
    """

    deck: dict
    G: dict
    rock: dict
    fluid: dict
    state0: dict
    model: object
    N: _np.ndarray
    T: _np.ndarray
    prefix: str
    restart_count: int

# The parameters FAHM's Parameter tab exposes, with the box limits its
# setDefaultParameterLimits assigns.
DEFAULT_PARAMETER_LIMITS = {
    'porevolume': (0.5, 2.0),
    'permx': (0.1, 10.0),
    'permy': (0.1, 10.0),
    'permz': (0.1, 10.0),
    'swl': (0.0, 0.4),
    'swcr': (0.0, 0.4),
    'swu': (0.6, 1.0),
    'sgl': (0.0, 0.2),
    'sgcr': (0.0, 0.2),
    'sgu': (0.6, 1.0),
    'sowcr': (0.0, 0.4),
    'sogcr': (0.0, 0.4),
    'krw': (0.1, 1.0),
    'kro': (0.1, 1.0),
    'krg': (0.1, 1.0),
}


@dataclass
class FahmConfig:
    """The state FAHM's tabs collect, as plain data.

    ``parameters`` names which quantities are tuned (the Parameter tab's
    checkboxes); ``weights`` is the Run tab's oil/water/gas emphasis.
    """
    deck_path: str
    work_dir: str
    case_name: str = 'CASE'
    simulator: str = DEFAULT_ECLIPSE
    parameters: Sequence[str] = field(default_factory=lambda: ['permx'])
    relative_limits: bool = True
    weights: dict = field(default_factory=lambda: {'oil': 1.0, 'water': 1.0,
                                                   'gas': 0.0, 'bhp': 0.0})
    max_iterations: int = 10
    # Retained for legacy headless callers only. FAHM itself matches the
    # wells selected by omega and does not apply a producer-only switch.
    match_only_producers: bool = False
    #: Per-parameter overrides of DEFAULT_PARAMETER_LIMITS, as the
    #: Parameter tab's limits tables supply them.
    parameter_limits: dict = field(default_factory=dict)
    #: Stage 6 App selection for rates/BHP/tracer/profile/saturation.  The
    #: forward pipeline consumes it in Stage 7; carrying it here does not
    #: change the current algorithm.
    monitoring: dict = field(default_factory=dict)
    #: Exact arguments produced by FAHM's dependent ``alpha``, ``beta``
    #: and ``omega`` properties.  ``None`` keeps the small headless API
    #: backward-compatible by deriving them from ``weights``/observed data.
    objective_weight: Optional[dict] = None
    normalization_factor: Optional[dict] = None
    wells_weight: Optional[dict] = None

    def limits_for(self, name):
        key = str(name).lower()
        if key in self.parameter_limits:
            lo, hi = self.parameter_limits[key]
            return float(lo), float(hi)
        return DEFAULT_PARAMETER_LIMITS.get(key, (0.1, 10.0))


def read_case(config):
    """Read, unit-convert and condition the selected Eclipse deck."""
    from PRSTCore.deckformat.deckinput.convert_deck_units import convert_deck_units
    from PRSTCore.deckformat.deckinput.read_eclipse_deck import read_eclipse_deck
    from PRSTCore.hm.utils.processEclipseDeck import processEclipseDeck

    deck = convert_deck_units(read_eclipse_deck(config.deck_path))
    deck = processEclipseDeck(deck)
    return deck


def base_case_paths(deck_path):
    """Return FAHM's ``directory.base``, ``filename`` and ``baseCase``.

    ``get.directory`` puts ``baseCase`` beside the selected DATA file and
    ``get.filename`` uses that file's stem.  ``app.baseCase`` preserves the
    stem's spelling even though ``writeDeck`` upper-cases the physical DATA
    filename.
    """
    source = _os.path.abspath(_os.fspath(deck_path))
    directory = _os.path.dirname(source)
    filename = _os.path.splitext(_os.path.basename(source))[0]
    base_dir = _os.path.join(directory, 'baseCase')
    base_case = _os.path.join(base_dir, filename + '.DATA')
    return base_dir, filename, base_case


def _first_welspec_report_step(deck):
    """Translate FAHM's two MATLAB ``find(..., 1, 'first')`` calls.

    PRST schedule controls are zero-based, while tNavigator's ``stop-step``
    is the one-based report-step position passed by MATLAB.
    """
    schedule = (deck or {}).get('SCHEDULE') or {}
    controls = schedule.get('control')
    if controls is None:
        controls = ()

    first_control = None
    for index, control in enumerate(controls):
        records = (control or {}).get('WELSPECS')
        if records is not None and len(records) > 0:
            first_control = index
            break
    if first_control is None:
        return None

    step = schedule.get('step') or {}
    step_controls = step.get('control')
    if step_controls is None:
        step_controls = ()
    for index, control_index in enumerate(step_controls):
        if int(control_index) == first_control:
            return index + 1
    return None


def build_base_case_command(simulator, base_case, deck):
    """Build the exact Create Project command as argv tokens.

    The branch checks intentionally remain case-sensitive and in MATLAB's
    order.  In particular, selecting ``eclipse.exe`` only identifies the
    branch: FAHM then invokes the fixed ``eclrun eclipse`` command rather
    than that selected executable path.
    """
    simulator = _os.fspath(simulator)
    base_case = _os.fspath(base_case)
    if 'eclipse' in simulator:
        return BaseCaseCommand(
            simulator_kind='ECLIPSE',
            argv=('eclrun', 'eclipse', base_case),
            nosim=True,
        )
    if 'e300' in simulator:
        return BaseCaseCommand(
            simulator_kind='E300',
            argv=('eclrun', 'e300', base_case),
            nosim=True,
        )
    if 'tNavigator' in simulator:
        stop_step = _first_welspec_report_step(deck)
        token = '--stop-step=%s' % ('' if stop_step is None else stop_step)
        return BaseCaseCommand(
            simulator_kind='tNavigator',
            argv=(simulator,) + _TNAVIGATOR_BASE_ARGS + (token, base_case),
            nosim=False,
            stop_step=stop_step,
        )
    raise UnsupportedSimulatorError('Unsupported simulator')


def _recreate_base_case_directory(base_dir):
    """Port FAHM.m:1710-1721 without following a symlink/junction target."""
    if _os.path.isdir(base_dir):
        if _os.path.islink(base_dir):
            _os.unlink(base_dir)
        else:
            _shutil.rmtree(base_dir)
    # A non-directory entry named baseCase is deliberately not removed:
    # MATLAB's isfolder is false and its following mkdir fails as well.
    _os.mkdir(base_dir)


def _selected_eclrun(simulator):
    """Find ECLIPSE's launcher from the executable selected in the App.

    FAHM uses the selected ``eclipse.exe``/``e300.exe`` only to choose a
    branch and then assumes that ``eclrun`` is on PATH.  A VS Code process
    commonly has an older PATH than an interactive terminal.  Search the
    selected installation tree as well, e.g.::

        C:\\ecl\\2022.2\\bin\\pc_x86_64\\eclipse.exe
        C:\\ecl\\macros\\eclrun.exe

    No unrelated installation tree is searched: the selected simulator is
    the sole anchor for this defect-corrected runtime lookup.
    """
    selected = _os.path.abspath(_os.fspath(simulator))
    if not _os.path.isfile(selected):
        return None
    directory = _os.path.dirname(selected)
    while True:
        candidates = (
            _os.path.join(directory, 'eclrun'),
            _os.path.join(directory, 'eclrun.exe'),
            _os.path.join(directory, 'eclrun.cmd'),
            _os.path.join(directory, 'eclrun.bat'),
            _os.path.join(directory, 'macros', 'eclrun'),
            _os.path.join(directory, 'macros', 'eclrun.exe'),
            _os.path.join(directory, 'macros', 'eclrun.cmd'),
            _os.path.join(directory, 'macros', 'eclrun.bat'),
        )
        for candidate in candidates:
            if _os.path.isfile(candidate):
                return candidate
        parent = _os.path.dirname(directory)
        if parent == directory:
            return None
        directory = parent


def _launch_argv(argv, env, simulator=None):
    """Resolve the logical oracle command to a runnable physical command.

    The returned logical command remains the exact MATLAB token contract.
    FAHM-FIX-021 only changes executable resolution: PATH is tried first,
    then an ``eclrun`` beside the selected ECLIPSE installation, and finally
    the selected simulator executable itself.
    """
    logical = tuple(_os.fspath(token) for token in argv)
    path = None if env is None else env.get('PATH')
    resolved = _shutil.which(logical[0], path=path)
    physical = None
    if resolved is not None:
        physical = (resolved,) + logical[1:]
    elif (simulator is not None and logical[0].lower() == 'eclrun'):
        selected = _os.path.abspath(_os.fspath(simulator))
        selected_launcher = _selected_eclrun(selected)
        if selected_launcher is not None:
            physical = (selected_launcher,) + logical[1:]
        elif _os.path.isfile(selected):
            # Direct launch is the last safe fallback.  Drop the eclrun
            # product selector because eclipse.exe/e300.exe take the case
            # path directly.
            physical = (selected,) + logical[2:]
    if physical is None:
        physical = logical
    if _os.name == 'nt' and str(physical[0]).lower().endswith(('.bat', '.cmd')):
        # Passing a nested quoted command as the final ``cmd /c`` argv item
        # makes Python backslash-escape its quotes, which cmd.exe interprets
        # literally. Let subprocess perform the one shell hand-off instead.
        return _subprocess.list2cmdline(list(physical)), True
    return physical, False


def create_base_case(deck, deck_path, simulator, *, runner=None,
                     deck_writer=None, cwd=None, env=None):
    """Execute FAHM.m:1707-1744 and validate its next required files.

    Existing ``baseCase`` is removed before simulator classification, just
    as in the callback.  A process return code is retained but is not itself
    fatal because MATLAB discards ``system(command)``'s status; observable
    success is the presence of INIT/EGRID/UNRST, which the next three source
    statements read unconditionally.
    """
    base_dir, filename, base_case = base_case_paths(deck_path)
    _recreate_base_case_directory(base_dir)
    command = build_base_case_command(simulator, base_case, deck)

    if deck_writer is None:
        from PRSTCore.deckformat.deckoutput.write_deck import write_deck
        deck_writer = write_deck
    written = deck_writer(deck, base_dir, filename=filename,
                          NOSIM=command.nosim)

    logical_argv = tuple(command.argv)
    run_argv, use_shell = _launch_argv(
        logical_argv, env, simulator=simulator)
    run_cwd = _os.path.abspath(_os.fspath(cwd) if cwd is not None
                               else _os.getcwd())
    if runner is None:
        runner = _subprocess.run
    try:
        completed = runner(
            run_argv if use_shell else list(run_argv), cwd=run_cwd, env=env,
            capture_output=True, text=True, check=False, shell=use_shell)
    except OSError as exc:
        attempted = ((run_argv,) if isinstance(run_argv, str)
                     else tuple(run_argv))
        raise BaseCaseLaunchError(attempted, base_case, exc) from exc

    output_files = tuple(
        _os.path.splitext(base_case)[0] + suffix
        for suffix in _BASE_CASE_OUTPUT_SUFFIXES)
    result = BaseCaseResult(
        base_dir=base_dir,
        base_case=base_case,
        written_data_file=_os.fspath(written),
        simulator_kind=command.simulator_kind,
        argv=logical_argv,
        nosim=command.nosim,
        stop_step=command.stop_step,
        cwd=run_cwd,
        returncode=int(completed.returncode),
        stdout=completed.stdout or '',
        stderr=completed.stderr or '',
        output_files=output_files,
    )
    missing = [path for path in output_files if not _os.path.isfile(path)]
    if missing:
        raise MissingBaseCaseOutputError(result, missing)
    return result


def initialize_fahm_project(deck, base_case):
    """Execute FAHM.m:1746-1824 on existing simulator result files.

    Grid/rock/transmissibility and state0 come exclusively from
    ``EGRID/INIT/UNRST``.  The processed deck supplies fluid/PVT and model
    configuration only; it is deep-copied before FAHM's MULTPV removal so
    the caller's original deck is never changed.
    """
    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import (
        _init_deck_adi_fluid, _select_model_from_deck)
    from PRSTCore.ad_core.operators_tpfa import setup_operators_tpfa
    from PRSTCore.ad_props.impose_relperm_scaling import \
        impose_relperm_scaling
    from PRSTCore.deckformat.resultinput import (
        convert_restart_to_states, init_grid_from_eclipse_output,
        process_eclipse_restart_spec, read_eclipse_output_file_unfmt)
    from PRSTCore.hm.utils.getCapPressScalingPoints import (
        as_dict as capillary_as_dict, getCapPressScalingPoints)
    from PRSTCore.hm.utils.getRelpermScalingPoints import (
        as_dict as relperm_as_dict, getRelpermScalingPoints)
    from PRSTCore.hm.utils.imposeCapPressScaling import \
        imposeCapPressScaling

    work_deck = _deepcopy(deck)
    for section in ('GRID', 'EDIT'):
        values = work_deck.get(section)
        if isinstance(values, dict):
            values.pop('MULTPV', None)

    prefix = _os.path.splitext(_os.path.abspath(_os.fspath(base_case)))[0]
    try:
        init = read_eclipse_output_file_unfmt(prefix + '.INIT')
        grid = read_eclipse_output_file_unfmt(prefix + '.EGRID')
        G, rock, N, T = init_grid_from_eclipse_output(
            init, grid, output_sim_grid=False)
        # These are explicit App fields in the oracle even though its
        # subsequent setupOperatorsTPFA call intentionally recomputes T.
        G['trans'] = {
            'neighbors': np_copy(N, dtype=int),
            'T': np_copy(T, dtype=float),
        }

        rsspec, _ = process_eclipse_restart_spec(prefix, 'all')
        states, _ = convert_restart_to_states(
            prefix, G, restart_info=rsspec,
            split_wells_on_sign_change=False,
            remove_closed_wells=False, remove_crossflow=False,
            include_well_sols=False, include_aquifers=True,
            include_components='COMPS' in (work_deck.get('RUNSPEC') or {}))
        if not states:
            raise ValueError('UNRST contains no state with PRESSURE')
        state0 = _deepcopy(states[0])

        fluid = _init_deck_adi_fluid(
            work_deck, G, useMex=True)
        # ``selectModelFromDeck`` receives MATLAB value arguments.  Copy
        # each mutable input before model setup/scaling so the App copies
        # above cannot be changed through model references.
        model = _select_model_from_deck(
            _deepcopy(G), _deepcopy(rock), _deepcopy(fluid),
            _deepcopy(work_deck), useNatural=False)
        model.operators = setup_operators_tpfa(
            model.G, model.rock,
            neighbors=model.G['trans']['neighbors'])
        model.porevolume = np_copy(model.operators['pv'], dtype=float)

        relperm = getRelpermScalingPoints(model)
        model = impose_relperm_scaling(model, **relperm_as_dict(relperm))
        capillary = getCapPressScalingPoints(model)
        model = imposeCapPressScaling(
            model, **capillary_as_dict(capillary))

        _fill_default_aquifer_pressures(model, state0)
        _apply_fahm_swatinit_scale(model, state0, work_deck)
        model = model.validateModel()
    except Exception as exc:
        if isinstance(exc, ProjectStateError):
            raise
        raise ProjectStateError(
            'Failed to build G/rock/fluid/model/state0 from simulator '
            'results for %s: %s' % (prefix, exc)) from exc

    return FahmProject(
        deck=work_deck, G=G, rock=rock, fluid=fluid,
        state0=state0, model=model,
        N=np_copy(N, dtype=int), T=np_copy(T, dtype=float),
        prefix=prefix, restart_count=len(states))


def np_copy(value, dtype=None):
    """Small local helper that always returns an owning C/F-neutral array."""
    return _np.array(value, dtype=dtype, copy=True)


def _fill_default_aquifer_pressures(model, state0):
    aquifer = getattr(model, 'AquiferModel', None)
    if aquifer is None:
        return
    pressure = _np.asarray(aquifer.initvals['pressures'], dtype=float)
    if not _np.any(_np.isnan(pressure)):
        return
    solution = state0.get('aquiferSol')
    if solution:
        ids = _np.asarray([entry['num'] for entry in solution], dtype=int)
        values = _np.asarray([entry['pressure'] for entry in solution],
                             dtype=float)
        values[values < 0.0] = 0.0
        aquifer.initvals['pressures'] = values[_np.argsort(ids)]
        aquifer.initval = aquifer.initvals
        return

    values = pressure.copy()
    aqid = aquifer.aquifers[:, aquifer.aquind['aquid']].astype(int)
    cells_all = aquifer.aquifers[:, aquifer.aquind['conn']].astype(int)
    reservoir_pressure = _np.asarray(state0['pressure'], dtype=float).ravel()
    for index in range(values.size):
        cells = cells_all[aqid == index + 1]
        values[index] = _np.mean(reservoir_pressure[cells])
    aquifer.initvals['pressures'] = values
    aquifer.initval = aquifer.initvals


def _apply_fahm_swatinit_scale(model, state0, deck):
    props = deck.get('PROPS', {}) if isinstance(deck, dict) else {}
    if 'SWATINIT' not in props:
        return
    if 'pcow' not in state0:
        raise KeyError('SWATINIT requires PCOW in the first restart state')
    pressure = _np.asarray(state0['pressure'], dtype=float).ravel()
    sw = _np.asarray(state0['sW'], dtype=float).ravel()
    sg = _np.asarray(state0['sG'], dtype=float).ravel()
    p_w, p_o, _ = model._phase_pressures(
        pressure, sw, sg, pcow_scale=None)
    pcow = _np.asarray(p_o - p_w, dtype=float).ravel()
    with _np.errstate(divide='ignore', invalid='ignore'):
        multiplier = _np.asarray(state0['pcow'], dtype=float).ravel() / pcow
    multiplier[~_np.isfinite(multiplier)] = 1.0
    model.rock['pcowScale'] = multiplier


def observed_from_history(prefix, unit=None):
    """Step 3, for a deck that carries its own history vectors.

    ECLIPSE writes the observed rates a history-matching deck declares
    (WCONHIST/WCONINJH) into the summary as the ``*H`` vectors, alongside
    the simulated ones. Reading those gives the observed container without
    any separate monitoring file -- which is how a deck like QIEDIE.DATA
    is meant to be matched.
    """
    from PRSTCore.deckformat.resultinput.read_eclipse_summary import (
        _get_units, read_eclipse_summary)

    smry = read_eclipse_summary(prefix)
    u = _get_units(unit) if unit else _get_units('metric')

    names = [n for n in smry['get_names']('WOPRH') or []]
    if not names:
        names = [n for n in smry['get_names']('WBHPH') or []]
    t = smry['get'](':+:+:+:+', 'TIME')
    time = _np.asarray(t, dtype=float) * u['t'] if t is not None else _np.zeros(0)
    nt = time.size

    observed = []
    for step in range(nt):
        sol = []
        for name in names:
            akw = set(smry['get_keywords'](name))
            qO = -_col(smry, name, 'WOPRH', nt)[step] * u['ql']
            qW = -_col(smry, name, 'WWPRH', nt)[step] * u['ql']
            qG = -_col(smry, name, 'WGPRH', nt)[step] * u['qg']
            if 'WWIRH' in akw:
                qW = qW + _col(smry, name, 'WWIRH', nt)[step] * u['ql']
            if 'WGIRH' in akw:
                qG = qG + _col(smry, name, 'WGIRH', nt)[step] * u['qg']
            bhp = _col(smry, name, 'WBHPH', nt)[step] * u['p']
            total = qO + qW + qG
            sol.append({'name': name, 'qOs': qO, 'qWs': qW, 'qGs': qG,
                        'bhp': bhp, 'sign': float(_np.sign(total)) or -1.0,
                        'status': bool(abs(total) > 0 or bhp > 0)})
        observed.append({'wellSol': sol, 'dt': (time[step] - time[step - 1])
                         if step else time[step]})
    return observed, time


def simulated_from_summary(prefix, unit=None):
    """Read a run's *simulated* well solutions -- the other half of the
    mismatch."""
    from PRSTCore.deckformat.resultinput.read_eclipse_summary import \
        convert_summary_to_well_sols

    well_sols, time = convert_summary_to_well_sols(prefix, unit)
    states = []
    for sols in well_sols:
        for s in sols:
            # Derive status the same way observed_from_history does, so the
            # two sides of the mismatch agree on which wells are open --
            # expandToFull blanks a well only when the sides disagree.
            total = s['qOs'] + s['qWs'] + s['qGs']
            s['status'] = bool(abs(total) > 0 or s['bhp'] > 0)
        states.append({'wellSol': sols})
    return states, time


def run_eclipse(config, run_dir, case_name=None, timeout=3600):
    """Step 6's external call: run the simulator on ``run_dir/case``.

    Returns the case prefix the summary readers take. Raises when the run
    produced no summary -- a silent failure here would otherwise be scored
    as a perfect match.
    """
    case = case_name or config.case_name
    prefix = _os.path.join(run_dir, case)
    result = _subprocess.run([config.simulator, case], cwd=run_dir,
                             capture_output=True, text=True, timeout=timeout)
    if not _os.path.exists(prefix + '.UNSMRY'):
        raise RuntimeError(
            'ECLIPSE produced no summary for %s (exit %d). Last output:\n%s'
            % (prefix, result.returncode, (result.stdout or '')[-2000:]))
    return prefix


def prepare_run_dir(config, run_dir):
    """Copy the deck and its INCLUDE tree into a fresh run directory."""
    source = _os.path.dirname(_os.path.abspath(config.deck_path))
    _os.makedirs(run_dir, exist_ok=True)
    for entry in _os.listdir(source):
        src = _os.path.join(source, entry)
        dst = _os.path.join(run_dir, entry)
        if _os.path.isdir(src):
            if _os.path.exists(dst):
                _shutil.rmtree(dst)
            _shutil.copytree(src, dst)
        else:
            _shutil.copy2(src, dst)
    return run_dir


def mismatch(observed, simulated, schedule, weights=None,
             match_only_producers=False, *, objective_weight=None,
             normalization_factor=None, wells_weight=None, model=None):
    """Evaluate FAHM's sole objective and sum its report-step scalars.

    The App supplies the exact ``alpha``/``beta``/``omega`` dictionaries.
    ``weights`` remains only as a compatibility input for headless callers;
    it is translated into FAHM's seven alpha fields before calling
    :func:`matchObservedOWGProfile` and never selects another objective.
    """
    from PRSTCore.hm.utils.evaluate.matchObservedOWGProfile import \
        matchObservedOWGProfile

    schedule, objective_weight, normalization_factor, wells_weight = \
        _objective_inputs(
            observed, schedule, weights, match_only_producers,
            objective_weight, normalization_factor, wells_weight)
    model = model or _SummaryObjectiveModel(objective_weight)
    terms = matchObservedOWGProfile(
        model, simulated, schedule, observed,
        ObjectiveWeight=objective_weight,
        NormalizationFactor=normalization_factor,
        WellsWeight=wells_weight)
    return float(_np.sum([float(_np.sum(term)) for term in terms]))


class _SummaryObjectiveModel:
    """Minimal model surface for a rates/BHP-only summary evaluation."""

    G = {'cells': {'num': 0}}

    def __init__(self, alpha):
        if any(float(alpha[key]) != 0.0 for key in ('wt', 'wf', 'ws')):
            raise ValueError(
                'Tracer/profile/saturation objectives require reservoir '
                'states and a model, not summary-only well data')

    @staticmethod
    def getActivePhases():
        return _np.ones(3, dtype=bool)

    @staticmethod
    def getPhaseNames():
        return ['W', 'O', 'G']


_OBJECTIVE_KEYS = ('ww', 'wo', 'wg', 'wp', 'wt', 'wf', 'ws')
_OBJECTIVE_NAMES = {
    'ww': 'Water', 'wo': 'Oil', 'wg': 'Gas', 'wp': 'BHP',
    'wt': 'Tracer', 'wf': 'Profile', 'ws': 'Saturation',
}


def _legacy_alpha(weights):
    weights = weights or {}
    alpha = {
        'ww': weights.get('water', 0.0),
        'wo': weights.get('oil', 0.0),
        'wg': weights.get('gas', 0.0),
        'wp': weights.get('bhp', 0.0),
        'wt': weights.get('tracer', 0.0),
        'wf': weights.get('profile', 0.0),
        'ws': weights.get('saturation', 0.0),
    }
    out = {key: float(value) for key, value in alpha.items()}
    if not _np.all(_np.isfinite(list(out.values()))):
        raise ValueError('objective weights must be finite')
    return out


def _schedule_with_objective_wells(schedule, observed):
    """Give summary-only callers the W list FAHM's objective indexes.

    This is schedule metadata construction, not value padding: one W entry
    is created for each observed well in its existing order.  Completion
    arrays remain empty because summary-only matching cannot score profile
    or saturation terms.
    """
    out = _deepcopy(schedule)
    controls = out.get('control') or []
    if not controls:
        controls = [{}]
        out['control'] = controls
    if not (controls[-1].get('W') or []):
        if not observed or not observed[0].get('wellSol'):
            raise ValueError('observed must contain at least one well')
        controls[-1]['W'] = [
            {'name': str(well.get('name', '')), 'cells': []}
            for well in observed[0]['wellSol']
        ]
    return out


def _objective_inputs(observed, schedule, weights,
                      match_only_producers, objective_weight,
                      normalization_factor, wells_weight):
    from PRSTCore.hm.utils.observed.getNormalizationFactors import \
        getNormalizationFactors

    schedule = _schedule_with_objective_wells(schedule, observed)
    alpha = (_legacy_alpha(weights) if objective_weight is None
             else dict(objective_weight))
    beta = (getNormalizationFactors(observed)
            if normalization_factor is None else dict(normalization_factor))
    nw = len(schedule['control'][-1]['W'])
    if wells_weight is None:
        omega = {key: _np.ones(nw) for key in _OBJECTIVE_KEYS}
        # Compatibility for direct callers of the former headless helper.
        # FAHM's App path always supplies omega from its seven list triples.
        if match_only_producers:
            sol = observed[0]['wellSol']
            if len(sol) != nw:
                raise ValueError('observed well order and schedule W disagree')
            producing = _np.asarray(
                [float(well.get('sign', 0.0)) == -1.0 for well in sol])
            omega = {key: value * producing for key, value in omega.items()}
    else:
        omega = {key: _np.asarray(value, dtype=float).copy()
                 for key, value in wells_weight.items()}
    return schedule, alpha, beta, omega


def _config_objective_kwargs(config):
    return {
        'objective_weight': config.objective_weight,
        'normalization_factor': config.normalization_factor,
        'wells_weight': config.wells_weight,
    }


#: What the Plots view draws, and where each comes from. Water cut and
#: gas-oil ratio are computed from the rates rather than read: they are
#: what an engineer looks at, not what the summary stores.
PLOT_SOURCE = {'Oil': 'qOs', 'Water': 'qWs', 'Gas': 'qGs', 'BHP': 'bhp'}


def well_series(observed, simulated, time):
    """The observed and simulated curves, per well, ready to plot.

    Returns ``{'time', 'wells', 'observed', 'simulated'}`` where the last
    two are ``{curve: array of shape (nsteps, nwells)}``. Rates come
    straight from the well solutions; ``WaterCut`` and ``GOR`` are
    derived, with a zero denominator giving zero rather than a nan, so a
    shut-in step draws a gap in the curve instead of breaking the axis.
    """
    wells = [w['name'] for w in observed[0]['wellSol']] if observed else []
    time = _np.asarray(time, dtype=float).ravel()

    def gather(container):
        out = {}
        for curve, key in PLOT_SOURCE.items():
            out[curve] = _np.array(
                [[float(w.get(key, 0.0)) for w in step['wellSol']]
                 for step in container], dtype=float)
        oil, water, gas = out['Oil'], out['Water'], out['Gas']
        liquid = _np.abs(oil) + _np.abs(water)
        with _np.errstate(divide='ignore', invalid='ignore'):
            out['WaterCut'] = _np.where(liquid > 0,
                                        _np.abs(water) / _np.maximum(liquid,
                                                                     1e-300),
                                        0.0)
            out['GOR'] = _np.where(_np.abs(oil) > 0,
                                   _np.abs(gas) / _np.maximum(_np.abs(oil),
                                                              1e-300),
                                   0.0)
        return out

    return {'time': time, 'wells': wells,
            'observed': gather(observed), 'simulated': gather(simulated)}


def mismatch_by_type(observed, simulated, schedule, weights=None,
                     match_only_producers=False, per_well=False, *,
                     objective_weight=None, normalization_factor=None,
                     wells_weight=None, model=None):
    """Read all seven score families from the sole FAHM objective."""
    from PRSTCore.hm.utils.evaluate.matchObservedOWGProfile import \
        matchObservedOWGProfile

    schedule, alpha, beta, omega = _objective_inputs(
        observed, schedule, weights, match_only_producers,
        objective_weight, normalization_factor, wells_weight)
    model = model or _SummaryObjectiveModel(alpha)
    _terms, breakdown = matchObservedOWGProfile(
        model, simulated, schedule, observed,
        ObjectiveWeight=alpha, NormalizationFactor=beta,
        WellsWeight=omega, return_breakdown=True)
    nw = len(schedule['control'][-1]['W'])
    scores = {key: _np.zeros(nw) for key in _OBJECTIVE_KEYS}
    for step in breakdown:
        for key in _OBJECTIVE_KEYS:
            values = _np.asarray(step[key], dtype=float).ravel()
            if values.size != nw:
                raise ValueError('%s breakdown has width %d; expected %d'
                                 % (key, values.size, nw))
            scores[key] += values
    named = {_OBJECTIVE_NAMES[key]: value for key, value in scores.items()}
    if per_well:
        return named
    return {name: float(_np.sum(value)) for name, value in named.items()}


def run_forward(config, run_dir=None, keep=True):
    """One full round trip: write, simulate, read, score.

    This is the inner evaluation an optimiser would call, and on its own
    it is the end-to-end test of the whole chain.
    """
    run_dir = run_dir or _os.path.join(config.work_dir, 'run')
    prepare_run_dir(config, run_dir)

    case = _os.path.splitext(_os.path.basename(config.deck_path))[0]
    prefix = run_eclipse(config, run_dir, case_name=case)

    observed, time = observed_from_history(prefix)
    simulated, _ = simulated_from_summary(prefix)

    # A summary's first record is the initial state at TIME=0, not a report
    # step: nothing has flowed yet, so every well has zero liquid and the
    # water-cut ratio is 0/0. Drop it, leaving dt = diff(time).
    if time.size and time[0] == 0.0:
        observed, simulated, time = observed[1:], simulated[1:], time[1:]
    dt = _np.diff(_np.concatenate([[0.0], time]))

    schedule = {'step': {'val': dt, 'control': _np.zeros(dt.size, dtype=int)},
                'control': [{'W': []}]}

    value = mismatch(observed, simulated, schedule, config.weights,
                     config.match_only_producers,
                     **_config_objective_kwargs(config))
    return {'misfit': value, 'prefix': prefix, 'observed': observed,
            'simulated': simulated, 'time': time, 'schedule': schedule}


class StoppedByUser(RuntimeError):
    """Raised inside the objective when the caller asked the loop to stop."""


def run_history_match(config, u0=None, gradient_step=0.05, verbose=True,
                      should_stop=None, gradient='adjoint'):
    """The whole loop -- FAHM's run block, headless.

    Scores the unperturbed case first, then optimises the multipliers
    against that baseline so the objective starts at 1.0.

    ``gradient`` picks how the derivative is obtained:

    ``'adjoint'``
        The default. One simulation per evaluation. The external
        simulator produces states; PRSTCore rebuilds the residual at
        them and sweeps an adjoint backwards for every parameter at
        once. Cost is independent of how many parameters are tuned.

    ``'fd'``
        One extra simulation per parameter per iteration. Kept because
        it needs nothing from the model -- a deck PRSTCore cannot build
        a model from can still be matched this way, and it is the
        reference the adjoint is checked against.

    ``should_stop`` is polled before each simulator run. This is what the
    Terminate button uses: a run cannot be interrupted safely once it is
    writing its output files, so the loop stops between evaluations
    rather than killing the process.
    """
    from PRSTCore.hm.utils.optimizer.optimizeBoundConstrainedForFAHM import \
        optimizeBoundConstrainedForFAHM

    names = list(config.parameters)
    if not names:
        raise ValueError('No parameters selected to tune.')
    if gradient not in ('adjoint', 'fd'):
        raise ValueError("gradient must be 'adjoint' or 'fd', not %r"
                         % gradient)

    base = run_forward(config, run_dir=_os.path.join(config.work_dir, 'base'))
    if gradient == 'adjoint':
        f = objective = make_adjoint_objective(
            config, base_misfit=base['misfit'], should_stop=should_stop)
    else:
        objective = make_objective(config, base_misfit=base['misfit'],
                                   should_stop=should_stop)
        f = with_finite_difference_gradient(objective, gradient_step)

    if u0 is None:
        # Start where the multiplier is 1.0, i.e. the untouched deck.
        lims = _np.array([config.limits_for(n) for n in names], dtype=float)
        u0 = (1.0 - lims[:, 0]) / (lims[:, 1] - lims[:, 0])
    u0 = _np.clip(_np.asarray(u0, dtype=float).ravel(), 0.0, 1.0)

    try:
        v, u, history = optimizeBoundConstrainedForFAHM(
            u0, f, {'work': config.work_dir},
            maximize=False,                # FAHM defaults to maximizing
            maxIt=config.max_iterations, objChangeTol=1e-10, gradTol=1e-6,
            lbfgsStrategy='dynamic', lbfgsNum=10, lineSearchMaxIt=5,
            verbose=verbose)
    except StoppedByUser:
        # Report what was reached rather than discarding it: the case
        # directories and the checkpointed history are on disk either way.
        history = _load_history(config.work_dir)
        u = history['u'][-1] if history['u'] else u0
        v = history['val'][-1] if history['val'] else float('nan')

    # The baseline run already holds everything the Mismatch Scores panel
    # needs, so the breakdown costs no further simulator run.
    scores = mismatch_by_type(base['observed'], base['simulated'],
                              base['schedule'], config.weights,
                              config.match_only_producers,
                              **_config_objective_kwargs(config))
    per_well = mismatch_by_type(base['observed'], base['simulated'],
                                base['schedule'], config.weights,
                                config.match_only_producers, per_well=True,
                                **_config_objective_kwargs(config))

    return {'value': v, 'u': u, 'multipliers': dict(zip(names,
                                                        objective.unscale(u))),
            'baseline': base['misfit'], 'history': history,
            'scores': scores, 'well_scores': per_well,
            'wells': [w['name'] for w in base['observed'][0]['wellSol']],
            'series': well_series(base['observed'], base['simulated'],
                                  base['time'])}


def _load_history(work_dir):
    """Read back the checkpoint optimizeBoundConstrainedForFAHM writes."""
    path = _os.path.join(work_dir, 'history.npz')
    if not _os.path.exists(path):
        return {'val': [], 'u': [], 'pg': []}
    saved = _np.load(path)
    return {'val': list(saved['val']), 'u': list(saved['u']),
            'pg': list(saved['pg']) if 'pg' in saved else []}


# The deck keyword each tunable parameter multiplies. Pore volume goes
# through MULTPV rather than PORO so it scales volume without also moving
# the saturation functions' porosity dependence.
_MULTIPLY_TARGET = {
    'porevolume': 'MULTPV', 'permx': 'PERMX', 'permy': 'PERMY',
    'permz': 'PERMZ', 'poro': 'PORO',
}

#: The saturation-function parameters, and the PROPS array each one is.
#: These cannot go through MULTIPLY: the arrays are optional and a deck
#: that leaves its endpoints to the saturation table has nothing there to
#: multiply. Each is written out explicitly instead, at the table's own
#: point times the tuned factor.
_ENDPOINT_KEYWORD = {
    'swl': 'SWL', 'swcr': 'SWCR', 'swu': 'SWU',
    'sgl': 'SGL', 'sgcr': 'SGCR', 'sgu': 'SGU',
    'sowcr': 'SOWCR', 'sogcr': 'SOGCR',
    'krw': 'KRW', 'kro': 'KRO', 'krg': 'KRG',
}

#: The endpoints that are saturations, and so must stay within [0, 1].
#: The three relative permeabilities are not bounded above by one --
#: ECLIPSE accepts a maximum relperm greater than unity.
_SATURATION_ENDPOINT = set(_ENDPOINT_KEYWORD) - {'krw', 'kro', 'krg'}

_SECTIONS = ('RUNSPEC', 'GRID', 'EDIT', 'PROPS', 'REGIONS', 'SOLUTION',
             'SUMMARY', 'SCHEDULE', 'END')


def apply_multipliers(deck_text, multipliers, endpoints=None, ncells=None):
    """Return the deck with the tuned parameters written into it.

    Writing the tuned properties this way rather than regenerating the
    whole deck is deliberate. PRSTCore's reader captures 35 of QIEDIE's
    123 keywords, and the 88 it drops include the entire SUMMARY section
    -- which is where the observed WOPRH/WWPRH history lives. Round-
    tripping the deck through the parser would therefore delete the very
    data being matched. Overlaying leaves the original bytes untouched
    and changes only what a parameter is supposed to change.

    ``multipliers`` maps a parameter name to its factor; a factor of 1.0
    is skipped, since applying one only adds noise to the deck.

    Rock properties go into a MULTIPLY block at the end of GRID. The
    saturation-function endpoints cannot: they are optional PROPS arrays
    and a deck that leaves them to its saturation table has nothing to
    multiply. Those are written out in full instead, which needs
    ``endpoints`` -- the table's own point per parameter, from
    :func:`PRSTCore.ad_props.kr_points.get_kr_points` -- and ``ncells``.
    Endscaling must also be enabled for ECLIPSE to read them, so
    ``ENDSCALE`` is added to RUNSPEC when the deck does not already
    declare it.

    A caution the parameterisation carries rather than this function:
    the endpoints are scaled independently, so nothing stops a factor
    combination from pushing Swcr past Swu. ECLIPSE rejects such a deck.
    MRST imposes no ordering either, and none is invented here -- keep
    the limits narrow enough that the order survives.
    """
    rows, arrays = [], []
    for name, factor in multipliers.items():
        key = str(name).lower()
        factor = float(factor)

        if key in _MULTIPLY_TARGET:
            if factor != 1.0:
                rows.append("  %-8s %.10g  /"
                            % (_MULTIPLY_TARGET[key], factor))
            continue

        if key in _ENDPOINT_KEYWORD:
            if factor == 1.0:
                continue
            base = (endpoints or {}).get(key)
            if base is None:
                raise ValueError(
                    'Tuning %r needs its saturation-table value; pass '
                    'endpoints=%s.' % (name, 'get_kr_points(...)'))
            if not ncells:
                raise ValueError('Writing %r needs the cell count.'
                                 % _ENDPOINT_KEYWORD[key])
            value = float(base) * factor
            if key in _SATURATION_ENDPOINT:
                value = min(max(value, 0.0), 1.0)
            arrays.append('%s\n  %d*%.10g  /\n'
                          % (_ENDPOINT_KEYWORD[key], int(ncells), value))
            continue

        raise ValueError('No deck keyword known for parameter %r; add it '
                         'to _MULTIPLY_TARGET or _ENDPOINT_KEYWORD.' % name)

    if rows:
        # MULTIPLY needs the array it scales to exist already, so this
        # goes at the end of GRID rather than the start.
        block = ('\n-- history-match multipliers, written by PRSTCore FAHM\n'
                 'MULTIPLY\n' + '\n'.join(rows) + '\n/\n')
        deck_text = _append_to_section(deck_text, 'GRID', block)

    if arrays:
        block = ('\n-- history-match endpoints, written by PRSTCore FAHM\n'
                 + ''.join(arrays))
        deck_text = _append_to_section(deck_text, 'PROPS', block)
        deck_text = _ensure_endscale(deck_text)

    return deck_text


def _append_to_section(deck_text, section, block):
    """Insert ``block`` at the end of ``section``, i.e. immediately before
    whatever section header comes next."""
    lines = deck_text.splitlines()
    inside = False
    for i, line in enumerate(lines):
        head = line.strip().upper()
        if head == section:
            inside = True
        elif inside and head in _SECTIONS:
            return '\n'.join(lines[:i]) + block + '\n'.join(lines[i:])
    if not inside:
        raise ValueError('Deck has no %s section to write into.' % section)
    return deck_text + block


def ensure_restart_output(deck_text, frequency=2):
    """Make the run emit a restart record at every report step.

    The adjoint differentiates the residual at each step, so it needs the
    cell state -- pressure, saturations, Rs -- that only the restart file
    carries. A summary alone gives well rates and nothing to assemble a
    Jacobian at. A deck written for history matching has no reason to ask
    for restarts, so this adds the request rather than requiring the user
    to.

    ``BASIC=2`` is one record per report step, which is exactly the set
    of states the adjoint sweeps back through. ``UNIFOUT`` collects them
    into a single ``.UNRST``; without it ECLIPSE writes one file per
    step, which the reader also accepts but which litters the run
    directory.
    """
    lines = deck_text.splitlines()
    for line in lines:
        if line.strip().upper().startswith('RPTRST'):
            return deck_text
    return _append_to_section(
        deck_text, 'SOLUTION',
        '\n-- restart records, required by the adjoint gradient\n'
        'RPTRST\n  BASIC=%d  /\n' % int(frequency))


def _ensure_endscale(deck_text):
    """ECLIPSE ignores the endpoint arrays unless endscaling is on, and
    the deck being matched need not have asked for it."""
    lines = deck_text.splitlines()
    for line in lines:
        head = line.strip().upper()
        if head.startswith('ENDSCALE'):
            return deck_text
        if head == 'GRID':
            break
    return _append_to_section(
        deck_text, 'RUNSPEC',
        '\n-- endpoint scaling, required by the tuned endpoints below\n'
        'ENDSCALE\n/\n')


#: Which ``krPts`` curve and column each endpoint parameter reads. The
#: four columns are [connate, critical, max-saturation, max-relperm].
_ENDPOINT_SOURCE = {
    'swl': ('w', 0), 'swcr': ('w', 1), 'swu': ('w', 2), 'krw': ('w', 3),
    'sgl': ('g', 0), 'sgcr': ('g', 1), 'sgu': ('g', 2), 'krg': ('g', 3),
    'sowcr': ('ow', 1), 'kro': ('ow', 3), 'sogcr': ('og', 1),
}


def read_endpoints(deck_path):
    """The saturation table's own endpoints, and the deck's cell count.

    Returns ``(endpoints, ncells)``. The endpoints are what an untuned
    deck would use, so a factor of 1.0 reproduces the original curves --
    which is what makes them a sound thing to multiply.

    A multi-region deck reports its first region: the parameterisation
    carries one factor per parameter, not one per region, so a single
    base value is all a factor can be applied to. The Parameter tab's
    per-region table shows the regions; the run collapses them, which is
    the same narrowing ``FahmConfig`` already does to the limits.
    """
    from PRSTCore.ad_props.kr_points import get_kr_points
    from PRSTCore.deckformat.deckinput.convert_deck_units import \
        convert_deck_units
    from PRSTCore.deckformat.deckinput.read_eclipse_deck import \
        read_eclipse_deck

    deck = convert_deck_units(read_eclipse_deck(deck_path)) or {}
    points = get_kr_points(deck.get('PROPS', {}))

    endpoints = {}
    for name, (curve, column) in _ENDPOINT_SOURCE.items():
        table = points.get(curve)
        if table is not None and len(table):
            endpoints[name] = float(_np.asarray(table)[0, column])

    dims = (deck.get('RUNSPEC') or {}).get('cartDims')
    ncells = int(_np.prod(_np.asarray(dims, dtype=int))) if dims is not None \
        else None
    return endpoints, ncells


def make_objective(config, run_dir=None, base_misfit=None, should_stop=None):
    """Build ``objective(u)`` for the optimiser.

    ``u`` is in the unit box; each entry is unscaled through its
    parameter's box limits into a multiplier, written into the deck, and
    scored by a full simulator run. Returns the misfit relative to
    ``base_misfit`` when given, so the optimiser sees an O(1) objective.
    """
    names = list(config.parameters)
    lims = _np.array([config.limits_for(n) for n in names], dtype=float)

    # The saturation-function endpoints are written out at the table's
    # own value times the factor, so those values are needed once up
    # front. Read here rather than per evaluation: they do not change.
    endpoints, ncells = ({}, None)
    if any(n in _ENDPOINT_KEYWORD for n in names):
        endpoints, ncells = read_endpoints(config.deck_path)

    def unscale(u):
        u = _np.clip(_np.asarray(u, dtype=float).ravel(), 0.0, 1.0)
        return lims[:, 0] + u * (lims[:, 1] - lims[:, 0])

    def objective(u, case_dir=None):
        if should_stop is not None and should_stop():
            raise StoppedByUser('history match stopped before evaluation')
        factors = unscale(u)
        target = case_dir or run_dir or _os.path.join(config.work_dir, 'run')
        prepare_run_dir(config, target)

        case = _os.path.splitext(_os.path.basename(config.deck_path))[0]
        deck_file = _os.path.join(target, _os.path.basename(config.deck_path))
        with open(deck_file, 'r', encoding='utf-8', errors='replace') as fh:
            text = fh.read()
        text = apply_multipliers(text, dict(zip(names, factors)),
                                 endpoints=endpoints, ncells=ncells)
        with open(deck_file, 'w', encoding='utf-8') as fh:
            fh.write(text)

        prefix = run_eclipse(config, target, case_name=case)
        observed, time = observed_from_history(prefix)
        simulated, _ = simulated_from_summary(prefix)
        if time.size and time[0] == 0.0:
            observed, simulated, time = observed[1:], simulated[1:], time[1:]
        dt = _np.diff(_np.concatenate([[0.0], time]))
        schedule = {'step': {'val': dt,
                             'control': _np.zeros(dt.size, dtype=int)},
                    'control': [{'W': []}]}
        value = mismatch(observed, simulated, schedule, config.weights,
                         config.match_only_producers,
                         **_config_objective_kwargs(config))
        return value / base_misfit if base_misfit else value

    objective.unscale = unscale
    objective.names = names
    return objective


def make_adjoint_objective(config, model=None, run_dir=None, base_misfit=None,
                           should_stop=None, linear_solver=None):
    """``f(u) -> (value, gradient)`` with the gradient from an adjoint.

    The loop this belongs to is FAHM's own, and the point of it is that
    the forward simulation stays with ECLIPSE or tNavigator while the
    derivatives come from PRSTCore:

    1. unscale ``u`` into one multiplier per parameter and write them
       into the deck -- MULTIPLY for the rock, explicit PROPS arrays for
       the saturation endpoints;
    2. run the external simulator;
    3. read its **restart** back as states, and its summary as well
       solutions -- the restart is what makes this possible at all,
       because the adjoint needs the cell state at every report step and
       a summary does not carry it;
    4. rebuild the residual and its Jacobians at those states with
       PRSTCore's own assembly, sweep the adjoint backwards, and contract
       the resulting field sensitivities onto the multipliers.

    So the simulator never has to supply a derivative. It supplies
    states; the derivatives are ours. That is what makes an adjoint
    possible against a closed-source simulator, and it is why the cost
    does not grow with the number of tuned parameters -- one simulation
    per iteration instead of one per parameter per iteration.

    ``model`` is built once by the caller and reused: it costs half a
    minute on a field-sized deck and nothing about it changes between
    evaluations except the operators this function scales.
    """
    from PRSTCore.ad_core.simulators.adjoint_sweep import adjoint_gradient
    from PRSTCore.hm.utils.evaluate.getEclipseSimResults import \
        getEclipseSimResults

    names = list(config.parameters)
    lims = _np.array([config.limits_for(n) for n in names], dtype=float)

    endpoints, ncells = ({}, None)
    if any(n in _ENDPOINT_KEYWORD for n in names):
        endpoints, ncells = read_endpoints(config.deck_path)

    if model is None:
        model = build_model(config)
    targets = sorted({_ADJOINT_TARGET.get(n, n) for n in names})

    def unscale(u):
        u = _np.clip(_np.asarray(u, dtype=float).ravel(), 0.0, 1.0)
        return lims[:, 0] + u * (lims[:, 1] - lims[:, 0])

    def objective(u, case_dir=None):
        if should_stop is not None and should_stop():
            raise StoppedByUser('history match stopped before evaluation')
        factors = dict(zip(names, unscale(u)))
        target = case_dir or run_dir or _os.path.join(config.work_dir, 'run')
        prepare_run_dir(config, target)

        case = _os.path.splitext(_os.path.basename(config.deck_path))[0]
        deck_file = _os.path.join(target, _os.path.basename(config.deck_path))
        with open(deck_file, 'r', encoding='utf-8', errors='replace') as fh:
            text = fh.read()
        text = apply_multipliers(text, factors, endpoints=endpoints,
                                 ncells=ncells)
        text = ensure_restart_output(text)
        with open(deck_file, 'w', encoding='utf-8') as fh:
            fh.write(text)

        prefix = run_eclipse(config, target, case_name=case)

        observed, time = observed_from_history(prefix)
        simulated, _ = simulated_from_summary(prefix)
        if time.size and time[0] == 0.0:
            observed, simulated, time = observed[1:], simulated[1:], time[1:]
        dt = _np.diff(_np.concatenate([[0.0], time]))
        schedule = {'step': {'val': dt,
                             'control': _np.zeros(dt.size, dtype=int)},
                    'control': [{'W': []}]}
        value = mismatch(observed, simulated, schedule, config.weights,
                         config.match_only_producers,
                         **_config_objective_kwargs(config))
        scale = base_misfit or 1.0

        # MRST's own reader, not convert_restart_to_states directly. It
        # reorders each restart record's well solutions into the
        # schedule's well order by name and its perforations by cell,
        # selects the records that fall on report times, and splits off
        # state0. Everything downstream then aligns by index, which is
        # what the facility model's active-well mask assumes -- doing the
        # matching per step instead, as this used to, re-derives at every
        # step what the reader establishes once.
        setup = {'model': model,
                 'schedule': (model.inputdata or {}).get('_schedule'),
                 'state0': None}
        states, _wellsols, setup = getEclipseSimResults(
            target, case, setup)
        if len(states) < 1:
            raise RuntimeError(
                'The run wrote no restart record at a report time; the '
                'adjoint needs one per step. Check that RPTRST survived '
                'into the deck.')
        state_first = setup['state0']

        with scaled_operators(model, factors):
            steps = min(len(states), dt.size)
            # One set of driving forces per step: a history-matching deck
            # restates WCONHIST at every report step, so the well targets
            # -- and which wells are open at all -- change under the
            # adjoint's feet. Differentiating all 63 steps against the
            # first step's controls would be a gradient for a schedule
            # that was never run.
            forces = _forces_per_step(model, steps)
            partials = _objective_partials(model, observed, schedule,
                                           config, scale, forces)
            fields = adjoint_gradient(model, state_first, states[:steps],
                                      dt[:steps], forces, targets, partials,
                                      linear_solver=linear_solver)
            gradient = multiplier_gradient(model, fields, names)

        span = lims[:, 1] - lims[:, 0]
        # u is the unit-box coordinate; the optimiser wants dJ/du, and
        # the multiplier is an affine function of it.
        return value / scale, _np.array([gradient[n] for n in names]) * span

    objective.unscale = unscale
    objective.names = names
    objective.model = model
    return objective


def _forces_per_step(model, steps):
    """Driving forces for each report step, from the real schedule."""
    schedule = (model.inputdata or {}).get('_schedule') or {}
    controls = schedule.get('control') or [{'W': []}]
    index = _np.asarray(schedule.get('step', {}).get(
        'control', _np.zeros(steps, dtype=int)), dtype=int).ravel()
    out = []
    for n in range(steps):
        which = int(index[n]) if n < index.size else 0
        out.append(model.getDrivingForces(
            controls[min(which, len(controls) - 1)]))
    return out


def _objective_partials(model, observed, schedule, config, scale, forces):
    """``dg_n/dx_n`` for the mismatch, as the adjoint sweep wants it.

    The same objective the loop minimises, asked for its derivative
    rather than its value. ``matchObservedOWGProfile`` reads the quantities
    through ``FacilityModel.getProp``; given a state whose facility
    variables carry derivatives, the scalar it returns carries the whole
    partial in its Jacobian row. One expression of the objective serves
    both the value and the derivative -- keeping a second,
    hand-differentiated one is how the two drift apart.

    Two alignments have to hold for that comparison to mean anything.
    The adjoint's unknown vector contains only the wells that are *open*
    at that step, while the summary reports every well the deck ever
    names, so the observed set is restricted and reordered to the open
    ones by name. And the restart states carry well solutions but none
    of the primary facility variables, so those are seeded first --
    without them a rate or bhp objective, whose derivative lives
    entirely in the well block, returns a zero gradient that reads as
    converged.
    """
    from PRSTCore.hm.utils.evaluate.matchObservedOWGProfile import \
        matchObservedOWGProfile

    objective_schedule, alpha, beta, omega = _objective_inputs(
        observed, schedule, config.weights, config.match_only_producers,
        config.objective_weight, config.normalization_factor,
        config.wells_weight)
    dts = _np.asarray(objective_schedule['step']['val'], dtype=float).ravel()
    all_wells = objective_schedule['control'][-1]['W']
    well_index = {str(well.get('name', '')): index
                  for index, well in enumerate(all_wells)}

    def partials(step, state):
        forces_n = forces[step] if isinstance(forces, list) else forces
        active_wells = list(model._mrst_active_wells(forces_n))
        active = [str(w['name']) for w in active_wells]
        if not active_wells:
            return _np.zeros(1)
        if len(set(active)) != len(active):
            raise ValueError('active well names must be unique')
        missing = [name for name in active if name not in well_index]
        if missing:
            raise ValueError('active wells absent from objective schedule: %s'
                             % ', '.join(missing))

        state = _seeded_state(model, state, forces_n, active)
        observed_n = [
            dict(entry, wellSol=_by_name(entry['wellSol'], active))
            for entry in observed
        ]
        model_schedule = (model.inputdata or {}).get('_schedule', {})
        model_controls = model_schedule.get('control') or [{}]
        model_mapping = _np.asarray(
            model_schedule.get('step', {}).get(
                'control', _np.zeros(dts.size, dtype=int)),
            dtype=int).ravel()
        control_index = int(model_mapping[step]) \
            if step < model_mapping.size else 0
        if control_index < 0 or control_index >= len(model_controls):
            raise IndexError('model schedule control index is invalid')
        current_control = _deepcopy(model_controls[control_index])
        current_control['W'] = _deepcopy(active_wells)
        partial_schedule = {
            'step': {'val': dts.copy(),
                     'control': _np.zeros(dts.size, dtype=int)},
            'control': [current_control],
        }
        take = _np.asarray([well_index[name] for name in active], dtype=int)
        omega_n = {key: _np.asarray(value, dtype=float).ravel()[take]
                   for key, value in omega.items()}
        terms = matchObservedOWGProfile(
            model, None, partial_schedule, observed_n,
            ObjectiveWeight=alpha, NormalizationFactor=beta,
            WellsWeight=omega_n, ComputePartials=True, tStep=step,
            state=state, from_states=False)
        return _partial_row(terms[0], scale)

    return partials


def _by_name(sols, names):
    """The well solutions for ``names``, in that order, all marked open.

    The open flag is set on every row deliberately. ``matchObserved*``
    otherwise reaches for ``expandToFull``, which scatters an open-well
    subset back across the full well list by *position* -- the alignment
    this function has already done by name, and done better, since a
    positional scatter cannot tell two wells apart if one of them is
    missing. Both sides here are the same wells in the same order, so
    there is nothing left to scatter; a well shut in the history simply
    arrives with the zero rates the summary reports for it, which is a
    real observation and not an absent one.
    """
    index = {str(w.get('name')): w for w in sols}
    out = []
    for name in names:
        found = index.get(name)
        if found is None:
            raise ValueError('well %r is missing from wellSol' % name)
        row = dict(found)
        row['status'] = True
        out.append(row)
    return out


def _seeded_state(model, state, forces, active):
    """A restart state with its primary variables carrying derivatives.

    A restart record holds pressure, saturations, Rs and the well
    solutions, but not the facility primaries the assembly seeds, so
    those are filled in first -- restricted to the wells that are open,
    because those are the ones the adjoint's unknown vector contains.
    """
    state = dict(state)
    sols = _by_name(state.get('wellSol') or [], active)
    state['wellSol'] = sols
    for name in ('qWs', 'qOs', 'qGs', 'bhp'):
        state['facility_' + name] = _np.asarray(
            [float(w.get(name, 0.0)) for w in sols], dtype=float)
    return model.getStateAD(state, True, drivingForces=forces)


def _partial_row(term, scale):
    """The Jacobian row of an AD scalar, as a dense vector."""
    jac = getattr(term, 'jac', None)
    if jac is None:
        return _np.zeros(1)
    dense = _np.asarray(jac.todense()).ravel() if hasattr(jac, 'todense') \
        else _np.asarray(jac).ravel()
    return dense / (scale or 1.0)


class scaled_operators:
    """Apply the tuned multipliers to a model, then put it back.

    The adjoint has to differentiate the model the simulator actually
    ran, which is the base model with these factors applied. Scaling the
    operators in place is far cheaper than rebuilding from the tuned
    deck -- half a minute per evaluation on a field-sized case -- and it
    is the same thing MRST's ``ModelParameter.setParameter`` does.
    """

    def __init__(self, model, factors):
        self.model = model
        self.factors = factors
        self.saved = {}

    def __enter__(self):
        model, factors = self.model, self.factors
        pv = factors.get('porevolume', 1.0)
        if pv != 1.0:
            self.saved['pv'] = model.porevolume
            model.porevolume = _np.asarray(
                model._porevolume_vector(), dtype=float).ravel() * pv

        perm = [(n, factors.get(n, 1.0)) for n in _PERM_AXIS]
        if any(f != 1.0 for _n, f in perm):
            base = _np.asarray(model.operators['T'], dtype=float).ravel()
            self.saved['T'] = model.operators['T']
            scale = _np.ones(base.size)
            for name, factor in perm:
                if factor == 1.0:
                    continue
                mask = _face_axis_mask(model, _PERM_AXIS[name], base.size)
                scale = scale * (1.0 + (factor - 1.0) * mask)
            model.operators['T'] = base * scale

        endpoints = [(n, f) for n, f in factors.items()
                     if n in _ENDPOINT_KEYWORD and f != 1.0]
        if endpoints:
            from PRSTCore.ad_core.simulators.adjoint_verification import \
                ENDPOINT_COLUMNS
            scaling = model._get_relperm_scaling(
                int(model.G['cells']['num']), model._get_relperm_tables())
            if scaling is not None:
                self.saved['target'] = {k: v.copy() for k, v
                                        in scaling['target'].items()}
                self.saved['scaling'] = scaling
                for name, factor in endpoints:
                    for phase, column in ENDPOINT_COLUMNS[name]:
                        scaling['target'][phase][:, column] *= factor
        return self

    def __exit__(self, *_exc):
        model = self.model
        if 'pv' in self.saved:
            model.porevolume = self.saved['pv']
        if 'T' in self.saved:
            model.operators['T'] = self.saved['T']
        if 'target' in self.saved:
            for phase, table in self.saved['target'].items():
                self.saved['scaling']['target'][phase][...] = table
        return False


def build_model(config):
    """The PRSTCore model the adjoint differentiates, from the base deck."""
    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import \
        init_eclipse_problem_ad
    _state0, model, schedule, _ = init_eclipse_problem_ad(config.deck_path)
    model.inputdata = dict(model.inputdata or {})
    # The whole schedule, not just its first control: a history-matching
    # deck restates WCONHIST every report step, so the adjoint needs to
    # know which controls were in force at each of them.
    model.inputdata['_schedule'] = schedule
    return model


#: What each tuned parameter multiplies inside the model, and which
#: quantity the adjoint differentiates for it. Permeability has no
#: derivative path of its own: it reaches the residual only through the
#: face transmissibilities, so its gradient is read off ``dJ/dT``.
_ADJOINT_TARGET = {
    'porevolume': 'porevolume',
    'permx': 'transmissibility',
    'permy': 'transmissibility',
    'permz': 'transmissibility',
}

#: The axis tag each permeability owns. PRSTCore's corner-point grids
#: label every face 1, 2 or 3 for I, J or K -- already the axis, not
#: MRST's six signed half-face tags.
_PERM_AXIS = {'permx': 1, 'permy': 2, 'permz': 3}


def multiplier_gradient(model, gradients, names):
    """Turn per-cell and per-face sensitivities into one number per
    tuned multiplier.

    FAHM tunes a single factor per parameter, applied to the whole
    field: ``p = m * p_base``. So ``dp_i/dm = p_base_i`` and the
    multiplier's derivative is the field gradient contracted with the
    field it scales -- an exact chain rule, not an approximation, and
    the reason no extra simulation is needed to get it.

    Permeability is the one that is not a plain contraction. It does not
    appear in the residual at all; it reaches it through the face
    transmissibilities. Scaling PERMX by ``m`` scales both half
    transmissibilities of every I-face by exactly ``m``, and leaves J-
    and K-faces untouched -- a harmonic mean of two quantities that both
    scale by ``m`` scales by ``m``. So ``dT_f/dm`` is ``T_f`` on that
    direction's faces and zero elsewhere, and the contraction runs over
    those faces only. Exact again, and it is why the axis has to be
    known rather than assumed.
    """
    out = {}
    for name in names:
        target = _ADJOINT_TARGET.get(name, name)
        field_gradient = gradients.get(target)
        if field_gradient is None:
            out[name] = 0.0
            continue
        field_gradient = _np.asarray(field_gradient, dtype=float).ravel()

        if name in _PERM_AXIS:
            base = _np.asarray(model.operators['T'], dtype=float).ravel()
            mask = _face_axis_mask(model, _PERM_AXIS[name], base.size)
        elif name == 'porevolume':
            base = _np.asarray(model._porevolume_vector(),
                               dtype=float).ravel()
            mask = None
        else:
            from PRSTCore.ad_core.simulators.adjoint_verification import \
                endpoint_base
            base = endpoint_base(model, name)
            mask = None
            if base is None:
                out[name] = 0.0
                continue

        n = min(field_gradient.size, base.size)
        product = field_gradient[:n] * base[:n]
        if mask is not None:
            product = product * mask[:n]
        out[name] = float(_np.sum(product))
    return out


def _face_axis_mask(model, axis, nfaces):
    """1.0 on the internal faces belonging to ``axis`` (1=I, 2=J, 3=K).

    ``operators['T']`` is indexed by *internal* face while ``faces.tag``
    is indexed by every face, so the tags are restricted to the internal
    ones before they line up. On QIEDIE's 52x52x20 that gives 53040 I-,
    53040 J- and 51376 K-faces, which is exactly the count a Cartesian
    grid of that size has -- a cheap way to know the two orderings agree.

    A grid with no tags leaves every face in. That overstates a
    directional permeability's gradient, which is wrong but visibly so;
    silently returning zero would read as a converged direction.
    """
    faces = model.G.get('faces', {})
    tag = faces.get('tag')
    neighbors = faces.get('neighbors')
    if tag is None or neighbors is None:
        return _np.ones(nfaces)

    tag = _np.asarray(tag).ravel()
    neighbors = _np.asarray(neighbors, dtype=int)
    if neighbors.ndim != 2 or neighbors.shape[1] != 2:
        return _np.ones(nfaces)

    internal = _np.flatnonzero((neighbors[:, 0] >= 0)
                               & (neighbors[:, 1] >= 0))
    if internal.size != nfaces or tag.size != neighbors.shape[0]:
        return _np.ones(nfaces)
    return (tag[internal] == int(axis)).astype(float)


def with_finite_difference_gradient(objective, h=0.05):
    """Wrap a value-only objective so it also returns a gradient.

    FAHM gets its gradient from an adjoint through the PRSTCore model.
    That path needs the model built from the deck, which QIEDIE's reader
    coverage does not currently support, so this uses one-sided
    differences instead: ``n+1`` simulator runs per gradient rather than
    two. The step defaults to 5% of the unit box, large enough that the
    difference is not swamped by the simulator's own convergence noise.
    """
    def f(u, *args):
        u = _np.asarray(u, dtype=float).ravel()
        v0 = objective(u, *args)
        g = _np.zeros(u.size)
        for k in range(u.size):
            step = -h if u[k] + h > 1.0 else h
            up = u.copy()
            up[k] += step
            g[k] = (objective(up, *args) - v0) / step
        return v0, g
    return f


def _col(smry, name, keyword, nt):
    values = smry['get'](name, keyword)
    if values is None:
        return _np.zeros(nt)
    values = _np.asarray(values, dtype=float).ravel()
    if values.size >= nt:
        return values[:nt]
    out = _np.zeros(nt)
    out[:values.size] = values
    return out
