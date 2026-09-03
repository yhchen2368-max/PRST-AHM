"""Stage 4 parity tests for FAHM Create Project's base-case preflight."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from PRSTCore.hm.APP.fahm import (
    BaseCaseResult,
    MissingBaseCaseOutputError,
    UnsupportedSimulatorError,
    base_case_paths,
    build_base_case_command,
    create_base_case,
)


ROOT = Path(__file__).resolve().parent
ORACLE = ROOT / 'fixtures' / 'fahm_oracle' / 'v1'
CONTRACT = ORACLE / 'stage4' / 'create_project_contract.json'
FINGERPRINT = ORACLE / 'static' / 'source_fingerprint.json'


def _schedule():
    return {
        'control': [{
            'WELSPECS': [[
                'P1', 'G1', 1, 1, 2000.0, 'OIL', 0, 'STD', 'SHUT',
                'YES', 0, 'SEG', 0,
            ]],
            'COMPDAT': [[
                'P1', 1, 1, 1, 1, 'OPEN', -1, 1e-12, 0.2, 0.0, 0.0,
                'Default', 'Z', 0.0,
            ]],
            'WCONPROD': [[
                'P1', 'OPEN', 'BHP', np.inf, np.inf, np.inf, np.inf,
                np.inf, 100.0, np.nan, 0, 0,
            ]],
        }],
        'step': {'val': [1.0, 2.0], 'control': [0, 0]},
    }


def _deck():
    nx, ny, nz = 2, 2, 1
    return {
        'RUNSPEC': {
            'cartDims': [nx, ny, nz], 'DIMENS': [nx, ny, nz],
            'METRIC': True, 'OIL': True, 'WATER': True,
            'TITLE': 'STAGE4',
        },
        'GRID': {
            'cartDims': [nx, ny, nz],
            'PERMX': np.array([100.0, 200.0, 300.0, 400.0]),
            'PORO': np.array([0.1, 0.2, 0.3, 0.4]),
            'ACTNUM': np.array([1, 1, 1, 1]),
        },
        'PROPS': {}, 'REGIONS': {}, 'SOLUTION': {},
        'SCHEDULE': _schedule(),
    }


def _fake_executable(tmp_path: Path, name: str) -> Path:
    """Install an actual subprocess that records argv/cwd and emits files."""
    bindir = tmp_path / 'fake bin'
    bindir.mkdir(exist_ok=True)
    helper = bindir / 'fake_fahm_simulator.py'
    helper.write_text(
        """\
import json
import os
from pathlib import Path
import sys

Path(os.environ['FAHM_FAKE_RECORD']).write_text(json.dumps({
    'argv': sys.argv[1:],
    'cwd': os.getcwd(),
}), encoding='utf-8')
data_file = Path(sys.argv[-1])
if os.environ.get('FAHM_FAKE_WRITE_OUTPUT', '1') == '1':
    prefix = data_file.with_suffix('')
    for suffix in ('.INIT', '.EGRID', '.UNRST'):
        Path(str(prefix) + suffix).write_bytes(suffix.encode('ascii'))
raise SystemExit(int(os.environ.get('FAHM_FAKE_EXIT', '0')))
""",
        encoding='utf-8')

    if os.name == 'nt':
        executable = bindir / (name + '.cmd')
        executable.write_text(
            '@echo off\r\n"%s" "%s" %%*\r\n'
            % (sys.executable, helper), encoding='utf-8')
    else:
        executable = bindir / name
        executable.write_text(
            '#!/bin/sh\nexec "%s" "%s" "$@"\n'
            % (sys.executable, helper), encoding='utf-8')
        executable.chmod(0o755)
    return executable


def _environment(monkeypatch, executable: Path, record: Path, *,
                 write_output=True, exit_status=0):
    bindir = str(executable.parent)
    monkeypatch.setenv('PATH', bindir + os.pathsep + os.environ.get('PATH', ''))
    monkeypatch.setenv('FAHM_FAKE_RECORD', str(record))
    monkeypatch.setenv('FAHM_FAKE_WRITE_OUTPUT', '1' if write_output else '0')
    monkeypatch.setenv('FAHM_FAKE_EXIT', str(exit_status))
    return os.environ.copy()


def test_contract_is_anchored_to_the_frozen_fahm_source():
    contract = json.loads(CONTRACT.read_text(encoding='utf-8'))
    fingerprint = json.loads(FINGERPRINT.read_text(encoding='utf-8'))
    assert contract['schema'] == 'fahm-stage4-create-project-v1'
    assert contract['oracle']['artifact'] == 'MRST/dev/APP/FAHM.mlapp'
    assert contract['oracle']['fahm_m_sha256'] == \
        fingerprint['fahm_m']['sha256']
    assert contract['execution']['system_return_status_used_by_mrst'] is False


def test_command_specs_match_the_machine_readable_oracle_contract():
    contract = json.loads(CONTRACT.read_text(encoding='utf-8'))
    base_case = r'C:\model\baseCase\CASE.DATA'
    deck = {'SCHEDULE': {
        'control': [{}, {'WELSPECS': [['P1']]}],
        'step': {'control': [0, 0, 1]},
    }}
    simulators = {
        'ECLIPSE': r'C:\sim\eclipse.exe',
        'E300': r'C:\sim\e300.exe',
        'tNavigator': r'C:\Program Files\tNavigator\tNavigator.exe',
    }
    for branch, simulator in simulators.items():
        expected = [
            token.format(
                baseCase=base_case,
                simulator=simulator,
                first_welspec_report_step_1based=3)
            for token in contract['branches'][branch]['argv_template']
        ]
        actual = build_base_case_command(simulator, base_case, deck)
        assert list(actual.argv) == expected
        assert actual.nosim is contract['branches'][branch]['nosim']


def test_base_case_path_is_the_model_sibling_and_preserves_stem(tmp_path):
    model = tmp_path / 'models' / 'MixedCase.data'
    base, filename, case = base_case_paths(model)
    assert Path(base) == model.parent / 'baseCase'
    assert filename == 'MixedCase'
    assert Path(case) == model.parent / 'baseCase' / 'MixedCase.DATA'


def test_tnavigator_stop_step_is_first_step_using_first_welspec_control():
    deck = {'SCHEDULE': {
        'control': [{}, {'WELSPECS': [['P1']]}],
        'step': {'control': [0, 0, 1, 1]},
    }}
    command = build_base_case_command(
        r'C:\sim\tNavigator.exe', r'C:\model\baseCase\CASE.DATA', deck)
    assert command.stop_step == 3
    assert command.argv[-2] == '--stop-step=3'
    assert command.nosim is False


@pytest.mark.parametrize(
    ('branch', 'selected_simulator', 'runner_name', 'expected_prefix',
     'nosim', 'stop_step'),
    [
        ('ECLIPSE', r'C:\sim\eclipse.exe', 'eclrun', ('eclrun', 'eclipse'),
         True, None),
        ('E300', r'C:\sim\e300.exe', 'eclrun', ('eclrun', 'e300'),
         True, None),
        ('tNavigator', None, 'fake_tNavigator', None, False, 1),
    ],
)
def test_three_simulator_branches_use_exact_argv_cwd_and_nosim(
        tmp_path, monkeypatch, branch, selected_simulator, runner_name,
        expected_prefix, nosim, stop_step):
    executable = _fake_executable(tmp_path, runner_name)
    record = tmp_path / ('record-' + branch + '.json')
    env = _environment(monkeypatch, executable, record)
    caller_cwd = tmp_path / 'caller cwd'
    caller_cwd.mkdir()
    monkeypatch.chdir(caller_cwd)

    model = tmp_path / 'model with spaces' / 'CASE.DATA'
    model.parent.mkdir()
    model.write_text('oracle path holder', encoding='ascii')
    simulator = str(executable) if branch == 'tNavigator' else selected_simulator
    result = create_base_case(_deck(), model, simulator, env=env)

    expected_case = str(model.parent / 'baseCase' / 'CASE.DATA')
    if expected_prefix is None:
        expected = (
            str(executable), '--no-dump-res', '--ecl-root', '-e', '-i', '-r',
            '-u', '--no-gui', '--ignore-lock', '--use-gpu', '--stop-step=1',
            expected_case,
        )
    else:
        expected = expected_prefix + (expected_case,)
    assert result.simulator_kind == branch
    assert result.argv == expected
    assert result.cwd == str(caller_cwd)
    assert result.nosim is nosim
    assert result.stop_step == stop_step
    assert all(Path(path).is_file() for path in result.output_files)

    actual = json.loads(record.read_text(encoding='utf-8'))
    assert actual['argv'] == list(expected[1:])
    assert Path(actual['cwd']) == caller_cwd
    text = Path(result.written_data_file).read_text(encoding='utf-8')
    assert ('\nNOSIM\n' in text) is nosim


def test_existing_base_case_tree_is_deleted_and_recreated(
        tmp_path, monkeypatch):
    executable = _fake_executable(tmp_path, 'eclrun')
    env = _environment(monkeypatch, executable, tmp_path / 'record.json')
    model = tmp_path / 'model' / 'CASE.DATA'
    model.parent.mkdir()
    model.write_text('source', encoding='ascii')
    old = model.parent / 'baseCase'
    (old / 'nested').mkdir(parents=True)
    (old / 'stale.UNRST').write_text('stale', encoding='ascii')
    (old / 'nested' / 'sentinel').write_text('stale', encoding='ascii')

    result = create_base_case(
        _deck(), model, r'C:\sim\eclipse.exe', env=env)
    assert Path(result.base_dir) == old
    assert not (old / 'stale.UNRST').exists()
    assert not (old / 'nested').exists()
    assert Path(result.written_data_file).is_file()


@pytest.mark.parametrize('exit_status', [0, 7])
def test_no_output_is_a_visible_stage4_failure_even_if_exit_is_zero(
        tmp_path, monkeypatch, exit_status):
    executable = _fake_executable(tmp_path, 'eclrun')
    record = tmp_path / 'record.json'
    env = _environment(
        monkeypatch, executable, record, write_output=False,
        exit_status=exit_status)
    model = tmp_path / 'model' / 'CASE.DATA'
    model.parent.mkdir()
    model.write_text('source', encoding='ascii')

    with pytest.raises(MissingBaseCaseOutputError) as caught:
        create_base_case(_deck(), model, r'C:\sim\eclipse.exe', env=env)
    failure = caught.value
    assert failure.result.returncode == exit_status
    assert [Path(path).suffix for path in failure.missing] == [
        '.INIT', '.EGRID', '.UNRST']
    assert Path(failure.result.written_data_file).is_file()
    assert not any(Path(path).exists() for path in failure.result.output_files)


def test_nonzero_status_with_all_required_outputs_follows_mrst_and_continues(
        tmp_path, monkeypatch):
    executable = _fake_executable(tmp_path, 'eclrun')
    env = _environment(
        monkeypatch, executable, tmp_path / 'record.json', exit_status=9)
    model = tmp_path / 'model' / 'CASE.DATA'
    model.parent.mkdir()
    model.write_text('source', encoding='ascii')
    result = create_base_case(_deck(), model, r'C:\sim\eclipse.exe', env=env)
    assert result.returncode == 9
    assert all(Path(path).is_file() for path in result.output_files)


def test_unsupported_simulator_happens_after_base_case_recreation(tmp_path):
    model = tmp_path / 'model' / 'CASE.DATA'
    model.parent.mkdir()
    model.write_text('source', encoding='ascii')
    old = model.parent / 'baseCase'
    old.mkdir()
    (old / 'stale').write_text('remove me', encoding='ascii')
    with pytest.raises(UnsupportedSimulatorError, match='Unsupported simulator'):
        create_base_case(_deck(), model, r'C:\sim\other.exe')
    assert old.is_dir()
    assert list(old.iterdir()) == []


@pytest.fixture
def app():
    tk = pytest.importorskip('tkinter')
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('desktop Tk session is unavailable')
    root.withdraw()
    from PRSTCore.hm.APP.fahm_app import FahmApp
    widget = FahmApp(
        root, startup_delay_ms=None, render_plot_canvases=False)
    root.update_idletasks()
    yield widget
    widget.destroy()
    root.destroy()


def test_app_routes_create_project_through_stage4_then_stage5_and_unlocks(
        app, monkeypatch):
    import PRSTCore.hm.APP.fahm_app as module

    deck = {'RUNSPEC': {}, 'SCHEDULE': {}}
    monkeypatch.setattr(module, 'read_case', lambda _config: deck)
    result = BaseCaseResult(
        base_dir='C:/model/baseCase',
        base_case='C:/model/baseCase/CASE.DATA',
        written_data_file='C:/model/baseCase/CASE.DATA',
        simulator_kind='ECLIPSE',
        argv=('eclrun', 'eclipse', 'C:/model/baseCase/CASE.DATA'),
        nosim=True,
        stop_step=None,
        cwd='C:/caller',
        returncode=0,
        stdout='',
        stderr='',
        output_files=(
            'C:/model/baseCase/CASE.INIT',
            'C:/model/baseCase/CASE.EGRID',
            'C:/model/baseCase/CASE.UNRST',
        ),
    )
    calls = []
    monkeypatch.setattr(
        module, 'create_base_case',
        lambda actual, path, simulator: calls.append(
            (actual, path, simulator)) or result)
    project = SimpleNamespace(
        deck={'WORKING': True}, G={'cells': {'num': 1}},
        rock={'poro': np.ones(1)}, fluid={'krPts': {}},
        state0={'pressure': np.ones(1)}, model=object(),
        N=np.zeros((0, 2), dtype=int), T=np.zeros(0))
    setup_calls = []
    monkeypatch.setattr(
        module, 'initialize_fahm_project',
        lambda actual, path: setup_calls.append((actual, path)) or project)
    app.ModelPath.set('C:/model/CASE.DATA')
    app.SimulatorPath.set('C:/sim/eclipse.exe')
    app._creat_project()
    assert calls == [(deck, 'C:/model/CASE.DATA', 'C:/sim/eclipse.exe')]
    assert setup_calls == [(deck, result.base_case)]
    assert app.base_case_result is result
    assert app.baseCase == result.base_case
    assert app.deck is project.deck
    assert app.G is project.G
    assert app.rock is project.rock
    assert app.fluid is project.fluid
    assert app.state0 is project.state0
    assert app.model is project.model
    assert app.N is project.N and app.T is project.T
    assert 'disabled' not in app.ModelProceedButton.state()
    assert 'disabled' not in app.ObjectiveProceedButton.state()
    assert 'disabled' not in app.ParameterProceedButton.state()


def test_app_reports_no_output_and_does_not_unlock(app, monkeypatch):
    import PRSTCore.hm.APP.fahm_app as module

    monkeypatch.setattr(
        module, 'read_case', lambda _config: {'RUNSPEC': {}, 'SCHEDULE': {}})
    failed_result = BaseCaseResult(
        base_dir='C:/model/baseCase',
        base_case='C:/model/baseCase/CASE.DATA',
        written_data_file='C:/model/baseCase/CASE.DATA',
        simulator_kind='ECLIPSE',
        argv=('eclrun', 'eclipse', 'C:/model/baseCase/CASE.DATA'),
        nosim=True,
        stop_step=None,
        cwd='C:/caller',
        returncode=7,
        stdout='',
        stderr='simulator failed',
        output_files=(
            'C:/model/baseCase/CASE.INIT',
            'C:/model/baseCase/CASE.EGRID',
            'C:/model/baseCase/CASE.UNRST',
        ),
    )
    failure = MissingBaseCaseOutputError(
        failed_result, failed_result.output_files)
    monkeypatch.setattr(
        module, 'create_base_case',
        lambda *_args: (_ for _ in ()).throw(failure))
    errors = []
    monkeypatch.setattr(
        module.messagebox, 'showerror',
        lambda title, message: errors.append((title, message)))
    app.ModelPath.set('C:/model/CASE.DATA')
    app.SimulatorPath.set('C:/sim/eclipse.exe')
    app._creat_project()
    assert errors == [('ERROR', str(failure))]
    assert app.baseCase == failed_result.base_case
    assert app.base_case_result is None
    assert 'disabled' in app.ModelProceedButton.state()


def test_app_reports_stage5_result_import_failure_and_does_not_unlock(
        app, monkeypatch):
    import PRSTCore.hm.APP.fahm_app as module

    deck = {'RUNSPEC': {}, 'SCHEDULE': {}}
    monkeypatch.setattr(module, 'read_case', lambda _config: deck)
    result = BaseCaseResult(
        base_dir='C:/model/baseCase',
        base_case='C:/model/baseCase/CASE.DATA',
        written_data_file='C:/model/baseCase/CASE.DATA',
        simulator_kind='ECLIPSE',
        argv=('eclrun', 'eclipse', 'C:/model/baseCase/CASE.DATA'),
        nosim=True, stop_step=None, cwd='C:/caller', returncode=0,
        stdout='', stderr='',
        output_files=('CASE.INIT', 'CASE.EGRID', 'CASE.UNRST'))
    monkeypatch.setattr(module, 'create_base_case', lambda *_args: result)
    failure = RuntimeError('bad INIT payload')
    monkeypatch.setattr(
        module, 'initialize_fahm_project',
        lambda *_args: (_ for _ in ()).throw(failure))
    errors = []
    monkeypatch.setattr(
        module.messagebox, 'showerror',
        lambda title, message: errors.append((title, message)))
    app.ModelPath.set('C:/model/CASE.DATA')
    app.SimulatorPath.set('C:/sim/eclipse.exe')
    app._creat_project()
    assert errors == [('ERROR', str(failure))]
    assert app.base_case_result is result
    assert app.baseCase == result.base_case
    assert app.model is None and app.state0 is None
    assert 'disabled' in app.ModelProceedButton.state()


def test_app_uses_fahm_import_failure_message(app, monkeypatch):
    import PRSTCore.hm.APP.fahm_app as module

    monkeypatch.setattr(
        module, 'read_case',
        lambda _config: (_ for _ in ()).throw(ValueError('parser detail')))
    errors = []
    monkeypatch.setattr(
        module.messagebox, 'showerror',
        lambda title, message: errors.append((title, message)))
    app.ModelPath.set('C:/model/BAD.DATA')
    app.SimulatorPath.set('C:/sim/eclipse.exe')
    app._creat_project()
    assert errors == [('ERROR', 'Failed in importing the Eclipse data.')]
    assert app.deck is None
    assert 'disabled' in app.ModelProceedButton.state()
