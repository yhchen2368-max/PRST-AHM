"""Stage 3 parity tests for the FAHM App shell and initial state machine.

The oracle is ``MRST/dev/APP/FAHM.m`` (the source embedded in
``FAHM.mlapp``). These tests stop before Create Project does deck processing
or simulation. Terminate, Mismatch result loading and View are labelled PRST
extensions and are not asserted as MATLAB behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

tk = pytest.importorskip('tkinter')

from PRSTCore.hm.APP.fahm_app import (
    APP_SIZE,
    APP_TITLE,
    COPY_SELECTION_ITEMS,
    DEFAULT_ON,
    MONITORING,
    PARAMETERS,
    PARAMETER_CHECKBOX_ROWS,
    PARAMETER_TABS,
    QUANTITIES,
    STARTUP_DELAY_MS,
    FahmApp,
)


ROOT = Path(__file__).resolve().parent
ORACLE = ROOT / 'fixtures' / 'fahm_oracle' / 'v1'
SNAPSHOT = ORACLE / 'stage3' / 'initial_ui_snapshot.json'
SOURCE_UI = ORACLE / 'static' / 'ui_controls.json'


def _tabs(notebook):
    return [notebook.tab(i, 'text') for i in range(notebook.index('end'))]


def _enabled(widget):
    return 'disabled' not in widget.state()


@pytest.fixture(scope='module')
def root():
    try:
        widget = tk.Tk()
    except tk.TclError:
        pytest.skip('desktop Tk session is unavailable')
    widget.withdraw()
    yield widget
    widget.destroy()


@pytest.fixture
def app(root):
    widget = FahmApp(root, startup_delay_ms=None,
                     render_plot_canvases=False)
    root.update_idletasks()
    yield widget
    widget.destroy()


def test_initial_component_tree_and_state_match_frozen_snapshot(app):
    expected = json.loads(SNAPSHOT.read_text(encoding='utf-8'))
    assert app.stage3_snapshot() == expected


def test_snapshot_is_tied_to_the_stage2_fahm_source_oracle():
    source = json.loads(SOURCE_UI.read_text(encoding='utf-8'))
    by_name = {component['name']: component
               for component in source['components']}

    def properties(name):
        return {entry['attribute']: entry['expression']
                for entry in by_name[name]['assignments']}

    assert properties('UIFigure')['Position'] == '[100 100 915 736]'
    assert properties('UIFigure')['Name'] == "'MATLAB App'"
    assert properties('RatesCheckBox')['Value'] == 'true'
    assert properties('RatesCheckBox')['Enable'] == "'off'"
    assert properties('SetUpTabGroup')['TabLocation'] == "'left'"
    assert properties('OilRateWeight')['Value'] == "'0'"
    assert properties('StartingCaseDropDown')['Items'] == '{}'
    assert properties('NumberofIterations')['Value'] == '50'


def test_window_title_and_size_match_create_components(app, root):
    app._finish_startup()
    root.deiconify()
    root.update_idletasks()
    assert root.title() == APP_TITLE == 'MATLAB App'
    assert (root.winfo_width(), root.winfo_height()) == APP_SIZE == (915, 736)
    root.withdraw()


def test_startup_visibility_transition_matches_startup_fcn(app):
    assert app.startup_visible
    assert not app.main_visible
    app._finish_startup()
    assert not app.startup_visible
    assert app.main_visible


def test_runtime_default_uses_the_authorized_two_second_splash():
    assert STARTUP_DELAY_MS == 2_000


def test_tab_order_is_exact(app):
    assert _tabs(app.MainTabGroup) == ['SetUp', 'Run', 'Mismatch', 'View']
    assert _tabs(app.SetUpTabGroup) == ['Model', 'Objective', 'Parameter']
    assert app.SetUpTabGroup.tab_location == 'left'
    assert _tabs(app.WellstoMatchTabGroup) == list(QUANTITIES)
    assert _tabs(app.ParameterModifyTabGroup) == list(PARAMETER_TABS)
    assert PARAMETER_TABS == PARAMETERS


def test_parameter_checkbox_rows_match_the_app_layout():
    assert PARAMETER_CHECKBOX_ROWS == (
        ('Porv', 'PermX', 'PermY', 'PermZ', 'krw', 'kro', 'krg'),
        ('Swl', 'Swcr', 'Swu', 'Sowcr', 'Sgl', 'Sgcr', 'Sgu', 'Sogcr'),
    )


def test_fahm_named_controls_are_exposed(app):
    names = {
        'UIFigure', 'MainTabGroup', 'SetUpTab', 'SetUpTabGroup', 'ModelTab',
        'SimulatorPathTextArea', 'SimulatorBrowseButton',
        'StartingModelPanel', 'ModelPathTextArea', 'ModelBrowseButton',
        'CreatProjectButton', 'MonitoringDataPanel', 'ModelProceedButton',
        'ObjectiveTab', 'ObjectiveProceedButton', 'ObjectiveButtonGroup',
        'UseWizardButton', 'SetObjectiveFunctionWeightsExplicitlyButton',
        'ObjectionFunctionUITable', 'WellstoMatchPanel',
        'WellstoMatchTabGroup', 'CopySelectionDropDown',
        'CopySelectionOKButton', 'RelativeWeightsPanel', 'ParameterTab',
        'SelectParametersPanel', 'ParameterModifyTabGroup',
        'ParameterProceedButton', 'RunTab', 'RunPanel', 'StartButton',
        'TerminateButton', 'SaveLoopParametersFileOnlyCheckBox',
        'StartingCaseDropDown', 'ProgressPanel', 'LoopProgressLabel',
        'TextArea', 'MismatchTab', 'ViewTab', 'StartupPanel', 'TileLabel',
        'VersionLabel', 'OwnerLabel',
    }
    assert all(hasattr(app, name) for name in names)
    for name, _ in MONITORING:
        prefix = 'Bhp' if name == 'BHP' else name
        assert hasattr(app, prefix + 'CheckBox')
        assert hasattr(app, prefix + 'PathTextArea')
        assert hasattr(app, prefix + 'BrowseButton')
    for name in PARAMETERS:
        assert hasattr(app, name + 'CheckBox')
        assert hasattr(app, name + 'Tab')
        assert hasattr(app, name + 'LimitsUITable')
        assert hasattr(app, 'UseRelativeValueButton_' + name)
        assert hasattr(app, 'UseAbsoluteValueButton_' + name)


def test_no_non_fahm_status_or_model_summary_widget_is_visible(app):
    assert not hasattr(app, 'status')
    assert not isinstance(app.ModelSummary, tk.Misc)
    assert set(app.winfo_children()) == {app.MainTabGroup, app.StartupPanel}


def test_model_paths_and_navigation_begin_locked(app):
    assert app.ModelPath.get() == ''
    assert app.SimulatorPath.get() == ''
    assert not _enabled(app.CreatProjectButton)
    assert not _enabled(app.ModelProceedButton)
    assert not _enabled(app.ObjectiveProceedButton)
    assert not _enabled(app.ParameterProceedButton)


def test_monitoring_defaults_are_exact(app):
    assert [name for name, _ in MONITORING
            if app.monitor_use[name].get()] == ['Rates']
    assert all(not _enabled(widgets['check'])
               for widgets in app.monitor_widgets.values())
    assert all(not _enabled(widgets['path'])
               for widgets in app.monitor_widgets.values())
    assert all(not _enabled(widgets['browse'])
               for widgets in app.monitor_widgets.values())
    assert app.monitor_source['Rates'].get() == 'model'
    assert app.monitor_source['BHP'].get() == 'model'


def test_objective_defaults_are_exact(app):
    assert app.ObjectiveMode.get() == 'wizard'
    assert not _enabled(app.SetObjectiveFunctionWeightsExplicitlyButton)
    assert not app.ObjectionFunctionUITable.winfo_manager()
    assert app.WellstoMatchPanel.winfo_manager()
    assert all(value.get() == '0' for value in app.weights.values())
    assert all(not _enabled(widget)
               for widget in app.weight_widgets.values())
    assert tuple(app.CopySelectionDropDown['values']) == COPY_SELECTION_ITEMS
    assert COPY_SELECTION_ITEMS[3] == 'Copy selection to Bhp'
    assert not _enabled(app.CopySelectionOKButton)


def test_parameter_defaults_are_exact(app):
    assert [name for name in PARAMETERS
            if app.param_on[name].get()] == list(DEFAULT_ON)
    assert all(app.param_mode[name].get() == 'useRel'
               for name in PARAMETERS)
    assert all(len(app.param_tables[name].get_children()) == 0
               for name in PARAMETERS)
    assert all(_enabled(app.param_check_widgets[name])
               for name in DEFAULT_ON)
    assert all(not _enabled(app.param_check_widgets[name])
               for name in PARAMETERS if name not in DEFAULT_ON)


def test_run_defaults_are_exact(app):
    assert app.RunDirectory.get() == ''
    assert app.SimulatorLanchCommand.get() == ''
    assert app.SimCaseDirectory.get() == ''
    assert app.SimBaseName.get() == ''
    assert app.NumberofIterations.get() == 50
    assert app.StartingCase.get() == ''
    assert tuple(app.StartingCaseDropDown['values']) == ()
    assert not app.SaveLoopParametersFileOnly.get()


def test_create_project_enables_only_after_both_paths_exist(app, monkeypatch):
    existing = {'C:/case/MODEL.DATA', 'C:/sim/eclipse.exe'}
    monkeypatch.setattr('os.path.isfile', lambda path: path in existing)
    app.ModelPath.set('C:/case/MODEL.DATA')
    assert not _enabled(app.CreatProjectButton)
    app.SimulatorPath.set('C:/sim/eclipse.exe')
    assert _enabled(app.CreatProjectButton)
    app.ModelPath.set('C:/case/missing.DATA')
    assert not _enabled(app.CreatProjectButton)


def test_model_browse_updates_path_and_create_gate(app, monkeypatch):
    model = 'C:/case/MODEL.DATA'
    simulator = 'C:/sim/eclipse.exe'
    monkeypatch.setattr('os.path.isfile',
                        lambda path: path in {model, simulator})
    monkeypatch.setattr('tkinter.filedialog.askopenfilename',
                        lambda **_kwargs: model)
    app.SimulatorPath.set(simulator)
    app._browse_model()
    assert app.ModelPath.get() == model
    # FAHM only updates ModelPath here. Run-directory/base fields are set by
    # RunDirectoryBrowseButtonPushed after Create Project has established
    # app.baseCase.
    assert app.SimBaseName.get() == ''
    assert app.SimCaseDirectory.get() == ''
    assert _enabled(app.CreatProjectButton)


def test_simulator_browse_updates_path_without_inventing_launch_command(
        app, monkeypatch):
    model = 'C:/case/MODEL.DATA'
    simulator = 'C:/sim/eclipse.exe'
    monkeypatch.setattr('os.path.isfile',
                        lambda path: path in {model, simulator})
    monkeypatch.setattr('tkinter.filedialog.askopenfilename',
                        lambda **_kwargs: simulator)
    app.ModelPath.set(model)
    app._browse_simulator()
    assert app.SimulatorPath.get() == simulator
    assert app.SimulatorLanchCommand.get() == ''
    assert _enabled(app.CreatProjectButton)


def test_create_project_ui_tail_unlocks_monitoring_but_not_phase_controls(app):
    app._unlock_after_project()
    assert all(_enabled(widgets['check'])
               for widgets in app.monitor_widgets.values())
    assert _enabled(app.monitor_widgets['Rates']['from_model'])
    assert _enabled(app.monitor_widgets['Rates']['from_file'])
    assert not _enabled(app.monitor_widgets['Rates']['path'])
    assert all(_enabled(widget) for widget in (
        app.ModelProceedButton, app.ObjectiveProceedButton,
        app.ParameterProceedButton))
    assert all(app.weights[name].get() == '0' for name in QUANTITIES)
    assert all(not _enabled(app.param_check_widgets[name])
               for name in PARAMETERS if name not in DEFAULT_ON)


@pytest.mark.parametrize(
    ('runspec', 'enabled_parameters', 'weights'),
    [
        ({'OIL': True, 'WATER': True, 'GAS': True},
         {'kro', 'krw', 'krg', 'Swu', 'Swl', 'Swcr', 'Sgu', 'Sgl',
          'Sgcr', 'Sowcr', 'Sogcr'},
         {'Oil': '1', 'Water': '1', 'Gas': '1'}),
        ({'OIL': True, 'WATER': True},
         {'kro', 'krw', 'Swu', 'Swl', 'Swcr', 'Sowcr'},
         {'Oil': '1', 'Water': '1', 'Gas': '0'}),
        ({'OIL': True, 'GAS': True},
         {'kro', 'krg', 'Sgu', 'Sgl', 'Sgcr', 'Sogcr'},
         {'Oil': '1', 'Water': '0', 'Gas': '1'}),
    ],
)
def test_model_proceed_applies_phase_gating_before_navigation(
        app, runspec, enabled_parameters, weights):
    app.deck = {'RUNSPEC': runspec, 'SCHEDULE': {}}
    app._unlock_after_project()
    app._model_proceed()
    actual = {name for name in PARAMETERS if name not in DEFAULT_ON
              and _enabled(app.param_check_widgets[name])}
    assert actual == enabled_parameters
    assert {name: app.weights[name].get()
            for name in ('Oil', 'Water', 'Gas')} == weights
    assert app.SetUpTabGroup.index(app.SetUpTabGroup.select()) == 1


def test_proceed_callbacks_follow_model_objective_parameter_run(app):
    app.deck = {'RUNSPEC': {'OIL': True, 'WATER': True, 'GAS': True},
                'SCHEDULE': {}}
    app.SimulatorPath.set('C:/sim/eclipse.exe')
    app._unlock_after_project()
    app._model_proceed()
    assert app.SetUpTabGroup.index(app.SetUpTabGroup.select()) == 1
    app._objective_proceed()
    assert app.SetUpTabGroup.index(app.SetUpTabGroup.select()) == 2
    app._parameter_proceed()
    assert app.MainTabGroup.index(app.MainTabGroup.select()) == 1
    assert app.SimulatorLanchCommand.get() == 'eclrun eclipse'


def test_monitoring_toggle_and_source_callbacks_update_enable_state(app):
    app.deck = {'RUNSPEC': {'OIL': True, 'WATER': True, 'GAS': False}}
    app._unlock_after_project()
    app.monitor_use['BHP'].set(True)
    app._monitor_toggled('BHP')
    assert _enabled(app.BhpExtractFromModelButton)
    assert _enabled(app.BhpFromFileButton)
    assert app.weights['BHP'].get() == '1'
    assert _enabled(app.BHPWeight)
    app.monitor_source['BHP'].set('file')
    app._monitor_source_changed('BHP')
    assert _enabled(app.BhpPathTextArea)
    assert _enabled(app.BhpBrowseButton)


def test_monitoring_gate_recomputes_instead_of_latching_old_truth(
        app, monkeypatch):
    existing = {'C:/case/MODEL.DATA', 'C:/sim/eclipse.exe'}
    monkeypatch.setattr('os.path.isfile', lambda path: path in existing)
    app.ModelPath.set('C:/case/MODEL.DATA')
    app.SimulatorPath.set('C:/sim/eclipse.exe')
    app._unlock_after_project()
    app._model_setup_check()
    assert _enabled(app.ModelProceedButton)
    app.monitor_use['Rates'].set(False)
    app._monitor_toggled('Rates')
    assert not _enabled(app.ModelProceedButton)


def test_well_transfer_matches_matlab_union_sorted(app):
    boxes = app.well_lists['Oil']
    for well in ('W3', 'W1', 'W2'):
        boxes['Match'].insert('end', well)
    boxes['Emphasize'].insert('end', 'W2')
    boxes['Match'].selection_set(0, 1)
    app._move_wells('Oil', 'Match', 'Emphasize')
    assert list(boxes['Emphasize'].get(0, 'end')) == ['W1', 'W2', 'W3']
    assert list(boxes['Match'].get(0, 'end')) == ['W2']


def test_proceed_callbacks_do_not_invent_validation_not_in_fahm(app):
    for value in app.param_on.values():
        value.set(False)
    app._objective_proceed()
    assert app.SetUpTabGroup.index(app.SetUpTabGroup.select()) == 2
    app._parameter_proceed()
    assert app.MainTabGroup.index(app.MainTabGroup.select()) == 1


def test_authorized_non_parity_functions_are_explicitly_labelled(app):
    assert app.stage3_snapshot()['extensions'] == {
        'terminate': 'PRST_EXTENSION',
        'mismatch_result_loading': 'PRST_EXTENSION',
        'view': 'PRST_EXTENSION',
        'startup_delay': 'PRST_EXTENSION',
    }
    assert app.ViewTab.winfo_children() == []


def test_resize_keeps_top_level_surface_filling_the_window(app, root):
    app._finish_startup()
    root.deiconify()
    root.geometry('1000x800')
    root.update_idletasks()
    assert app.winfo_width() == 1000
    assert app.winfo_height() == 800
    assert app.MainTabGroup.winfo_width() == 1000
    assert app.MainTabGroup.winfo_height() == 800
    root.geometry('%dx%d' % APP_SIZE)
    root.withdraw()


def test_stage3_screenshot_has_the_frozen_canvas_size():
    import hashlib

    image = ORACLE / 'stage3' / 'prst_initial_915x736.png'
    manifest = ORACLE / 'stage3' / 'screenshot.json'
    if not image.exists() or not manifest.exists():
        pytest.fail('run tests/fahm_oracle/tools/capture_stage3_ui.py')
    from PIL import Image
    with Image.open(image) as opened:
        assert opened.size == APP_SIZE
    info = json.loads(manifest.read_text(encoding='utf-8'))
    assert info['width'] == APP_SIZE[0]
    assert info['height'] == APP_SIZE[1]
    assert info['classification'] == 'PARITY'
    assert hashlib.sha256(image.read_bytes()).hexdigest() == info['sha256']
