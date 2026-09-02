"""FAHM's interface, rebuilt on tkinter.

The parity source is exclusively ``MRST/dev/APP/FAHM.mlapp`` and its
byte-frozen extracted source ``FAHM.m``.  The window, tab hierarchy,
controls, strings, initial enable/visible state, startup transition and
navigation callbacks follow ``createComponents``/``startupFcn``.

    MainTabGroup
      SetUp
        Model      -- Starting Model | Monitoring Data | Simulator
        Objective  -- objective weights table | Wells to Match |
                      Relative Weights | Copy Selection
        Parameter  -- Select Parameters | fifteen per-parameter tabs
      Run          -- Run | Progress | Log
      Mismatch     -- Plots | WellBars | CaseBars
      View         -- empty, as FAHM leaves it

The fifteen tunable quantities are MRST's: pore volume, the three
permeability directions, the three relative permeabilities, and the
eight saturation endpoints. Each has its own tab with a relative /
absolute choice and a limits table carrying **one row per region** --
FIPNUM for pore volume and permeability, SATNUM for everything derived
from the saturation functions. The defaults come from
:mod:`fahm_parameters`, which ports MRST's own.

Where FAHM.m spells each of those tabs and each of the seven phase
panels out longhand -- App Designer generates code, it does not write it
-- this builds them from a table. The widget tree is the same; only the
source is shorter.

Terminate, Mismatch result loading and View are explicitly classified as
``PRST_EXTENSION`` by the migration contract.  Their Python behavior is not
presented as MATLAB parity.  Simulation and history-matching behavior is
outside the Stage 3 shell/state-machine contract.

Run it with::

    python -m PRSTCore.hm.APP.fahm_app
"""

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from PRSTCore.hm.APP.fahm import FahmConfig, read_case, run_history_match
from PRSTCore.hm.APP.fahm_parameters import (BACKEND_NAME, DEFAULTS,
                                             config_row, default_limits)

#: Window metadata frozen from ``FAHM.createComponents``.
APP_TITLE = 'MATLAB App'
APP_SIZE = (915, 736)
STARTUP_DELAY_MS = 2_000
STARTUP_TITLE = ('油气藏数字孪生体',
                 '--快速自动历史拟合及模型迭代更新系统')
STARTUP_VERSION = 'PRST-FAHM'

#: FAHM concatenates ``trainRock`` and ``trainFluid`` in this order.  The
#: parameter tabs follow the same order; the generated App Designer source
#: merely declares the checkbox properties in a different order.
PARAMETERS = ('Porv', 'PermX', 'PermY', 'PermZ', 'krw', 'kro', 'krg',
              'Swl', 'Swcr', 'Swu', 'Sowcr', 'Sgl', 'Sgcr', 'Sgu',
              'Sogcr')
PARAMETER_TABS = PARAMETERS

#: Visual rows in ``SelectParametersPanel``.
PARAMETER_CHECKBOX_ROWS = (
    ('Porv', 'PermX', 'PermY', 'PermZ', 'krw', 'kro', 'krg'),
    ('Swl', 'Swcr', 'Swu', 'Sowcr', 'Sgl', 'Sgcr', 'Sgu', 'Sogcr'),
)

#: Checked by default in FAHM.m.
DEFAULT_ON = ('Porv', 'PermX', 'PermY', 'PermZ')

#: The Wells-to-Match sub-tabs and the Relative Weights fields.
QUANTITIES = ('Oil', 'Water', 'Gas', 'BHP', 'Tracer', 'Profile', 'Saturation')

#: The objective table's columns.
OBJECTIVE_COLUMNS = ('Well',) + QUANTITIES

#: The Mismatch tab's three views, in FAHM's own order. The toggle group
#: and the tab group both show this list.
MISMATCH_VIEWS = ('Plots', 'WellBars', 'CaseBars')

#: What the Plots view can draw against time. Water cut and GOR are
#: ratios computed from the rates rather than matched quantities of their
#: own, which is why they appear here but not among QUANTITIES.
PLOT_CURVES = ('Oil', 'Water', 'Gas', 'WaterCut', 'GOR', 'BHP', 'Tracer')

#: The bar views add the two that have a score but no time series.
BAR_CURVES = PLOT_CURVES + ('Profile', 'Saturation')

#: FAHM spells two of these with a space.
CURVE_LABELS = {'WaterCut': 'Water Cut'}

#: The Copy Selection dropdown's entries, one per quantity.
COPY_SELECTION = 'Copy selection to %s'
COPY_SELECTION_NAMES = {
    'Oil': 'Oil', 'Water': 'Water', 'Gas': 'Gas', 'BHP': 'Bhp',
    'Tracer': 'Tracer', 'Profile': 'Profile', 'Saturation': 'Saturation',
}
COPY_SELECTION_ITEMS = tuple(
    COPY_SELECTION % COPY_SELECTION_NAMES[name] for name in QUANTITIES)

#: The Mismatch Scores rows -- QUANTITIES, under FAHM's own labels.
SCORE_LABELS = {'Oil': 'Oil Rate', 'Water': 'Water Rate',
                'Gas': 'Gas Rate', 'BHP': 'BHP', 'Tracer': 'Tracer',
                'Profile': 'Profile', 'Saturation': 'Saturation'}

def _axes_in(panel):
    """Embed a matplotlib figure in a panel and return its axes.

    Returns ``None`` if matplotlib is not installed, and every drawing
    routine below treats that as "no canvas" rather than failing: the
    rest of the interface -- which is what drives the simulator -- has no
    business depending on a plotting library being present.
    """
    try:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
    except ImportError:
        return None

    figure = Figure(figsize=(3.2, 2.1), dpi=88)
    axes = figure.add_subplot(111)
    figure.subplots_adjust(left=0.22, right=0.97, top=0.88, bottom=0.22)
    canvas = FigureCanvasTkAgg(figure, master=panel)
    canvas.get_tk_widget().pack(fill='both', expand=True)
    axes._prst_canvas = canvas
    return axes


def _draw(axes):
    if axes is not None:
        axes._prst_canvas.draw_idle()


#: The Monitoring Data rows. The flag says whether the row offers the
#: "Extract from the Model" / "From File" choice, which only Rates and
#: BHP do -- the other three can only come from a file.
MONITORING = (('Rates', True), ('BHP', True), ('Tracer', False),
              ('Profile', False), ('Saturation', False))


class _TextBuffer:
    """Small non-widget compatibility buffer.

    Earlier PRST builds displayed a model-summary text box that does not
    exist in FAHM.  Keeping this buffer lets non-UI callers inspect the
    summary without adding an extra visible component to the parity tree.
    """

    def __init__(self):
        self.value = ''

    def delete(self, *_args):
        self.value = ''

    def insert(self, _index, value):
        self.value = str(value)

    def get(self, *_args):
        return self.value


class _ProgressValue:
    """Dictionary-like progress value without a non-FAHM progress bar."""

    def __init__(self):
        self.value = 0

    def __getitem__(self, key):
        if key != 'value':
            raise KeyError(key)
        return self.value

    def __setitem__(self, key, value):
        if key != 'value':
            raise KeyError(key)
        self.value = value


class _LeftTabGroup(ttk.Frame):
    """Small notebook-compatible tab group with tabs on the left.

    Tk's native ttk Notebook has no portable left-tab placement option,
    while FAHM fixes ``SetUpTabGroup.TabLocation`` to ``'left'``.  This
    wrapper implements the subset of Notebook used by the App and tests.
    """

    def __init__(self, master):
        super().__init__(master)
        self.tab_location = 'left'
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._rail = ttk.Frame(self)
        self._rail.grid(row=0, column=0, sticky='ns')
        self._rail.configure(width=85)
        self._rail.grid_propagate(False)
        self.content = ttk.Frame(self)
        self.content.grid(row=0, column=1, sticky='nsew')
        self.content.columnconfigure(0, weight=1)
        self.content.rowconfigure(0, weight=1)
        self._tabs = []
        self._labels = []
        self._buttons = []
        self._selected = tk.IntVar(master=self, value=0)

    def add(self, child, *, text):
        index = len(self._tabs)
        self._tabs.append(child)
        self._labels.append(str(text))
        button = ttk.Button(
            self._rail, text=text, style='Toolbutton',
            command=lambda i=index: self.select(i))
        button.grid(row=index, column=0, sticky='ew', padx=(0, 4), pady=1)
        self._buttons.append(button)
        if index == 0:
            button.state(['selected'])
            child.grid(row=0, column=0, sticky='nsew')
        else:
            child.grid_remove()

    def _resolve(self, spec):
        if isinstance(spec, int):
            return spec
        if isinstance(spec, tk.Misc):
            return self._tabs.index(spec)
        value = str(spec)
        for index, child in enumerate(self._tabs):
            if value == str(child):
                return index
        return int(value)

    def select(self, tab_id=None):
        if tab_id is None:
            return str(self._tabs[self._selected.get()])
        index = self._resolve(tab_id)
        if not 0 <= index < len(self._tabs):
            raise tk.TclError('tab index out of range')
        old = self._selected.get()
        self._tabs[old].grid_remove()
        self._buttons[old].state(['!selected'])
        self._selected.set(index)
        self._buttons[index].state(['selected'])
        self._tabs[index].grid(row=0, column=0, sticky='nsew')
        return str(self._tabs[index])

    def index(self, tab_id):
        if str(tab_id) == 'end':
            return len(self._tabs)
        return self._resolve(tab_id)

    def tab(self, tab_id, option=None, **_kwargs):
        index = self._resolve(tab_id)
        data = {'text': self._labels[index]}
        return data[option] if option else data


class FahmApp(ttk.Frame):
    """The application window."""

    def __init__(self, master, *, startup_delay_ms=STARTUP_DELAY_MS,
                 render_plot_canvases=True):
        super().__init__(master, padding=0)
        self.grid(sticky='nsew')
        self.UIFigure = master
        master.title(APP_TITLE)
        master.geometry('%dx%d' % APP_SIZE)
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._messages = queue.Queue()
        self._worker = None
        self._stop = threading.Event()
        self.result = None
        self.deck = None
        self.model = None
        self._status_text = ''
        self._startup_after_id = None
        self._render_plot_canvases = bool(render_plot_canvases)

        self._build_state()

        self.MainTabGroup = ttk.Notebook(self)
        self.MainTabGroup.grid(row=0, column=0, sticky='nsew')
        self.SetUpTab = self._setup_tab(self.MainTabGroup)
        self.RunTab = self._run_tab(self.MainTabGroup)
        self.MismatchTab = self._mismatch_tab(self.MainTabGroup)
        self.MainTabGroup.add(self.SetUpTab, text='SetUp')
        self.MainTabGroup.add(self.RunTab, text='Run')
        self.MainTabGroup.add(self.MismatchTab, text='Mismatch')
        # FAHM's fourth tab is declared and left empty -- no children are
        # created for it. Reproduced as it is rather than filled in.
        self.ViewTab = ttk.Frame(self.MainTabGroup)
        self.MainTabGroup.add(self.ViewTab, text='View')

        self._objective_mode_changed()
        for name in PARAMETER_TABS:
            self._parameter_toggled(name, populate=False)
        self._build_startup_panel()
        self._begin_startup(startup_delay_ms)
        self._drain_after_id = self.after(100, self._drain)

    # ------------------------------------------------------------ state --

    def _build_state(self):
        """Every widget's backing variable, named after FAHM's control."""
        self.ModelPath = tk.StringVar()
        self.SimulatorPath = tk.StringVar()

        self.monitor_use = {n: tk.BooleanVar(value=(n == 'Rates'))
                            for n, _ in MONITORING}
        self.monitor_path = {n: tk.StringVar() for n, _ in MONITORING}
        # 'model' == Extract from the Model, FAHM's default for both.
        self.monitor_source = {n: tk.StringVar(value='model')
                               for n, has in MONITORING if has}

        self.ObjectiveMode = tk.StringVar(value='wizard')
        self.weights = {q: tk.StringVar(value='0') for q in QUANTITIES}

        self.param_on = {n: tk.BooleanVar(value=(n in DEFAULT_ON))
                         for n in PARAMETERS}
        self.param_mode = {n: tk.StringVar(value='useRel')
                           for n in PARAMETER_TABS}
        self.param_tables = {}        # name -> Treeview of region rows
        self.param_widgets = {}       # name -> widgets to enable/disable

        self.RunDirectory = tk.StringVar()
        self.SimulatorLanchCommand = tk.StringVar()
        self.SimCaseDirectory = tk.StringVar()
        self.SimBaseName = tk.StringVar()
        self.NumberofIterations = tk.IntVar(value=50)
        self.StartingCase = tk.StringVar()
        self.SaveLoopParametersFileOnly = tk.BooleanVar(value=False)

        self.MismatchView = tk.StringVar(value=MISMATCH_VIEWS[0])
        self.view_cases = {}          # view -> Listbox of case names
        self.view_wells = {}          # view -> Listbox of well names
        self.view_curves = {}         # (view, curve) -> BooleanVar
        self.bar_panels = {}          # view -> the panel bars are drawn in
        self.bar_axes = {}            # view -> its axes, or None headless
        self.CopySelection = tk.StringVar(
            value=COPY_SELECTION_ITEMS[0])

        self.dataCheck = {
            'startingModel': False,
            'simulator': False,
            'monitorData': {name.lower(): False for name, _ in MONITORING},
        }

    # ----------------------------------------------------------- startup --

    def _build_startup_panel(self):
        """Build the three-component splash panel from ``startupFcn``."""
        self.StartupPanel = ttk.Frame(self)
        self.StartupPanel.columnconfigure(0, weight=1)
        self.StartupPanel.rowconfigure(0, weight=1)
        self.StartupPanel.rowconfigure(1, weight=1)
        self.StartupPanel.rowconfigure(2, weight=1)

        self.TileLabel = ttk.Label(
            self.StartupPanel, text='\n'.join(STARTUP_TITLE),
            anchor='center', justify='center',
            font=('TkDefaultFont', 18, 'bold'))
        self.TileLabel.grid(row=0, column=0, sticky='sew', padx=24, pady=12)
        self.VersionLabel = ttk.Label(
            self.StartupPanel, text=STARTUP_VERSION, anchor='center')
        self.VersionLabel.grid(row=1, column=0, sticky='new', padx=24, pady=12)
        self.OwnerLabel = ttk.Label(self.StartupPanel, text='', anchor='center')
        self.OwnerLabel.grid(row=2, column=0, sticky='sew', padx=24, pady=12)

    def _begin_startup(self, delay_ms):
        """Port ``startupFcn`` without blocking Tk's event loop."""
        self.StartupPanel.grid(row=0, column=0, sticky='nsew')
        self.StartupPanel.tkraise()
        self.MainTabGroup.grid_remove()
        if delay_ms is not None:
            self._startup_after_id = self.after(
                max(0, int(delay_ms)), self._finish_startup)

    def _finish_startup(self):
        """Perform the visibility transition after FAHM's 20-second splash."""
        self._startup_after_id = None
        if not self.winfo_exists():
            return
        self.StartupPanel.grid_remove()
        self.MainTabGroup.grid(row=0, column=0, sticky='nsew')
        self.MainTabGroup.tkraise()

    @property
    def startup_visible(self):
        return bool(self.StartupPanel.winfo_manager())

    @property
    def main_visible(self):
        return bool(self.MainTabGroup.winfo_manager())

    @staticmethod
    def _enabled(widget):
        return 'disabled' not in widget.state()

    @staticmethod
    def _tabs(notebook):
        return [notebook.tab(i, 'text')
                for i in range(notebook.index('end'))]

    def stage3_snapshot(self):
        """Return a deterministic snapshot of the Stage 3 parity surface."""
        return {
            'schema': 'fahm-stage3-ui-v1',
            'window': {
                'title': self.winfo_toplevel().title(),
                'size': list(APP_SIZE),
                'startup_delay_ms': STARTUP_DELAY_MS,
            },
            'visibility': {
                'startup': self.startup_visible,
                'main': self.main_visible,
            },
            'tabs': {
                'main': self._tabs(self.MainTabGroup),
                'setup': self._tabs(self.SetUpTabGroup),
                'setup_location': self.SetUpTabGroup.tab_location,
                'wells': self._tabs(self.WellstoMatchTabGroup),
                'parameters': self._tabs(self.ParameterModifyTabGroup),
            },
            'model': {
                'simulator_path': self.SimulatorPath.get(),
                'model_path': self.ModelPath.get(),
                'create_enabled': self._enabled(self.CreatProjectButton),
                'proceed_enabled': self._enabled(self.ModelProceedButton),
            },
            'monitoring': {
                name: {
                    'checked': bool(self.monitor_use[name].get()),
                    'check_enabled': self._enabled(
                        self.monitor_widgets[name]['check']),
                    'path': self.monitor_path[name].get(),
                    'path_enabled': self._enabled(
                        self.monitor_widgets[name]['path']),
                    'browse_enabled': self._enabled(
                        self.monitor_widgets[name]['browse']),
                    **({
                        'source': self.monitor_source[name].get(),
                        'source_enabled': self._enabled(
                            self.monitor_widgets[name]['from_model']),
                    } if name in self.monitor_source else {}),
                }
                for name, _ in MONITORING
            },
            'objective': {
                'mode': self.ObjectiveMode.get(),
                'manual_enabled': self._enabled(
                    self.SetObjectiveFunctionWeightsExplicitlyButton),
                'manual_table_visible': bool(
                    self.ObjectionFunctionUITable.winfo_manager()),
                'wizard_visible': bool(self.WellstoMatchPanel.winfo_manager()),
                'weights': {name: self.weights[name].get()
                            for name in QUANTITIES},
                'weight_enabled': {
                    name: self._enabled(self.weight_widgets[name])
                    for name in QUANTITIES
                },
                'proceed_enabled': self._enabled(self.ObjectiveProceedButton),
                'copy_items': list(self.CopySelectionDropDown['values']),
                'copy_value': self.CopySelection.get(),
                'copy_ok_enabled': self._enabled(self.CopySelectionOKButton),
            },
            'parameters': {
                'checked': [name for name in PARAMETERS
                            if self.param_on[name].get()],
                'checkbox_enabled': {
                    name: self._enabled(self.param_check_widgets[name])
                    for name in PARAMETERS
                },
                'mode': {name: self.param_mode[name].get()
                         for name in PARAMETERS},
                'table_rows': {
                    name: len(self.param_tables[name].get_children())
                    for name in PARAMETERS
                },
                'proceed_enabled': self._enabled(self.ParameterProceedButton),
            },
            'run': {
                'work_directory': self.RunDirectory.get(),
                'launch_command': self.SimulatorLanchCommand.get(),
                'case_directory': self.SimCaseDirectory.get(),
                'base_name': self.SimBaseName.get(),
                'iterations': self.NumberofIterations.get(),
                'starting_case': self.StartingCase.get(),
                'starting_case_items': list(
                    self.StartingCaseDropDown['values']),
                'save_loop_parameters_only': bool(
                    self.SaveLoopParametersFileOnly.get()),
            },
            'extensions': {
                'terminate': 'PRST_EXTENSION',
                'mismatch_result_loading': 'PRST_EXTENSION',
                'view': 'PRST_EXTENSION',
            },
        }

    # ------------------------------------------------------------- SetUp --

    def _setup_tab(self, master):
        tab = ttk.Frame(master)
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
        self.SetUpTabGroup = _LeftTabGroup(tab)
        self.SetUpTabGroup.grid(row=0, column=0, sticky='nsew')
        self.ModelTab = self._model_tab(self.SetUpTabGroup.content)
        self.ObjectiveTab = self._objective_tab(self.SetUpTabGroup.content)
        self.ParameterTab = self._parameter_tab(self.SetUpTabGroup.content)
        self.SetUpTabGroup.add(self.ModelTab, text='Model')
        self.SetUpTabGroup.add(self.ObjectiveTab, text='Objective')
        self.SetUpTabGroup.add(self.ParameterTab, text='Parameter')
        return tab

    def _model_tab(self, master):
        tab = ttk.Frame(master, padding=15)
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(4, weight=1)

        self.SimulatorPanel = ttk.LabelFrame(
            tab, text='Simulator', padding=(14, 10), height=80)
        self.SimulatorPanel.grid(row=0, column=0, sticky='ew', pady=(0, 35))
        self.SimulatorPanel.grid_propagate(False)
        self.SimulatorPanel.columnconfigure(0, weight=1)
        self.SimulatorPathTextArea = ttk.Entry(
            self.SimulatorPanel, textvariable=self.SimulatorPath)
        self.SimulatorPathTextArea.grid(
            row=0, column=0, sticky='ew', padx=(0, 4))
        self.SimulatorBrowseButton = ttk.Button(
            self.SimulatorPanel, text='Browse', command=self._browse_simulator)
        self.SimulatorBrowseButton.grid(row=0, column=1)

        self.StartingModelPanel = ttk.LabelFrame(
            tab, text='Starting Model (Case 0)', padding=(14, 6), height=80)
        self.StartingModelPanel.grid(
            row=1, column=0, sticky='ew', pady=(0, 35))
        self.StartingModelPanel.grid_propagate(False)
        self.StartingModelPanel.columnconfigure(0, weight=1)
        self.ModelPathTextArea = ttk.Entry(
            self.StartingModelPanel, textvariable=self.ModelPath)
        self.ModelPathTextArea.grid(
            row=0, column=0, sticky='ew', padx=(0, 4))
        self.ModelBrowseButton = ttk.Button(
            self.StartingModelPanel, text='Browse', command=self._browse_model)
        self.ModelBrowseButton.grid(row=0, column=1)
        self.CreatProjectButton = ttk.Button(
            self.StartingModelPanel, text='Creat the Project',
            command=self._creat_project, state='disabled')
        self.CreatProjectButton.grid(
            row=1, column=0, columnspan=2, pady=(4, 0))

        self.MonitoringDataPanel = ttk.LabelFrame(
            tab, text=' Monitoring Data', padding=(14, 8), height=230)
        self.MonitoringDataPanel.grid(
            row=2, column=0, sticky='ew', pady=(0, 35))
        self.MonitoringDataPanel.grid_propagate(False)
        self.MonitoringDataPanel.columnconfigure(1, minsize=240)
        self.MonitoringDataPanel.columnconfigure(2, weight=1)
        self.monitor_widgets = {}
        for row, (name, has_source) in enumerate(MONITORING):
            prefix = 'Bhp' if name == 'BHP' else name
            check = ttk.Checkbutton(
                self.MonitoringDataPanel, text=name,
                variable=self.monitor_use[name], state='disabled',
                command=lambda n=name: self._monitor_toggled(n))
            check.grid(row=row, column=0, sticky='w', pady=7)
            path = ttk.Entry(
                self.MonitoringDataPanel, textvariable=self.monitor_path[name],
                state='disabled')
            path_column = 2 if has_source else 1
            path_span = 1 if has_source else 2
            path.grid(row=row, column=path_column, columnspan=path_span,
                      sticky='ew', padx=4)
            browse = ttk.Button(
                self.MonitoringDataPanel, text='Browse', state='disabled',
                command=lambda n=name: self._browse_monitor(n))
            browse.grid(row=row, column=3)
            widgets = {'check': check, 'path': path, 'browse': browse}
            setattr(self, prefix + 'CheckBox', check)
            setattr(self, prefix + 'PathTextArea', path)
            setattr(self, prefix + 'BrowseButton', browse)
            if has_source:
                var = self.monitor_source[name]
                source_group = ttk.Frame(self.MonitoringDataPanel)
                source_group.grid(row=row, column=1, sticky='w')
                from_model = ttk.Radiobutton(
                    source_group, text='Extract from the Model',
                    variable=var, value='model', state='disabled',
                    command=lambda n=name: self._monitor_source_changed(n))
                from_model.grid(row=0, column=0, sticky='w')
                from_file = ttk.Radiobutton(
                    source_group, text='From File', variable=var,
                    value='file', state='disabled',
                    command=lambda n=name: self._monitor_source_changed(n))
                from_file.grid(row=0, column=1, sticky='w', padx=(4, 0))
                widgets.update(from_model=from_model, from_file=from_file)
                setattr(self, prefix + 'ExtractFromModelButton', from_model)
                setattr(self, prefix + 'FromFileButton', from_file)
            self.monitor_widgets[name] = widgets

        self.ModelProceedButton = ttk.Button(tab, text='Proceed',
                                             command=self._model_proceed,
                                             state='disabled')
        self.ModelProceedButton.grid(row=3, column=0)

        # PRST used to display this as an extra text widget.  FAHM has no
        # such control, so retain only an off-tree compatibility buffer.
        self.ModelSummary = _TextBuffer()

        self.ModelPath.trace_add('write', self._path_state_changed)
        self.SimulatorPath.trace_add('write', self._path_state_changed)
        return tab

    def _objective_tab(self, master):
        tab = ttk.Frame(master, padding=8)
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)

        self.ObjectiveButtonGroup = ttk.Frame(tab)
        self.ObjectiveButtonGroup.grid(row=0, column=0, sticky='w')
        self.UseWizardButton = ttk.Radiobutton(
            self.ObjectiveButtonGroup, text='Use Wizard',
            variable=self.ObjectiveMode, value='wizard',
            command=self._objective_mode_changed)
        self.UseWizardButton.pack(side='left', padx=(0, 12))
        self.SetObjectiveFunctionWeightsExplicitlyButton = ttk.Radiobutton(
            self.ObjectiveButtonGroup,
            text='Set Objective Function Weights Explicitly',
            variable=self.ObjectiveMode, value='explicit', state='disabled',
            command=self._objective_mode_changed)
        self.SetObjectiveFunctionWeightsExplicitlyButton.pack(side='left')

        # The explicit alternative to the wizard: a weight per well per
        # quantity, which is what the wizard's lists and relative weights
        # amount to once resolved.
        self.ObjectionFunctionUITable = ttk.Treeview(
            tab, columns=OBJECTIVE_COLUMNS, show='headings', height=6)
        for column in OBJECTIVE_COLUMNS:
            self.ObjectionFunctionUITable.heading(column, text=column)
            self.ObjectionFunctionUITable.column(
                column, width=80 if column == 'Saturation' else 60,
                anchor='center')
        self.ObjectionFunctionUITable.grid(row=1, column=0, sticky='nsew',
                                           pady=(6, 0))

        self.WellstoMatchPanel = ttk.LabelFrame(tab, text='Wells to Match',
                                                padding=4)
        self.WellstoMatchPanel.grid(row=2, column=0, sticky='nsew',
                                    pady=(6, 0))
        self.WellstoMatchPanel.columnconfigure(0, weight=1)
        self.WellstoMatchPanel.rowconfigure(0, weight=1)
        self.WellstoMatchTabGroup = ttk.Notebook(self.WellstoMatchPanel)
        self.WellstoMatchTabGroup.grid(row=0, column=0, sticky='nsew')
        self.WellstoMatchTabGroup.bind(
            '<<NotebookTabChanged>>',
            lambda _e: self._copy_selection_changed())
        self.well_lists = {}
        for q in QUANTITIES:
            quantity_tab = self._wells_panel(self.WellstoMatchTabGroup, q)
            prefix = 'Bhp' if q == 'BHP' else q
            setattr(self, prefix + 'Tab', quantity_tab)
            self.WellstoMatchTabGroup.add(quantity_tab, text=q)

        # Seven quantities each need the same wells sorted into the same
        # three lists; this copies the tab in front onto another.
        copy = ttk.Frame(self.WellstoMatchPanel)
        copy.grid(row=1, column=0, sticky='e', pady=(4, 0))
        self.CopySelectionDropDown = ttk.Combobox(
            copy, textvariable=self.CopySelection, width=28, state='readonly',
            values=COPY_SELECTION_ITEMS)
        self.CopySelectionDropDown.pack(side='left')
        self.CopySelectionDropDown.bind(
            '<<ComboboxSelected>>', lambda _e: self._copy_selection_changed())
        self.CopySelectionOKButton = ttk.Button(copy, text='OK', width=6,
                                                command=self._copy_selection,
                                                state='disabled')
        self.CopySelectionOKButton.pack(side='left', padx=(4, 0))

        self.RelativeWeightsPanel = ttk.LabelFrame(
            tab, text='Relative Weights', padding=6)
        self.RelativeWeightsPanel.grid(row=3, column=0, sticky='ew',
                                       pady=(6, 0))
        self.weight_widgets = {}
        for col, q in enumerate(QUANTITIES):
            prefix = {'Oil': 'OilRate', 'Water': 'WaterRate',
                      'Gas': 'GasRate'}.get(q, q)
            label_name = {'Oil': 'OilEditFieldLabel',
                          'Water': 'WaterEditFieldLabel',
                          'Gas': 'GasEditFieldLabel',
                          'BHP': 'BHPEditFieldLabel'}.get(
                              q, q + 'EditFieldLabel')
            label = ttk.Label(self.RelativeWeightsPanel, text=q)
            label.grid(row=0, column=col, padx=4)
            entry = ttk.Entry(
                self.RelativeWeightsPanel, textvariable=self.weights[q],
                width=8, state='disabled')
            entry.grid(row=1, column=col, padx=4)
            setattr(self, label_name, label)
            setattr(self, prefix + 'Weight', entry)
            self.weight_widgets[q] = entry

        self.ObjectiveProceedButton = ttk.Button(
            tab, text='Proceed', command=self._objective_proceed,
            state='disabled')
        self.ObjectiveProceedButton.grid(row=4, column=0, sticky='e',
                                         pady=(6, 0))
        return tab

    def _wells_panel(self, master, quantity):
        """One phase's Ignore / Match / Emphasize triple."""
        panel = ttk.Frame(master, padding=6)
        prefix = 'Bhp' if quantity == 'BHP' else quantity
        for col in (0, 2, 4):
            panel.columnconfigure(col, weight=1)
        panel.rowconfigure(1, weight=1)

        boxes = {}
        for col, label in ((0, 'Ignore'), (2, 'Match'), (4, 'Emphasize')):
            label_widget = ttk.Label(panel, text=label)
            label_widget.grid(row=0, column=col)
            box = tk.Listbox(panel, selectmode='extended', height=7,
                             exportselection=False)
            box.grid(row=1, column=col, sticky='nsew')
            boxes[label] = box
            setattr(self, prefix + label + 'Label', label_widget)
            setattr(self, prefix + label + 'ListBox', box)
        self.well_lists[quantity] = boxes

        for col, (src, dst) in ((1, ('Ignore', 'Match')),
                                (3, ('Match', 'Emphasize'))):
            frame = ttk.Frame(panel)
            frame.grid(row=1, column=col, padx=2)
            forward = ttk.Button(
                frame, text='＞＞', width=4,
                command=lambda q=quantity, a=src, b=dst:
                    self._move_wells(q, a, b))
            forward.pack(pady=2)
            backward = ttk.Button(
                frame, text='＜＜', width=4,
                command=lambda q=quantity, a=dst, b=src:
                    self._move_wells(q, a, b))
            backward.pack(pady=2)
            setattr(self, prefix + src + 'To' + dst + 'Button', forward)
            setattr(self, prefix + dst + 'To' + src + 'Button', backward)
        setattr(self, prefix + 'Panel', panel)
        return panel

    def _parameter_tab(self, master):
        tab = ttk.Frame(master, padding=8)
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)

        self.SelectParametersPanel = ttk.LabelFrame(
            tab, text='Select the Parameters to Modify', padding=6)
        self.SelectParametersPanel.grid(row=0, column=0, sticky='ew')
        self.param_check_widgets = {}
        for row, names in enumerate(PARAMETER_CHECKBOX_ROWS):
            for column, name in enumerate(names):
                check = ttk.Checkbutton(
                    self.SelectParametersPanel, text=name,
                    variable=self.param_on[name],
                    command=lambda n=name: self._parameter_toggled(n))
                if name not in DEFAULT_ON:
                    check.state(['disabled'])
                check.grid(row=row, column=column, sticky='w', padx=4)
                self.param_check_widgets[name] = check
                setattr(self, name + 'CheckBox', check)

        self.ParameterModifyTabGroup = ttk.Notebook(tab)
        self.ParameterModifyTabGroup.grid(row=1, column=0, sticky='nsew',
                                          pady=(6, 0))
        for name in PARAMETER_TABS:
            parameter_tab = self._parameter_panel(
                self.ParameterModifyTabGroup, name)
            setattr(self, name + 'Tab', parameter_tab)
            self.ParameterModifyTabGroup.add(parameter_tab, text=name)

        self.ParameterProceedButton = ttk.Button(
            tab, text='Proceed', command=self._parameter_proceed,
            state='disabled')
        self.ParameterProceedButton.grid(row=2, column=0, sticky='e',
                                         pady=(6, 0))
        return tab

    def _parameter_panel(self, master, name):
        """One parameter's relative/absolute choice and its limits table."""
        panel = ttk.Frame(master, padding=6)
        panel.columnconfigure(1, weight=1)
        panel.rowconfigure(1, weight=1)

        kind, lb, ub = DEFAULTS[name]
        label = ttk.Label(panel, text=name, font=('TkDefaultFont', 10, 'bold'),
                          foreground='blue')
        label.grid(row=0, column=0, columnspan=2)
        setattr(self, name + 'Label', label)

        group = ttk.LabelFrame(panel, text='%s regions' % kind.upper()[:3],
                               padding=4)
        group.grid(row=1, column=0, sticky='nw', padx=(0, 8))
        setattr(self, name + 'ButtonGroup', group)
        var = self.param_mode[name]
        buttons = []
        for text, value in (('Use Relative Value', 'useRel'),
                            ('Use Absolute Value', 'useAbs')):
            button = ttk.Radiobutton(
                group, text=text, variable=var, value=value,
                command=lambda n=name: self._parameter_mode_changed(n))
            button.pack(anchor='w', pady=6)
            buttons.append(button)
            prefix = ('UseRelativeValueButton_' if value == 'useRel'
                      else 'UseAbsoluteValueButton_')
            setattr(self, prefix + name, button)
        ttk.Label(group, text='default  %g .. %g' % (lb, ub),
                  foreground='grey').pack(anchor='w', pady=(6, 0))

        table = ttk.Treeview(panel, columns=('region', 'min', 'max'),
                             show='headings', height=7)
        for column, title in (('region', 'region'), ('min', 'Min. Value'),
                              ('max', 'Max. Value')):
            table.heading(column, text=title)
            table.column(column, width=110, anchor='center')
        table.grid(row=1, column=1, sticky='nsew')
        table.bind('<Double-1>', lambda e, n=name: self._edit_limit(e, n))

        self.param_tables[name] = table
        setattr(self, name + 'LimitsUITable', table)
        self.param_widgets[name] = buttons + [table]
        return panel

    # --------------------------------------------------------------- Run --

    def _run_tab(self, master):
        tab = ttk.Frame(master, padding=8)
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)

        self.RunPanel = ttk.Frame(tab, padding=6)
        self.RunPanel.grid(row=0, column=0, sticky='ew')
        self.RunPanel.columnconfigure(1, weight=1)

        rows = (
            ('Simulation Work Directory', self.RunDirectory, True,
             'SimulationWorkDirectoryEditField', True),
            ('Simulator Lanch Command', self.SimulatorLanchCommand, False,
             'SimulatorLanchCommandEditField', True),
            ('Sim Case Directory', self.SimCaseDirectory, False,
             'SimCaseDirectoryEditField', False),
            ('Sim Base Name', self.SimBaseName, False,
             'SimBaseNameEditField', False),
        )
        for r, (label, var, browse, field_name, editable) in enumerate(rows):
            label_widget = ttk.Label(self.RunPanel, text=label)
            label_widget.grid(row=r, column=0, sticky='w', pady=1)
            entry = ttk.Entry(self.RunPanel, textvariable=var,
                              state='normal' if editable else 'readonly')
            entry.grid(row=r, column=1, sticky='ew', padx=4)
            setattr(self, field_name, entry)
            setattr(self, field_name + 'Label', label_widget)
            if browse:
                self.RunDirectoryBrowseButton = ttk.Button(
                    self.RunPanel, text='Browse', command=self._browse_rundir)
                self.RunDirectoryBrowseButton.grid(row=r, column=2)

        r = len(rows)
        self.NumberofIterationsEditFieldLabel = ttk.Label(
            self.RunPanel, text='Number of Iterations')
        self.NumberofIterationsEditFieldLabel.grid(
            row=r, column=0, sticky='w')
        self.NumberofIterationsEditField = ttk.Entry(
            self.RunPanel, textvariable=self.NumberofIterations, width=10)
        self.NumberofIterationsEditField.grid(
            row=r, column=1, sticky='w', padx=4)
        self.StartingCaseDropDownLabel = ttk.Label(
            self.RunPanel, text='Starting Case')
        self.StartingCaseDropDownLabel.grid(
            row=r + 1, column=0, sticky='w')
        self.StartingCaseDropDown = ttk.Combobox(
            self.RunPanel, textvariable=self.StartingCase, values=(), width=12,
            state='readonly')
        self.StartingCaseDropDown.grid(row=r + 1, column=1, sticky='w',
                                       padx=4)
        self.SaveLoopParametersFileOnlyCheckBox = ttk.Checkbutton(
            self.RunPanel, text='Save Loop Parameters File Only',
            variable=self.SaveLoopParametersFileOnly)
        self.SaveLoopParametersFileOnlyCheckBox.grid(
            row=r + 2, column=0, columnspan=2, sticky='w', pady=(4, 0))

        buttons = ttk.Frame(self.RunPanel)
        buttons.grid(row=r + 3, column=0, columnspan=3, sticky='w',
                     pady=(6, 0))
        self.StartButton = ttk.Button(buttons, text='Start',
                                      command=self._start)
        self.StartButton.pack(side='left')
        self.TerminateButton = ttk.Button(buttons, text='Terminate',
                                          command=self._terminate,
                                          state='disabled')
        self.TerminateButton.pack(side='left', padx=(6, 0))

        self.ProgressPanel = ttk.Frame(tab, padding=6)
        self.ProgressPanel.grid(row=1, column=0, sticky='nsew', pady=(6, 0))
        self.ProgressPanel.columnconfigure(0, weight=1)
        self.ProgressPanel.rowconfigure(1, weight=1)
        self.LoopProgressLabel = ttk.Label(
            self.ProgressPanel, text='Loop Progress:',
            font=('TkDefaultFont', 10, 'bold'), foreground='blue')
        self.LoopProgressLabel.grid(row=0, column=0, sticky='w')
        self.TextArea = tk.Text(self.ProgressPanel, height=10, wrap='word')
        self.TextArea.grid(row=1, column=0, sticky='nsew')
        self.log = self.TextArea
        self.LoopProgress = _ProgressValue()
        return tab

    # ---------------------------------------------------------- Mismatch --

    def _mismatch_tab(self, master):
        """FAHM's third main tab: three views of the same misfit.

        A toggle group picks one of ``Plots``, ``WellBars`` and
        ``CaseBars``, and the tab group below follows it -- the two are
        one control shown twice, which is what
        ``MismatchButtonGroupSelectionChanged`` keeps in step.
        """
        tab = ttk.Frame(master, padding=8)
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)

        self.MismatchButtonGroup = ttk.Frame(tab)
        self.MismatchButtonGroup.grid(row=0, column=0, sticky='w')
        for column, name in enumerate(MISMATCH_VIEWS):
            ttk.Radiobutton(self.MismatchButtonGroup, text=name,
                            variable=self.MismatchView, value=name,
                            command=self._mismatch_view_changed).grid(
                row=0, column=column, padx=(0, 8))

        self.MismatchTabGroup = ttk.Notebook(tab)
        self.MismatchTabGroup.grid(row=1, column=0, sticky='nsew',
                                   pady=(6, 0))
        self.MismatchTabGroup.add(self._plots_tab(self.MismatchTabGroup),
                                  text='Plots')
        self.MismatchTabGroup.add(
            self._bars_tab(self.MismatchTabGroup, 'WellBars'),
            text='WellBars')
        self.MismatchTabGroup.add(
            self._bars_tab(self.MismatchTabGroup, 'CaseBars'),
            text='CaseBars')
        return tab

    def _plots_tab(self, master):
        tab = ttk.Frame(master, padding=6)
        tab.columnconfigure(2, weight=1)
        tab.rowconfigure(0, weight=1)

        self.PlotsIndexSelectPanel = self._index_select(tab, 'Plots',
                                                        wells=True)
        self.PlotsIndexSelectPanel.grid(row=0, column=0, sticky='nsw')

        scores = ttk.LabelFrame(tab, text='Mismatch Scores', padding=6)
        scores.grid(row=0, column=1, sticky='nsw', padx=(8, 8))
        ttk.Label(scores, text='Well').grid(row=0, column=1)
        ttk.Label(scores, text='Case').grid(row=0, column=2)
        # One row per quantity, a well score beside a whole-case score.
        self.score_fields = {}
        for row, quantity in enumerate(QUANTITIES, start=1):
            ttk.Label(scores, text=SCORE_LABELS[quantity]).grid(
                row=row, column=0, sticky='w', pady=1)
            for column, scope in ((1, 'Well'), (2, 'Case')):
                var = tk.StringVar()
                ttk.Entry(scores, textvariable=var, width=12,
                          state='readonly').grid(row=row, column=column,
                                                 padx=2, pady=1)
                self.score_fields[(scope, quantity)] = var

        # Plot1Panel .. Plot4Panel, two by two.
        self.plot_panels = []
        self.plot_axes = []
        grid = ttk.Frame(tab)
        grid.grid(row=0, column=2, sticky='nsew')
        for i in range(4):
            panel = ttk.LabelFrame(grid, text='Plot%d' % (i + 1), padding=2)
            panel.grid(row=i // 2, column=i % 2, sticky='nsew', padx=2,
                       pady=2)
            grid.columnconfigure(i % 2, weight=1)
            grid.rowconfigure(i // 2, weight=1)
            self.plot_panels.append(panel)
            self.plot_axes.append(
                _axes_in(panel) if self._render_plot_canvases else None)
        return tab

    def _bars_tab(self, master, view):
        """``WellBars`` and ``CaseBars``: one bar panel and its selection.

        The two differ only in that case bars are not per well, so
        ``CaseBars`` has no well list -- FAHM omits
        ``CaseBarsWellsListBox`` for the same reason.
        """
        tab = ttk.Frame(master, padding=6)
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=1)
        panel = self._index_select(tab, view, wells=(view == 'WellBars'))
        panel.grid(row=0, column=0, sticky='nsw')
        bars = ttk.LabelFrame(tab, text='Bars', padding=2)
        bars.grid(row=0, column=1, sticky='nsew', padx=(8, 0))
        self.bar_panels[view] = bars
        self.bar_axes[view] = (
            _axes_in(bars) if self._render_plot_canvases else None)
        return tab

    def _index_select(self, master, view, wells):
        """``<view>IndexSelectPanel``: what to draw, and for what.

        ``Plots`` offers seven quantities; the two bar views add Profile
        and Saturation, which have no time series to plot but do have a
        score to compare.
        """
        panel = ttk.LabelFrame(master, text='%s' % view, padding=6)

        ttk.Label(panel, text='Cases').grid(row=0, column=0, sticky='w')
        self.view_cases[view] = tk.Listbox(panel, height=8, width=14,
                                           exportselection=False)
        self.view_cases[view].grid(row=1, column=0, sticky='ns', padx=(0, 6))

        if wells:
            ttk.Label(panel, text='Wells').grid(row=0, column=1, sticky='w')
            self.view_wells[view] = tk.Listbox(panel, height=8, width=14,
                                               exportselection=False)
            self.view_wells[view].grid(row=1, column=1, sticky='ns')
            self.view_wells[view].bind(
                '<<ListboxSelect>>',
                lambda _e, v=view: self._selection_changed(v))

        boxes = ttk.Frame(panel)
        boxes.grid(row=2, column=0, columnspan=2, sticky='w', pady=(6, 0))
        names = PLOT_CURVES if view == 'Plots' else BAR_CURVES
        for i, name in enumerate(names):
            var = tk.BooleanVar(value=(name in ('Oil', 'Water')))
            ttk.Checkbutton(boxes, text=CURVE_LABELS.get(name, name),
                            variable=var,
                            command=lambda v=view:
                                self._selection_changed(v)).grid(
                row=i // 3, column=i % 3, sticky='w', padx=(0, 8))
            self.view_curves[(view, name)] = var
        return panel

    def _selection_changed(self, view):
        """A different well or a different curve redraws that view."""
        if view == 'Plots':
            self._show_scores()
            self.draw_plots()
        else:
            self.draw_bars()

    def _mismatch_view_changed(self):
        """``MismatchButtonGroupSelectionChanged``: the toggle group and
        the tab group show the same choice, so moving one moves the
        other."""
        self.MismatchTabGroup.select(MISMATCH_VIEWS.index(
            self.MismatchView.get()))

    # ------------------------------------------------- parameter tables --

    def _refresh_limits(self, name):
        """Fill a parameter's table with one row per region."""
        table = self.param_tables.get(name)
        if table is None:
            return
        table.delete(*table.get_children())
        absolute = self.param_mode[name].get() == 'useAbs'
        for region, low, high in default_limits(self.model, name, absolute):
            table.insert('', 'end', values=(region, '%g' % low, '%g' % high))

    def _parameter_mode_changed(self, name):
        """Port of ``ParameterButtonGroupSelectionChanged``: the table is
        rebuilt, because relative and absolute limits are different
        quantities rather than different renderings of one."""
        self._refresh_limits(name)

    def _parameter_toggled(self, name, *, populate=True):
        """Port of the checkbox callback: an unselected parameter's
        controls are disabled and its table emptied."""
        if name not in self.param_widgets:
            return
        on = bool(self.param_on[name].get())
        for widget in self.param_widgets[name]:
            try:
                widget.state(['!disabled'] if on else ['disabled'])
            except tk.TclError:
                widget.configure(state='normal' if on else 'disabled')
        if on:
            if populate and not self.param_tables[name].get_children():
                self._refresh_limits(name)
        else:
            self.param_tables[name].delete(
                *self.param_tables[name].get_children())

    def _edit_limit(self, event, name):
        """Edit one region's limits in place. The table is editable in
        FAHM (ColumnEditable = [false true true]), so it is here."""
        table = self.param_tables[name]
        row = table.identify_row(event.y)
        column = table.identify_column(event.x)
        if not row or column == '#1':          # the region column is fixed
            return
        index = int(column[1:]) - 1
        values = list(table.item(row, 'values'))

        popup = tk.Toplevel(self)
        popup.title('%s %s' % (name, table.heading(column)['text']))
        var = tk.StringVar(value=values[index])
        ttk.Entry(popup, textvariable=var, width=18).pack(padx=8, pady=8)

        def apply():
            try:
                float(var.get())
            except ValueError:
                messagebox.showerror('Not a number', var.get(), parent=popup)
                return
            values[index] = var.get()
            table.item(row, values=values)
            popup.destroy()

        ttk.Button(popup, text='OK', command=apply).pack(pady=(0, 8))
        popup.transient(self.winfo_toplevel())
        popup.grab_set()

    def limits_of(self, name):
        """The rows currently shown for a parameter."""
        table = self.param_tables.get(name)
        if table is None:
            return []
        rows = []
        for item in table.get_children():
            region, low, high = table.item(item, 'values')
            rows.append((int(float(region)), float(low), float(high)))
        return rows

    # -------------------------------------------------------- callbacks --

    def _objective_mode_changed(self):
        """Port of ``ObjectiveButtonGroupSelectionChanged``: the wizard
        and the explicit table are alternatives, never both."""
        wizard = self.ObjectiveMode.get() == 'wizard'
        if wizard:
            self.WellstoMatchPanel.grid()
            self.RelativeWeightsPanel.grid()
            self.ObjectionFunctionUITable.grid_remove()
        else:
            self.WellstoMatchPanel.grid_remove()
            self.RelativeWeightsPanel.grid_remove()
            self.ObjectionFunctionUITable.grid()

    @staticmethod
    def _set_enabled(widget, enabled):
        widget.state(['!disabled'] if enabled else ['disabled'])

    @staticmethod
    def _phase_enabled(deck, phase):
        runspec = (deck or {}).get('RUNSPEC') or {}
        for key, value in runspec.items():
            if str(key).upper() == phase.upper():
                if value is None:
                    return True
                try:
                    return bool(value)
                except (TypeError, ValueError):
                    return True
        return False

    def _path_state_changed(self, *_args):
        """Enable Create Project iff both selected files exist."""
        ready = (os.path.isfile(self.ModelPath.get()) and
                 os.path.isfile(self.SimulatorPath.get()))
        self.dataCheck['startingModel'] = os.path.isfile(self.ModelPath.get())
        self.dataCheck['simulator'] = os.path.isfile(self.SimulatorPath.get())
        self._set_enabled(self.CreatProjectButton, ready)

    def _monitor_source_changed(self, name):
        widgets = self.monitor_widgets[name]
        enabled = bool(self.monitor_use[name].get())
        from_file = self.monitor_source[name].get() == 'file'
        self._set_enabled(widgets['path'], enabled and from_file)
        self._set_enabled(widgets['browse'], enabled and from_file)
        self._model_setup_check()

    def _monitor_toggled(self, name):
        """Port the five monitoring checkbox callbacks."""
        enabled = bool(self.monitor_use[name].get())
        widgets = self.monitor_widgets[name]
        if name in self.monitor_source:
            self._set_enabled(widgets['from_model'], enabled)
            self._set_enabled(widgets['from_file'], enabled)
            self.monitor_source[name].set('model')
            self._set_enabled(widgets['path'], False)
            self._set_enabled(widgets['browse'], False)
        else:
            self._set_enabled(widgets['path'], enabled)
            self._set_enabled(widgets['browse'], enabled)

        affected = ('Oil', 'Water', 'Gas') if name == 'Rates' else (name,)
        phases = {'Oil': 'OIL', 'Water': 'WATER', 'Gas': 'GAS'}
        for quantity in affected:
            active = enabled and (
                self._phase_enabled(self.deck, phases[quantity])
                if quantity in phases else True)
            self.weights[quantity].set('1' if active else '0')
            self._set_enabled(self.weight_widgets[quantity], active)
        self._model_setup_check()

    def _model_setup_check(self):
        """Stateless form of ``modelSetUpCheck``.

        FAHM leaves prior ``dataCheck.monitorData`` values latched.  That
        source defect is registered as FAHM-FIX-008; PRST recomputes every
        flag so unchecking the final monitoring source disables Proceed.
        """
        self.dataCheck['startingModel'] = os.path.isfile(self.ModelPath.get())
        self.dataCheck['simulator'] = os.path.isfile(self.SimulatorPath.get())
        flags = {}
        for name, _has_source in MONITORING:
            selected = bool(self.monitor_use[name].get())
            if name in self.monitor_source:
                valid_source = (
                    self.monitor_source[name].get() == 'model' or
                    os.path.isfile(self.monitor_path[name].get()))
            else:
                valid_source = os.path.isfile(self.monitor_path[name].get())
            flags[name.lower()] = selected and valid_source
        self.dataCheck['monitorData'] = flags
        ready = (self.dataCheck['startingModel'] and
                 self.dataCheck['simulator'] and any(flags.values()))
        for widget in (self.ModelProceedButton, self.ObjectiveProceedButton,
                       self.ParameterProceedButton):
            self._set_enabled(widget, ready)

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title='Select Eclipse data file',
            filetypes=[('ECLIPSE deck', '*.DATA *.data')])
        if path:
            self.ModelPath.set(path)
            self.SimBaseName.set(os.path.splitext(os.path.basename(path))[0])
            self.SimCaseDirectory.set(os.path.dirname(path))
        self._path_state_changed()

    def _browse_simulator(self):
        path = filedialog.askopenfilename(title='Select the simulator')
        if path:
            self.SimulatorPath.set(path)
        self._path_state_changed()

    def _browse_rundir(self):
        path = filedialog.askdirectory(title='Select a work directory')
        if path:
            self.RunDirectory.set(path)

    def _browse_monitor(self, name):
        paths = filedialog.askopenfilenames(
            title='Select the %s data' % name,
            filetypes=[('Monitoring data', '*.xls *.xlsx *.csv *.txt')])
        if paths:
            self.monitor_path[name].set(';'.join(paths))
            if name in self.monitor_source:
                self.monitor_source[name].set('file')
                self._monitor_source_changed(name)
        self._model_setup_check()

    def _current_quantity(self):
        """The Wells-to-Match tab in front."""
        return self.WellstoMatchTabGroup.tab(
            self.WellstoMatchTabGroup.select(), 'text')

    def _copy_selection_changed(self):
        """Port of ``CopySelectionDropDownValueChanged`` and
        ``WellstoMatchTabGroupSelectionChanged``, which share a body:
        copying a tab onto itself does nothing, so OK is disabled for
        it."""
        if not hasattr(self, 'CopySelectionOKButton'):
            return
        target = self._copy_target_quantity()
        self.CopySelectionOKButton.state(
            ['disabled'] if target == self._current_quantity()
            else ['!disabled'])

    def _copy_target_quantity(self):
        tag = self.CopySelection.get().replace(COPY_SELECTION % '', '')
        for quantity, source_tag in COPY_SELECTION_NAMES.items():
            if tag == source_tag:
                return quantity
        return tag

    def _copy_selection(self):
        """Port of ``CopySelectionOKButtonPushed``: the tab in front
        replaces the target's three lists outright."""
        source = self._current_quantity()
        target = self._copy_target_quantity()
        if target == source or target not in self.well_lists:
            return
        for name in ('Ignore', 'Match', 'Emphasize'):
            box = self.well_lists[target][name]
            box.delete(0, 'end')
            for well in self.well_lists[source][name].get(0, 'end'):
                box.insert('end', well)
        self._say('Copied %s to %s.' % (source, target))

    def _move_wells(self, quantity, source, target):
        """Port of ``WellsToMatchTransfer``."""
        boxes = self.well_lists[quantity]
        src, dst = boxes[source], boxes[target]
        selected = src.curselection()
        # MATLAB ``union(final, select, 'sorted')`` sorts and removes
        # duplicates before replacing the destination Items.
        merged = sorted(set(dst.get(0, 'end')).union(src.get(i)
                                                       for i in selected))
        dst.delete(0, 'end')
        for name in merged:
            dst.insert('end', name)
        for index in reversed(selected):
            src.delete(index)

    def _creat_project(self):
        """Read the deck and populate everything that depends on it."""
        if not self.ModelPath.get():
            messagebox.showwarning('No model', 'Select a deck first.')
            return
        try:
            self.deck = read_case(self.config())
            self.model = self._build_model()
        except Exception as exc:
            messagebox.showerror('Could not read the deck', str(exc))
            self.deck = None
            return

        wells = self._well_names(self.deck)
        self._unlock_after_project()

        dims = (self.deck.get('RUNSPEC') or {}).get('cartDims')
        lines = ['Grid            : %s' % (dims,),
                 'Wells           : %d' % len(wells),
                 'Model           : %s' % ('built'
                                           if self.model else 'not available')]
        for section in ('RUNSPEC', 'GRID', 'PROPS', 'REGIONS', 'SOLUTION',
                        'SCHEDULE'):
            keys = sorted(self.deck.get(section, {}) or {})
            lines.append('%-15s : %d keywords' % (section, len(keys)))
            if keys:
                lines.append('                  %s' % ', '.join(keys[:14]))
        self.ModelSummary.delete('1.0', 'end')
        self.ModelSummary.insert('1.0', '\n'.join(lines))
        self._say('Project created: %d wells.' % len(wells))

    def _unlock_after_project(self):
        """Apply the UI-only tail of ``CreatProjectButtonPushed``."""
        for name, _ in MONITORING:
            self._set_enabled(self.monitor_widgets[name]['check'], True)
        self.monitor_use['Rates'].set(True)
        self.monitor_source['Rates'].set('model')
        self._set_enabled(self.monitor_widgets['Rates']['from_model'], True)
        self._set_enabled(self.monitor_widgets['Rates']['from_file'], True)
        for widget in (self.ModelProceedButton, self.ObjectiveProceedButton,
                       self.ParameterProceedButton):
            self._set_enabled(widget, True)

    def _build_model(self):
        """The model, so the limits tables can show real regions and
        values. A deck the model builder cannot handle still gives a
        usable project -- the tables just fall back to multipliers."""
        try:
            from PRSTCore.ad_core.initialization.init_eclipse_problem_ad \
                import init_eclipse_problem_ad
            _, model, _, _ = init_eclipse_problem_ad(self.ModelPath.get())
            return model
        except Exception:
            return None

    def _model_proceed(self):
        if self.deck is None:
            return
        self._apply_phase_gating()
        wells = self._well_names(self.deck)
        for boxes in self.well_lists.values():
            for box in boxes.values():
                box.delete(0, 'end')
            for well in wells:
                boxes['Match'].insert('end', well)

        for name in PARAMETER_TABS:
            if self.param_on[name].get() and not self.param_tables[
                    name].get_children():
                self._refresh_limits(name)
        self.SetUpTabGroup.select(1)

    def _apply_phase_gating(self):
        """Port the phase-dependent UI portion of Model Proceed."""
        phases = {
            'Oil': self._phase_enabled(self.deck, 'OIL'),
            'Water': self._phase_enabled(self.deck, 'WATER'),
            'Gas': self._phase_enabled(self.deck, 'GAS'),
        }
        for quantity, enabled in phases.items():
            self.weights[quantity].set('1' if enabled else '0')
            self._set_enabled(self.weight_widgets[quantity], enabled)

        fluid_enabled = {
            'kro': phases['Oil'],
            'krw': phases['Water'],
            'Swu': phases['Water'],
            'Swl': phases['Water'],
            'Swcr': phases['Water'],
            'krg': phases['Gas'],
            'Sgu': phases['Gas'],
            'Sgl': phases['Gas'],
            'Sgcr': phases['Gas'],
            'Sowcr': phases['Water'] and phases['Oil'],
            'Sogcr': phases['Gas'] and phases['Oil'],
        }
        for name, enabled in fluid_enabled.items():
            self._set_enabled(self.param_check_widgets[name], enabled)

    def _objective_proceed(self):
        self.SetUpTabGroup.select(2)

    def _parameter_proceed(self):
        self.MainTabGroup.select(1)
        simulator = self.SimulatorPath.get()
        low = simulator.lower()
        if 'eclipse' in low:
            command = 'eclrun eclipse'
        elif 'e300' in low:
            command = 'eclrun e300'
        elif 'tnavigator' in low:
            command = (simulator + ' --no-dump-res --ecl-root -e -i -r -u '
                       '--no-gui --ignore-lock --use-gpu ')
        else:
            command = ''
        self.SimulatorLanchCommand.set(command)

    def _start(self):
        if not self.ModelPath.get():
            messagebox.showwarning('No model', 'Select a deck first.')
            return
        config = self.config()
        if not config.parameters:
            messagebox.showwarning('No parameters',
                                   'Select at least one parameter on the '
                                   'Parameter tab.')
            return
        if self.SaveLoopParametersFileOnly.get():
            self._save_loop_parameters(config)
            return

        self._stop.clear()
        self.StartButton.state(['disabled'])
        self.TerminateButton.state(['!disabled'])
        self.LoopProgress['value'] = 0
        self._say('Running %s ...' % os.path.basename(config.simulator))
        self._worker = threading.Thread(target=self._work, args=(config,),
                                        daemon=True)
        self._worker.start()

    def _terminate(self):
        """Ask the loop to stop after the evaluation in flight.

        A simulator run cannot be interrupted safely mid-write, so this
        sets a flag the objective checks between evaluations rather than
        killing the process.
        """
        self._stop.set()
        self._say('Terminating after the current simulator run ...')

    def _work(self, config):
        try:
            out = run_history_match(config, verbose=False,
                                    should_stop=self._stop.is_set)
            self.result = out
            trace = ', '.join('%.4f' % v for v in out['history']['val'])
            self._messages.put(('done',
                                'baseline misfit %.6e -> %.4f of it\n'
                                '  multipliers: %s\n  objective: %s'
                                % (out['baseline'], out['value'],
                                   {k: round(v, 4) for k, v in
                                    out['multipliers'].items()}, trace)))
        except Exception as exc:
            self._messages.put(('error', '%s: %s' % (type(exc).__name__, exc)))

    def _save_loop_parameters(self, config):
        """Write the configuration without running anything."""
        os.makedirs(config.work_dir, exist_ok=True)
        path = os.path.join(config.work_dir, 'loop_parameters.txt')
        with open(path, 'w') as fh:
            fh.write('deck        = %s\n' % config.deck_path)
            fh.write('simulator   = %s\n' % config.simulator)
            fh.write('iterations  = %d\n' % config.max_iterations)
            fh.write('weights     = %s\n' % config.weights)
            for row in self.parameter_config():
                name, on, scaling, box, rel, uniform = row[:5] + row[6:]
                if not on:
                    continue
                fh.write('%-12s scaling=%-6s %s=%s uniform=%s\n'
                         % (name, scaling,
                            'box' if box else 'relative', box or rel,
                            uniform))
        self._say('Wrote %s' % path)

    def _drain(self):
        """Move worker-thread results onto the Tk thread."""
        while True:
            try:
                kind, text = self._messages.get_nowait()
            except queue.Empty:
                break
            self.log.insert('end', text + '\n')
            self.log.see('end')
            self._say(text.splitlines()[0] if kind == 'done'
                      else 'Failed: ' + text)
            self.LoopProgress['value'] = 100 if kind == 'done' else 0
            self.StartButton.state(['!disabled'])
            self.TerminateButton.state(['disabled'])
            if kind == 'done':
                self._show_mismatch()
        self._drain_after_id = self.after(100, self._drain)

    # --------------------------------------------------- mismatch results --

    def _show_mismatch(self):
        """Fill the Mismatch tab from the run that just finished."""
        result = self.result or {}
        cases = ['case%d' % i
                 for i in range(len(result.get('history', {}).get('val', [])))]
        for view in MISMATCH_VIEWS:
            box = self.view_cases[view]
            box.delete(0, 'end')
            for name in cases or ['case0']:
                box.insert('end', name)
            if view in self.view_wells:
                wells = self.view_wells[view]
                wells.delete(0, 'end')
                for name in result.get('wells', []):
                    wells.insert('end', name)
        self._show_scores()
        self.draw_plots()
        self.draw_bars()
        self.MainTabGroup.select(2)

    def draw_plots(self):
        """History against simulation, one curve per panel.

        Four panels for however many curves are ticked: the first four
        win, which is what having four panels means. Observed is drawn as
        points and simulated as a line, so a step the history does not
        cover is visibly absent rather than interpolated through.
        """
        series = (self.result or {}).get('series')
        available = (series or {}).get('observed') or {}
        # Tracer has a checkbox because FAHM has one, but nothing
        # produces a tracer series yet: the summary reader does not carry
        # it. Ticking it leaves its panel blank rather than drawing a
        # flat zero that would read as a matched tracer.
        selected = [c for c in PLOT_CURVES
                    if self.view_curves[('Plots', c)].get()
                    and c in available]
        well = self._selected_well('Plots')

        for i, axes in enumerate(self.plot_axes):
            if axes is None:
                continue
            axes.clear()
            if series is None or well is None or i >= len(selected):
                axes.set_xticks([])
                axes.set_yticks([])
                _draw(axes)
                continue

            curve = selected[i]
            column = series['wells'].index(well)
            time = series['time'] / 86400.0        # days, as FAHM plots
            axes.plot(time, series['observed'][curve][:, column], 'o',
                      markersize=3, label='history')
            axes.plot(time, series['simulated'][curve][:, column], '-',
                      linewidth=1.2, label='simulated')
            axes.set_title('%s  %s' % (well, CURVE_LABELS.get(curve, curve)),
                           fontsize=8)
            axes.tick_params(labelsize=7)
            axes.set_xlabel('days', fontsize=7)
            axes.legend(fontsize=6, loc='best')
            _draw(axes)

    def draw_bars(self):
        """The two bar views.

        ``WellBars`` compares the wells against each other at the
        baseline; ``CaseBars`` compares the iterations against each
        other. Both answer "where is the misfit", at different
        granularity.
        """
        result = self.result or {}

        axes = self.bar_axes.get('WellBars')
        if axes is not None:
            axes.clear()
            wells = result.get('wells') or []
            scores = result.get('well_scores') or {}
            selected = [c for c in BAR_CURVES
                        if self.view_curves[('WellBars', c)].get()
                        and c in scores]
            total = None
            for curve in selected:
                values = np.asarray(scores[curve], dtype=float)
                total = values if total is None else total + values
            if wells and total is not None:
                axes.bar(range(len(wells)), total[:len(wells)])
                axes.set_xticks(range(len(wells)))
                axes.set_xticklabels(wells, rotation=60, fontsize=6)
                axes.set_title('misfit by well', fontsize=8)
            else:
                axes.set_xticks([])
                axes.set_yticks([])
            axes.tick_params(labelsize=7)
            _draw(axes)

        axes = self.bar_axes.get('CaseBars')
        if axes is not None:
            axes.clear()
            values = list((result.get('history') or {}).get('val') or [])
            if values:
                axes.bar(range(len(values)), values)
                axes.set_xticks(range(len(values)))
                axes.set_xticklabels(['case%d' % i for i in
                                      range(len(values))], rotation=60,
                                     fontsize=6)
                axes.set_title('objective by case', fontsize=8)
            else:
                axes.set_xticks([])
                axes.set_yticks([])
            axes.tick_params(labelsize=7)
            _draw(axes)

    def _selected_well(self, view):
        """The well highlighted in a view's list, or the first one."""
        box = self.view_wells.get(view)
        if box is None:
            return None
        names = list(box.get(0, 'end'))
        if not names:
            return None
        selection = box.curselection()
        return names[selection[0]] if selection else names[0]

    def _show_scores(self):
        """The Mismatch Scores panel: a per-well figure beside the
        whole-case one, per quantity.

        The Well column follows the Plots view's well selection; with no
        well selected there is nothing well-specific to show and it stays
        empty rather than repeating the case figure.
        """
        result = self.result or {}
        case = result.get('scores') or {}
        well = result.get('well_scores') or {}

        selection = self.view_wells['Plots'].curselection()
        index = selection[0] if selection else None

        for quantity in QUANTITIES:
            self.score_fields[('Case', quantity)].set(
                '%.6g' % case[quantity] if quantity in case else '')
            values = well.get(quantity)
            self.score_fields[('Well', quantity)].set(
                '%.6g' % values[index]
                if index is not None and values is not None
                and index < len(values) else '')

    def _say(self, text):
        # FAHM has no status bar.  Keep the latest message as application
        # state while the visible progress log remains ``TextArea``.
        self._status_text = str(text)

    def destroy(self):
        for after_id in (self._startup_after_id,
                         getattr(self, '_drain_after_id', None)):
            if after_id is not None:
                try:
                    self.after_cancel(after_id)
                except tk.TclError:
                    pass
        super().destroy()

    # ------------------------------------------------------------ config --

    def parameter_config(self):
        """FAHM's ``app.config``: one row per tunable parameter.

        Columns, as ``StartButtonPushed`` reads them::

            name, enabled, scaling, boxLims, relativeLimits, subset,
            uniformLimits
        """
        rows = []
        for name in PARAMETERS:
            enabled = bool(self.param_on[name].get())
            absolute = self.param_mode[name].get() == 'useAbs'
            limits = self.limits_of(name) or \
                default_limits(self.model, name, absolute)
            rows.append(config_row(name, enabled, absolute, limits))
        return rows

    def config(self):
        """Collect the widget state into a :class:`FahmConfig`.

        FahmConfig carries one interval per parameter, so a multi-region
        parameter is narrowed to the tightest limits any region declares
        -- the backend applies a single multiplier per parameter, and
        offering per-region control it cannot honour would be worse than
        saying so.
        """
        limits = {}
        for name in PARAMETERS:
            if not self.param_on[name].get():
                continue
            rows = self.limits_of(name)
            if not rows:
                continue
            key = BACKEND_NAME.get(name, name.lower())
            limits[key] = (max(r[1] for r in rows), min(r[2] for r in rows)) \
                if len(rows) > 1 else (rows[0][1], rows[0][2])

        weights = {}
        for key, q in (('oil', 'Oil'), ('water', 'Water'), ('gas', 'Gas'),
                       ('bhp', 'BHP')):
            try:
                weights[key] = float(self.weights[q].get())
            except ValueError:
                weights[key] = 0.0

        return FahmConfig(
            deck_path=self.ModelPath.get(),
            work_dir=self.RunDirectory.get(),
            simulator=self.SimulatorPath.get(),
            parameters=[BACKEND_NAME.get(n, n.lower()) for n in PARAMETERS
                        if self.param_on[n].get()],
            weights=weights,
            max_iterations=int(self.NumberofIterations.get()),
            parameter_limits=limits)

    @staticmethod
    def _well_names(deck):
        """The well names the schedule declares.

        FAHM reads ``deck.SCHEDULE.control(end).WELSPECS``. PRSTCore's
        reader produces a flat ``SCHEDULE.WELSPECS`` instead, so both
        shapes are accepted. Empty rows are skipped -- the reader emits
        one for the keyword line itself -- and WCONHIST/WCONINJH serve as
        a fallback for a deck that only names its wells there.
        """
        schedule = deck.get('SCHEDULE') or {}

        sources = []
        for control in reversed(schedule.get('control') or []):
            sources.append((control or {}).get('WELSPECS'))
        sources.extend(schedule.get(k) for k in ('WELSPECS', 'WCONHIST',
                                                 'WCONINJH', 'WCONPROD',
                                                 'WCONINJE'))

        for rows in sources:
            names = []
            for row in rows or []:
                if not row:
                    continue
                name = str(row[0]).strip().strip("'").strip('"')
                if name and name not in names:
                    names.append(name)
            if names:
                return names
        return []


def main():
    root = tk.Tk()
    root.title(APP_TITLE)
    root.geometry('%dx%d' % APP_SIZE)
    FahmApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
