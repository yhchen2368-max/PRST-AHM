"""Network topology builder for well-based networks.

1:1 Python translation of MRST modules/network-models/Network.m

Creates a network (graph) between nodes defined by a set of wells.
Supports multiple topology types: all-to-all, injectors-to-producers,
flow-diagnostics-based, and user-defined edges.
"""

import numpy as np
import networkx as nx

from PRSTCore.visualization.diagnostics.preprocessorGUI.utils import (
    computePressureAndDiagnostics,
)
from PRSTCore.visualization.diagnostics.utils.helpers import (
    get_field,
    normalize_cell_indices,
    num_cells,
)


class Network:
    """Network of wells with graph-based topology.

    Parameters
    ----------
    W : list of dict
        Well structures.
    G : dict
        Grid structure with cells.centroids.
    type : str
        Network type: 'all_to_all', 'injectors_to_producers',
        'fd_preprocessor', 'fd_postprocessor', 'user_defined_edges'.
    injectors : list, optional
        Injector well indices.
    producers : list, optional
        Producer well indices.
    edges : ndarray, optional
        User-defined edges (Nx2).
    problem : dict, optional
        Packed simulation problem (for flow diagnostics).
    flow_filter : float, optional
        Flow threshold for filtering connections.
    state_number : int, optional
        State index for flow diagnostics.
    """

    def __init__(self, W, G, type="all_to_all", injectors=None,
                 producers=None, edges=None, problem=None,
                 flow_filter=0.0, state_number=None, **kwargs):
        self.G = G
        self.W = W

        if edges is not None:
            type = "user_defined_edges"
        if injectors is not None and producers is not None:
            type = "injectors_to_producers"

        if state_number is None and problem is not None:
            schedule = _setup_field(problem, "schedule")
            step = _field(schedule, "step", {})
            state_number = len(_field(step, "val", []))

        # Count wells and cells
        nWcells = [len(_well_cell_vector(w)) for w in W]
        numNodes = sum(nWcells)
        grid_cells = num_cells(G)

        # Build node table
        nodes_data = []
        node_idx = 0
        for iw, w in enumerate(W):
            w_cells = _well_cell_vector(w)
            zero_based_cells = normalize_cell_indices(w_cells, grid_cells)
            for ic, cell_no in enumerate(w_cells):
                centroids_arr = np.asarray(G["cells"]["centroids"])
                zcell = int(zero_based_cells[ic]) if ic < zero_based_cells.size else int(cell_no)
                if 0 <= zcell < centroids_arr.shape[0]:
                    centroid = centroids_arr[zcell]
                else:
                    centroid = np.zeros(3)
                nodes_data.append({
                    "node": node_idx,
                    "well": iw,
                    "subwell": ic,
                    "name": w.get("name", f"W{iw}"),
                    "cells": [],
                    "type": 1,
                    "XData": float(centroid[0]),
                    "YData": float(centroid[1]) if len(centroid) > 1 else 0.0,
                    "ZData": float(centroid[2]) if len(centroid) > 2 else 0.0,
                })
                node_idx += 1

        # Build graph based on topology type
        G_nx = nx.Graph()
        G_nx.add_nodes_from(range(numNodes))

        fd_types = ["fd_preprocessor", "fd_postprocessor"]
        if type == "all_to_all":
            # ``graph(ones(numNodes) - eye(numNodes))``: the complete
            # graph on every *node*, and a node is a well perforation, not
            # a well. Two perforations of the same well are therefore
            # joined as well -- excluding them (as this used to) silently
            # drops edges for any multi-perforation well, and GPSNet then
            # builds a network with fewer flow paths than MRST's.
            for i in range(numNodes):
                for j in range(i + 1, numNodes):
                    G_nx.add_edge(i, j)
        elif type == "user_defined_edges":
            assert edges is not None
            edges_arr = _normalise_node_edges(edges, numNodes)
            # ``accumarray(opt.edges(:),1)``: MRST checks both that no edge
            # names a node that does not exist and that every node is
            # reached by at least one edge. An unreferenced node has no
            # flow path, so it would sit in the network contributing
            # nothing while still being counted.
            referenced = np.asarray(edges_arr, dtype=int).ravel()
            if referenced.size:
                counts = np.bincount(referenced)
                if counts.size > numNodes:
                    raise ValueError("Edges refer to non-existing node(s)")
                if counts.size != numNodes or not np.all(counts > 0):
                    raise ValueError("Each node must have at least one edge")
            for e in edges_arr:
                G_nx.add_edge(int(e[0]), int(e[1]))
        elif type == "injectors_to_producers":
            assert injectors is not None and producers is not None
            injectors = _normalise_node_indices(injectors, numNodes)
            producers = _normalise_node_indices(producers, numNodes)
            for inj in injectors:
                for prod in producers:
                    G_nx.add_edge(int(inj), int(prod))
        elif type in fd_types:
            G_nx = self._build_flow_diagnostics_network(
                problem=problem,
                ftype=type,
                flow_filter=flow_filter,
                state_number=state_number,
                W=W,
                nWcells=nWcells,
            )
        else:
            raise ValueError(f"Type of network: {type} is not implemented")

        self.network = G_nx
        self._nodes_data = nodes_data
        self.type = type

    def _build_flow_diagnostics_network(self, problem, ftype, flow_filter, state_number, W, nWcells):
        """Build a flow-diagnostics-derived well network.

        Mirrors MRST ``Network.m`` for ``fd_preprocessor`` and
        ``fd_postprocessor``.  Edges connect injector wells to producer wells
        where ``diagnostics.wellCommunication`` exceeds ``flow_filter``.
        """
        if problem is None:
            raise ValueError("Flow-diagnostics network types require a packed problem")
        if not all(n == 1 for n in nWcells):
            raise AssertionError(
                "Flow diagnostics analysis to multiple connections between wells is not yet supported."
            )

        setup = _field(problem, "SimulatorSetup", _field(problem, "simulator_setup", None))
        if setup is None:
            raise ValueError("problem must contain SimulatorSetup")
        model = _field(setup, "model")
        schedule = _field(setup, "schedule")
        if model is None or schedule is None:
            raise ValueError("problem.SimulatorSetup must contain model and schedule")

        if ftype == "fd_preprocessor":
            state = None
            pressure_field = "pressure"
            ctrl_no = 0
        else:
            states = _output_states(problem)
            if states is None:
                raise ValueError("fd_postprocessor requires problem.OutputHandlers.states")
            state_ix = _normalise_state_index(state_number, len(states))
            state = states[state_ix]
            pressure_field = "bhp"
            ctrl_no = _control_number(schedule, state_ix)

        Wdiag = _schedule_wells(schedule, ctrl_no, fallback=W)
        state, diagnostics = computePressureAndDiagnostics(
            model,
            wells=Wdiag,
            state=state,
            firstArrival=False,
        )

        G_nx = nx.Graph()
        G_nx.add_nodes_from(range(len(W)))
        communication = np.asarray(diagnostics.wellCommunication, dtype=float)
        flat_communication = communication.ravel(order="F")
        pair_ix = np.asarray(diagnostics.WP.pairIx, dtype=int)
        vols = np.asarray(diagnostics.WP.vols, dtype=float).ravel()
        active = flat_communication > float(flow_filter)
        if active.size != pair_ix.shape[0]:
            active = active[: pair_ix.shape[0]]
        if not hasattr(G_nx, "edges_data"):
            G_nx.edges_data = {}
        for pair_no in np.flatnonzero(active):
            inj_local, prod_local = pair_ix[int(pair_no)]
            inj_well = int(diagnostics.D.inj[int(inj_local)])
            prod_well = int(diagnostics.D.prod[int(prod_local)])
            flux = float(flat_communication[int(pair_no)])
            dp = _well_pressure(state, inj_well, pressure_field) - _well_pressure(state, prod_well, pressure_field)
            trans = flux / dp if abs(dp) > np.finfo(float).eps else np.inf
            pv = float(vols[int(pair_no)]) if pair_no < vols.size else 0.0
            G_nx.add_edge(inj_well, prod_well, T=trans, pv=pv, flux=flux, dP=dp)
            key = (min(inj_well, prod_well), max(inj_well, prod_well))
            G_nx.edges_data[key] = {"T": trans, "pv": pv, "flux": flux, "dP": dp}
        return G_nx

    def plot_network(self, plottype="default", *, ax=None, colors=True,
                     on_grid=True, data=None, max_width=6.0,
                     face_color="none", edge_alpha=0.05):
        """Port of ``Network.plotNetwork``.

        ``plottype`` selects the layout and what sets the line widths:

        ``'default'``/``'spacegraph'``
            Nodes at their well perforations' grid coordinates.
        ``'circle'``
            A circular layout, with the grid suppressed -- the point is
            the topology, not where the wells are.
        ``'transmissibility'``/``'porevolume'``
            Node positions as above, but line width proportional to the
            edge's ``T`` or ``pv``. Only a flow-diagnostics network
            carries those, and asking for them on any other kind is an
            error rather than a plot of nothing.

        ``data`` overrides the widths directly and must have one entry
        per edge. Returns ``(ax, colors)``, the second being the per-edge
        colour list MRST returns as ``omap``.
        """
        import matplotlib.pyplot as plt

        edges = list(self.network.edges())
        ne = len(edges)

        line_width = 2.0
        if colors:
            cmap = plt.get_cmap('tab20')
            edge_colors = [cmap(i % cmap.N) for i in range(ne)]
            line_width = 3.0
        else:
            edge_colors = None

        if data is not None:
            data = np.atleast_1d(np.asarray(data, dtype=float)).ravel()
            if data.size != ne:
                raise ValueError('Size of data does not match number of edges')
            line_width = _scaled_widths(data, max_width)

        positions = {int(n['node']): (n['XData'], n['YData'])
                     for n in self._nodes_data}

        if plottype in ('default', 'spacegraph'):
            pass
        elif plottype == 'circle':
            positions = nx.circular_layout(self.network)
            on_grid = False
        elif plottype in ('transmissibility', 'porevolume'):
            key = 'T' if plottype == 'transmissibility' else 'pv'
            values = [self.network.get_edge_data(u, v, default={}).get(key)
                      for u, v in edges]
            if not edges or any(v is None for v in values):
                raise ValueError('Network contains no %s'
                                 % ('transmissibilities' if key == 'T'
                                    else 'pore volumes'))
            line_width = _scaled_widths(np.asarray(values, dtype=float),
                                        max_width)
        else:
            raise ValueError('Plot type not defined')

        if ax is None:
            _, ax = plt.subplots()

        if on_grid:
            from PRSTCore.visualization.grid_plots import plot_grid
            plot_grid(self.G, ax=ax, facecolor=face_color, edgecolor='k',
                      alpha=1.0, linewidth=0.3)

        nx.draw_networkx_edges(self.network, positions, edgelist=edges, ax=ax,
                               width=line_width, edge_color=edge_colors)
        nx.draw_networkx_nodes(self.network, positions, ax=ax, node_size=40)
        # ``labelnode(pg, Nodes.well, Nodes.name)``: one label per well,
        # placed at the node the well owns.
        labels = {int(n['node']): n['name'] for n in self._nodes_data}
        nx.draw_networkx_labels(self.network, positions, labels=labels,
                                font_size=10, ax=ax)
        ax.set_axis_off()
        return ax, edge_colors

    def get_edge_data(self):
        """Return edge transmissibilities and pore volumes if available."""
        T = []
        pv = []
        for u, v in self.network.edges():
            data = self.network.get_edge_data(u, v, default={}) or {}
            if not data:
                data = getattr(self.network, "edges_data", {}).get((u, v), {})
            if not data:
                data = getattr(self.network, "edges_data", {}).get((v, u), {})
            T.append(data.get("T", 1.0))
            pv.append(data.get("pv", 1.0))
        return np.array(T), np.array(pv)

    @property
    def num_edges(self):
        return self.network.number_of_edges()

    @property
    def num_nodes(self):
        return self.network.number_of_nodes()


def _scaled_widths(values, max_width):
    """``maxWidth*data/max(data)``, guarding the all-zero case MATLAB
    would turn into NaN."""
    values = np.asarray(values, dtype=float).ravel()
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    if peak == 0.0:
        return np.full(values.size, float(max_width))
    return float(max_width) * values / peak


def _field(obj, name, default=None):
    return get_field(obj, name, default)


def _setup_field(problem, name):
    setup = _field(problem, "SimulatorSetup", _field(problem, "simulator_setup", {}))
    return _field(setup, name, {})


def _well_cell_vector(well):
    cells = _field(well, "cells", [0])
    arr = np.asarray(cells, dtype=int).ravel()
    if arr.size == 0:
        return [0]
    return arr.tolist()


def _normalise_node_indices(values, num_nodes):
    arr = np.asarray(values, dtype=int).ravel()
    if arr.size and np.min(arr) >= 1 and np.max(arr) >= num_nodes:
        arr = arr - 1
    if np.any((arr < 0) | (arr >= num_nodes)):
        raise ValueError("Network node indices refer to non-existing node(s)")
    return arr


def _normalise_node_edges(edges, num_nodes):
    arr = np.asarray(edges, dtype=int)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("edges must be an nedge x 2 array")
    if arr.size and np.min(arr) >= 1 and np.max(arr) >= num_nodes:
        arr = arr - 1
    if np.any((arr < 0) | (arr >= num_nodes)):
        raise ValueError("Edges refer to non-existing node(s)")
    return arr


def _output_states(problem):
    handlers = _field(problem, "OutputHandlers", _field(problem, "output_handlers", None))
    if handlers is None:
        return None
    states = _field(handlers, "states", None)
    if states is None:
        return None
    if isinstance(states, dict) and "data" in states:
        return states["data"]
    return states


def _normalise_state_index(state_number, nstate):
    if nstate <= 0:
        raise ValueError("No states available for fd_postprocessor")
    if state_number is None:
        return nstate - 1
    value = int(state_number)
    if 0 <= value < nstate:
        return value
    if 1 <= value <= nstate:
        return value - 1
    raise IndexError("state_number is outside the available state range")


def _control_number(schedule, state_ix):
    step = _field(schedule, "step", {})
    control = np.asarray(_field(step, "control", []), dtype=int).ravel()
    controls = _field(schedule, "control", [])
    if control.size == 0:
        return 0
    ctrl = int(control[min(state_ix, control.size - 1)])
    nctrl = len(controls) if hasattr(controls, "__len__") else ctrl
    if 1 <= ctrl <= nctrl:
        return ctrl - 1
    return ctrl


def _schedule_wells(schedule, ctrl_no, fallback):
    controls = _field(schedule, "control", [])
    if controls is not None and len(controls) > 0:
        ctrl = controls[int(ctrl_no)]
        W = _field(ctrl, "W", None)
        if W is not None:
            return W
    return fallback


def _well_pressure(state, well_index, field):
    well_sols = _field(state, "wellSol", []) or []
    if well_index < len(well_sols):
        ws = well_sols[well_index]
        value = _field(ws, field, None)
        if value is None and field == "bhp":
            value = _field(ws, "pressure", None)
        if value is None and field == "pressure":
            value = _field(ws, "bhp", None)
        if value is not None:
            arr = np.asarray(value, dtype=float).ravel()
            if arr.size:
                return float(arr[0])
    return 0.0
