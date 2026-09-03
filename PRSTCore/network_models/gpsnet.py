"""GPSNet - Graph-based Production System Network model.

1:1 Python translation of MRST modules/network-models/GPSNet.m

Creates a reduced network model where each edge in a well connectivity
graph is subgridded and mapped onto a row in a rectangular Cartesian grid.
"""

import numpy as np


class GPSNet:
    """GPSNet type of network model for reservoir simulation.

    Each edge in the network graph is subgridded into nc cells and
    mapped onto a row in a 2D Cartesian grid. The grid has zero
    transmissibility in the vertical direction.

    Parameters
    ----------
    model_true : dict
        True fine-scale model with rock, operators, fluid, G.
    network : Network
        Network object defining well connectivity graph.
    W_in : list of dict
        Input well structures.
    nc : int, optional
        Number of cells per connection (default 10).
    p0 : float, optional
        Initial pressure (default 100 bar = 100e5 Pa).
    S0 : ndarray, optional
        Initial saturations.
    fluid : dict, optional
        Custom fluid model.
    scaling : list, optional
        Relperm scaling arguments.
    verbose : bool
        Verbose output.
    """

    def __init__(self, model_true, network, W_in, nc=10, p0=100e5,
                 S0=None, fluid=None, scaling=None, verbose=False):
        self.nc = nc
        self.type = network.type
        self.W_in = W_in

        graph = network.network
        self.graph = graph
        self.network_input = network
        self._fd_edge_T, self._fd_edge_pv = network.get_edge_data()
        num_edges = graph.number_of_edges()
        num_nodes = graph.number_of_nodes()
        node_data = getattr(network, "_nodes_data", [])
        node_to_well = [
            int(node_data[i].get("well", i)) if i < len(node_data) else i
            for i in range(num_nodes)
        ]

        # Build a Cartesian grid: nc cells per row, 1 column, num_edges rows
        # L approximated from bulk volume
        pv = np.asarray(model_true["operators"]["pv"]).ravel()
        poro = np.asarray(model_true["rock"]["poro"]).ravel()
        bulk_vol = np.sum(pv / np.maximum(poro, 1e-6))
        L = np.power(bulk_vol * 25, 1.0 / 3.0)

        nx_c, ny_c, nz_c = nc, 1, num_edges
        n_cells = nx_c * ny_c * nz_c
        G = {
            "cells": {
                "num": n_cells,
                "centroids": np.column_stack([
                    np.tile(np.linspace(0, L, nx_c), ny_c * nz_c),
                    np.zeros(n_cells),
                    np.repeat(np.linspace(0, L / 5, nz_c), nx_c * ny_c),
                ]),
            },
            "cartDims": [nx_c, ny_c, nz_c],
        }

        # Rock properties
        if fluid is None:
            fluid = {}
        poro_avg = float(np.mean(poro))
        perm_avg = float(np.mean(np.asarray(model_true["rock"]["perm"]).ravel()))
        rock = {"poro": np.full(n_cells, poro_avg),
                "perm": np.full((n_cells, 1), perm_avg)}

        # Model
        model = {
            "G": G,
            "rock": rock,
            "fluid": fluid,
            "operators": {
                "N": np.zeros((0, 2), dtype=int),
                "T": np.ones(n_cells * 3),
                "pv": np.full(n_cells, poro_avg * (L / nx_c) * (L / 5) * (L / 5 / nz_c)),
            },
            "OutputStateFunctions": [],
        }

        # Set all transmissibility to zero except x-direction
        # Internal faces: (nc-1) * num_edges in x-direction
        T = model["operators"]["T"].copy()
        n_internal_faces = (nc - 1) * num_edges
        T[n_internal_faces:] = 0.0

        # Initialize face indices per node
        node_face_ix = -np.ones(num_nodes, dtype=int)
        node_cells = [[] for _ in range(num_nodes)]
        w_cells_init = [[] for _ in W_in]

        nf = nc - 1
        edges_list = list(graph.edges())
        edge_cell_ix = {}
        edge_face_ix = {}

        N_list = []
        T_list = list(T[:n_internal_faces])

        for i, (nodeL, nodeR) in enumerate(edges_list):
            cellL = i * nc
            cellR = (i + 1) * nc - 1
            edge_cell_ix[i] = list(range(cellL + 1, cellR))
            nodeL_well = node_to_well[nodeL]
            nodeR_well = node_to_well[nodeR]

            # Left node
            if node_face_ix[nodeL] < 0:
                first_ix = i * nf
                node_face_ix[nodeL] = first_ix
                node_cells[nodeL].append(cellL)
                w_cells_init[nodeL_well].append(cellL)
            else:
                # Add non-neighboring connection
                last_cell_of_node = node_cells[nodeL][-1] if node_cells[nodeL] else cellL
                N_list.append([last_cell_of_node, cellL + 1])
                T_list.append(T[0])
                first_ix = len(T_list) - 1

            # Internal faces
            intern_ix = list(range(i * nf + 1, i * nf + nf - 1))

            # Right node
            if node_face_ix[nodeR] < 0:
                last_ix = i * nf + nf - 1
                node_face_ix[nodeR] = last_ix
                node_cells[nodeR].append(cellR)
                w_cells_init[nodeR_well].append(cellR)
            else:
                target_cell = node_cells[nodeR][-1] if node_cells[nodeR] else cellR
                N_list.append([cellR - 1, target_cell])
                T_list.append(T[0])
                last_ix = len(T_list) - 1

            edge_face_ix[i] = [first_ix] + intern_ix + [last_ix]

        # Build final N and T
        N_arr = np.zeros((n_internal_faces + len(N_list), 2), dtype=int)
        N_arr[:n_internal_faces] = np.column_stack([
            np.arange(0, n_cells - 1)[np.arange(0, n_cells - 1) % nc != nc - 1],
            np.arange(1, n_cells)[np.arange(0, n_cells - 1) % nc != nc - 1],
        ])
        for idx, (a, b) in enumerate(N_list):
            N_arr[n_internal_faces + idx] = [a, b]
        T_arr = np.asarray(T_list, dtype=float)

        model["operators"]["N"] = N_arr
        model["operators"]["T"] = T_arr
        self.model = model

        # Well structure
        W = []
        for iw, w in enumerate(W_in):
            wcells = w_cells_init[iw]
            wi = {
                "cells": wcells,
                "type": w.get("type", "bhp"),
                "val": w.get("val", 200e5),
                "radius": w.get("r", 0.1),
                "name": w.get("name", f"W{iw}"),
                "compi": w.get("compi", [1, 0]),
                "sign": w.get("sign", -1),
                "lims": w.get("lims", None),
                "status": w.get("status", True),
                "WI": w.get("WI", 1.0),
            }
            W.append(wi)
        self.W = W

        # Initial state
        if S0 is None:
            S0 = np.array([0.0, 1.0])
        pressure = np.full(n_cells, p0)
        s = np.tile(S0, (n_cells, 1))
        self.state0 = {"pressure": pressure, "s": s}

        # Store graph data
        self._edge_cell_ix = edge_cell_ix
        self._edge_face_ix = edge_face_ix
        self._node_face_ix = node_face_ix
        self._node_cells = node_cells

        self.model["toleranceCNV"] = 1e-6

    def plot_grid(self, data=None, *, ax=None):
        """Port of ``GPSNet.plotGrid``.

        Draws the 1-D network grid: the layer index colours the cells when
        no data is given, the two end columns and the well cells are
        outlined, and each node carries its well's name -- offset to
        whichever side of the cell keeps the label off the grid.
        """
        import matplotlib.pyplot as plt

        from PRSTCore.nwm._core import gridLogicalIndices
        from PRSTCore.visualization.grid_plots import plot_cell_data, plot_grid

        G = self.model["G"] if isinstance(self.model, dict) else self.model.G
        I, _J, K = gridLogicalIndices(G)
        I = np.asarray(I, dtype=int).ravel()
        K = np.asarray(K, dtype=int).ravel()
        ends = np.flatnonzero((I == I.min()) | (I == I.max()))
        well_cells = np.concatenate(
            [np.atleast_1d(np.asarray(w["cells"], dtype=int)).ravel()
             for w in self.W]) if self.W else np.zeros(0, dtype=int)

        if ax is None:
            _, ax = plt.subplots()

        if data is None:
            plot_cell_data(G, K, ax=ax)
            plot_grid(G, ends, ax=ax, facecolor=(0.9, 0.9, 0.9))
            plot_grid(G, well_cells, ax=ax, facecolor=(0.7, 0.7, 0.7))
        else:
            plot_cell_data(G, np.asarray(data, dtype=float).ravel(), ax=ax,
                           alpha=0.9)
            plot_grid(G, ends, ax=ax, facecolor='none', linewidth=1.0)
            plot_grid(G, well_cells, ax=ax, facecolor='none', linewidth=2.0)

        centroids = np.asarray(G["cells"]["centroids"], dtype=float)
        for node in getattr(self.network_input, '_nodes_data', []):
            cells = self._node_cells.get(int(node['node'])) \
                if isinstance(self._node_cells, dict) else None
            if not cells:
                continue
            cell_no = int(np.atleast_1d(np.asarray(cells, dtype=int)).ravel()[0])
            if not (0 <= cell_no < centroids.shape[0]):
                continue
            x, y = centroids[cell_no, 0], centroids[cell_no, 1]
            # Label on whichever side is further from the nearer end of
            # the grid, so it does not overlap the network.
            near_first = abs(x - centroids[0, 0]) < abs(x - centroids[-1, 0])
            ax.text(x - 70.0 if near_first else x + 20.0, y, node['name'])

        ax.set_axis_off()
        return ax

    def get_mapping(self, map_type="cells"):
        """Get mapping between graph edges and cell/face indices.

        Parameters
        ----------
        map_type : str
            'cells' or 'faces'.

        Returns
        -------
        edge : ndarray
            Edge number for each cell/face.
        subset : ndarray
            Cell/face indices for each edge.
        """
        if map_type == "cells":
            indices = self._edge_cell_ix
        elif map_type == "faces":
            indices = self._edge_face_ix
        else:
            raise ValueError("Unknown mapping type")

        edge = []
        subset = []
        for e_idx, idx_list in indices.items():
            edge.extend([e_idx] * len(idx_list))
            subset.extend(idx_list)
        return np.array(edge, dtype=int), np.array(subset, dtype=int)

    def get_scaled_parameter_vector(self, setup, params, connscale=None, rand_sw=False):
        """Get parameters scaled to unit interval.

        Extended version for GPSNet - overwrites pore volumes and
        transmissibilities from graph data if available.
        """
        u = []
        if connscale is None:
            connscale = np.ones(len(params))
        elif np.isscalar(connscale):
            connscale = connscale * np.ones(len(params))

        is_diag = self.type in ("fd_preprocessor", "fd_postprocessor")

        for k, param in enumerate(params):
            val = np.asarray(setup["model"].get(param.name,
                              np.zeros(param.n_param)), dtype=float).ravel()
            val = val[:param.n_param]

            if param.name == "porevolume" and is_diag:
                val = self._flow_diagnostics_porevolume_values(param, setup)
            elif param.name == "transmissibility" and is_diag:
                val = self._flow_diagnostics_transmissibility_values(param, setup)
            elif param.name == "conntrans":
                val = connscale[k] * val
            elif param.name == "sw" and rand_sw:
                val = np.random.rand(len(val))

            u.append(param.scale(val))

        return np.concatenate(u)

    def _flow_diagnostics_porevolume_values(self, param, setup):
        edge_pv = np.asarray(self._fd_edge_pv, dtype=float).ravel()
        if edge_pv.size == 0:
            return np.asarray(setup["model"].get("porevolume", np.zeros(param.n_param)), dtype=float).ravel()[:param.n_param]
        per_cell = edge_pv / max(int(self.nc), 1)
        if param.n_param == per_cell.size:
            return per_cell
        base = _model_vector(setup["model"], "porevolume", param.n_param)
        if base.size < param.n_param:
            base = np.resize(base, param.n_param)
        for edge_no, value in enumerate(per_cell):
            start = edge_no * self.nc
            stop = min(start + self.nc, base.size)
            if start < base.size:
                base[start:stop] = value
        return base[:param.n_param]

    def _flow_diagnostics_transmissibility_values(self, param, setup):
        edge_T = np.asarray(self._fd_edge_T, dtype=float).ravel()
        if edge_T.size == 0:
            return _model_vector(setup["model"], "transmissibility", param.n_param)
        if param.n_param == edge_T.size:
            return edge_T
        base = _model_vector(setup["model"], "transmissibility", param.n_param)
        if base.size < param.n_param:
            base = np.resize(base, param.n_param)
        for edge_no, value in enumerate(edge_T):
            for face in self._edge_face_ix.get(edge_no, []):
                if 0 <= int(face) < base.size:
                    base[int(face)] = value
        return base[:param.n_param]


def _model_vector(model, name, n_param):
    if name == "porevolume":
        value = model.get("porevolume", None)
        if value is None:
            value = model.get("operators", {}).get("pv", np.zeros(n_param))
    elif name == "transmissibility":
        value = model.get("transmissibility", None)
        if value is None:
            value = model.get("operators", {}).get("T", np.zeros(n_param))
    else:
        value = model.get(name, np.zeros(n_param))
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size == 0:
        return np.zeros(n_param, dtype=float)
    return arr.copy()
