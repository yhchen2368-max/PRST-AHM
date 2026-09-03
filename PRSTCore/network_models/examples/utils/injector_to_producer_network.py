"""Injector-to-producer network factory.

1:1 Python translation of MRST
modules/network-models/examples/utils/injector_to_producer_network.m
"""

import numpy as np


def injector_to_producer_network(reference_case, cells_per_connection=10,
                                  p0=400e5, s0=None, perm=200e-15,
                                  poro=0.1, plot_network=False,
                                  plot_type="default"):
    """Create a GPSNet injector-to-producer network from a reference case.

    Parameters
    ----------
    reference_case : dict
        Reference case with model, schedule, state0.
    cells_per_connection : int
        Number of cells per network edge.
    p0 : float
        Initial pressure.
    s0 : list
        Initial saturations [sw, so].
    perm : float
        Permeability in connections.
    poro : float
        Porosity.
    plot_network : bool
        Whether to plot the network.
    plot_type : str
        Layout for plotting.

    Returns
    -------
    dict
        Setup with name, description, options, state0, model, schedule.
    """
    from PRSTCore.network_models import Network, GPSNet

    if s0 is None:
        s0 = [0.2, 0.8]

    ref_model = reference_case["model"]
    ref_G = ref_model["G"]
    ref_schedule = reference_case["schedule"]
    W_original = ref_schedule["control"][0]["W"]

    # Single perforation per well (top)
    Wnetwork = [dict(w) for w in W_original]
    for w in Wnetwork:
        cells = w.get("cells", [0])
        if isinstance(cells, list) and len(cells) > 0:
            w["cells"] = [cells[-1]]

    injectors = [i for i, w in enumerate(Wnetwork) if w.get("sign", 0) > 0]
    producers = [i for i, w in enumerate(Wnetwork) if w.get("sign", 0) < 0]

    network = Network(
        Wnetwork, ref_G, type="injectors_to_producers",
        injectors=injectors, producers=producers,
    )

    gps_net = GPSNet(
        ref_model, network, W_original,
        nc=cells_per_connection, p0=p0, S0=s0,
    )

    return {
        "name": f"injector_to_producer_{reference_case.get('name', 'case')}",
        "description": f"Creates an injector-to-producer network of {reference_case.get('name', 'case')}",
        "model": gps_net.model,
        "state0": gps_net.state0,
        "schedule": ref_schedule,
        "gps_net": gps_net,
        "network": network,
    }
