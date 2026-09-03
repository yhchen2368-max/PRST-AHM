import matplotlib.pyplot as plt
import numpy as np


def plot_well_sols(well_sols_list, time_vectors=None, dataset_names=None,
                   zoom=True, field="qOs", selected_wells=None, **kwargs):
    """MRST-compatible well solution plotting."""
    if dataset_names is None:
        dataset_names = [f"dataset {i}" for i in range(len(well_sols_list))]
    if time_vectors is None:
        time_vectors = [np.cumsum(np.ones(len(ws))) for ws in well_sols_list]

    n_wells = max(len(ws[0]) if ws else 0 for ws in well_sols_list)
    if selected_wells is None:
        selected_wells = list(range(min(n_wells, 5)))

    field_label = {"qWs": "Water Rate", "qOs": "Oil Rate", "bhp": "BHP"}
    fig, axes = plt.subplots(len(selected_wells), 1 + zoom,
                             figsize=(6 * (1 + zoom), 2.5 * len(selected_wells)),
                             squeeze=False)
    fig.suptitle(f"{field_label.get(field, field)} comparison", fontsize=12)

    for wi_idx, well_idx in enumerate(selected_wells):
        ax_full = axes[wi_idx, 0]
        for ds_idx, well_sols in enumerate(well_sols_list):
            t = time_vectors[ds_idx]
            values = [float(ws[well_idx].get(field, 0.0)) if well_idx < len(ws) else 0.0
                      for ws in well_sols]
            ax_full.plot(t, values, label=dataset_names[ds_idx])
        ax_full.set_ylabel(f"Well {well_idx + 1}")
        ax_full.legend(fontsize=7, loc="best")
        ax_full.grid(True, alpha=0.3)

        if zoom:
            ax_zoom = axes[wi_idx, 1]
            for ds_idx, well_sols in enumerate(well_sols_list):
                t = time_vectors[ds_idx]
                values = [float(ws[well_idx].get(field, 0.0)) if well_idx < len(ws) else 0.0
                          for ws in well_sols]
                ax_zoom.plot(t, values, label=dataset_names[ds_idx])
            ax_zoom.set_ylabel(f"Well {well_idx + 1} (zoom)")
            ax_zoom.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
