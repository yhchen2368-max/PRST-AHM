"""MRST deckformat resultinput - ECLIPSE output/result reading."""
from .read_eclipse_output_file_unfmt import read_eclipse_output_file_unfmt
from .read_eclipse_output_file_fmt import read_eclipse_output_file_fmt
from .init_grid_from_eclipse_output import init_grid_from_eclipse_output
from .convert_restart_to_states import convert_restart_to_states
from .process_eclipse_restart_spec import process_eclipse_restart_spec
from .read_eclipse_summary import read_eclipse_summary, convert_summary_to_well_sols

__all__ = [
    "read_eclipse_output_file_unfmt",
    "read_eclipse_output_file_fmt",
    "init_grid_from_eclipse_output",
    "convert_restart_to_states",
    "process_eclipse_restart_spec",
    "read_eclipse_summary",
    "convert_summary_to_well_sols",
]
