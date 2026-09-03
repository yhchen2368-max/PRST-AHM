"""MRST deckformat deckinput - ECLIPSE deck reading."""
from .read_eclipse_deck import read_eclipse_deck
from .convert_deck_units import convert_deck_units
from .initialize_deck import initialize_deck
from .match_wells import match_wells
from .read_grid import read_grid
from .read_props import read_props
from .read_regions import read_regions
from .read_runspec import read_runspec
from .read_schedule import read_schedule
from .read_solution import read_solution
from .read_summary import read_summary

__all__ = [
    "read_eclipse_deck",
    "convert_deck_units",
    "initialize_deck",
    "match_wells",
    "read_grid",
    "read_props",
    "read_regions",
    "read_runspec",
    "read_schedule",
    "read_solution",
    "read_summary",
]
