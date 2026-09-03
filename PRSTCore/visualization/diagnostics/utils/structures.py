"""Small MRST-like structures used by flow diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np


class Struct(SimpleNamespace):
    """Simple MATLAB-struct style object with light dict interoperability."""

    def get(self, name: str, default: Any = None) -> Any:
        return getattr(self, name, default)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(slots=True)
class TOFDiagnostics:
    inj: np.ndarray
    prod: np.ndarray
    tof: np.ndarray
    itracer: np.ndarray
    ipart: np.ndarray
    ptracer: np.ndarray
    ppart: np.ndarray
    itof: np.ndarray | None = None
    ptof: np.ndarray | None = None
    ifa: np.ndarray | None = None
    pfa: np.ndarray | None = None


@dataclass(slots=True)
class WellAllocation:
    alloc: np.ndarray
    ralloc: np.ndarray
    z: np.ndarray
    name: str


@dataclass(slots=True)
class WellPairDiagnostics:
    pairs: list[str]
    pairIx: np.ndarray
    vols: np.ndarray
    inj: list[WellAllocation] = field(default_factory=list)
    prod: list[WellAllocation] = field(default_factory=list)


@dataclass(slots=True)
class DiagnosticsStruct:
    D: TOFDiagnostics
    WP: WellPairDiagnostics
    wellCommunication: np.ndarray

