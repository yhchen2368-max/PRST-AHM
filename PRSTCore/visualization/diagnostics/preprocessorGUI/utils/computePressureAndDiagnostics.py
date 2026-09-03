"""Compatibility wrapper matching MRST's ``computePressureAndDiagnostics.m`` name."""

from .compute_pressure_and_diagnostics import (
    computePressureAndDiagnostics,
    compute_pressure_and_diagnostics,
)

__all__ = ["computePressureAndDiagnostics", "compute_pressure_and_diagnostics"]

