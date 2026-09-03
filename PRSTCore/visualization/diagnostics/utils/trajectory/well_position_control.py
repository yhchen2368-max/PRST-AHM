"""Python counterpart for MRST trajectory ``WellPositionControl.m``."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WellPositionControl:
    """Lightweight well trajectory control object."""

    points: np.ndarray
    parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=float)
        self.parameters.setdefault("trajectory", self.points.copy())
        if "perturbation" not in self.parameters:
            self.parameters["perturbation"] = np.eye(self.points.size).reshape((-1, *self.points.shape))

    def getTrajectory(self):
        return np.asarray(self.parameters.get("trajectory", self.points), dtype=float)

    def setTrajectory(self, trajectory):
        self.parameters["trajectory"] = np.asarray(trajectory, dtype=float)
        return self

    def update(self, W, index, amount):
        traj = self.getTrajectory().copy()
        flat = traj.reshape(-1)
        if 0 <= int(index) < flat.size:
            flat[int(index)] += float(amount)
        self.setTrajectory(flat.reshape(traj.shape))
        return W


