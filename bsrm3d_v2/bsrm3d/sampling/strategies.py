"""Direction-sampling strategies for the 3D beam fan."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


def fibonacci_sphere_directions(num_samples: int) -> np.ndarray:
    """Near-uniform unit vectors on S^2 via the Fibonacci lattice."""
    if num_samples < 1:
        return np.zeros((0, 3), dtype=float)
    i = np.arange(num_samples, dtype=float)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (2.0 * i + 1.0) / num_samples
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = phi * i
    x = r * np.cos(theta)
    z = r * np.sin(theta)
    dirs = np.stack([x, y, z], axis=1)
    dirs /= np.clip(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12, None)
    return dirs


class DirectionSampler(ABC):
    @abstractmethod
    def sample(self, num_samples: int) -> np.ndarray: ...


@dataclass
class FibonacciDirectionSampler(DirectionSampler):
    def sample(self, num_samples: int) -> np.ndarray:
        return fibonacci_sphere_directions(num_samples)


@dataclass
class RandomDirectionSampler(DirectionSampler):
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def sample(self, num_samples: int) -> np.ndarray:
        if num_samples < 1:
            return np.zeros((0, 3), dtype=float)
        s = self._rng.normal(size=(num_samples, 3))
        s /= np.clip(np.linalg.norm(s, axis=1, keepdims=True), 1e-12, None)
        return s


def directions_from_angular_step(step_deg: float) -> np.ndarray:
    """
    Pick a Fibonacci sphere sampling whose average neighbour angle
    matches the requested ``step_deg``. Roughly:

        N ~= 4 * pi / angle_rad**2
    """
    step_rad = max(np.deg2rad(step_deg), np.deg2rad(2.0))
    n = max(8, int(round(4.0 * np.pi / (step_rad * step_rad))))
    # sensible cap to keep the beam fan affordable
    n = min(n, 1024)
    return fibonacci_sphere_directions(n)
