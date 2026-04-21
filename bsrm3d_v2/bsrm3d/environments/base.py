"""Abstract 3D environment interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from ..config import EnvironmentBounds
from ..types import Point3D


class Environment3D(ABC):
    """Generic 3D free-space / collision interface."""

    bounds: EnvironmentBounds

    @abstractmethod
    def is_free(self, point: Point3D, radius: float = 0.0) -> bool: ...

    @abstractmethod
    def segment_is_free(self, a: Point3D, b: Point3D,
                        radius: float = 0.0, step: float = 0.1) -> bool: ...

    @abstractmethod
    def sample_free(self, n: int, rng=None) -> Iterable[Point3D]: ...

    # Optional: used by the beam caster for a fast ray-march.
    def first_hit_distance(self, origin: Point3D, direction, max_length: float,
                           step: float = 0.1, radius: float = 0.0) -> float:
        """
        Return the free-flight distance along ``origin + t * direction`` before
        the first collision (or ``max_length`` if the ray never hits anything).

        Generic fallback implementation via stepping; subclasses may override.
        """
        import numpy as np
        d = np.asarray(direction, dtype=float)
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            return 0.0
        d = d / n
        t = step
        last_free = 0.0
        while t <= max_length:
            p = (origin[0] + d[0] * t,
                 origin[1] + d[1] * t,
                 origin[2] + d[2] * t)
            if not self.is_free(p, radius=radius):
                return last_free
            last_free = t
            t += step
        return max_length
