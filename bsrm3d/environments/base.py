from abc import ABC, abstractmethod
from typing import Iterable

from bsrm3d.config import EnvironmentBounds
from bsrm3d.types import Point3D


class Environment3D(ABC):
    """Common interface for 3D collision checking and free-space sampling."""

    bounds: EnvironmentBounds

    @abstractmethod
    def is_free(self, point: Point3D, radius: float = 0.0) -> bool:
        raise NotImplementedError

    @abstractmethod
    def segment_is_free(self, a: Point3D, b: Point3D, radius: float = 0.0, step: float = 0.1) -> bool:
        raise NotImplementedError

    @abstractmethod
    def sample_free(self, n: int) -> Iterable[Point3D]:
        raise NotImplementedError
