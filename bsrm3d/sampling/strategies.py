from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from .fibonacci_sphere import fibonacci_sphere_directions


class DirectionSampler(ABC):
    @abstractmethod
    def sample(self, num_samples: int) -> np.ndarray:
        raise NotImplementedError


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

        samples = self._rng.normal(size=(num_samples, 3))
        norms = np.linalg.norm(samples, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return samples / norms
