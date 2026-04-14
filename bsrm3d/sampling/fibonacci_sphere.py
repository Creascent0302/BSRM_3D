import numpy as np


def fibonacci_sphere_directions(num_samples: int) -> np.ndarray:
    """Generate near-uniform unit vectors on a sphere using Fibonacci lattice."""
    if num_samples < 1:
        return np.zeros((0, 3), dtype=float)

    i = np.arange(num_samples, dtype=float)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (2.0 * i + 1.0) / num_samples
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = phi * i

    x = r * np.cos(theta)
    z = r * np.sin(theta)
    directions = np.stack([x, y, z], axis=1)

    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return directions / norms
