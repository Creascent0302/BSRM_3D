from __future__ import annotations

from typing import Dict, List

from bsrm3d.config import BenchmarkConfig, EnvironmentBounds
from .voxel_env import VoxelEnvironment3D


def available_benchmarks() -> List[str]:
    return ["cluttered", "tunnel", "rooms"]


def _default_configs() -> Dict[str, BenchmarkConfig]:
    bounds = EnvironmentBounds(0.0, 10.0, 0.0, 10.0, 0.0, 6.0)
    return {
        "cluttered": BenchmarkConfig("cluttered", bounds=bounds, voxel_size=0.2, seed=42),
        "tunnel": BenchmarkConfig("tunnel", bounds=bounds, voxel_size=0.2, seed=42),
        "rooms": BenchmarkConfig("rooms", bounds=bounds, voxel_size=0.2, seed=42),
    }


def build_benchmark_environment(name: str, config: BenchmarkConfig | None = None) -> VoxelEnvironment3D:
    configs = _default_configs()
    if name not in configs:
        raise ValueError(f"Unknown benchmark '{name}', options: {available_benchmarks()}")

    cfg = config or configs[name]
    env = VoxelEnvironment3D(bounds=cfg.bounds, voxel_size=cfg.voxel_size, seed=cfg.seed)

    if name == "cluttered":
        # Sparse random cuboids for generic UAV-like navigation.
        env.add_random_boxes(count=36, min_size=0.5, max_size=1.6)

    elif name == "tunnel":
        # Build a narrow passage by blocking space except central corridor.
        env.add_box_obstacle((5.0, 5.0, 1.0), (10.0, 10.0, 1.8))
        env.add_box_obstacle((5.0, 5.0, 5.0), (10.0, 10.0, 2.0))
        env.add_box_obstacle((5.0, 1.2, 3.0), (10.0, 2.2, 6.0))
        env.add_box_obstacle((5.0, 8.8, 3.0), (10.0, 2.2, 6.0))

    elif name == "rooms":
        # Multi-room structure with door-like openings.
        env.add_box_obstacle((5.0, 5.0, 3.0), (0.8, 10.0, 6.0))
        env.add_box_obstacle((5.0, 3.0, 3.0), (0.8, 3.4, 2.8))
        env.add_box_obstacle((5.0, 7.0, 3.0), (0.8, 3.4, 2.8))
        env.add_random_boxes(count=12, min_size=0.6, max_size=1.2)

    return env
