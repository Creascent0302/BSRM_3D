from dataclasses import dataclass


@dataclass
class EnvironmentBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


@dataclass
class Planner3DConfig:
    num_nodes: int = 1200
    connection_radius: float = 1.8
    min_connection_radius: float = 0.8
    collision_radius: float = 0.15
    step_size: float = 0.1
    frontier_top_k: int = 24
    beam_solid_angle_deg: float = 80.0
    ray_step: float = 0.15
    ray_length: float = 2.0
    direction_samples: int = 120
    neighbor_k: int = 20
    frontier_info_weight: float = 1.0
    frontier_visibility_weight: float = 1.2
    visibility_probe_dirs: int = 20
    visibility_probe_length: float = 2.2
    seed: int = 42


@dataclass
class BenchmarkConfig:
    name: str
    bounds: EnvironmentBounds
    voxel_size: float = 0.2
    seed: int = 42
