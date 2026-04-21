"""
Configuration dataclasses for BSRM-3D.

The knobs here mirror those of the 2D BSRM paper but generalised to 3D:

- ``beam_step`` (Δr)   : ray marching step along a beam.
- ``angular_step_deg`` (Δθ): angular resolution of the Fibonacci-sphere beam fan.
- ``min_spacing`` (r_min): Poisson-disk spacing that caps local node density.
- ``connection_radius`` (r_conn): neighbor search radius when wiring edges.
- ``discontinuity_abs`` (d0) and ``discontinuity_rel`` (k): the adaptive
  threshold that decides whether two neighbouring beams mark a geometric
  singularity (corner / passage entrance).
- ``sector_half_angle_deg`` (φ/2): the half-angle of the directional sector
  that gets suppressed on an explored direction.
- ``sketch_visual_radius`` (r_v): graph-radius used by the NMS sketch phase.
- ``sketch_link_radius`` (r_l): radius used when reconnecting skeletal nodes.

All distances are in world (metre) units; the bounds and voxel size of the
environment control the physical extent.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnvironmentBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @property
    def size(self) -> tuple[float, float, float]:
        return (self.x_max - self.x_min,
                self.y_max - self.y_min,
                self.z_max - self.z_min)

    @property
    def diagonal(self) -> float:
        import math
        sx, sy, sz = self.size
        return math.sqrt(sx * sx + sy * sy + sz * sz)


@dataclass
class Planner3DConfig:
    """Parameters for BSRM-3D (beam-driven + structural sketching)."""

    # --- node budget (upper cap; BSRM typically terminates far below this) ---
    max_nodes: int = 1200

    # --- beam casting ---
    angular_step_deg: float = 15.0           # Δθ; beam fan resolution
    beam_step: float = 0.18                  # Δr; ray-march step
    beam_max_length: float = 12.0            # safety cap on beam length (>> diag for 10m map)

    # --- discontinuity detection (paper's adaptive τ) ---
    discontinuity_abs: float = 0.45          # d0 in the paper
    discontinuity_rel: float = 0.4           # k in the paper
    short_beam_fraction: float = 0.55        # where on the shorter beam we seed
    long_beam_fraction: float = 0.5          # fraction along the longer beam

    # --- frontier + spacing ---
    min_spacing: float = 0.4                 # r_min; Poisson-disk gate
    connection_radius: float = 2.0           # r_conn
    sector_half_angle_deg: float = 22.0      # half-angle of sector suppressed on connection
    max_frontier_expansions: int = 6000      # hard stop on the while-loop
    initial_seeds: int = 20                  # random seeds to bootstrap exploration

    # --- robot / collision ---
    collision_radius: float = 0.15
    collision_step: float = 0.14             # step for segment collision checks

    # --- sketching (NMS) ---
    enable_sketch: bool = True
    sketch_visual_radius: float = 1.2        # r_v (graph radius)
    sketch_link_radius: float = 2.4          # r_l
    sketch_clearance_probes: int = 32        # rays used to estimate clearance

    # --- query ---
    neighbor_k: int = 16                     # k used when attaching start/goal

    # --- adaptive beam expansion ---
    enable_two_pass: bool = True             # beam-informed gap-aware second pass

    # --- misc ---
    seed: int = 42


@dataclass
class BenchmarkConfig:
    name: str
    bounds: EnvironmentBounds
    voxel_size: float = 0.2
    seed: int = 42


# =========================================================================
#                   Per-scenario BSRM-3D parameter presets
# =========================================================================
#
# Mirrors the paper's Table I: smaller min_spacing + finer Δθ in tight
# scenarios (narrow, clutter), coarser for open ones. Each preset is
# applied on top of the Planner3DConfig defaults.
#
# The presets were selected by a small sensitivity sweep (see
# scripts/run_sensitivity.py) and hit 100 % on all moderate scenarios
# and ≥75 % on aggressively-hard ones. Extra-hard scenarios use a
# larger max_nodes to let BSRM explore farther.
# =========================================================================

_PER_ENV_PRESETS = {
    "cluttered":    dict(min_spacing=0.30, angular_step_deg=10, max_nodes=1500),
    "forest":       dict(min_spacing=0.35, angular_step_deg=15, max_nodes=1200),
    "maze3d":       dict(min_spacing=0.40, angular_step_deg=20, max_nodes=1200),
    "indoor":       dict(min_spacing=0.40, angular_step_deg=20, max_nodes=1200),
    "narrow":       dict(min_spacing=0.35, angular_step_deg=10, max_nodes=1500),
    "bugtrap":      dict(min_spacing=0.30, angular_step_deg=10, max_nodes=1500),
    "narrow_tight": dict(min_spacing=0.25, angular_step_deg=8,  max_nodes=3000),
    "maze_hard":    dict(min_spacing=0.30, angular_step_deg=15, max_nodes=3000),
    "building_3floor": dict(min_spacing=0.50, angular_step_deg=15, max_nodes=5000,
                            connection_radius=3.0, beam_max_length=35.0),
}


def bsrm_config_for_env(env_name: str, seed: int = 42) -> "Planner3DConfig":
    """Return a Planner3DConfig pre-tuned for the named benchmark environment.

    Falls back to defaults if the environment is unknown.
    """
    cfg = Planner3DConfig(seed=seed)
    preset = _PER_ENV_PRESETS.get(env_name, {})
    for k, v in preset.items():
        setattr(cfg, k, v)
    # also scale sketch radii with min_spacing
    if "min_spacing" in preset:
        rm = preset["min_spacing"]
        cfg.sketch_visual_radius = max(1.0, rm * 2.5)
        cfg.sketch_link_radius = max(2.0, rm * 5.0)
    return cfg
