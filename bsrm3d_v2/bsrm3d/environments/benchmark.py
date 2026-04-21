"""
Benchmark environments for BSRM-3D.

We extend the paper's 2D suite (Cluttered / Maze / Indoor / Narrow-Passage)
to 3D with five scenarios, each chosen to exercise a distinct capability:

* ``cluttered``  - sparse random obstacles; sanity check.
* ``forest``     - dense vertical pillars (think UAV flying through trees).
* ``maze3d``     - multi-level maze walls with bridges between layers.
* ``indoor``     - "building" partitioned into rooms with doorways.
* ``narrow``     - two halls separated by a tight window; stresses passage discovery.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..config import BenchmarkConfig, EnvironmentBounds
from .voxel_env import VoxelEnvironment3D


def available_benchmarks() -> List[str]:
    return ["cluttered", "forest", "maze3d", "indoor", "narrow",
            "narrow_tight", "maze_hard", "bugtrap", "building_3floor"]


def _default_configs() -> Dict[str, BenchmarkConfig]:
    bounds = EnvironmentBounds(0.0, 10.0, 0.0, 10.0, 0.0, 6.0)
    bounds_large = EnvironmentBounds(0.0, 30.0, 0.0, 30.0, 0.0, 12.0)
    return {
        "cluttered":    BenchmarkConfig("cluttered",    bounds=bounds, voxel_size=0.2, seed=42),
        "forest":       BenchmarkConfig("forest",       bounds=bounds, voxel_size=0.2, seed=42),
        "maze3d":       BenchmarkConfig("maze3d",       bounds=bounds, voxel_size=0.2, seed=42),
        "indoor":       BenchmarkConfig("indoor",       bounds=bounds, voxel_size=0.2, seed=42),
        "narrow":       BenchmarkConfig("narrow",       bounds=bounds, voxel_size=0.2, seed=42),
        "narrow_tight": BenchmarkConfig("narrow_tight", bounds=bounds, voxel_size=0.1, seed=42),
        "maze_hard":    BenchmarkConfig("maze_hard",    bounds=bounds, voxel_size=0.1, seed=42),
        "bugtrap":      BenchmarkConfig("bugtrap",      bounds=bounds, voxel_size=0.2, seed=42),
        "building_3floor": BenchmarkConfig("building_3floor", bounds=bounds_large, voxel_size=0.2, seed=42),
    }


# ------------------------------------------------------------ scene builders --
def _build_cluttered(env: VoxelEnvironment3D) -> None:
    env.add_random_boxes(count=42, min_size=0.5, max_size=1.6)


def _build_forest(env: VoxelEnvironment3D) -> None:
    """Dense vertical pillars. Narrow lanes with free ceiling and floor bands."""
    rng = np.random.default_rng(env.seed)
    b = env.bounds
    # tall pillars
    for _ in range(55):
        cx = float(rng.uniform(b.x_min + 0.5, b.x_max - 0.5))
        cy = float(rng.uniform(b.y_min + 0.5, b.y_max - 0.5))
        w = float(rng.uniform(0.35, 0.8))
        d = float(rng.uniform(0.35, 0.8))
        h = float(rng.uniform(3.0, (b.z_max - b.z_min) - 0.6))
        cz = b.z_min + h * 0.5
        env.add_box_obstacle((cx, cy, cz), (w, d, h))
    # a scattering of floating obstacles
    env.add_random_boxes(count=10, min_size=0.3, max_size=0.8)


def _build_maze3d(env: VoxelEnvironment3D) -> None:
    """
    Two horizontal floors of maze-like walls plus floor/ceiling separators
    with a 'stairwell' column that is free.

    The walls alternate orientation on the two levels to force 3D routing.
    """
    b = env.bounds
    # Middle slab (separator) with one opening
    mid_z = (b.z_min + b.z_max) * 0.5
    # slab thickness
    slab = 0.4
    # fill entire x-y slice at mid_z
    env.add_box_obstacle(((b.x_min + b.x_max) * 0.5,
                          (b.y_min + b.y_max) * 0.5,
                          mid_z),
                         (b.x_max - b.x_min, b.y_max - b.y_min, slab))
    # punch a stairwell hole in the slab
    hole_cx, hole_cy = 5.0, 5.0
    hole_w, hole_d = 1.4, 1.4
    # We can't "subtract" voxels easily; do it by direct grid edit.
    ix0 = int((hole_cx - hole_w / 2 - b.x_min) / env.voxel_size)
    ix1 = int((hole_cx + hole_w / 2 - b.x_min) / env.voxel_size)
    iy0 = int((hole_cy - hole_d / 2 - b.y_min) / env.voxel_size)
    iy1 = int((hole_cy + hole_d / 2 - b.y_min) / env.voxel_size)
    iz0 = int((mid_z - slab / 2 - b.z_min) / env.voxel_size)
    iz1 = int((mid_z + slab / 2 - b.z_min) / env.voxel_size) + 1
    env._grid[ix0:ix1 + 1, iy0:iy1 + 1, iz0:iz1 + 1] = False  # noqa: SLF001

    # Lower-level walls running along Y, with gaps
    lower_cz = (b.z_min + (mid_z - slab / 2)) * 0.5
    lower_h = (mid_z - slab / 2) - b.z_min
    for wx in (2.5, 7.5):
        env.add_box_obstacle((wx, 3.0, lower_cz), (0.4, 5.4, lower_h - 0.2))
        env.add_box_obstacle((wx, 8.0, lower_cz), (0.4, 2.6, lower_h - 0.2))

    # Upper-level walls running along X, with gaps
    upper_cz = ((mid_z + slab / 2) + b.z_max) * 0.5
    upper_h = b.z_max - (mid_z + slab / 2)
    for wy in (3.0, 7.0):
        env.add_box_obstacle((3.0, wy, upper_cz), (5.4, 0.4, upper_h - 0.2))
        env.add_box_obstacle((8.0, wy, upper_cz), (2.6, 0.4, upper_h - 0.2))


def _build_indoor(env: VoxelEnvironment3D) -> None:
    """3 rooms connected by doorways, with some furniture clutter."""
    b = env.bounds
    mid_z = (b.z_min + b.z_max) * 0.5
    room_h = b.z_max - b.z_min - 0.4

    # Two vertical walls dividing into 3 rooms
    env.add_box_obstacle((3.33, 5.0, mid_z), (0.3, 10.0, room_h))
    env.add_box_obstacle((6.66, 5.0, mid_z), (0.3, 10.0, room_h))

    # Doors (knock out holes in the walls)
    for wall_x in (3.33, 6.66):
        # door at y=3.0, full height 2.4
        for (dy, dh) in [(3.0, 2.4), (7.0, 2.4)]:
            ix0 = int((wall_x - 0.2 - b.x_min) / env.voxel_size)
            ix1 = int((wall_x + 0.2 - b.x_min) / env.voxel_size)
            iy0 = int((dy - 0.9 - b.y_min) / env.voxel_size)
            iy1 = int((dy + 0.9 - b.y_min) / env.voxel_size)
            iz0 = int((b.z_min + 0.2 - b.z_min) / env.voxel_size)
            iz1 = int((b.z_min + 0.2 + dh - b.z_min) / env.voxel_size)
            env._grid[ix0:ix1 + 1, iy0:iy1 + 1, iz0:iz1 + 1] = False  # noqa: SLF001

    # Furniture: a few boxes per room
    keep = [(1.5, 2.0, 0.5), (5.0, 2.0, 0.5), (8.5, 2.0, 0.5),
            (1.5, 8.0, 0.5), (5.0, 8.0, 0.5), (8.5, 8.0, 0.5)]
    env.add_random_boxes(count=16, min_size=0.5, max_size=1.1, keep_free_around=keep)


def _build_narrow(env: VoxelEnvironment3D) -> None:
    """Two chambers separated by a thin wall with a small window."""
    b = env.bounds
    mid_x = (b.x_min + b.x_max) * 0.5
    # Solid wall across Y and Z
    env.add_box_obstacle((mid_x, (b.y_min + b.y_max) * 0.5, (b.z_min + b.z_max) * 0.5),
                         (0.35, b.y_max - b.y_min, b.z_max - b.z_min))
    # Punch a small window at roughly (5.0, 5.0, 3.0)
    window = (1.2, 1.2)  # y,z extent
    cx, cy, cz = mid_x, 5.0, 3.0
    ix0 = int((cx - 0.25 - b.x_min) / env.voxel_size)
    ix1 = int((cx + 0.25 - b.x_min) / env.voxel_size)
    iy0 = int((cy - window[0] / 2 - b.y_min) / env.voxel_size)
    iy1 = int((cy + window[0] / 2 - b.y_min) / env.voxel_size)
    iz0 = int((cz - window[1] / 2 - b.z_min) / env.voxel_size)
    iz1 = int((cz + window[1] / 2 - b.z_min) / env.voxel_size)
    env._grid[ix0:ix1 + 1, iy0:iy1 + 1, iz0:iz1 + 1] = False  # noqa: SLF001

    # A second window up high
    cx, cy, cz = mid_x, 2.5, 4.8
    iy0 = int((cy - 0.6 - b.y_min) / env.voxel_size)
    iy1 = int((cy + 0.6 - b.y_min) / env.voxel_size)
    iz0 = int((cz - 0.6 - b.z_min) / env.voxel_size)
    iz1 = int((cz + 0.6 - b.z_min) / env.voxel_size)
    env._grid[ix0:ix1 + 1, iy0:iy1 + 1, iz0:iz1 + 1] = False  # noqa: SLF001

    # Some light clutter in both chambers
    env.add_random_boxes(count=12, min_size=0.5, max_size=1.2)


def _build_narrow_tight(env: VoxelEnvironment3D) -> None:
    """Like narrow but with a much smaller 0.6m window. Aggressively tests
    narrow-passage discovery; random sampling-based methods struggle here."""
    b = env.bounds
    mid_x = (b.x_min + b.x_max) * 0.5
    env.add_box_obstacle((mid_x, (b.y_min + b.y_max) * 0.5, (b.z_min + b.z_max) * 0.5),
                         (0.35, b.y_max - b.y_min, b.z_max - b.z_min))
    # ONE 0.6 x 0.6 m window (vs 1.2 x 1.2 in the easy version)
    window = (0.6, 0.6)
    cx, cy, cz = mid_x, 5.0, 3.0
    ix0 = int((cx - 0.25 - b.x_min) / env.voxel_size)
    ix1 = int((cx + 0.25 - b.x_min) / env.voxel_size)
    iy0 = int((cy - window[0] / 2 - b.y_min) / env.voxel_size)
    iy1 = int((cy + window[0] / 2 - b.y_min) / env.voxel_size)
    iz0 = int((cz - window[1] / 2 - b.z_min) / env.voxel_size)
    iz1 = int((cz + window[1] / 2 - b.z_min) / env.voxel_size)
    env._grid[ix0:ix1 + 1, iy0:iy1 + 1, iz0:iz1 + 1] = False  # noqa: SLF001
    # light clutter
    env.add_random_boxes(count=10, min_size=0.4, max_size=1.0)


def _build_maze3d_hard(env: VoxelEnvironment3D) -> None:
    """Two floors with a SINGLE 0.8m stairwell (vs 1.4m easy version).
    Each floor has more interior walls to further restrict routing."""
    b = env.bounds
    mid_z = (b.z_min + b.z_max) * 0.5
    slab = 0.4
    env.add_box_obstacle(((b.x_min + b.x_max) * 0.5,
                          (b.y_min + b.y_max) * 0.5,
                          mid_z),
                         (b.x_max - b.x_min, b.y_max - b.y_min, slab))
    # ONE 0.8 x 0.8 m stairwell (down from 1.4)
    hole_cx, hole_cy = 5.0, 5.0
    hole_w, hole_d = 0.8, 0.8
    ix0 = int((hole_cx - hole_w / 2 - b.x_min) / env.voxel_size)
    ix1 = int((hole_cx + hole_w / 2 - b.x_min) / env.voxel_size)
    iy0 = int((hole_cy - hole_d / 2 - b.y_min) / env.voxel_size)
    iy1 = int((hole_cy + hole_d / 2 - b.y_min) / env.voxel_size)
    iz0 = int((mid_z - slab / 2 - b.z_min) / env.voxel_size)
    iz1 = int((mid_z + slab / 2 - b.z_min) / env.voxel_size) + 1
    env._grid[ix0:ix1 + 1, iy0:iy1 + 1, iz0:iz1 + 1] = False  # noqa: SLF001

    # lower floor: narrower corridor
    lower_cz = (b.z_min + (mid_z - slab / 2)) * 0.5
    lower_h = (mid_z - slab / 2) - b.z_min
    for wx in (2.5, 7.5):
        env.add_box_obstacle((wx, 3.0, lower_cz), (0.4, 5.8, lower_h - 0.2))
        env.add_box_obstacle((wx, 8.0, lower_cz), (0.4, 3.4, lower_h - 0.2))
    # middle wall on lower level
    env.add_box_obstacle((5.0, 2.0, lower_cz), (3.0, 0.4, lower_h - 0.2))

    # upper floor: similarly constrained
    upper_cz = ((mid_z + slab / 2) + b.z_max) * 0.5
    upper_h = b.z_max - (mid_z + slab / 2)
    for wy in (3.0, 7.0):
        env.add_box_obstacle((3.0, wy, upper_cz), (5.8, 0.4, upper_h - 0.2))
        env.add_box_obstacle((8.0, wy, upper_cz), (3.4, 0.4, upper_h - 0.2))
    env.add_box_obstacle((2.0, 5.0, upper_cz), (0.4, 3.0, upper_h - 0.2))


def _build_bugtrap(env: VoxelEnvironment3D) -> None:
    """Classical bug-trap scenario - a concave C-shape that pure-random
    sampling has a hard time escaping. Combined with clutter outside."""
    b = env.bounds
    cx, cy = 3.0, 5.0
    mz = (b.z_min + b.z_max) * 0.5
    h = 3.0
    # C-shape: back wall + top wall + bottom wall; open on +x side
    env.add_box_obstacle((cx - 1.3, cy, mz), (0.4, 3.0, h))       # back
    env.add_box_obstacle((cx - 0.5, cy + 1.3, mz), (2.0, 0.4, h)) # top
    env.add_box_obstacle((cx - 0.5, cy - 1.3, mz), (2.0, 0.4, h)) # bottom
    # small lip to make the opening tight (0.8m wide)
    env.add_box_obstacle((cx + 0.45, cy + 0.9, mz), (0.4, 0.4, h))
    env.add_box_obstacle((cx + 0.45, cy - 0.9, mz), (0.4, 0.4, h))

    # Second bug-trap on the other side, oriented the other way
    cx2 = 7.5
    env.add_box_obstacle((cx2 + 1.3, cy, mz), (0.4, 3.0, h))
    env.add_box_obstacle((cx2 + 0.5, cy + 1.3, mz), (2.0, 0.4, h))
    env.add_box_obstacle((cx2 + 0.5, cy - 1.3, mz), (2.0, 0.4, h))
    env.add_box_obstacle((cx2 - 0.45, cy + 0.9, mz), (0.4, 0.4, h))
    env.add_box_obstacle((cx2 - 0.45, cy - 0.9, mz), (0.4, 0.4, h))

    # Some clutter in the middle to force some routing
    env.add_random_boxes(count=8, min_size=0.5, max_size=1.0)


_BUILDERS = {
    "cluttered":    _build_cluttered,
    "forest":       _build_forest,
    "maze3d":       _build_maze3d,
    "indoor":       _build_indoor,
    "narrow":       _build_narrow,
    "narrow_tight": _build_narrow_tight,
    "maze_hard":    _build_maze3d_hard,
    "bugtrap":      _build_bugtrap,
}


def _build_building_3floor(env: VoxelEnvironment3D) -> None:
    """
    Large-scale 3-floor building (30×30×12 m) for scalability testing.
    Each floor 4 m tall, separated by concrete slabs. Two stairwells
    (1.2 m wide) connect floors. Rooms with doorways + random furniture.
    27× larger volume than the standard 10×10×6 m scenes.
    """
    vs = env.voxel_size

    # Floor slabs at z=4 and z=8 (0.4 m thick)
    for z_slab in [4.0, 8.0]:
        env.add_box_obstacle((15.0, 15.0, z_slab), (30.0, 30.0, 0.4))

    # Stairwells: carve holes in slabs
    stairwells = [(5.0, 5.0), (25.0, 20.0)]
    sw_half = 0.6
    for sx, sy in stairwells:
        for z_slab in [4.0, 8.0]:
            x0 = int((sx - sw_half - env.bounds.x_min) / vs)
            x1 = int((sx + sw_half - env.bounds.x_min) / vs) + 1
            y0 = int((sy - sw_half - env.bounds.y_min) / vs)
            y1 = int((sy + sw_half - env.bounds.y_min) / vs) + 1
            z0 = int((z_slab - 0.2 - env.bounds.z_min) / vs)
            z1 = int((z_slab + 0.2 - env.bounds.z_min) / vs) + 1
            env._grid[x0:x1, y0:y1, z0:z1] = 0

    # Interior walls per floor
    for z_base in [0.0, 4.4, 8.4]:
        z_mid = z_base + 1.8
        fh = 3.2
        env.add_box_obstacle((15.0, 15.0, z_mid), (0.3, 30.0, fh))
        env.add_box_obstacle((15.0, 15.0, z_mid), (30.0, 0.3, fh))
        # Doorways: carve wall segments
        for dy in [7.0, 23.0]:
            x0 = int((14.7 - env.bounds.x_min) / vs)
            x1 = int((15.3 - env.bounds.x_min) / vs) + 1
            y0 = int((dy - 0.5 - env.bounds.y_min) / vs)
            y1 = int((dy + 0.5 - env.bounds.y_min) / vs) + 1
            z0 = int((z_base + 0.1 - env.bounds.z_min) / vs)
            z1 = int((z_base + 2.2 - env.bounds.z_min) / vs) + 1
            env._grid[x0:x1, y0:y1, z0:z1] = 0
        for dx in [7.0, 23.0]:
            x0 = int((dx - 0.5 - env.bounds.x_min) / vs)
            x1 = int((dx + 0.5 - env.bounds.x_min) / vs) + 1
            y0 = int((14.7 - env.bounds.y_min) / vs)
            y1 = int((15.3 - env.bounds.y_min) / vs) + 1
            z0 = int((z_base + 0.1 - env.bounds.z_min) / vs)
            z1 = int((z_base + 2.2 - env.bounds.z_min) / vs) + 1
            env._grid[x0:x1, y0:y1, z0:z1] = 0

    # Random furniture
    env.add_random_boxes(count=60, min_size=0.4, max_size=1.5)


_BUILDERS["building_3floor"] = _build_building_3floor


def build_benchmark_environment(name: str,
                                config: BenchmarkConfig | None = None) -> VoxelEnvironment3D:
    configs = _default_configs()
    if name not in configs:
        raise ValueError(f"Unknown benchmark '{name}'. Available: {available_benchmarks()}")
    cfg = config or configs[name]
    env = VoxelEnvironment3D(bounds=cfg.bounds, voxel_size=cfg.voxel_size, seed=cfg.seed)
    _BUILDERS[name](env)
    return env
