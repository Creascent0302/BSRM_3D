from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Set, Tuple
import numpy as np

from bsrm3d.config import EnvironmentBounds
from bsrm3d.types import Point3D
from .base import Environment3D

Voxel = Tuple[int, int, int]


@dataclass
class VoxelEnvironment3D(Environment3D):
    bounds: EnvironmentBounds
    voxel_size: float = 0.2
    seed: int = 42
    occupied_voxels: Set[Voxel] = field(default_factory=set)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def _to_voxel(self, p: Point3D) -> Voxel:
        bx = int((p[0] - self.bounds.x_min) / self.voxel_size)
        by = int((p[1] - self.bounds.y_min) / self.voxel_size)
        bz = int((p[2] - self.bounds.z_min) / self.voxel_size)
        return bx, by, bz

    def _in_bounds(self, p: Point3D, margin: float = 0.0) -> bool:
        return (
            self.bounds.x_min + margin <= p[0] <= self.bounds.x_max - margin
            and self.bounds.y_min + margin <= p[1] <= self.bounds.y_max - margin
            and self.bounds.z_min + margin <= p[2] <= self.bounds.z_max - margin
        )

    def is_free(self, point: Point3D, radius: float = 0.0) -> bool:
        if not self._in_bounds(point, margin=radius):
            return False

        if radius <= 0:
            return self._to_voxel(point) not in self.occupied_voxels

        # Check local neighborhood voxels to account for robot radius.
        vx, vy, vz = self._to_voxel(point)
        r = int(np.ceil(radius / self.voxel_size))
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if (vx + dx, vy + dy, vz + dz) in self.occupied_voxels:
                        return False
        return True

    def segment_is_free(self, a: Point3D, b: Point3D, radius: float = 0.0, step: float = 0.1) -> bool:
        aa = np.asarray(a, dtype=float)
        bb = np.asarray(b, dtype=float)
        length = float(np.linalg.norm(bb - aa))
        if length < 1e-12:
            return self.is_free(a, radius=radius)

        n = max(2, int(np.ceil(length / max(step, 1e-3))))
        for t in np.linspace(0.0, 1.0, n):
            p = tuple(aa + t * (bb - aa))
            if not self.is_free(p, radius=radius):
                return False
        return True

    def sample_free(self, n: int) -> Iterable[Point3D]:
        points = []
        max_attempts = max(1000, n * 100)
        attempts = 0
        while len(points) < n and attempts < max_attempts:
            attempts += 1
            p = (
                float(self._rng.uniform(self.bounds.x_min, self.bounds.x_max)),
                float(self._rng.uniform(self.bounds.y_min, self.bounds.y_max)),
                float(self._rng.uniform(self.bounds.z_min, self.bounds.z_max)),
            )
            if self.is_free(p):
                points.append(p)
        return points

    def add_box_obstacle(self, center: Point3D, size: Point3D) -> None:
        hx, hy, hz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
        x0, x1 = center[0] - hx, center[0] + hx
        y0, y1 = center[1] - hy, center[1] + hy
        z0, z1 = center[2] - hz, center[2] + hz

        xi0 = int((x0 - self.bounds.x_min) / self.voxel_size)
        xi1 = int((x1 - self.bounds.x_min) / self.voxel_size)
        yi0 = int((y0 - self.bounds.y_min) / self.voxel_size)
        yi1 = int((y1 - self.bounds.y_min) / self.voxel_size)
        zi0 = int((z0 - self.bounds.z_min) / self.voxel_size)
        zi1 = int((z1 - self.bounds.z_min) / self.voxel_size)

        for ix in range(xi0, xi1 + 1):
            for iy in range(yi0, yi1 + 1):
                for iz in range(zi0, zi1 + 1):
                    self.occupied_voxels.add((ix, iy, iz))

    def add_random_boxes(self, count: int, min_size: float, max_size: float) -> None:
        for _ in range(count):
            cx = float(self._rng.uniform(self.bounds.x_min, self.bounds.x_max))
            cy = float(self._rng.uniform(self.bounds.y_min, self.bounds.y_max))
            cz = float(self._rng.uniform(self.bounds.z_min, self.bounds.z_max))
            sx = float(self._rng.uniform(min_size, max_size))
            sy = float(self._rng.uniform(min_size, max_size))
            sz = float(self._rng.uniform(min_size, max_size))
            self.add_box_obstacle((cx, cy, cz), (sx, sy, sz))

    def voxel_center(self, voxel: Voxel) -> Point3D:
        return (
            self.bounds.x_min + (voxel[0] + 0.5) * self.voxel_size,
            self.bounds.y_min + (voxel[1] + 0.5) * self.voxel_size,
            self.bounds.z_min + (voxel[2] + 0.5) * self.voxel_size,
        )

    def occupied_centers(self, max_points: int | None = None) -> List[Point3D]:
        voxels = list(self.occupied_voxels)
        if max_points is not None and max_points > 0 and len(voxels) > max_points:
            step = max(1, len(voxels) // max_points)
            voxels = voxels[::step]
        return [self.voxel_center(v) for v in voxels]
