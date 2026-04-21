"""Shared type aliases used across BSRM-3D."""
from typing import Tuple, List

Point3D = Tuple[float, float, float]
Edge3D = Tuple[Point3D, Point3D]
Path3D = List[Point3D]
