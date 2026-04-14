from __future__ import annotations

from abc import ABC, abstractmethod
import heapq
import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import KDTree

from bsrm3d.config import Planner3DConfig
from bsrm3d.environments.base import Environment3D
from bsrm3d.types import Edge3D, Path3D, Point3D


class BasePlanner3D(ABC):
    """Base class for 3D roadmap planners."""

    def __init__(self, env: Environment3D, config: Planner3DConfig):
        self.env = env
        self.config = config
        self.nodes: List[Point3D] = []
        self.edges: List[Edge3D] = []
        self._adj: Dict[Point3D, Dict[Point3D, float]] = {}

    @staticmethod
    def distance(a: Point3D, b: Point3D) -> float:
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

    def _rebuild_adjacency(self) -> None:
        self._adj = {n: {} for n in self.nodes}
        for a, b in self.edges:
            w = self.distance(a, b)
            self._adj.setdefault(a, {})[b] = w
            self._adj.setdefault(b, {})[a] = w

    def _connect_start_goal(self, start: Point3D, goal: Point3D) -> Dict[Point3D, Dict[Point3D, float]]:
        graph = {k: dict(v) for k, v in self._adj.items()}
        graph[start] = {}
        graph[goal] = {}

        if not self.nodes:
            return graph

        k = min(max(self.config.neighbor_k, 8), len(self.nodes))
        tree = KDTree(np.asarray(self.nodes))

        for q in [start, goal]:
            _, idx = tree.query(np.asarray(q), k=k)
            idx = np.atleast_1d(idx)
            for i in idx:
                n = self.nodes[int(i)]
                if self.distance(q, n) > self.config.connection_radius:
                    continue
                if not self.env.segment_is_free(q, n, radius=self.config.collision_radius, step=self.config.step_size):
                    continue
                w = self.distance(q, n)
                graph[q][n] = w
                graph.setdefault(n, {})[q] = w

        return graph

    def find_path(self, start: Point3D, goal: Point3D) -> Tuple[Path3D, float]:
        if not self.env.is_free(start, radius=self.config.collision_radius):
            return [], math.inf
        if not self.env.is_free(goal, radius=self.config.collision_radius):
            return [], math.inf
        if not self.nodes:
            return [], math.inf

        graph = self._connect_start_goal(start, goal)
        if not graph[start] or not graph[goal]:
            return [], math.inf

        open_heap = [(self.distance(start, goal), 0.0, start)]
        g = {start: 0.0}
        parent: Dict[Point3D, Point3D] = {}
        visited = set()

        while open_heap:
            _, g_cur, cur = heapq.heappop(open_heap)
            if cur in visited:
                continue
            visited.add(cur)

            if cur == goal:
                path = [goal]
                while path[-1] in parent:
                    path.append(parent[path[-1]])
                path.reverse()
                return path, g_cur

            for nxt, w in graph.get(cur, {}).items():
                ng = g_cur + w
                if ng >= g.get(nxt, math.inf):
                    continue
                g[nxt] = ng
                parent[nxt] = cur
                f = ng + self.distance(nxt, goal)
                heapq.heappush(open_heap, (f, ng, nxt))

        return [], math.inf

    @abstractmethod
    def generate_roadmap(self) -> Tuple[List[Point3D], List[Edge3D]]:
        raise NotImplementedError
