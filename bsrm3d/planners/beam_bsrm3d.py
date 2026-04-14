from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

import numpy as np
from scipy.spatial import KDTree

from bsrm3d.config import Planner3DConfig
from bsrm3d.environments.base import Environment3D
from bsrm3d.sampling.strategies import DirectionSampler, FibonacciDirectionSampler
from bsrm3d.types import Edge3D, Point3D
from .base_planner import BasePlanner3D


@dataclass
class FrontierNodeState:
    explored: np.ndarray


class BeamBSRM3D(BasePlanner3D):
    """
    3D Beam-Sketch Roadmap:
    - Use near-uniform sphere directions (default Fibonacci sphere)
    - Maintain frontier nodes with unexplored directional sectors
    - Cast beams in high-priority directions and add valid samples
    """

    def __init__(
        self,
        env: Environment3D,
        config: Planner3DConfig,
        direction_sampler: DirectionSampler | None = None,
    ):
        super().__init__(env=env, config=config)
        self.direction_sampler = direction_sampler or FibonacciDirectionSampler()
        self._rng = np.random.default_rng(config.seed)
        self._directions = self.direction_sampler.sample(config.direction_samples)
        self._frontier_state: Dict[Point3D, FrontierNodeState] = {}

    def _ray_free_length(self, origin: Point3D, direction: np.ndarray, max_length: float | None = None) -> float:
        d = direction / max(np.linalg.norm(direction), 1e-12)
        step = max(self.config.ray_step, 1e-3)
        ray_len = self.config.ray_length if max_length is None else max(0.05, max_length)
        free_len = 0.0

        for t in np.arange(step, ray_len + step, step):
            p = (
                float(origin[0] + d[0] * t),
                float(origin[1] + d[1] * t),
                float(origin[2] + d[2] * t),
            )
            if not self.env.is_free(p, radius=self.config.collision_radius):
                break
            free_len = float(t)
        return free_len

    def _angle_threshold_cos(self) -> float:
        angle_rad = math.radians(max(1.0, self.config.beam_solid_angle_deg))
        return float(math.cos(angle_rad * 0.5))

    def _mark_explored_sector(self, node: Point3D, direction: np.ndarray) -> None:
        if node not in self._frontier_state:
            self._frontier_state[node] = FrontierNodeState(
                explored=np.zeros(len(self._directions), dtype=bool)
            )

        d = direction / max(np.linalg.norm(direction), 1e-12)
        cos_th = self._angle_threshold_cos()
        dots = self._directions @ d
        self._frontier_state[node].explored |= (dots >= cos_th)

    def _local_visibility_gain(self, node: Point3D) -> float:
        state = self._frontier_state.get(node)
        if state is None:
            return 0.0

        candidate_idx = np.where(~state.explored)[0]
        if len(candidate_idx) == 0:
            return 0.0

        probe_count = min(self.config.visibility_probe_dirs, len(candidate_idx))
        if probe_count <= 0:
            return 0.0

        choice = self._rng.choice(candidate_idx, size=probe_count, replace=False)
        gains = []
        max_len = max(1e-6, self.config.visibility_probe_length)
        for idx in choice:
            free_len = self._ray_free_length(node, self._directions[int(idx)], max_length=max_len)
            gains.append((free_len / max_len) ** 3)
        return float(np.mean(gains)) if gains else 0.0

    def _frontier_score(self, node: Point3D) -> float:
        st = self._frontier_state.get(node)
        if st is None:
            return 0.0

        unexplored = int(np.sum(~st.explored))
        if unexplored <= 0:
            return 0.0

        # Information gain from unexplored directional sectors.
        ratio = unexplored / max(1, len(self._directions))
        info_gain = ratio * math.log1p(unexplored)

        # Local visible volume gain estimated by short ray probes.
        visible_gain = self._local_visibility_gain(node)

        return (
            self.config.frontier_info_weight * info_gain
            + self.config.frontier_visibility_weight * visible_gain
        )

    def _trace_ray(self, origin: Point3D, direction: np.ndarray) -> Point3D | None:
        free_len = self._ray_free_length(origin, direction)
        if free_len <= 0.0:
            return None

        d = direction / max(np.linalg.norm(direction), 1e-12)
        last_free = (
            float(origin[0] + d[0] * free_len),
            float(origin[1] + d[1] * free_len),
            float(origin[2] + d[2] * free_len),
        )

        if self.distance(origin, last_free) < self.config.min_connection_radius:
            return None

        if not self.env.segment_is_free(
            origin,
            last_free,
            radius=self.config.collision_radius,
            step=self.config.step_size,
        ):
            return None

        return last_free

    def _seed_nodes(self, count: int = 16) -> None:
        seeds = list(self.env.sample_free(count))
        for s in seeds:
            if s not in self._frontier_state:
                self.nodes.append(s)
                self._frontier_state[s] = FrontierNodeState(
                    explored=np.zeros(len(self._directions), dtype=bool)
                )

    def _is_far_enough(self, p: Point3D, tree: KDTree | None) -> bool:
        if tree is None or not self.nodes:
            return True
        dist, _ = tree.query(np.asarray(p), k=1)
        return float(dist) >= self.config.min_connection_radius

    def _pick_frontiers(self) -> List[Point3D]:
        ranked = sorted(self.nodes, key=self._frontier_score, reverse=True)
        return ranked[: max(1, self.config.frontier_top_k)]

    def _beam_expand_once(self) -> int:
        if not self.nodes:
            return 0

        added = 0
        tree = KDTree(np.asarray(self.nodes)) if self.nodes else None
        frontiers = self._pick_frontiers()

        for node in frontiers:
            state = self._frontier_state[node]
            candidate_idx = np.where(~state.explored)[0]
            if len(candidate_idx) == 0:
                continue

            # Score directions using novelty + local free-ray gain, then try high-gain ones first.
            candidate_scores = []
            for idx in candidate_idx:
                d = self._directions[int(idx)]
                free_len = self._ray_free_length(node, d, max_length=self.config.visibility_probe_length)
                dir_gain = 1.0 + (free_len / max(1e-6, self.config.visibility_probe_length))
                candidate_scores.append((dir_gain, int(idx)))

            candidate_scores.sort(key=lambda x: x[0], reverse=True)
            max_trials = min(10, len(candidate_scores))
            for _, j in candidate_scores[:max_trials]:
                d = self._directions[int(j)]
                sampled = self._trace_ray(node, d)
                self._mark_explored_sector(node, d)
                if sampled is None:
                    continue
                if not self._is_far_enough(sampled, tree):
                    continue

                self.nodes.append(sampled)
                self._frontier_state[sampled] = FrontierNodeState(
                    explored=np.zeros(len(self._directions), dtype=bool)
                )
                added += 1
                tree = KDTree(np.asarray(self.nodes))
                if len(self.nodes) >= self.config.num_nodes:
                    return added

        return added

    def _connect_edges(self) -> None:
        self.edges = []
        if len(self.nodes) < 2:
            self._rebuild_adjacency()
            return

        arr = np.asarray(self.nodes)
        tree = KDTree(arr)
        used = set()

        for i, node in enumerate(self.nodes):
            k = min(self.config.neighbor_k, len(self.nodes) - 1)
            if k <= 0:
                continue
            _, idx = tree.query(arr[i], k=k + 1)
            idx = np.atleast_1d(idx)

            for j in idx[1:]:
                j = int(j)
                other = self.nodes[j]
                key = (i, j) if i < j else (j, i)
                if key in used:
                    continue

                dist = self.distance(node, other)
                if dist > self.config.connection_radius:
                    continue
                if not self.env.segment_is_free(
                    node,
                    other,
                    radius=self.config.collision_radius,
                    step=self.config.step_size,
                ):
                    continue

                self.edges.append((node, other))
                used.add(key)

        self._rebuild_adjacency()

    def generate_roadmap(self) -> Tuple[List[Point3D], List[Edge3D]]:
        self.nodes = []
        self.edges = []
        self._frontier_state = {}

        self._seed_nodes(count=16)
        if not self.nodes:
            return self.nodes, self.edges

        stagnation = 0
        while len(self.nodes) < self.config.num_nodes:
            added = self._beam_expand_once()
            if added == 0:
                stagnation += 1
            else:
                stagnation = 0

            if stagnation >= 12:
                # If frontier expansion stalls, inject random free nodes.
                inject = list(self.env.sample_free(12))
                for p in inject:
                    self.nodes.append(p)
                    self._frontier_state[p] = FrontierNodeState(
                        explored=np.zeros(len(self._directions), dtype=bool)
                    )
                    if len(self.nodes) >= self.config.num_nodes:
                        break
                stagnation = 0

        self._connect_edges()
        return self.nodes, self.edges
