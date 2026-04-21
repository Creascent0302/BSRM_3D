"""
Pure-Python baseline planners used for benchmarking BSRM-3D.

Why reimplement? OMPL Python bindings require a conda install that is not
always available (and not available in our dev container). The four
baselines below are faithful textbook implementations that plug into the
same Environment3D interface and produce the same TrialResult format.

Planners provided:

* ``DeltaPRM``     - classical PRM with fixed connection radius. Uniform
                     sampling + kNN connection. Corresponds to "δ-PRM"
                     in the paper.
* ``PRMStar``      - asymptotically optimal PRM* with log-scaling radius
                     r_n = gamma_PRM * (log n / n)^(1/d).
* ``RRT``          - baseline single-tree RRT.
* ``RRTConnect``   - bidirectional RRT-Connect (what most benchmarks call
                     just "RRT-Connect"; typically the fastest of this group).

None of these need external dependencies beyond numpy + scipy.
"""
from __future__ import annotations

import math
import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree

from ..config import Planner3DConfig
from ..environments.base import Environment3D
from ..types import Edge3D, Path3D, Point3D
from .base_planner import BasePlanner3D


# =========================================================================
#                       Probabilistic Roadmap planners
# =========================================================================
class DeltaPRM(BasePlanner3D):
    """Classical δ-PRM with a fixed connection radius."""

    def __init__(self, env: Environment3D, config: Planner3DConfig,
                 num_samples: int = 900,
                 connection_radius: Optional[float] = None):
        super().__init__(env, config)
        self.num_samples = num_samples
        self.connection_radius = (connection_radius
                                  if connection_radius is not None
                                  else config.connection_radius)
        self._rng = np.random.default_rng(config.seed)

    def generate_roadmap(self):
        cfg = self.config
        self.nodes = list(self.env.sample_free(self.num_samples, rng=self._rng,
                                               radius=cfg.collision_radius))
        self.edges = []
        if len(self.nodes) < 2:
            self._rebuild_adjacency()
            return self.nodes, self.edges
        arr = np.asarray(self.nodes)
        tree = KDTree(arr)
        seen = set()
        # Use radius query so the neighbourhood scales with N correctly.
        # At large N, a fixed k=16 shrinks the effective radius and breaks
        # narrow-passage connectivity. Using query_ball_point captures
        # every node within connection_radius and makes the graph
        # monotone in N.
        for i, n in enumerate(self.nodes):
            js = tree.query_ball_point(arr[i], r=self.connection_radius)
            for j in js:
                if j == i:
                    continue
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in seen:
                    continue
                other = self.nodes[j]
                if self.env.segment_is_free(n, other,
                                            radius=cfg.collision_radius,
                                            step=cfg.collision_step):
                    self.edges.append((n, other))
                    seen.add((a, b))
        self._rebuild_adjacency()
        return self.nodes, self.edges


class PRMStar(BasePlanner3D):
    """PRM* with r_n = gamma * (log n / n)^(1/3) in 3D (asymptotically optimal)."""

    def __init__(self, env: Environment3D, config: Planner3DConfig,
                 num_samples: int = 900,
                 gamma: Optional[float] = None):
        super().__init__(env, config)
        self.num_samples = num_samples
        # Karaman-Frazzoli: gamma_PRM* > 2 (1 + 1/d)^(1/d) * (mu(Xfree)/zeta_d)^(1/d)
        # For a ~600 m^3 box in 3D and mu(B)=4pi/3, a safe default is ~5.
        self.gamma = gamma if gamma is not None else 5.0
        self._rng = np.random.default_rng(config.seed)

    def generate_roadmap(self):
        cfg = self.config
        self.nodes = list(self.env.sample_free(self.num_samples, rng=self._rng,
                                               radius=cfg.collision_radius))
        self.edges = []
        n = len(self.nodes)
        if n < 2:
            self._rebuild_adjacency()
            return self.nodes, self.edges
        r_n = self.gamma * (math.log(max(n, 2)) / max(n, 2)) ** (1 / 3)
        arr = np.asarray(self.nodes)
        tree = KDTree(arr)
        seen = set()
        for i, a in enumerate(self.nodes):
            neigh = tree.query_ball_point(arr[i], r=r_n)
            for jj in neigh:
                j = int(jj)
                if j == i:
                    continue
                ia, ib = (i, j) if i < j else (j, i)
                if (ia, ib) in seen:
                    continue
                b = self.nodes[j]
                if self.env.segment_is_free(a, b,
                                            radius=cfg.collision_radius,
                                            step=cfg.collision_step):
                    self.edges.append((a, b))
                    seen.add((ia, ib))
        self._rebuild_adjacency()
        return self.nodes, self.edges


class LazyPRM(BasePlanner3D):
    """
    Lazy PRM (Bohlin & Kavraki, 2000).

    Builds the full nearest-neighbour graph WITHOUT collision-checking edges.
    Collision checks are deferred to query time: A* runs on the full graph,
    and each traversed edge is validated lazily. Invalid edges are removed
    and the search continues.

    This is typically much faster than DeltaPRM on environments where most
    random edges are obstacle-free.
    """

    def __init__(self, env: Environment3D, config: Planner3DConfig,
                 num_samples: int = 900,
                 connection_radius: Optional[float] = None):
        super().__init__(env, config)
        self.num_samples = num_samples
        self.connection_radius = (connection_radius
                                  if connection_radius is not None
                                  else config.connection_radius)
        self._rng = np.random.default_rng(config.seed)
        # lazy state: mapping (i, j) -> True if validated free, False if
        # known-blocked, missing if unvalidated
        self._edge_status: Dict[Tuple[int, int], bool] = {}
        self._blocked: set = set()

    def generate_roadmap(self):
        cfg = self.config
        # sample points (no collision checking while building edges)
        self.nodes = list(self.env.sample_free(self.num_samples, rng=self._rng,
                                               radius=cfg.collision_radius))
        self.edges = []  # we don't maintain a validated edge list
        if len(self.nodes) < 2:
            self._rebuild_adjacency()
            return self.nodes, self.edges

        arr = np.asarray(self.nodes)
        tree = KDTree(arr)
        # Radius-based (see DeltaPRM comment).
        self._adj = {i: {} for i in range(len(self.nodes))}
        for i in range(len(self.nodes)):
            js = tree.query_ball_point(arr[i], r=self.connection_radius)
            for j in js:
                j = int(j)
                if j == i:
                    continue
                d = float(np.linalg.norm(arr[i] - arr[j]))
                self._adj[i][j] = d
                self._adj[j][i] = d
                if i < j:
                    self.edges.append((self.nodes[i], self.nodes[j]))
        # invalidate the base class's tree cache so find_path will refresh
        self._arr_cache = arr
        self._tree_cache = tree
        self._edge_status = {}
        self._blocked = set()
        return self.nodes, self.edges

    def _edge_free(self, i: int, j: int) -> bool:
        cfg = self.config
        key = (i, j) if i < j else (j, i)
        if key in self._edge_status:
            return self._edge_status[key]
        ok = self.env.segment_is_free(
            self.nodes[i], self.nodes[j],
            radius=cfg.collision_radius,
            step=cfg.collision_step,
        )
        self._edge_status[key] = ok
        if not ok:
            self._blocked.add(key)
        return ok

    def find_path(self, start: Point3D, goal: Point3D):
        """
        Override the base class's A*: we want lazy edge validation.
        """
        cfg = self.config
        if not self.env.is_free(start, radius=cfg.collision_radius):
            return [], math.inf
        if not self.env.is_free(goal, radius=cfg.collision_radius):
            return [], math.inf
        if not self.nodes:
            return [], math.inf

        arr, tree = self._ensure_tree()
        qr = max(cfg.connection_radius,
                 getattr(cfg, "sketch_link_radius", 0.0)) * 1.6

        def connect(q, k):
            k = min(max(k, 4), len(self.nodes))
            dists, idx = tree.query(np.asarray(q), k=k)
            out = []
            for d, i in zip(np.atleast_1d(dists), np.atleast_1d(idx)):
                if d > qr:
                    continue
                if self.env.segment_is_free(q, self.nodes[int(i)],
                                             radius=cfg.collision_radius,
                                             step=cfg.collision_step):
                    out.append((int(i), float(d)))
            return out

        sc, gc = [], []
        kk = cfg.neighbor_k
        while kk <= len(self.nodes):
            if not sc:
                sc = connect(start, kk)
            if not gc:
                gc = connect(goal, kk)
            if sc and gc:
                break
            if kk >= len(self.nodes):
                break
            kk *= 2

        if not sc or not gc:
            return [], math.inf

        S, G = -1, -2
        goal_map = {i: w for i, w in gc}

        for _ in range(6):  # lazy outer-loop retries if edges fall out
            import heapq as _h
            heap = [(self.dist(start, goal), 0.0, S)]
            g_cost = {S: 0.0}
            parent = {}
            found = False
            while heap:
                _, gc_cur, cur = _h.heappop(heap)
                if gc_cur > g_cost.get(cur, math.inf):
                    continue
                if cur == G:
                    found = True
                    break
                if cur == S:
                    cands = sc
                else:
                    cands = list(self._adj.get(cur, {}).items())
                    if cur in goal_map:
                        cands.append((G, goal_map[cur]))
                for nxt, w in cands:
                    ng = gc_cur + w
                    if ng >= g_cost.get(nxt, math.inf):
                        continue
                    g_cost[nxt] = ng
                    parent[nxt] = cur
                    h = 0.0 if nxt == G else self.dist(self.nodes[nxt], goal)
                    _h.heappush(heap, (ng + h, ng, nxt))

            if not found:
                return [], math.inf

            # reconstruct
            seq = [G]
            while seq[-1] in parent:
                seq.append(parent[seq[-1]])
            seq.reverse()
            # lazy validate internal edges
            need_retry = False
            for a, b in zip(seq[:-1], seq[1:]):
                if a in (S, G) or b in (S, G):
                    continue
                if not self._edge_free(a, b):
                    # drop from adjacency and retry
                    self._adj[a].pop(b, None)
                    self._adj[b].pop(a, None)
                    need_retry = True
            if not need_retry:
                out = []
                for x in seq:
                    if x == S: out.append(start)
                    elif x == G: out.append(goal)
                    else: out.append(self.nodes[x])
                return out, g_cost[G]
        return [], math.inf


class SPARS2(BasePlanner3D):
    """
    SPARS2 (Dobson & Bekris, 2013) - Sparse Roadmap Spanner 2.

    Core idea: incrementally sample and *add only when necessary*. Every
    sample `q` triggers at most one of four rules:

      1. COVERAGE: `q` cannot see any existing vertex within
         visibility range -> add it as a guard.
      2. CONNECTIVITY: `q` sees >= 2 vertices belonging to DIFFERENT
         connected components -> add q, create bridge edges.
      3. INTERFACE: `q` sees two vertices v1, v2 that are in the same
         component but the path v1 -> q -> v2 via direct edges beats
         the best existing path by a factor >= stretch_t. Edges added.
      4. Otherwise: discard.

    We also enforce a local Poisson-disk gap to avoid oversampling.

    This is the direct competitor of BSRM-3D -- both produce sparse
    structure-aware roadmaps. SPARS2 relies heavily on visibility tests
    and so its runtime grows with node count.
    """

    def __init__(self, env: Environment3D, config: Planner3DConfig,
                 num_samples: int = 2000,
                 visibility_range: Optional[float] = None,
                 stretch_t: float = 3.0,
                 max_failures: int = 200):
        super().__init__(env, config)
        self.num_samples = num_samples
        self.visibility_range = (visibility_range
                                  if visibility_range is not None
                                  else config.connection_radius)
        self.stretch_t = stretch_t
        self.max_failures = max_failures
        self._rng = np.random.default_rng(config.seed)

    # --------------------------------------------------------- connectivity --
    def _components(self):
        """Return a dict node_idx -> component_id using union-find."""
        n = len(self.nodes)
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        for (a, b) in self.edges:
            ia = self._node_idx[a]; ib = self._node_idx[b]
            union(ia, ib)
        return [find(i) for i in range(n)]

    def _rebuild_node_index(self):
        self._node_idx = {n: i for i, n in enumerate(self.nodes)}

    def _visible_nodes(self, q: Point3D, tree: KDTree):
        """Return list of (idx, distance) for nodes within visibility_range
        that are collision-free line-of-sight."""
        if not self.nodes:
            return []
        idx = tree.query_ball_point(np.asarray(q), r=self.visibility_range)
        if not idx:
            return []
        cfg = self.config
        visible = []
        # batch the visibility checks
        qarr = np.tile(np.asarray(q, dtype=float), (len(idx), 1))
        tarr = np.asarray([self.nodes[int(i)] for i in idx], dtype=float)
        if hasattr(self.env, "segments_are_free_batch"):
            ok = self.env.segments_are_free_batch(
                qarr, tarr,
                radius=cfg.collision_radius,
                step=cfg.collision_step,
            )
        else:
            ok = np.array([self.env.segment_is_free(
                q, self.nodes[int(i)],
                radius=cfg.collision_radius,
                step=cfg.collision_step) for i in idx], dtype=bool)
        for i, free in zip(idx, ok):
            if free:
                d = float(np.linalg.norm(
                    np.asarray(q) - np.asarray(self.nodes[int(i)])))
                visible.append((int(i), d))
        return visible

    def generate_roadmap(self):
        cfg = self.config
        self.nodes = []
        self.edges = []
        self._rebuild_node_index()

        # bootstrap with one free seed so the first sample is tested against something
        seeds = self.env.sample_free(1, rng=self._rng, radius=cfg.collision_radius)
        if seeds:
            self.nodes.append(seeds[0])
            self._rebuild_node_index()

        failures = 0
        tree: Optional[KDTree] = None
        if self.nodes:
            tree = KDTree(np.asarray(self.nodes))

        for attempt in range(self.num_samples):
            if failures >= self.max_failures:
                break
            q_list = self.env.sample_free(1, rng=self._rng,
                                          radius=cfg.collision_radius)
            if not q_list:
                continue
            q = q_list[0]
            if not self.nodes:
                self.nodes.append(q)
                self._rebuild_node_index()
                tree = KDTree(np.asarray(self.nodes))
                continue

            visible = self._visible_nodes(q, tree)

            if not visible:
                # RULE 1: COVERAGE
                self.nodes.append(q)
                self._rebuild_node_index()
                tree = KDTree(np.asarray(self.nodes))
                failures = 0
                continue

            # find components among visible set
            comps = self._components()
            visible_comps = {comps[i]: (i, d) for i, d in visible}
            if len(visible_comps) >= 2:
                # RULE 2: CONNECTIVITY
                self.nodes.append(q)
                self._rebuild_node_index()
                q_idx = self._node_idx[q]
                for (vi, d) in visible:
                    self.edges.append((self.nodes[vi], q))
                tree = KDTree(np.asarray(self.nodes))
                failures = 0
                continue

            # RULE 3: INTERFACE (quality path)
            # Consider the closest two visible nodes v1,v2 (in the same component).
            # Check if going v1 -> q -> v2 is shorter than the graph shortest path
            # between them by factor t. If so, add q + edges.
            if len(visible) >= 2:
                # sort by distance and try a few close pairs
                visible_sorted = sorted(visible, key=lambda x: x[1])[:6]
                improved = False
                for i in range(len(visible_sorted)):
                    for j in range(i + 1, len(visible_sorted)):
                        v1, d1 = visible_sorted[i]
                        v2, d2 = visible_sorted[j]
                        # direct path length via q
                        via_q = d1 + d2
                        # graph shortest path between v1 and v2
                        sp = self._shortest_path_len(v1, v2)
                        if sp == math.inf:
                            continue
                        if via_q * self.stretch_t < sp:
                            self.nodes.append(q)
                            self._rebuild_node_index()
                            self.edges.append((self.nodes[v1], q))
                            self.edges.append((self.nodes[v2], q))
                            tree = KDTree(np.asarray(self.nodes))
                            failures = 0
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    continue

            # sample did nothing
            failures += 1

        # final: rebuild adjacency from accumulated edges
        self._rebuild_adjacency()
        return self.nodes, self.edges

    def _shortest_path_len(self, src: int, dst: int) -> float:
        """Bounded Dijkstra shortest-path length between two node indices."""
        import heapq
        dist = {src: 0.0}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if u == dst:
                return d
            for v, w in self._adj.get(u, {}).items():
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        # adj was rebuilt? fallback: rebuild quickly
        if src not in self._adj:
            self._rebuild_adjacency()
            return self._shortest_path_len(src, dst)
        return math.inf


class HaltonPRM(BasePlanner3D):
    """
    Deterministic PRM using a Halton quasi-random sequence instead of
    uniform random sampling. A common "deterministic roadmap" baseline.
    Because Halton points are low-discrepancy, the roadmap coverage is
    more uniform than a random PRM at the same sample count, which often
    gives better empirical success in moderate-clutter environments.
    """

    def __init__(self, env: Environment3D, config: Planner3DConfig,
                 num_samples: int = 900,
                 connection_radius: Optional[float] = None):
        super().__init__(env, config)
        self.num_samples = num_samples
        self.connection_radius = (connection_radius
                                  if connection_radius is not None
                                  else config.connection_radius)

    @staticmethod
    def _halton(index: int, base: int) -> float:
        f = 1.0; r = 0.0; i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def _halton_points(self, n: int) -> List[Point3D]:
        b = self.env.bounds
        points: List[Point3D] = []
        skip = 50        # conventional skip for low-discrepancy bias
        i = skip
        while len(points) < n and i < skip + 20 * n:
            u = self._halton(i, 2)
            v = self._halton(i, 3)
            w = self._halton(i, 5)
            p = (b.x_min + u * (b.x_max - b.x_min),
                 b.y_min + v * (b.y_max - b.y_min),
                 b.z_min + w * (b.z_max - b.z_min))
            if self.env.is_free(p, radius=self.config.collision_radius):
                points.append(p)
            i += 1
        return points

    def generate_roadmap(self):
        cfg = self.config
        self.nodes = self._halton_points(self.num_samples)
        self.edges = []
        n = len(self.nodes)
        if n < 2:
            self._rebuild_adjacency()
            return self.nodes, self.edges

        arr = np.asarray(self.nodes)
        tree = KDTree(arr)
        seen = set()
        # Radius-based connection (see DeltaPRM comment): ensures monotone
        # behaviour as N grows and prevents the kNN-pathology where large
        # N actually drops success rate in narrow-passage scenarios.
        pair_list: List[Tuple[int, int]] = []
        for i in range(n):
            js = tree.query_ball_point(arr[i], r=self.connection_radius)
            for j in js:
                j = int(j)
                if j == i:
                    continue
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                pair_list.append((a, b))
        if pair_list:
            pair_arr = np.asarray(pair_list, dtype=np.int64)
            starts = arr[pair_arr[:, 0]]
            ends = arr[pair_arr[:, 1]]
            if hasattr(self.env, "segments_are_free_batch"):
                ok = self.env.segments_are_free_batch(
                    starts, ends,
                    radius=cfg.collision_radius,
                    step=cfg.collision_step,
                )
            else:
                ok = np.array([
                    self.env.segment_is_free(
                        tuple(s), tuple(e),
                        radius=cfg.collision_radius,
                        step=cfg.collision_step)
                    for s, e in zip(starts, ends)
                ], dtype=bool)
            self.edges = [
                (self.nodes[int(a)], self.nodes[int(b)])
                for (a, b), keep in zip(pair_list, ok) if keep
            ]
        self._rebuild_adjacency()
        return self.nodes, self.edges


# =========================================================================
#                          RRT planners (unchanged)
# =========================================================================
@dataclass
class _Tree:
    points: List[Point3D]
    parent: List[int]


class RRT(BasePlanner3D):
    """
    Baseline single-tree RRT for multi-query comparison. Since RRT is
    inherently single-query, ``generate_roadmap`` is a no-op and the path
    is produced on the fly inside ``find_path``.
    """

    def __init__(self, env: Environment3D, config: Planner3DConfig,
                 max_iters: int = 5000,
                 goal_bias: float = 0.1,
                 extend_step: float = 0.5):
        super().__init__(env, config)
        self.max_iters = max_iters
        self.goal_bias = goal_bias
        self.extend_step = extend_step
        self._rng = np.random.default_rng(config.seed)

    def generate_roadmap(self):
        # nothing to pre-build; keep the graph empty
        self.nodes = []
        self.edges = []
        self._rebuild_adjacency()
        return self.nodes, self.edges

    def _steer(self, a: Point3D, b: Point3D) -> Point3D:
        aa = np.asarray(a, dtype=float)
        bb = np.asarray(b, dtype=float)
        v = bb - aa
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            return a
        if n <= self.extend_step:
            return tuple(bb)
        return tuple(aa + (v / n) * self.extend_step)

    def find_path(self, start: Point3D, goal: Point3D) -> Tuple[Path3D, float]:
        cfg = self.config
        if not self.env.is_free(start, radius=cfg.collision_radius):
            return [], math.inf
        if not self.env.is_free(goal, radius=cfg.collision_radius):
            return [], math.inf

        T = _Tree(points=[start], parent=[-1])
        goal_arr = np.asarray(goal)
        b = self.env.bounds
        best_goal_idx = -1
        goal_reach_dist = self.extend_step

        for _ in range(self.max_iters):
            if self._rng.random() < self.goal_bias:
                q_rand = goal
            else:
                q_rand = (float(self._rng.uniform(b.x_min, b.x_max)),
                          float(self._rng.uniform(b.y_min, b.y_max)),
                          float(self._rng.uniform(b.z_min, b.z_max)))
            # nearest
            pts = np.asarray(T.points)
            d2 = np.sum((pts - np.asarray(q_rand)) ** 2, axis=1)
            idx_near = int(np.argmin(d2))
            q_near = T.points[idx_near]
            q_new = self._steer(q_near, q_rand)
            if not self.env.is_free(q_new, radius=cfg.collision_radius):
                continue
            if not self.env.segment_is_free(q_near, q_new,
                                            radius=cfg.collision_radius,
                                            step=cfg.collision_step):
                continue
            T.points.append(q_new)
            T.parent.append(idx_near)
            if self.dist(q_new, goal) <= goal_reach_dist and self.env.segment_is_free(
                    q_new, goal, radius=cfg.collision_radius,
                    step=cfg.collision_step):
                T.points.append(goal)
                T.parent.append(len(T.points) - 2)
                best_goal_idx = len(T.points) - 1
                break

        if best_goal_idx < 0:
            return [], math.inf

        # reconstruct
        path = []
        cur = best_goal_idx
        while cur != -1:
            path.append(T.points[cur])
            cur = T.parent[cur]
        path.reverse()
        return path, self._path_len(path)

    @staticmethod
    def _path_len(p: List[Point3D]) -> float:
        if len(p) < 2:
            return math.inf
        a = np.asarray(p)
        return float(np.sum(np.linalg.norm(a[1:] - a[:-1], axis=1)))


class RRTConnect(BasePlanner3D):
    """Bidirectional RRT-Connect."""

    def __init__(self, env: Environment3D, config: Planner3DConfig,
                 max_iters: int = 4000,
                 extend_step: float = 0.5):
        super().__init__(env, config)
        self.max_iters = max_iters
        self.extend_step = extend_step
        self._rng = np.random.default_rng(config.seed)

    def generate_roadmap(self):
        self.nodes = []
        self.edges = []
        self._rebuild_adjacency()
        return self.nodes, self.edges

    def _steer(self, a: Point3D, b: Point3D) -> Point3D:
        aa = np.asarray(a, dtype=float)
        bb = np.asarray(b, dtype=float)
        v = bb - aa
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            return a
        if n <= self.extend_step:
            return tuple(bb)
        return tuple(aa + (v / n) * self.extend_step)

    def _extend(self, T: _Tree, q: Point3D):
        cfg = self.config
        pts = np.asarray(T.points)
        d2 = np.sum((pts - np.asarray(q)) ** 2, axis=1)
        idx = int(np.argmin(d2))
        q_near = T.points[idx]
        q_new = self._steer(q_near, q)
        if not self.env.is_free(q_new, radius=cfg.collision_radius):
            return "Trapped", None
        if not self.env.segment_is_free(q_near, q_new,
                                        radius=cfg.collision_radius,
                                        step=cfg.collision_step):
            return "Trapped", None
        T.points.append(q_new)
        T.parent.append(idx)
        if np.allclose(q_new, q):
            return "Reached", len(T.points) - 1
        return "Advanced", len(T.points) - 1

    def _connect(self, T: _Tree, q: Point3D):
        status = "Advanced"
        last_idx = None
        while status == "Advanced":
            status, last_idx = self._extend(T, q)
        return status, last_idx

    def find_path(self, start: Point3D, goal: Point3D):
        cfg = self.config
        if not self.env.is_free(start, radius=cfg.collision_radius):
            return [], math.inf
        if not self.env.is_free(goal, radius=cfg.collision_radius):
            return [], math.inf

        Ta = _Tree([start], [-1])
        Tb = _Tree([goal], [-1])
        b = self.env.bounds

        for i in range(self.max_iters):
            q_rand = (float(self._rng.uniform(b.x_min, b.x_max)),
                      float(self._rng.uniform(b.y_min, b.y_max)),
                      float(self._rng.uniform(b.z_min, b.z_max)))
            status, last = self._extend(Ta, q_rand)
            if status != "Trapped":
                q_new = Ta.points[last]
                st2, idx2 = self._connect(Tb, q_new)
                if st2 == "Reached":
                    # reconstruct
                    def trace(T, i):
                        out = []
                        while i != -1:
                            out.append(T.points[i])
                            i = T.parent[i]
                        return out
                    path_a = list(reversed(trace(Ta, last)))
                    path_b = trace(Tb, idx2)
                    # de-duplicate join point
                    full = path_a + path_b[1:]
                    return full, RRT._path_len(full)
            # swap
            Ta, Tb = Tb, Ta

        return [], math.inf
