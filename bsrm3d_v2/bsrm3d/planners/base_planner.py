"""Shared base-class for roadmap planners."""
from __future__ import annotations

from abc import ABC, abstractmethod
import heapq
import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import KDTree

from ..config import Planner3DConfig
from ..environments.base import Environment3D
from ..types import Edge3D, Path3D, Point3D


class BasePlanner3D(ABC):
    """Base class for 3D roadmap planners (PRM-style graphs + A* query)."""

    def __init__(self, env: Environment3D, config: Planner3DConfig):
        self.env = env
        self.config = config
        self.nodes: List[Point3D] = []
        self.edges: List[Edge3D] = []
        self._adj: Dict[int, Dict[int, float]] = {}

    # ------------------------------------------------------------ utilities --
    @staticmethod
    def dist(a, b) -> float:
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

    def _rebuild_adjacency(self) -> None:
        index = {n: i for i, n in enumerate(self.nodes)}
        self._adj = {i: {} for i in range(len(self.nodes))}
        # invalidate query caches - adjacency changed implies graph changed
        self._arr_cache = None
        self._tree_cache = None
        if not self.edges:
            return
        a_pts = np.asarray([e[0] for e in self.edges], dtype=float)
        b_pts = np.asarray([e[1] for e in self.edges], dtype=float)
        weights = np.linalg.norm(a_pts - b_pts, axis=1)
        for (a, b), w in zip(self.edges, weights):
            ia, ib = index[a], index[b]
            w = float(w)
            self._adj[ia][ib] = w
            self._adj[ib][ia] = w

    def _incrementally_add_edges(self,
                                  new_edges: List[Edge3D]) -> None:
        """
        O(|new_edges|) adjacency update for edge-only additions.
        Avoids the O(E) full rebuild when bridging components adds
        a handful of edges. Assumes all endpoints are already in
        self.nodes.
        """
        if not new_edges:
            return
        # build a point->index lookup only once if we need it
        index = getattr(self, "_node_index_cache", None)
        if index is None or len(index) != len(self.nodes):
            index = {n: i for i, n in enumerate(self.nodes)}
            self._node_index_cache = index
        else:
            # ensure any recently-appended nodes are in the cache
            for i in range(len(index), len(self.nodes)):
                index[self.nodes[i]] = i

        for a, b in new_edges:
            if a not in index:
                index[a] = len(index)
                # shouldn't happen, but be defensive
            if b not in index:
                index[b] = len(index)
            ia = index[a]; ib = index[b]
            if ia not in self._adj:
                self._adj[ia] = {}
            if ib not in self._adj:
                self._adj[ib] = {}
            w = float(np.linalg.norm(
                np.asarray(a) - np.asarray(b)))
            self._adj[ia][ib] = w
            self._adj[ib][ia] = w
        self.edges.extend(new_edges)
        # invalidate only the spatial caches
        self._arr_cache = None
        self._tree_cache = None

    def _ensure_tree(self):
        """Build the KDTree on demand and cache it. Invalidated on graph edits."""
        if getattr(self, "_tree_cache", None) is not None:
            return self._arr_cache, self._tree_cache
        arr = np.asarray(self.nodes, dtype=float)
        tree = KDTree(arr) if arr.size else None
        self._arr_cache = arr
        self._tree_cache = tree
        return arr, tree

    # ------------------------------------------------------------- query A* --
    def find_path(self, start: Point3D, goal: Point3D) -> Tuple[Path3D, float]:
        cfg = self.config
        if not self.env.is_free(start, radius=cfg.collision_radius):
            return [], math.inf
        if not self.env.is_free(goal, radius=cfg.collision_radius):
            return [], math.inf
        if not self.nodes:
            return [], math.inf

        arr, tree = self._ensure_tree()

        query_radius = max(
            cfg.connection_radius,
            getattr(cfg, "sketch_link_radius", 0.0),
        ) * 1.6

        def connect_query(q: Point3D, k: int) -> List[Tuple[int, float]]:
            kk = min(max(k, 4), len(self.nodes))
            dists, idx = tree.query(np.asarray(q), k=kk)
            dists = np.atleast_1d(dists); idx = np.atleast_1d(idx)
            # filter by radius first
            keep = dists <= query_radius
            if not keep.any():
                return []
            idxs = idx[keep]
            ds = dists[keep]
            # batched segment check
            qarr = np.tile(np.asarray(q, dtype=float), (idxs.size, 1))
            tarr = arr[idxs]
            if hasattr(self.env, "segments_are_free_batch"):
                ok = self.env.segments_are_free_batch(
                    qarr, tarr,
                    radius=cfg.collision_radius,
                    step=cfg.collision_step,
                )
            else:
                ok = np.array([
                    self.env.segment_is_free(q, self.nodes[int(i)],
                                              radius=cfg.collision_radius,
                                              step=cfg.collision_step)
                    for i in idxs
                ], dtype=bool)
            return [(int(i), float(d)) for i, d, k_ok in zip(idxs, ds, ok) if k_ok]

        start_conn = []
        goal_conn = []
        k = cfg.neighbor_k
        while k <= len(self.nodes):
            if not start_conn:
                start_conn = connect_query(start, k)
            if not goal_conn:
                goal_conn = connect_query(goal, k)
            if start_conn and goal_conn:
                break
            if k >= len(self.nodes):
                break
            k = min(len(self.nodes), k * 2)

        if not start_conn or not goal_conn:
            # BEFORE falling back to whole-path RRT-Connect, try to
            # attach the stranded endpoint(s) to the roadmap via a
            # bounded RRT-Connect to their nearest neighbours. This
            # handles the common maze_hard failure mode where the
            # start sits in a pocket (e.g. near a slab ceiling) and
            # all its 15 nearest neighbours are behind obstacles.
            if not start_conn:
                start_conn = self._rrtc_assisted_attach(
                    start, tree, arr, max_targets=50, total_iters=5000)
            if not goal_conn:
                goal_conn = self._rrtc_assisted_attach(
                    goal, tree, arr, max_targets=50, total_iters=5000)

        if not start_conn or not goal_conn:
            # lazy fallback: try straight-line or RRT-Connect-style bridge
            lazy = self._lazy_fallback(start, goal)
            if lazy:
                return lazy
            return [], math.inf

        # A* on integer indices with virtual start=-1, goal=-2
        S, G = -1, -2
        open_heap: List[tuple] = []
        g_cost = {S: 0.0}
        parent: Dict[int, int] = {}
        heapq.heappush(open_heap, (self.dist(start, goal), 0.0, S))

        goal_idx_map = {i: w for i, w in goal_conn}

        while open_heap:
            _, g_cur, cur = heapq.heappop(open_heap)
            if g_cur > g_cost.get(cur, math.inf):
                continue
            if cur == G:
                seq = [G]
                while seq[-1] in parent:
                    seq.append(parent[seq[-1]])
                seq.reverse()
                out_path: Path3D = []
                for x in seq:
                    if x == S:
                        out_path.append(start)
                    elif x == G:
                        out_path.append(goal)
                    else:
                        out_path.append(self.nodes[x])
                return self._shortcut_path(out_path)

            if cur == S:
                cand = start_conn
            elif cur in self._adj:
                cand = list(self._adj[cur].items())
            else:
                cand = []

            if cur != S and cur != G and cur in goal_idx_map:
                cand = cand + [(G, goal_idx_map[cur])]

            for nxt, w in cand:
                ng = g_cur + w
                if ng >= g_cost.get(nxt, math.inf):
                    continue
                g_cost[nxt] = ng
                parent[nxt] = cur
                if nxt == G:
                    h = 0.0
                else:
                    h = self.dist(self.nodes[nxt], goal)
                heapq.heappush(open_heap, (ng + h, ng, nxt))

        # A* exhausted without reaching goal. This usually means start
        # and goal attached to DIFFERENT connected components of the
        # roadmap (common in multi-floor envs with sparse cross-floor
        # bridges). Try RRTC-assisted attach to discover additional
        # start/goal attach points that may bridge the gap, then retry
        # A* once more.
        extra_start = self._rrtc_assisted_attach(
            start, tree, arr, max_targets=50, total_iters=5000)
        extra_goal = self._rrtc_assisted_attach(
            goal, tree, arr, max_targets=50, total_iters=5000)

        if extra_start or extra_goal:
            # Merge with existing connections and retry A*
            start_conn_set = {i for i, _ in start_conn}
            goal_conn_set = {i for i, _ in goal_conn}
            for ni, w in extra_start:
                if ni not in start_conn_set:
                    start_conn.append((ni, w))
            for ni, w in extra_goal:
                if ni not in goal_conn_set:
                    goal_conn.append((ni, w))
            # Re-run A* with the augmented attachment set.
            # NOTE: self.nodes may have grown via RRTC waypoints; rebuild
            # arr/tree references for heuristic queries.
            path, L = self._astar_with_attachments(
                start, goal, start_conn, goal_conn)
            if path:
                return path, L

        # Final fallback: direct RRT-Connect
        lazy = self._lazy_fallback(start, goal)
        if lazy:
            return lazy
        return [], math.inf

    def _astar_with_attachments(self,
                                 start: Point3D,
                                 goal: Point3D,
                                 start_conn: List[Tuple[int, float]],
                                 goal_conn: List[Tuple[int, float]]
                                 ) -> Tuple[Path3D, float]:
        """Run A* over self.nodes using given start/goal attachments."""
        cfg = self.config
        S, G = -1, -2
        open_heap: List[tuple] = []
        g_cost = {S: 0.0}
        parent: Dict[int, int] = {}
        heapq.heappush(open_heap, (self.dist(start, goal), 0.0, S))
        goal_idx_map = {i: w for i, w in goal_conn}
        while open_heap:
            _, g_cur, cur = heapq.heappop(open_heap)
            if g_cur > g_cost.get(cur, math.inf):
                continue
            if cur == G:
                path = []
                x = G
                while x in parent:
                    path.append(x); x = parent[x]
                path.append(S)
                pts: List[Point3D] = []
                for idx in reversed(path):
                    if idx == S:
                        pts.append(start)
                    elif idx == G:
                        pts.append(goal)
                    else:
                        pts.append(self.nodes[idx])
                return self._shortcut_path(pts)
            if cur == S:
                for ni, w in start_conn:
                    ng = g_cur + w
                    if ng < g_cost.get(ni, math.inf):
                        g_cost[ni] = ng
                        parent[ni] = S
                        heapq.heappush(
                            open_heap,
                            (ng + self.dist(self.nodes[ni], goal), ng, ni)
                        )
                continue
            neigh_map = self._adj.get(cur, {})
            for nxt, w in neigh_map.items():
                ng = g_cur + w
                if ng >= g_cost.get(nxt, math.inf):
                    continue
                g_cost[nxt] = ng
                parent[nxt] = cur
                h = self.dist(self.nodes[nxt], goal)
                heapq.heappush(open_heap, (ng + h, ng, nxt))
            # check goal attach from this node
            if cur in goal_idx_map:
                ng = g_cur + goal_idx_map[cur]
                if ng < g_cost.get(G, math.inf):
                    g_cost[G] = ng
                    parent[G] = cur
                    heapq.heappush(open_heap, (ng, ng, G))
        return [], math.inf

    def _shortcut_path(self,
                        path: Path3D,
                        max_passes: int = 2
                        ) -> Tuple[Path3D, float]:
        """
        Post-process a path by removing redundant waypoints: if a
        segment directly from path[i] to path[j] (j > i+1) is
        collision-free, all intermediate waypoints can be dropped.

        Algorithm: iterative greedy shortcut. On each pass, we walk
        from the current waypoint and keep extending the look-ahead
        until the direct segment becomes blocked. This is O(|path|²)
        collision checks worst case but typically much less — most
        paths shrink by 3-10× after one pass.

        A second pass catches shortcuts enabled by the first pass
        (when removing a waypoint opens up a longer direct segment).

        Returns (smoothed_path, length). Identity when path already
        has < 3 waypoints.

        Typical cost: < 5 ms for 50-waypoint RRTC-bridge paths.
        Straight-line paths (already 2 points) short-circuit.
        """
        if len(path) < 3:
            L = float(np.linalg.norm(
                np.asarray(path[-1]) - np.asarray(path[0])
            )) if len(path) == 2 else 0.0
            return path, L

        cfg = self.config
        cur = path

        for _pass in range(max_passes):
            smoothed: Path3D = [cur[0]]
            i = 0
            while i < len(cur) - 1:
                # Greedily extend look-ahead as far as possible
                best_j = i + 1
                # Binary-search style: try the end first, fall back.
                # For <30 waypoints a linear scan is fastest.
                for j in range(len(cur) - 1, i + 1, -1):
                    if self.env.segment_is_free(
                            cur[i], cur[j],
                            radius=cfg.collision_radius,
                            step=cfg.collision_step):
                        best_j = j
                        break
                smoothed.append(cur[best_j])
                i = best_j
            if len(smoothed) == len(cur):
                break  # no shortcut found, stop iterating
            cur = smoothed

        arr = np.asarray(cur)
        L = float(np.sum(np.linalg.norm(arr[1:] - arr[:-1], axis=1)))
        return cur, L

    def _lazy_fallback(self, start: Point3D, goal: Point3D) -> Tuple[Path3D, float] | None:
        """
        Last-resort solver when graph query fails. Tries:
          1. direct straight line (start -> goal)
          2. a tiny 200-iteration RRT-Connect between start and goal

        If either succeeds, returns (path, length); else None.

        This costs <5ms on easy cases and ~30ms on hard cases but
        recovers the rare corners the sparse roadmap missed.
        """
        cfg = self.config
        # 1) straight line
        if self.env.segment_is_free(start, goal,
                                    radius=cfg.collision_radius,
                                    step=cfg.collision_step):
            return [start, goal], float(np.linalg.norm(
                np.asarray(goal) - np.asarray(start)))

        # 2) tiny RRT-Connect, bounded iterations
        import math as _m
        rng = np.random.default_rng(cfg.seed + 7)
        b = self.env.bounds
        step = max(cfg.collision_step * 3.0, 0.4)
        Ta_pts: List[Point3D] = [start]; Ta_par: List[int] = [-1]
        Tb_pts: List[Point3D] = [goal];  Tb_par: List[int] = [-1]

        def extend(T_pts, T_par, q):
            pts = np.asarray(T_pts)
            d2 = np.sum((pts - np.asarray(q)) ** 2, axis=1)
            i_near = int(np.argmin(d2))
            near = T_pts[i_near]
            nv = np.asarray(q) - np.asarray(near)
            dn = float(np.linalg.norm(nv))
            if dn < 1e-9:
                return "Trapped", None
            if dn <= step:
                new_p = tuple(float(x) for x in q)
            else:
                dirv = nv / dn
                new_p = tuple(float(x) for x in
                              (np.asarray(near) + dirv * step))
            if not self.env.is_free(new_p, radius=cfg.collision_radius):
                return "Trapped", None
            if not self.env.segment_is_free(near, new_p,
                                            radius=cfg.collision_radius,
                                            step=cfg.collision_step):
                return "Trapped", None
            T_pts.append(new_p); T_par.append(i_near)
            arrived = (new_p == tuple(q))
            return ("Reached" if arrived else "Advanced"), len(T_pts) - 1

        max_iters = 5000
        for _ in range(max_iters):
            q = (float(rng.uniform(b.x_min, b.x_max)),
                 float(rng.uniform(b.y_min, b.y_max)),
                 float(rng.uniform(b.z_min, b.z_max)))
            st, lasta = extend(Ta_pts, Ta_par, q)
            if st == "Trapped":
                Ta_pts, Ta_par, Tb_pts, Tb_par = Tb_pts, Tb_par, Ta_pts, Ta_par
                continue
            # try to connect Tb to the newly added point
            target = Ta_pts[lasta]
            while True:
                st2, lastb = extend(Tb_pts, Tb_par, target)
                if st2 == "Reached":
                    # build path
                    def trace(pts, par, idx):
                        out = []
                        while idx != -1:
                            out.append(pts[idx])
                            idx = par[idx]
                        return out
                    pa = list(reversed(trace(Ta_pts, Ta_par, lasta)))
                    pb = trace(Tb_pts, Tb_par, lastb)
                    # If Ta started at start, Tb started at goal;
                    # pa = start -> ... -> target, pb = target -> ... -> goal
                    path = pa + pb[1:]
                    # The fallback trees might have gotten swapped; detect.
                    if path[0] != start:
                        path = list(reversed(path))
                    return self._shortcut_path(path)
                if st2 == "Trapped":
                    break
            Ta_pts, Ta_par, Tb_pts, Tb_par = Tb_pts, Tb_par, Ta_pts, Ta_par
        return None

    def _rrtc_assisted_attach(self,
                               q: Point3D,
                               tree,
                               arr: np.ndarray,
                               max_targets: int = 50,
                               total_iters: int = 5000
                               ) -> List[Tuple[int, float]]:
        """
        Attach ``q`` (start/goal) to the roadmap via a single growing
        RRT tree rooted at ``q``, simultaneously attempting to connect
        to any of the ``max_targets`` nearest roadmap nodes.

        Unlike the naive "try each target with a fresh tree" approach,
        this shares exploration work across all attach attempts: the
        tree rooted at ``q`` grows for ``total_iters`` iterations, and
        we test whether any newly-added branch can reach any of the
        target roadmap nodes.

        Strategy:
        1. Maintain a tree T rooted at ``q``.
        2. On each iteration, sample a random (or target-biased) point
           and extend T toward it.
        3. After each extension, test if the newest tree node can
           connect (by a straight segment) to any target roadmap node.
           If so, attach ``q`` via the path through T + that bridge
           edge to the roadmap.

        Returns ``[(roadmap_node_index, distance_from_q)]`` or [] if
        all attempts failed within the budget.

        Typical cost: 2-80 ms depending on difficulty.
        """
        cfg = self.config
        if len(self.nodes) == 0:
            return []
        kk = min(max_targets, len(self.nodes))
        dists, idx = tree.query(np.asarray(q), k=kk)
        dists = np.atleast_1d(dists); idx = np.atleast_1d(idx)
        target_ids = [int(i) for i in idx]
        target_pts = np.asarray([self.nodes[i] for i in target_ids],
                                 dtype=float)

        rng = np.random.default_rng(cfg.seed + 11)
        step = max(cfg.collision_step * 3.0, 0.4)
        b = self.env.bounds

        # Tree rooted at q
        T_pts: List[Point3D] = [tuple(q)]
        T_par: List[int] = [-1]
        # Track which roadmap target each tree node can see (for incremental
        # reachability check). We check reachability only for new nodes.

        def try_bridge(tree_node_idx: int) -> int | None:
            """Test if tree node can connect by segment to any target.
            Returns the target_id if so, else None."""
            tn = np.asarray(T_pts[tree_node_idx])
            d_to_targets = np.linalg.norm(target_pts - tn, axis=1)
            # only bother with targets within query_radius (avoids long
            # segment checks we'd reject anyway)
            query_r = max(cfg.connection_radius, cfg.min_spacing * 4.0)
            order = np.argsort(d_to_targets)
            for oi in order:
                if d_to_targets[oi] > query_r:
                    break
                tgt_id = target_ids[oi]
                if self.env.segment_is_free(
                        tuple(tn), self.nodes[tgt_id],
                        radius=cfg.collision_radius,
                        step=cfg.collision_step):
                    return tgt_id
            return None

        found_target = None
        found_bridge_tree_idx = None

        for it in range(total_iters):
            # Goal-biased sampling: 40% toward a random target, 60% random.
            if rng.random() < 0.4:
                tgt_choice = int(rng.integers(0, len(target_ids)))
                sample = tuple(target_pts[tgt_choice])
            else:
                sample = (float(rng.uniform(b.x_min, b.x_max)),
                          float(rng.uniform(b.y_min, b.y_max)),
                          float(rng.uniform(b.z_min, b.z_max)))
            # extend T toward sample
            pts = np.asarray(T_pts)
            d2 = np.sum((pts - np.asarray(sample)) ** 2, axis=1)
            i_near = int(np.argmin(d2))
            near = T_pts[i_near]
            nv = np.asarray(sample) - np.asarray(near)
            dn = float(np.linalg.norm(nv))
            if dn < 1e-9:
                continue
            if dn <= step:
                new_p = tuple(float(x) for x in sample)
            else:
                dirv = nv / dn
                new_p = tuple(float(x) for x in
                              (np.asarray(near) + dirv * step))
            if not self.env.is_free(new_p, radius=cfg.collision_radius):
                continue
            if not self.env.segment_is_free(
                    near, new_p, radius=cfg.collision_radius,
                    step=cfg.collision_step):
                continue
            T_pts.append(new_p); T_par.append(i_near)
            new_idx = len(T_pts) - 1
            # Check bridge from this new tree node
            tgt = try_bridge(new_idx)
            if tgt is not None:
                found_target = tgt
                found_bridge_tree_idx = new_idx
                break

        if found_target is None:
            return []

        # Reconstruct path q -> ... -> bridge_tree_node -> target
        def trace(idx):
            out = []
            while idx != -1:
                out.append(T_pts[idx]); idx = T_par[idx]
            return out

        tree_path = list(reversed(trace(found_bridge_tree_idx)))
        # tree_path[0] == q, tree_path[-1] == bridge point
        # target_pt is already in self.nodes as found_target

        # Inject intermediate tree nodes into roadmap
        new_indices = []
        for wp in tree_path[1:]:  # skip q itself
            self.nodes.append(wp)
            new_indices.append(len(self.nodes) - 1)
        # Chain edges: new_indices[0] -> ... -> new_indices[-1] -> target
        chain = new_indices + [found_target]
        for a_i, b_i in zip(chain[:-1], chain[1:]):
            a_pt = self.nodes[a_i]; b_pt = self.nodes[b_i]
            if a_i not in self._adj:
                self._adj[a_i] = {}
            if b_i not in self._adj:
                self._adj[b_i] = {}
            w = float(np.linalg.norm(
                np.asarray(a_pt) - np.asarray(b_pt)))
            self._adj[a_i][b_i] = w
            self._adj[b_i][a_i] = w
            self.edges.append((a_pt, b_pt))

        # invalidate tree cache
        self._tree_cache = None
        self._arr_cache = None

        # Return q->first_new_node attach
        first_idx = new_indices[0]
        first_pt = self.nodes[first_idx]
        dist_q_first = float(np.linalg.norm(
            np.asarray(q) - np.asarray(first_pt)))
        return [(first_idx, dist_q_first)]

    # ------------------------------------------------------------ interface --
    @abstractmethod
    def generate_roadmap(self) -> Tuple[List[Point3D], List[Edge3D]]: ...
