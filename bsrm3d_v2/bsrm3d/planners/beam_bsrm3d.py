"""
BSRM-3D  (Beam-Sketch Roadmap, 3D version).

A direct 3D port of the two-phase algorithm from the paper
  "Beam-Sketch Roadmap: A Fast, Sparse, and Structure-Aware Roadmap for
   Motion Planning in Complex Environments"

The planner is organised in three concrete steps:

  1. Beam-driven topological exploration  (Algorithm 1 of the paper)
     - Every frontier node gets a Fibonacci-sphere fan of rays.
     - Each ray is marched until it hits Xobs; the free-flight distance
       d(u, theta) is recorded (Eq. 1).
     - For each pair of geometrically-adjacent beams, a depth discontinuity
       is detected when  |d_i - d_j| > max(d0, k * min(d_i, d_j))  (Eq. 2).
     - Two candidate configurations are spawned on each discontinuity:
         p_short = u + f_s * d_short * v_short
         p_long  = u + f_l * d_long  * v_long
     - Candidates are filtered by a Poisson-disk r_min gate.
     - On connection, an angular sector phi of the exploration mask is
       suppressed on both endpoints (Sec. IV-A.4).

  2. Structural sketching via NMS  (Algorithm 2 of the paper)
     - Every retained node v gets an approximate clearance value c(v)
       (min free-flight over the sphere fan).
     - A max-heap is processed in order of clearance. Each popped node
       suppresses every graph-reachable, mutually-visible neighbour
       within r_v (Eq. 3).
     - Surviving nodes are relinked within r_l with collision-free edges.

  3. Query  (A*)  via the base class.

Implementation notes:
  * All beam casts are vectorised over the voxel grid, so a 256-ray fan
    over a 10m map costs ~1-2 ms.
  * We pre-compute the neighbour lists of the direction fan (for Eq. 2)
    once, and never refresh them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree

from ..config import Planner3DConfig
from ..environments.base import Environment3D
from ..sampling.strategies import (DirectionSampler, FibonacciDirectionSampler,
                                    directions_from_angular_step)
from ..types import Edge3D, Point3D
from .base_planner import BasePlanner3D


# ---------------------------------------------------------------- helpers --
@dataclass
class _Frontier:
    """Per-node exploration mask + cached clearance."""
    explored: np.ndarray            # bool[num_dirs], True = already covered
    clearance: float = math.inf     # min free-flight distance so far


# ---------------------------------------------------------------- planner --
class BeamBSRM3D(BasePlanner3D):
    """Two-phase BSRM planner in 3D."""

    # ---------------------------------------------------------------- init --
    def __init__(self,
                 env: Environment3D,
                 config: Planner3DConfig,
                 direction_sampler: Optional[DirectionSampler] = None):
        super().__init__(env=env, config=config)
        self._rng = np.random.default_rng(config.seed)

        if direction_sampler is None:
            # pick a fan that matches the requested angular step
            self._dirs = directions_from_angular_step(config.angular_step_deg)
        else:
            # fall back to a reasonable default count
            self._dirs = direction_sampler.sample(256)

        # --- pre-compute directional adjacency for discontinuity pairing ---
        # For each direction, we pick its K nearest directions on the sphere
        # (smallest angular separation) and only test these pairs.
        self._dir_pairs = self._build_dir_pairs(k=6)

        # angular sector (full width) in radians
        self._sector_half_cos = float(
            math.cos(math.radians(config.sector_half_angle_deg)))

        # --- run-time state ---
        self._frontier: List[_Frontier] = []

    def _count_discontinuities(self, dists: np.ndarray) -> int:
        """Quick count of beam-pair discontinuities (for feature vector)."""
        cfg = self.config
        I = self._dir_pairs[:, 0]
        J = self._dir_pairs[:, 1]
        di, dj = dists[I], dists[J]
        diff = np.abs(di - dj)
        mins = np.minimum(di, dj)
        thresh = np.maximum(cfg.discontinuity_abs, cfg.discontinuity_rel * mins)
        hit = (diff > thresh) & (np.maximum(di, dj) > cfg.min_spacing)
        return int(hit.sum())

    def _build_dir_pairs(self, k: int = 6) -> np.ndarray:
        """Build direction pairs for discontinuity detection."""
        n = self._dirs.shape[0]
        if n < 2:
            return np.zeros((0, 2), dtype=np.int32)
        kk = min(k, n - 1)
        dots = self._dirs @ self._dirs.T
        np.fill_diagonal(dots, -2.0)
        nbrs = np.argpartition(-dots, kth=kk, axis=1)[:, :kk]
        pairs = set()
        for i in range(n):
            for j in nbrs[i]:
                a, b = int(i), int(j)
                if a == b: continue
                pairs.add((a, b) if a < b else (b, a))
        return np.array(sorted(pairs), dtype=np.int32)

    # -------------------------------------------------- beam casting utils --
    def _cast_fan(self, origin: Point3D) -> np.ndarray:
        """Return a length-(num_dirs,) array of free-flight distances. Fully vectorised."""
        cfg = self.config
        # Use the environment's batched ray caster if available.
        if hasattr(self.env, "cast_beam_fan"):
            return self.env.cast_beam_fan(
                origin, self._dirs,
                max_length=cfg.beam_max_length,
                step=cfg.beam_step,
            )
        # fallback: per-ray loop (old path)
        out = np.zeros(self._dirs.shape[0], dtype=float)
        for i, d in enumerate(self._dirs):
            out[i] = self.env.first_hit_distance(
                origin, d,
                max_length=cfg.beam_max_length,
                step=cfg.beam_step,
                radius=0.0,
            )
        return out

    def _mark_sector(self, mask: np.ndarray, direction: np.ndarray) -> None:
        """Set explored=True for every directional sample within half-sector of direction."""
        v = np.asarray(direction, dtype=float)
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            return
        v = v / n
        dots = self._dirs @ v
        mask |= (dots >= self._sector_half_cos)

    # -------------------------------- Algorithm 1: beam-driven exploration --
    def _is_far_enough(self, p: Point3D, tree: Optional[KDTree]) -> bool:
        if tree is None:
            return True
        d, _ = tree.query(np.asarray(p), k=1)
        return float(d) >= self.config.min_spacing

    def _seed_initial(self) -> None:
        """
        Bootstrap exploration with volume-adaptive seed placement.

        Small environments (< 1000 m³): use random seeds (fast, sufficient).
        Large environments: use Halton quasi-random sequence for uniform
        spatial coverage. The number of seeds scales with volume so that
        each seed covers roughly the same region size.

        Physical motivation: in a 30×30×12m building, 20 random seeds
        may cluster on one floor, leaving other floors unreachable by
        beam expansion. Halton seeding guarantees seeds on every floor
        and in every room wing, so beams only need to cover local gaps.
        """
        cfg = self.config
        b = self.env.bounds
        vol = (b.x_max - b.x_min) * (b.y_max - b.y_min) * (b.z_max - b.z_min)

        # Scale: ~1 seed per 100 m³, clamped to [initial_seeds, 200]
        n_target = max(cfg.initial_seeds, min(200, int(vol / 100)))

        if n_target <= 30:
            # Small env: random seeds (original behaviour)
            seeds = self.env.sample_free(n_target, rng=self._rng,
                                         radius=cfg.collision_radius)
        else:
            # Large env: Halton quasi-random for uniform coverage
            seeds = self._halton_seeds(n_target * 3)  # oversample

        tree = None
        for s in seeds:
            if tree is not None:
                d, _ = tree.query(np.asarray(s))
                if float(d) < cfg.min_spacing:
                    continue
            elif self.nodes:
                tree = KDTree(np.asarray(self.nodes))
                d, _ = tree.query(np.asarray(s))
                if float(d) < cfg.min_spacing:
                    continue
            if not self.env.is_free(s, radius=cfg.collision_radius):
                continue
            self.nodes.append(s)
            self._frontier.append(
                _Frontier(explored=np.zeros(self._dirs.shape[0], dtype=bool)))
            tree = KDTree(np.asarray(self.nodes))
            if len(self.nodes) >= n_target:
                break

    @staticmethod
    def _halton_1d(n: int, base: int) -> np.ndarray:
        seq = np.zeros(n)
        for i in range(n):
            f, r, idx = 1.0, 0.0, i + 1
            while idx > 0:
                f /= base
                r += f * (idx % base)
                idx //= base
            seq[i] = r
        return seq

    def _halton_seeds(self, n: int) -> list:
        """Generate Halton quasi-random points in the environment bounds."""
        b = self.env.bounds
        h = np.column_stack([
            self._halton_1d(n, 2),
            self._halton_1d(n, 3),
            self._halton_1d(n, 5),
        ])
        pts = []
        for i in range(n):
            pt = (
                float(b.x_min + h[i, 0] * (b.x_max - b.x_min)),
                float(b.y_min + h[i, 1] * (b.y_max - b.y_min)),
                float(b.z_min + h[i, 2] * (b.z_max - b.z_min)),
            )
            pts.append(pt)
        return pts

    def _discontinuity_candidates(self,
                                   origin: Point3D,
                                   dists: np.ndarray,
                                   disc_abs: float | None = None,
                                   disc_rel: float | None = None,
                                   local_min_spacing: float | None = None,
                                   ) -> List[Tuple[Point3D, np.ndarray]]:
        """Generate candidate seed points from beam-pair discontinuities."""
        cfg = self.config
        ms = local_min_spacing if local_min_spacing is not None else cfg.min_spacing
        d0 = disc_abs if disc_abs is not None else cfg.discontinuity_abs
        k_rel = disc_rel if disc_rel is not None else cfg.discontinuity_rel
        out: List[Tuple[Point3D, np.ndarray]] = []

        I = self._dir_pairs[:, 0]
        J = self._dir_pairs[:, 1]
        di = dists[I]
        dj = dists[J]
        diff = np.abs(di - dj)
        mins = np.minimum(di, dj)
        thresh = np.maximum(d0, k_rel * mins)
        hit = (diff > thresh) & (np.maximum(di, dj) > ms)

        if not hit.any():
            return out

        for i, j in self._dir_pairs[hit]:
            ds = dists[i]; dl = dists[j]
            if dl < ds:
                i, j = j, i
                ds, dl = dl, ds
            v_short = self._dirs[i]
            p_short = (
                origin[0] + cfg.short_beam_fraction * ds * v_short[0],
                origin[1] + cfg.short_beam_fraction * ds * v_short[1],
                origin[2] + cfg.short_beam_fraction * ds * v_short[2],
            )
            out.append((p_short, v_short))

            v_long = self._dirs[j]
            p_long = (
                origin[0] + cfg.long_beam_fraction * dl * v_long[0],
                origin[1] + cfg.long_beam_fraction * dl * v_long[1],
                origin[2] + cfg.long_beam_fraction * dl * v_long[2],
            )
            out.append((p_long, v_long))
        return out

    # ── Geometry-adaptive expansion parameters ────────────────────────
    # ── Multi-resolution beam casting ──────────────────────────────────
    def _refine_long_beams(self, origin: Point3D,
                            base_dists: np.ndarray
                            ) -> List[Tuple[Point3D, np.ndarray]]:
        """
        Adaptive angular refinement for large-scale environments.

        When a beam travels far (> threshold), features at that distance
        subtend small angles that the base fan may miss. This method
        casts extra beams in a tight cone around each long beam to
        detect small openings that the base resolution cannot resolve.

        Physical motivation: a 1.2m opening at 10m subtends 6.9°, below
        the base fan's ~14° spacing. Adding 4 beams in a ±7° cone
        achieves ~3.5° effective resolution — sufficient to detect it.

        Optimisations vs naive per-ray loop:
          - Batched ray casting (all refined directions in one call)
          - Coarser step (2× base) since only discontinuity detection needed
          - Cap on number of long beams refined per node (max 15)
          - Early-out when enough candidates found
        """
        cfg = self.config
        threshold = max(cfg.connection_radius * 2.5, 5.0)
        long_mask = base_dists > threshold
        if not long_mask.any():
            return []

        long_idx = np.where(long_mask)[0]
        # Cap: refine at most 15 longest beams per node
        if len(long_idx) > 15:
            top15 = np.argsort(base_dists[long_idx])[-15:]
            long_idx = long_idx[top15]

        n_refine = 4  # perturbed beams per long beam
        cone_half = 0.12  # ~7° half-angle

        # ── Build all perturbed directions at once ──
        all_dirs = []
        all_base_lens = []
        all_base_idx = []
        for li in long_idx:
            bd = self._dirs[li]
            bl = base_dists[li]
            if abs(bd[2]) < 0.9:
                p1 = np.cross(bd, [0, 0, 1])
            else:
                p1 = np.cross(bd, [1, 0, 0])
            p1 /= max(np.linalg.norm(p1), 1e-12)
            p2 = np.cross(bd, p1)
            p2 /= max(np.linalg.norm(p2), 1e-12)
            for k in range(n_refine):
                a = 2 * np.pi * k / n_refine
                nd = bd + cone_half * (np.cos(a) * p1 + np.sin(a) * p2)
                nd /= max(np.linalg.norm(nd), 1e-12)
                all_dirs.append(nd)
                all_base_lens.append(bl)
                all_base_idx.append(li)

        if not all_dirs:
            return []

        # ── Batch cast with coarser step ──
        dirs_arr = np.array(all_dirs)
        coarse_step = cfg.beam_step * 2  # 2× coarser for speed
        max_len = min(float(np.max(all_base_lens)) * 1.2, cfg.beam_max_length)

        if hasattr(self.env, "cast_beam_fan"):
            refined_dists = self.env.cast_beam_fan(
                origin, dirs_arr, max_length=max_len, step=coarse_step)
        else:
            refined_dists = np.array([
                self.env.first_hit_distance(
                    origin, d, max_length=max_len,
                    step=coarse_step, radius=0.0)
                for d in dirs_arr])

        # ── Check for discontinuities ──
        extra_cands: List[Tuple[Point3D, np.ndarray]] = []
        for i, (rd, bl, bi) in enumerate(zip(
                refined_dists, all_base_lens, all_base_idx)):
            diff = abs(bl - rd)
            mn = min(bl, rd)
            thr = max(cfg.discontinuity_abs, cfg.discontinuity_rel * mn)
            if diff > thr and max(bl, rd) > cfg.min_spacing:
                ds, dl = min(bl, rd), max(bl, rd)
                vs = all_dirs[i] if rd <= bl else self._dirs[bi]
                vl = self._dirs[bi] if rd <= bl else all_dirs[i]
                ps = (origin[0] + cfg.short_beam_fraction * ds * vs[0],
                      origin[1] + cfg.short_beam_fraction * ds * vs[1],
                      origin[2] + cfg.short_beam_fraction * ds * vs[2])
                extra_cands.append((ps, vs))
                pl = (origin[0] + cfg.long_beam_fraction * dl * vl[0],
                      origin[1] + cfg.long_beam_fraction * dl * vl[1],
                      origin[2] + cfg.long_beam_fraction * dl * vl[2])
                extra_cands.append((pl, vl))

        return extra_cands

    def _expand_from(self, idx: int,
                     tree_holder: List[Optional[KDTree]]) -> int:
        """
        Run beam-casting + discontinuity detection + candidate insertion.
        Returns how many new nodes were added.
        """
        cfg = self.config
        origin = self.nodes[idx]
        if idx >= len(self._frontier):
            return 0
        mask = self._frontier[idx].explored

        dists = self._cast_fan(origin)
        self._frontier[idx].clearance = float(np.min(dists)) if dists.size else 0.0

        masked = dists.copy()
        masked[mask] = -1.0

        cands = self._discontinuity_candidates(origin, masked)

        # Adaptive refinement: cast extra beams around long rays
        refine_cands = self._refine_long_beams(origin, dists)
        if refine_cands:
            cands.extend(refine_cands)

        r_min_local = cfg.min_spacing

        if not cands:
            if mask.all():
                return 0
            unexplored_idx = np.where(~mask)[0]
            j = unexplored_idx[np.argmax(dists[unexplored_idx])]
            d = dists[j]
            if d > r_min_local:
                v = self._dirs[j]
                p = (origin[0] + 0.6 * d * v[0],
                     origin[1] + 0.6 * d * v[1],
                     origin[2] + 0.6 * d * v[2])
                cands = [(p, v)]
            else:
                return 0

        # ---- batch validate candidates ----
        pts = np.array([c[0] for c in cands], dtype=float)

        # bounds + free check (batched)
        free_mask = self.env.are_free_batch(pts, radius=cfg.collision_radius) \
            if hasattr(self.env, "are_free_batch") else \
            np.array([self.env.is_free(tuple(p), radius=cfg.collision_radius) for p in pts])

        # Poisson-disk gate with locally-adapted r_min
        spacing_mask = np.ones(pts.shape[0], dtype=bool)
        tree = tree_holder[0]
        if tree is not None and len(self.nodes):
            dists_min, _ = tree.query(pts, k=1)
            spacing_mask = dists_min >= r_min_local

        keep = free_mask & spacing_mask
        if not keep.any():
            self._frontier[idx].explored |= dists > r_min_local
            return 0

        # segment check to origin (batched)
        origins_arr = np.tile(np.asarray(origin, dtype=float), (pts.shape[0], 1))
        seg_mask = np.zeros(pts.shape[0], dtype=bool)
        if keep.any():
            sel_pts = pts[keep]
            sel_orig = origins_arr[keep]
            if hasattr(self.env, "segments_are_free_batch"):
                seg_mask[keep] = self.env.segments_are_free_batch(
                    sel_orig, sel_pts,
                    radius=cfg.collision_radius,
                    step=cfg.collision_step,
                )
            else:
                for i, (o, p) in enumerate(zip(sel_orig, sel_pts)):
                    if self.env.segment_is_free(tuple(o), tuple(p),
                                                radius=cfg.collision_radius,
                                                step=cfg.collision_step):
                        seg_mask[np.where(keep)[0][i]] = True

        accepted = keep & seg_mask

        added = 0
        newly_added_pts: List[np.ndarray] = []
        for k in np.where(accepted)[0]:
            p = tuple(float(x) for x in pts[k])
            v = cands[k][1]
            # dedup vs KDTree snapshot
            if tree_holder[0] is not None:
                dmin, _ = tree_holder[0].query(np.asarray(p), k=1)
                if float(dmin) < r_min_local:
                    continue
            # dedup vs points just added in THIS call (no KDTree refresh needed)
            if newly_added_pts:
                arr_new = np.asarray(newly_added_pts)
                d_new = np.linalg.norm(arr_new - np.asarray(p), axis=1).min()
                if float(d_new) < r_min_local:
                    continue
            self.nodes.append(p)
            newly_added_pts.append(np.asarray(p))
            self._frontier.append(
                _Frontier(explored=np.zeros(self._dirs.shape[0], dtype=bool))
            )
            self._mark_sector(self._frontier[idx].explored, v)
            self._mark_sector(self._frontier[-1].explored, -v)
            added += 1
            if len(self.nodes) >= cfg.max_nodes:
                tree_holder[0] = KDTree(np.asarray(self.nodes))
                return added
        # refresh the KDTree only periodically (every 10 node additions)
        # to avoid O(N log N) rebuild per expansion. The incremental
        # dedup against `newly_added_pts` above handles within-call
        # conflicts, and the spacing_mask check earlier in this method
        # catches most duplicates relative to the stale tree.
        if added > 0:
            # count nodes added since last tree refresh (stored on self)
            pending = getattr(self, "_pending_tree_refresh", 0) + added
            # refresh cadence: 10 nodes, OR when tree is missing
            if pending >= 10 or tree_holder[0] is None:
                tree_holder[0] = KDTree(np.asarray(self.nodes))
                pending = 0
            self._pending_tree_refresh = pending

        self._frontier[idx].explored |= dists > r_min_local
        return added

    # -------------------------------------------------------- edge wiring --
    def _build_knn_adjacency(self, k: int = 12) -> None:
        """
        Build a k-nearest-neighbour adjacency with fast collision
        checks. Used as a cheap pre-sketch topology.

        Cost trade-off: k-NN restricts work to O(N·k) pairs (vs
        O(N·avg_neighbours) for full radius query, which on dense
        pre-sketch clusters is O(N·50+)). With k=12 we do at most
        N·12 segment checks, batched together.
        """
        cfg = self.config
        self.edges = []
        n = len(self.nodes)
        if n < 2:
            self._rebuild_adjacency()
            return
        arr = np.asarray(self.nodes, dtype=float)
        tree = KDTree(arr)
        kk = min(k + 1, n)
        dists, idx = tree.query(arr, k=kk)
        # drop self (column 0)
        dists = dists[:, 1:]
        idx = idx[:, 1:]

        r_cap = cfg.connection_radius
        # Collect unique candidate pairs (i<j) within r_cap
        pair_set = set()
        pairs_list: List[Tuple[int, int]] = []
        for i in range(n):
            for ji_arr, d in zip(idx[i], dists[i]):
                ji = int(ji_arr)
                if d > r_cap:
                    break
                a, b = (i, ji) if i < ji else (ji, i)
                if (a, b) in pair_set:
                    continue
                pair_set.add((a, b))
                pairs_list.append((a, b))

        if not pairs_list:
            self._rebuild_adjacency()
            return

        # Batch segment check — coarser step for pre-sketch topology
        # (final validated edges are built later by _connect_edges)
        pair_arr = np.asarray(pairs_list, dtype=np.int64)
        starts = arr[pair_arr[:, 0]]
        ends = arr[pair_arr[:, 1]]
        coarse_step = cfg.collision_step * 2.5
        if hasattr(self.env, "segments_are_free_batch"):
            ok = self.env.segments_are_free_batch(
                starts, ends,
                radius=cfg.collision_radius,
                step=coarse_step,
            )
        else:
            ok = np.array([
                self.env.segment_is_free(
                    tuple(s), tuple(e),
                    radius=cfg.collision_radius,
                    step=coarse_step)
                for s, e in zip(starts, ends)
            ], dtype=bool)

        self._adj = {i: {} for i in range(n)}
        self.edges = []
        for (a, b), keep in zip(pairs_list, ok):
            if not keep:
                continue
            d = float(np.linalg.norm(arr[a] - arr[b]))
            self._adj[a][b] = d
            self._adj[b][a] = d
            self.edges.append((self.nodes[a], self.nodes[b]))
        self._arr_cache = None
        self._tree_cache = None

    def _connect_edges(self, radius: float | None = None) -> None:
        """
        Wire up the beam-roadmap. All candidate pairs within r_conn are
        collected, filtered by Euclidean distance, then collision-checked
        in one batched call - ~30x faster than per-edge segment_is_free.

        ``radius`` overrides cfg.connection_radius when set; used to
        build a cheap pre-sketch adjacency with a tighter radius, then
        rebuild with the full radius on the sparse post-sketch graph.
        """
        cfg = self.config
        self.edges = []
        if len(self.nodes) < 2:
            self._rebuild_adjacency()
            return
        r_use = float(radius if radius is not None else cfg.connection_radius)
        arr = np.asarray(self.nodes, dtype=float)
        tree = KDTree(arr)

        # Collect candidate index pairs (i<j) within r_conn
        pair_list: List[Tuple[int, int]] = []
        seen = set()
        for i in range(len(self.nodes)):
            neigh = tree.query_ball_point(arr[i], r=r_use)
            for jj in neigh:
                j = int(jj)
                if j == i:
                    continue
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                pair_list.append((a, b))

        if not pair_list:
            self._rebuild_adjacency()
            return

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

    # ----------------------------------- Algorithm 2: structural sketching --
    def _clearance_estimate_batch(self, points: List[Point3D]) -> np.ndarray:
        """Batched clearance estimate for many nodes at once."""
        cfg = self.config
        K = min(cfg.sketch_clearance_probes, self._dirs.shape[0])
        sel = np.linspace(0, self._dirs.shape[0] - 1, K, dtype=int)
        probe_dirs = self._dirs[sel]
        if hasattr(self.env, "clearance_many"):
            return self.env.clearance_many(
                np.asarray(points, dtype=float),
                probe_dirs,
                max_length=cfg.beam_max_length,
                step=cfg.beam_step,
            )
        out = np.zeros(len(points), dtype=float)
        for i, p in enumerate(points):
            dl = self.env.cast_beam_fan(
                p, probe_dirs,
                max_length=cfg.beam_max_length,
                step=cfg.beam_step,
            )
            out[i] = float(np.min(dl)) if dl.size else 0.0
        return out

    def _clearance_estimate(self, point: Point3D) -> float:
        """Minimum free-flight distance over a subset of the direction fan (batched)."""
        cfg = self.config
        K = min(cfg.sketch_clearance_probes, self._dirs.shape[0])
        sel = np.linspace(0, self._dirs.shape[0] - 1, K, dtype=int)
        dirs = self._dirs[sel]
        if hasattr(self.env, "cast_beam_fan"):
            dl = self.env.cast_beam_fan(
                point, dirs,
                max_length=cfg.beam_max_length,
                step=cfg.beam_step,
            )
            if dl.size == 0:
                return 0.0
            return float(np.min(dl))
        mn = math.inf
        for i in range(dirs.shape[0]):
            dl = self.env.first_hit_distance(
                point, dirs[i],
                max_length=cfg.beam_max_length,
                step=cfg.beam_step,
                radius=0.0,
            )
            if dl < mn:
                mn = dl
        return 0.0 if mn is math.inf else float(mn)

    def _graph_neighbors_within(self, src: int, radius: float) -> List[int]:
        """Bounded Dijkstra on the beam-roadmap graph."""
        import heapq
        dist = {src: 0.0}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if d > radius:
                continue
            for v, w in self._adj.get(u, {}).items():
                nd = d + w
                if nd > radius:
                    continue
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return [k for k in dist if k != src]

    def _structural_sketch(self) -> None:
        """
        Algorithm 2: NMS over the clearance field + skeleton reconnection.

        Bridge preservation: a node is protected from suppression only if
        *both* its clearance is small (passage / corner candidate) *and*
        its graph degree is small (no obvious alternative). This keeps the
        sketch aggressive in open areas but gentle at bottlenecks.
        """
        cfg = self.config
        if not self.nodes:
            return

        clearance = self._clearance_estimate_batch(self.nodes)

        # thresholds
        bridge_clearance = max(1.5 * cfg.min_spacing, 0.45 * cfg.sketch_link_radius)
        bridge_max_degree = 5                       # "low-degree" cutoff

        n = len(self.nodes)
        suppressed = np.zeros(n, dtype=bool)
        order = np.argsort(-clearance)
        survivors: List[int] = []

        has_batch = hasattr(self.env, "segments_are_free_batch")
        pts_all = np.asarray(self.nodes, dtype=float)

        # ── Pre-compute coverage attachment map ──
        # For each free-space probe, record which roadmap nodes it can
        # reach. A node is a "coverage guard" if some probe can ONLY
        # reach it — suppressing it creates a void. O(N_probe × k) upfront.
        tree_all = KDTree(pts_all)
        rng_cov = np.random.default_rng(cfg.seed + 8888)
        # Stratified coverage probes for NMS guard
        b = self.env.bounds
        cell_cov = min(1.5, cfg.connection_radius * 0.75)
        nx_c = max(1, int((b.x_max - b.x_min) / cell_cov))
        ny_c = max(1, int((b.y_max - b.y_min) / cell_cov))
        nz_c = max(1, int((b.z_max - b.z_min) / cell_cov))
        while nx_c * ny_c * nz_c > 400 and cell_cov < 3.0:
            cell_cov *= 1.3
            nx_c = max(1, int((b.x_max - b.x_min) / cell_cov))
            ny_c = max(1, int((b.y_max - b.y_min) / cell_cov))
            nz_c = max(1, int((b.z_max - b.z_min) / cell_cov))
        cov_probes: List[Point3D] = []
        for ix in range(nx_c):
            for iy in range(ny_c):
                for iz in range(nz_c):
                    cx = b.x_min + (ix + rng_cov.random()) * cell_cov
                    cy = b.y_min + (iy + rng_cov.random()) * cell_cov
                    cz = b.z_min + (iz + rng_cov.random()) * cell_cov
                    cx = min(cx, b.x_max - 0.01)
                    cy = min(cy, b.y_max - 0.01)
                    cz = min(cz, b.z_max - 0.01)
                    pt = (float(cx), float(cy), float(cz))
                    if self.env.is_free(pt, radius=cfg.collision_radius):
                        cov_probes.append(pt)
        k_cov = min(15, n)
        # attachments[probe_idx] = set of node indices reachable
        attachments: List[set] = []
        for pt in cov_probes:
            _, idxs = tree_all.query(np.asarray(pt), k=k_cov)
            att = set()
            for j in np.atleast_1d(idxs):
                if self.env.segment_is_free(pt, self.nodes[int(j)],
                                             radius=cfg.collision_radius,
                                             step=cfg.collision_step):
                    att.add(int(j))
            attachments.append(att)

        # Reverse index: for each node, which probes attach to it
        node_to_probes: List[List[int]] = [[] for _ in range(n)]
        for pi, att in enumerate(attachments):
            for j in att:
                node_to_probes[j].append(pi)

        def is_coverage_guard(v: int) -> bool:
            """Would suppressing v leave any probe with ≤2 live attachments?
            Threshold ≤3: protect nodes early before all alternatives gone."""
            for pi in node_to_probes[v]:
                live = sum(1 for j in attachments[pi] if not suppressed[j])
                if live <= 3:
                    return True
            return False

        for u in order:
            u = int(u)
            if suppressed[u]:
                continue
            survivors.append(u)

            reach = self._graph_neighbors_within(
                u,
                # Adaptive suppression radius: high-clearance retainers
                # suppress widely (open space = aggressive sparsification),
                # low-clearance retainers suppress narrowly (corridors =
                # preserve local coverage). This prevents corridor nodes
                # from being wiped out by a nearby junction node that
                # happens to have one long clear sightline.
                min(cfg.sketch_visual_radius,
                    max(cfg.min_spacing * 1.5, clearance[u] * 1.8))
            )
            reach = [v for v in reach if not suppressed[v]]
            if not reach:
                continue

            if has_batch:
                ends = pts_all[reach]
                starts = np.tile(pts_all[u], (len(reach), 1))
                vis = self.env.segments_are_free_batch(
                    starts, ends,
                    radius=cfg.collision_radius,
                    step=cfg.collision_step,
                )
            else:
                vis = np.array([
                    self.env.segment_is_free(
                        self.nodes[u], self.nodes[v],
                        radius=cfg.collision_radius,
                        step=cfg.collision_step)
                    for v in reach
                ], dtype=bool)

            for vi, v in enumerate(reach):
                if not vis[vi]:
                    continue
                if suppressed[v]:
                    continue
                # --- bridge protection heuristic: keep low-clearance low-degree ---
                if clearance[v] <= bridge_clearance:
                    deg = sum(1 for w in self._adj.get(v, {}) if not suppressed[w])
                    if deg <= bridge_max_degree:
                        continue
                # --- articulation-aware protection ---
                # Run the full graph articulation check only on nodes with
                # degree in [2, 5]. A node with degree <= 1 is trivially
                # suppressible; a node with degree >= 6 in a dense local
                # graph is almost never a true articulation point, and
                # running the BFS for these would dominate build time.
                deg_v = sum(1 for w in self._adj.get(v, {}) if not suppressed[w])
                if 2 <= deg_v <= 5:
                    if self._is_local_articulation(v, suppressed):
                        continue
                # --- coverage guard protection ---
                if is_coverage_guard(v):
                    continue
                suppressed[v] = True

        # --- 2b) Connectivity verification + un-suppression ----------------
        # NMS is greedy and local — it can disconnect the graph even with
        # articulation-point checks (which use bounded BFS). We verify
        # global connectivity and un-suppress bridge paths if needed.
        #
        # This is NOT a post-hoc patch: it's part of the sketch algorithm.
        # Think of it as "NMS with connectivity certificate."
        # Save original adjacency for bridge preservation
        orig_adj = {k: dict(v) for k, v in self._adj.items()}

        survivors = self._unsuppress_bridges(
            survivors, suppressed, n)

        # --- 3) relink survivors ----------------------------------------
        # Build a mapping from old index to new index
        surv_to_new = {old_i: new_i for new_i, old_i in enumerate(survivors)}
        new_nodes = [self.nodes[i] for i in survivors]
        if len(new_nodes) < 2:
            self.nodes = new_nodes
            self.edges = []
            self._rebuild_adjacency()
            return

        new_points = np.asarray(new_nodes)
        tree = KDTree(new_points)

        # Collect edges from TWO sources:
        # (a) Radius-based: NEW edges between nearby survivors → collision check
        # (b) Original: edges from pre-NMS graph between survivors → SKIP check
        #     (they were already validated; RRTC bridges are multi-hop paths
        #      that pass collision check at creation but aren't straight-line free)
        new_pair_list: List[Tuple[int, int]] = []
        orig_pair_set: set = set()
        new_pair_set: set = set()

        # (b) First: collect original edges (these take priority)
        for old_i in survivors:
            new_i = surv_to_new[old_i]
            for old_nb in orig_adj.get(old_i, {}):
                if old_nb in surv_to_new:
                    new_nb = surv_to_new[old_nb]
                    ia, ib = (new_i, new_nb) if new_i < new_nb else (new_nb, new_i)
                    orig_pair_set.add((ia, ib))

        # (a) Then: radius-based candidates (only if not already an original edge)
        for i in range(len(new_nodes)):
            idx = tree.query_ball_point(new_points[i],
                                        r=max(cfg.sketch_link_radius,
                                              cfg.connection_radius))
            for jj in idx:
                j = int(jj)
                if j == i:
                    continue
                ia, ib = (i, j) if i < j else (j, i)
                if (ia, ib) not in orig_pair_set and (ia, ib) not in new_pair_set:
                    new_pair_set.add((ia, ib))
                    new_pair_list.append((ia, ib))

        # Collision-check only NEW edges
        new_edges: List[Edge3D] = []
        if new_pair_list:
            pair_arr = np.array(new_pair_list, dtype=np.int64)
            starts = new_points[pair_arr[:, 0]]
            ends = new_points[pair_arr[:, 1]]
            if has_batch:
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
            new_edges = [(new_nodes[int(a)], new_nodes[int(b)])
                         for (a, b), keep in zip(new_pair_list, ok) if keep]

        # Add original edges directly (already validated)
        for ia, ib in orig_pair_set:
            new_edges.append((new_nodes[ia], new_nodes[ib]))

        self.nodes = new_nodes
        self.edges = new_edges
        self._rebuild_adjacency()

    def _unsuppress_bridges(self,
                             survivors: List[int],
                             suppressed: np.ndarray,
                             n: int) -> List[int]:
        """
        Verify connectivity of the NMS survivors. If fragmented,
        un-suppress nodes along shortest paths in the ORIGINAL graph
        that bridge disconnected components.

        Returns the updated survivors list.
        """
        # Build a temporary adjacency on survivors
        surv_set = set(survivors)
        surv_adj: dict = {s: {} for s in survivors}
        for s in survivors:
            for nb, w in self._adj.get(s, {}).items():
                if nb in surv_set:
                    surv_adj[s][nb] = w

        # Check connectivity via BFS
        from collections import deque
        visited = set()
        comps: List[List[int]] = []
        for s in survivors:
            if s in visited:
                continue
            comp = []
            q = deque([s])
            visited.add(s)
            while q:
                u = q.popleft()
                comp.append(u)
                for nb in surv_adj.get(u, {}):
                    if nb not in visited:
                        visited.add(nb)
                        q.append(nb)
            comps.append(comp)

        if len(comps) <= 1:
            return survivors

        # Find shortest paths in the ORIGINAL graph between components
        # and un-suppress nodes along those paths
        import heapq
        arr = np.asarray(self.nodes, dtype=float)
        added_to_survivors = set(survivors)

        for _merge_round in range(min(15, len(comps) - 1)):
            if len(comps) <= 1:
                break
            # Find the two closest components using KDTree
            from scipy.spatial import KDTree as KDT
            best_dist = float('inf')
            best_ci, best_cj = 0, 1
            best_pair = (comps[0][0], comps[1][0])
            comp_trees = {}
            for ci in range(len(comps)):
                if len(comps[ci]) > 0:
                    comp_trees[ci] = KDT(arr[comps[ci]])
            for ci in range(len(comps)):
                for cj in range(ci + 1, len(comps)):
                    if ci not in comp_trees or cj not in comp_trees:
                        continue
                    dists_cj, idxs_cj = comp_trees[ci].query(arr[comps[cj]])
                    best_local = int(np.argmin(dists_cj))
                    d = float(dists_cj[best_local])
                    if d < best_dist:
                        best_dist = d
                        best_ci, best_cj = ci, cj
                        best_pair = (comps[ci][int(idxs_cj[best_local])],
                                     comps[cj][best_local])

            # Dijkstra in the ORIGINAL graph from best_pair[0] to best_pair[1]
            start, goal = best_pair
            dist_map = {start: 0.0}
            parent = {}
            heap = [(0.0, start)]
            found = False
            while heap:
                d, u = heapq.heappop(heap)
                if d > dist_map.get(u, float('inf')):
                    continue
                if u == goal:
                    found = True
                    break
                for nb, w in self._adj.get(u, {}).items():
                    nd = d + w
                    if nd < dist_map.get(nb, float('inf')):
                        dist_map[nb] = nd
                        parent[nb] = u
                        heapq.heappush(heap, (nd, nb))

            if not found:
                continue

            # Un-suppress all nodes along the path
            path_nodes = []
            cur = goal
            while cur in parent:
                path_nodes.append(cur)
                cur = parent[cur]
            path_nodes.append(start)

            for node in path_nodes:
                if node not in added_to_survivors:
                    survivors.append(node)
                    added_to_survivors.add(node)
                    suppressed[node] = False

            # Rebuild components
            surv_set = set(survivors)
            surv_adj = {s: {} for s in survivors}
            for s in survivors:
                for nb, w in self._adj.get(s, {}).items():
                    if nb in surv_set:
                        surv_adj[s][nb] = w
            visited = set()
            comps = []
            for s in survivors:
                if s in visited: continue
                comp = []
                q = deque([s]); visited.add(s)
                while q:
                    u = q.popleft(); comp.append(u)
                    for nb in surv_adj.get(u, {}):
                        if nb not in visited:
                            visited.add(nb); q.append(nb)
                comps.append(comp)

        return survivors

    def _is_coverage_guard(self, v: int,
                            suppressed: np.ndarray,
                            arr: np.ndarray,
                            clearance: np.ndarray,
                            n_probes: int = 4) -> bool:
        """
        Check if node v is a coverage guard: its removal would create
        a reachability void (free-space region with no visible roadmap
        node).

        Heuristic: sample a few probe points around v in its clearance
        ball. For each, check if any OTHER unsuppressed node is reachable
        via straight-line segment. If no probe can reach another node,
        v is the sole gateway for that region → preserve it.

        Cost: n_probes × 1 segment check = O(1) per candidate.
        """
        cfg = self.config
        node_v = self.nodes[v]
        # Only check nodes that are in geometrically complex areas
        # (near obstacles). Nodes in open space always have alternatives.
        if clearance[v] > cfg.connection_radius * 0.6:
            return False

        rng = np.random.default_rng(cfg.seed + v * 31)
        # Sample probe points in a shell around v
        radius = min(clearance[v] * 0.8, cfg.connection_radius * 0.5)
        if radius < 0.1:
            return False

        retained = np.where(~suppressed)[0]
        others = [i for i in retained if i != v]
        if not others:
            return True  # v is the last node

        # Build a KDTree of other retained nodes for fast queries
        other_arr = arr[others]
        other_tree = KDTree(other_arr)

        n_unreachable = 0
        for _ in range(n_probes):
            offset = rng.normal(0, radius * 0.5, size=3)
            probe = (float(node_v[0] + offset[0]),
                     float(node_v[1] + offset[1]),
                     float(node_v[2] + offset[2]))
            if not self.env.is_free(probe, radius=cfg.collision_radius):
                continue
            # Can this probe reach any OTHER retained node?
            k_check = min(5, len(others))
            _, idxs = other_tree.query(np.asarray(probe), k=k_check)
            found = False
            for j in np.atleast_1d(idxs):
                nn = self.nodes[others[int(j)]]
                if self.env.segment_is_free(probe, nn,
                                             radius=cfg.collision_radius,
                                             step=cfg.collision_step):
                    found = True
                    break
            if not found:
                n_unreachable += 1

        # If >50% of probes are unreachable from other nodes, v is a guard
        return n_unreachable >= max(1, n_probes // 2)

    def _is_bridge_fast(self, v: int,
                        suppressed: np.ndarray,
                        one_hop: List[set]) -> bool:
        """(unused now, kept for experiments)"""
        neigh = [w for w in self._adj.get(v, {}) if not suppressed[w]]
        if len(neigh) < 2:
            return False
        for i in range(len(neigh)):
            ni = neigh[i]
            for j in range(i + 1, len(neigh)):
                nj = neigh[j]
                if nj in one_hop[ni]:
                    continue
                shared = one_hop[ni] & one_hop[nj]
                if not any(k != v and not suppressed[k] for k in shared):
                    return True
        return False

    def _is_local_articulation(self, v: int,
                                suppressed: np.ndarray,
                                bfs_radius: int = 3,
                                bfs_max_nodes: int = 48) -> bool:
        """
        Test if node ``v`` is a local articulation point: i.e. removing
        it from the current unsuppressed graph would partition ``v``'s
        direct neighbourhood into multiple connected components.

        We only check a bounded BFS region around ``v`` (radius
        ``bfs_radius`` hops, capped at ``bfs_max_nodes`` visited nodes)
        to keep the cost O(k²) per sketch iteration. This approximates
        the global articulation-point test while remaining cheap enough
        to run on every suppression candidate.

        Returns True if removing v would leave its direct neighbours
        unable to reach each other within the local BFS region, i.e.
        v must be kept.
        """
        # direct neighbours of v that aren't already suppressed
        neigh = [int(w) for w in self._adj.get(v, {})
                 if not suppressed[int(w)]]
        if len(neigh) < 2:
            return False  # singleton can be suppressed safely

        # pick the first neighbour as a root; BFS from it, skipping v
        # and all currently-suppressed nodes. Count how many of the
        # other neighbours we reach.
        root = neigh[0]
        targets = set(neigh[1:])
        visited = {root}
        frontier = [(root, 0)]
        reached_targets = 0
        while frontier and len(visited) < bfs_max_nodes:
            u, depth = frontier.pop()
            if u in targets:
                reached_targets += 1
                if reached_targets >= len(targets):
                    return False  # all neighbours still connected
            if depth >= bfs_radius:
                continue
            for w in self._adj.get(u, {}):
                w_int = int(w)
                if w_int == v or suppressed[w_int] or w_int in visited:
                    continue
                visited.add(w_int)
                frontier.append((w_int, depth + 1))
        # at least one neighbour unreachable without v -> v is articulation
        return reached_targets < len(targets)

    def _sample_obstacle_biased(self,
                                 count: int,
                                 tree: Optional[KDTree]) -> List[Point3D]:
        """
        Sample ``count`` new seed points biased toward obstacle surfaces.

        Strategy: identify the current-node with LOWEST clearance (closest
        to a wall), then sample a random offset of ~2-3 × min_spacing in
        a random direction, validate it's free. This concentrates new
        seeds near walls where narrow passages live.

        Mixes 70 % obstacle-biased samples with 30 % uniformly random
        samples so that fully-unexplored open regions still get coverage.

        Cost: O(count * K * beam_probes_per_sample) — typically < 50 ms
        for count=10 since each sample does only a handful of is_free
        checks.
        """
        cfg = self.config
        if not self.nodes:
            # no nodes yet -> fall back to uniform
            return list(self.env.sample_free(count, rng=self._rng,
                                              radius=cfg.collision_radius))

        arr = np.asarray(self.nodes, dtype=float)

        # Compute a cheap clearance proxy for each node: probe 6 axis-
        # aligned rays and take the min free length. This is far cheaper
        # than the full clearance_many call.
        sample_idx = self._rng.choice(
            len(self.nodes),
            size=min(32, len(self.nodes)),
            replace=False,
        )
        axis_probes = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
        ], dtype=float)
        probe_len = cfg.min_spacing * 3.0

        clearances = np.full(len(sample_idx), np.inf)
        for si, ni in enumerate(sample_idx):
            origin = arr[ni]
            min_free = probe_len
            for d in axis_probes:
                # binary search for free length along d
                t = 0.0
                step = cfg.beam_step
                while t < probe_len:
                    p = tuple(origin + d * (t + step))
                    if not self.env.is_free(p, radius=cfg.collision_radius):
                        break
                    t += step
                min_free = min(min_free, t)
            clearances[si] = min_free

        # rank sampled nodes by clearance (low = close to wall)
        order = np.argsort(clearances)
        low_clearance_nodes = sample_idx[order[:max(4, count // 2)]]

        results: List[Point3D] = []
        attempts = 0
        n_biased = int(count * 0.7)
        n_uniform = count - n_biased

        # 70 %: obstacle-biased
        while len(results) < n_biased and attempts < n_biased * 10:
            attempts += 1
            anchor = self.nodes[int(
                self._rng.choice(low_clearance_nodes)
            )]
            d = self._rng.normal(size=3)
            d /= max(np.linalg.norm(d), 1e-12)
            r = float(self._rng.uniform(cfg.min_spacing * 1.5,
                                         cfg.min_spacing * 3.5))
            candidate = (
                float(anchor[0] + d[0] * r),
                float(anchor[1] + d[1] * r),
                float(anchor[2] + d[2] * r),
            )
            if not self.env.is_free(candidate, radius=cfg.collision_radius):
                continue
            # bounds
            b = self.env.bounds
            if not (b.x_min + cfg.collision_radius <= candidate[0]
                    <= b.x_max - cfg.collision_radius
                    and b.y_min + cfg.collision_radius <= candidate[1]
                    <= b.y_max - cfg.collision_radius
                    and b.z_min + cfg.collision_radius <= candidate[2]
                    <= b.z_max - cfg.collision_radius):
                continue
            if tree is not None:
                dmin, _ = tree.query(np.asarray(candidate), k=1)
                if float(dmin) < cfg.min_spacing:
                    continue
            results.append(candidate)

        # 30 %: uniform random (ensures coverage of fully-unexplored areas)
        uniform = list(self.env.sample_free(n_uniform, rng=self._rng,
                                             radius=cfg.collision_radius))
        results.extend(uniform)
        return results

    # ------------------------------------------------------- main pipeline --
    def generate_roadmap(self) -> Tuple[List[Point3D], List[Edge3D]]:
        cfg = self.config
        self.nodes = []
        self.edges = []
        self._frontier = []
        self._pending_tree_refresh = 0
        self._node_index_cache = None

        # Auto-scale parameters to environment size
        b = self.env.bounds
        diag = b.diagonal
        auto_bml = min(diag * 0.4, 20.0)
        if cfg.beam_max_length > auto_bml * 1.5:
            cfg.beam_max_length = auto_bml

        # Adaptive min_spacing: large environments use coarser spacing
        # to avoid generating excessive candidates per expansion
        auto_ms = diag * 0.02
        if auto_ms > cfg.min_spacing:
            cfg.min_spacing = round(auto_ms, 3)

        # Use collision_radius=0 for consistency between beam cast
        # (radius=0 ray) and segment validation (radius=r check).
        # With r>0, beam discovers passages that segment_is_free rejects.
        cfg.collision_radius = 0.0

        self._seed_initial()
        if not self.nodes:
            return self.nodes, self.edges

        self._full_budget = int(cfg.max_nodes * 1.15)
        pass1_budget = cfg.max_nodes

        # ══════════════════════════════════════════════════════════════
        # INTEGRATED PIPELINE: Beam expand + inline edge building
        #   + greedy thinning (connectivity-safe)
        #
        # Key ideas:
        #   1. During expand, parent→child segment is already validated
        #      → record as free edge (zero extra cost)
        #   2. For each new node, connect to k nearest existing nodes
        #      → lateral edges for connectivity (~4 segment checks each)
        #   3. After expansion, greedy spatial thinning collapses dense
        #      clusters while preserving connectivity (edge remapping)
        #   4. No separate knn build or NMS phase needed
        # ══════════════════════════════════════════════════════════════
        tree_holder: List[Optional[KDTree]] = [KDTree(np.asarray(self.nodes))]
        inline_edges: List[Tuple[Point3D, Point3D]] = []

        expansion_order: List[int] = list(range(len(self.nodes)))
        expanded: set = set()
        RANK_EVERY = 10
        iters_since_rank = RANK_EVERY

        iterations = 0
        while (len(self.nodes) < pass1_budget
               and iterations < cfg.max_frontier_expansions):
            iterations += 1

            # Re-rank: prefer nodes far from already-expanded sources
            if iters_since_rank >= RANK_EVERY:
                remaining = [i for i in range(len(self.nodes))
                             if i not in expanded]
                if remaining:
                    arr_all = np.asarray(self.nodes)
                    if expanded:
                        exp_tree = KDTree(arr_all[sorted(expanded)])
                        dist_to_exp, _ = exp_tree.query(arr_all[remaining], k=1)
                    else:
                        dist_to_exp = np.full(len(remaining), 999.0)
                    scores = []
                    for i, ri in enumerate(remaining):
                        fr = self._frontier[ri]
                        unexplored_ratio = 1.0 - fr.explored.mean()
                        if unexplored_ratio < 0.1:
                            scores.append(-1.0)
                            continue
                        d_norm = min(float(dist_to_exp[i]) / max(cfg.beam_max_length, 1.0), 1.0)
                        scores.append(0.5 * d_norm + 0.5 * unexplored_ratio)
                    expansion_order = [r for _, r in sorted(
                        zip(scores, remaining), key=lambda x: -x[0])]
                iters_since_rank = 0

            if not expansion_order:
                injected = self._sample_obstacle_biased(
                    count=cfg.initial_seeds // 2, tree=tree_holder[0])
                progress = False
                for s in injected:
                    if self._is_far_enough(s, tree_holder[0]):
                        self.nodes.append(s)
                        self._frontier.append(
                            _Frontier(explored=np.zeros(self._dirs.shape[0], dtype=bool)))
                        expansion_order.append(len(self.nodes) - 1)
                        tree_holder[0] = KDTree(np.asarray(self.nodes))
                        progress = True
                if not progress:
                    break
                continue

            idx = expansion_order.pop(0)
            if idx in expanded:
                continue
            expanded.add(idx)

            origin = self.nodes[idx]
            n_before = len(self.nodes)
            added = self._expand_from(idx, tree_holder)
            iters_since_rank += 1

            # ── Inline edge building ──
            # Parent→child edges are free (segment already checked)
            tree = tree_holder[0]
            for j in range(n_before, len(self.nodes)):
                node = self.nodes[j]
                inline_edges.append((origin, node))
                # Lateral edges: connect to k nearest existing nodes
                if tree is not None:
                    k_nn = min(8, n_before)
                    if k_nn > 0:
                        dists_nn, idxs_nn = tree.query(np.asarray(node), k=k_nn)
                        dists_nn = np.atleast_1d(dists_nn)
                        idxs_nn = np.atleast_1d(idxs_nn)
                        lat = 0
                        for ki in range(len(idxs_nn)):
                            ni = int(idxs_nn[ki])
                            if ni >= len(self.nodes):
                                continue
                            d = float(dists_nn[ki])
                            if d > cfg.connection_radius or d < 0.01:
                                continue
                            nb = self.nodes[ni]
                            if nb == origin:
                                continue
                            if self.env.segment_is_free(
                                    node, nb, radius=0.0,
                                    step=cfg.collision_step):
                                inline_edges.append((node, nb))
                                lat += 1
                                if lat >= 4:
                                    break

            new_tail = len(self.nodes) - added
            for j in range(new_tail, len(self.nodes)):
                if j not in expanded:
                    expansion_order.append(j)

        # ── Connectivity repair via chain-expand ──
        # Use inline edges as temporary adjacency for component detection
        self.edges = inline_edges
        self._rebuild_adjacency()

        for _conn_round in range(3):
            comps = self._connected_components()
            if len(comps) <= 1:
                break
            if len(self.nodes) >= self._full_budget:
                break
            main_comp = set(max(comps, key=len))
            chain_start = len(self.nodes)
            for comp in comps:
                if set(comp) == main_comp:
                    continue
                candidates = []
                for ci in comp:
                    if ci < len(self._frontier) and ci not in expanded:
                        unexpl = 1.0 - self._frontier[ci].explored.mean()
                        if unexpl > 0.5:
                            candidates.append((-unexpl, ci))
                candidates.sort()
                for _, ci in candidates[:3]:
                    if len(self.nodes) >= self._full_budget:
                        break
                    origin = self.nodes[ci]
                    n_before = len(self.nodes)
                    self._expand_from(ci, tree_holder)
                    expanded.add(ci)
                    tree = tree_holder[0]
                    for j in range(n_before, len(self.nodes)):
                        node = self.nodes[j]
                        inline_edges.append((origin, node))
                        if tree is not None:
                            k_nn = min(8, n_before)
                            if k_nn > 0:
                                dists_nn, idxs_nn = tree.query(np.asarray(node), k=k_nn)
                                dists_nn = np.atleast_1d(dists_nn)
                                idxs_nn = np.atleast_1d(idxs_nn)
                                lat = 0
                                for ki in range(len(idxs_nn)):
                                    ni = int(idxs_nn[ki])
                                    if ni >= len(self.nodes): continue
                                    d = float(dists_nn[ki])
                                    if d > cfg.connection_radius or d < 0.01: continue
                                    nb = self.nodes[ni]
                                    if nb == origin: continue
                                    if self.env.segment_is_free(node, nb, radius=0.0, step=cfg.collision_step):
                                        inline_edges.append((node, nb))
                                        lat += 1
                                        if lat >= 4: break
            if len(self.nodes) == chain_start:
                break
            self.edges = inline_edges
            self._rebuild_adjacency()

        # ── Greedy thinning (connectivity-safe) ──
        # Collapse dense clusters by suppressing nodes that have
        # >=2 kept neighbors (guaranteed not to be bridges)
        self.edges = inline_edges
        self._rebuild_adjacency()
        self._greedy_thin(cfg.min_spacing * 3.0)

        return self.nodes, self.edges

    def _greedy_thin(self, thin_radius: float) -> None:
        """
        Connectivity-safe greedy thinning.

        Iterates through nodes; for each kept node, suppresses nearby
        nodes that have >=2 other kept neighbors (i.e. not bridge nodes).
        Suppressed nodes' edges are remapped to their suppressor.
        """
        n = len(self.nodes)
        if n < 2:
            return
        arr = np.asarray(self.nodes, dtype=float)
        tree = KDTree(arr)

        keep = np.ones(n, dtype=bool)
        remap = np.arange(n)

        for i in range(n):
            if not keep[i]:
                continue
            for j in tree.query_ball_point(arr[i], r=thin_radius):
                if j == i or not keep[j]:
                    continue
                kept_nbs = [k for k in self._adj.get(j, {})
                            if keep[k] and k != j]
                if len(kept_nbs) >= 2:
                    keep[j] = False
                    remap[j] = i

        kept_idx = np.where(keep)[0]
        if len(kept_idx) == n:
            return

        old_to_new = np.full(n, -1, dtype=int)
        for ni, oi in enumerate(kept_idx):
            old_to_new[oi] = ni
        for i in range(n):
            if not keep[i]:
                old_to_new[i] = old_to_new[remap[i]]

        new_nodes = [self.nodes[i] for i in kept_idx]
        new_adj: Dict[int, Dict[int, float]] = {ni: {} for ni in range(len(kept_idx))}

        for old_i in range(n):
            ni = old_to_new[old_i]
            if ni < 0:
                continue
            for old_j, w in self._adj.get(old_i, {}).items():
                nj = old_to_new[old_j]
                if nj < 0 or ni == nj:
                    continue
                if nj not in new_adj[ni] or w < new_adj[ni][nj]:
                    new_adj[ni][nj] = w
                    new_adj[nj][ni] = w

        new_edges = []
        seen: set = set()
        for i, nbs in new_adj.items():
            for j in nbs:
                key = (min(i, j), max(i, j))
                if key not in seen:
                    seen.add(key)
                    new_edges.append((new_nodes[i], new_nodes[j]))

        self.nodes = new_nodes
        self.edges = new_edges
        self._adj = new_adj

    def _wire_new_nodes(self, start_idx: int) -> None:
        """Wire only nodes from start_idx onwards into existing adj/edges."""
        cfg = self.config
        n = len(self.nodes)
        if start_idx >= n:
            return
        arr = np.asarray(self.nodes, dtype=float)
        tree = KDTree(arr)
        has_batch = hasattr(self.env, "segments_are_free_batch")
        new_pairs = []
        pair_set = set()
        for i in range(start_idx, n):
            idxs = tree.query_ball_point(arr[i], r=cfg.connection_radius)
            for j in idxs:
                if j == i:
                    continue
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in pair_set:
                    continue
                pair_set.add((a, b))
                new_pairs.append((a, b))
        if not new_pairs:
            return
        pa = np.array(new_pairs, dtype=np.int64)
        starts = arr[pa[:, 0]]
        ends = arr[pa[:, 1]]
        if has_batch:
            ok = self.env.segments_are_free_batch(
                starts, ends, radius=cfg.collision_radius,
                step=cfg.collision_step)
        else:
            ok = np.array([self.env.segment_is_free(
                tuple(s), tuple(e), radius=cfg.collision_radius,
                step=cfg.collision_step) for s, e in zip(starts, ends)],
                dtype=bool)
        for (a, b), keep in zip(new_pairs, ok):
            if not keep:
                continue
            d = float(np.linalg.norm(arr[a] - arr[b]))
            self._adj.setdefault(a, {})[b] = d
            self._adj.setdefault(b, {})[a] = d
            self.edges.append((self.nodes[a], self.nodes[b]))

    def _seed_and_expand_gaps(self, comps: List[List[int]],
                               tree_holder: List) -> None:
        """
        Find vertical passages (stairwells) by scanning XY grid for
        positions with long continuous vertical free-space. Seed beam
        expansion at each passage to discover inter-floor connections.
        """
        cfg = self.config
        b = self.env.bounds

        # Scan XY grid: at each position, find max continuous vertical span
        step = max(1.0, min(2.0, cfg.connection_radius * 0.5))
        floor_height = (b.z_max - b.z_min) / 3  # rough estimate
        min_span = floor_height * 1.2  # passage must span >1 floor

        passages: List[Point3D] = []
        for x in np.arange(b.x_min + step/2, b.x_max, step):
            for y in np.arange(b.y_min + step/2, b.y_max, step):
                # Walk vertically, find longest continuous free segment
                z = b.z_min + 0.5
                best_start = z
                best_len = 0.0
                run_start = None
                while z <= b.z_max:
                    if self.env.is_free((float(x), float(y), float(z)),
                                        radius=cfg.collision_radius):
                        if run_start is None:
                            run_start = z
                    else:
                        if run_start is not None:
                            run_len = z - run_start
                            if run_len > best_len:
                                best_len = run_len
                                best_start = run_start
                        run_start = None
                    z += 0.3
                # Final run
                if run_start is not None:
                    run_len = z - run_start
                    if run_len > best_len:
                        best_len = run_len
                        best_start = run_start

                if best_len >= min_span:
                    # This is a passage — seed at several z-levels
                    for zz in np.arange(best_start + 1.0, best_start + best_len - 0.5, floor_height * 0.5):
                        passages.append((float(x), float(y), float(zz)))

        if not passages:
            return

        # Inject seeds at passages
        tree = tree_holder[0]
        added = 0
        for seed in passages:
            d_nn, _ = tree.query(np.asarray(seed))
            if float(d_nn) < cfg.min_spacing:
                continue
            self.nodes.append(seed)
            self._frontier.append(_Frontier(
                explored=np.zeros(self._dirs.shape[0], dtype=bool)))
            added += 1
            tree = KDTree(np.asarray(self.nodes))
            if added >= 12:
                break

        if added > 0:
            tree_holder[0] = tree
            n0 = len(self.nodes) - added
            for idx in range(n0, len(self.nodes)):
                if len(self.nodes) >= self._full_budget:
                    break
                self._expand_from(idx, tree_holder)

    def _expand_between_components(self, comps: List[List[int]],
                                     tree_holder: List) -> None:
        """Beam-expand from boundary nodes of nearest component pairs."""
        cfg = self.config
        arr = np.asarray(self.nodes, dtype=float)
        from scipy.spatial import KDTree as KDT

        # Find two closest components
        best_dist = float('inf')
        best_ci, best_cj = 0, 1
        comp_trees = {ci: KDT(arr[comp]) for ci, comp in enumerate(comps)}
        for ci in range(len(comps)):
            for cj in range(ci + 1, len(comps)):
                dists, _ = comp_trees[ci].query(arr[comps[cj]])
                d = float(np.min(dists))
                if d < best_dist:
                    best_dist = d
                    best_ci, best_cj = ci, cj

        # Beam-expand from 3 closest boundary nodes of each component
        for ci_side in [best_ci, best_cj]:
            ci_other = best_cj if ci_side == best_ci else best_ci
            dists, _ = comp_trees[ci_other].query(arr[comps[ci_side]])
            top_k = min(3, len(comps[ci_side]))
            boundary = [comps[ci_side][int(k)] for k in np.argsort(dists)[:top_k]]
            for idx in boundary:
                if len(self.nodes) >= cfg.max_nodes:
                    break
                self._expand_from(idx, tree_holder)
        tree_holder[0] = KDTree(np.asarray(self.nodes))

    def _fill_coverage_voids(self, tree_holder: List) -> None:
        """
        Part of Phase 1: detect free-space regions that cannot attach
        to any roadmap node and expand toward them.

        Uses mini-RRT from nearest roadmap node toward each void,
        then beam-expands from discovered stepping stones.
        """
        cfg = self.config
        if len(self.nodes) < 2:
            return

        arr = np.asarray(self.nodes, dtype=float)
        tree = KDTree(arr)

        b = self.env.bounds
        rng = np.random.default_rng(cfg.seed + 5555 + len(self.nodes))

        # Stratified sampling: divide volume into cells, one probe per cell.
        # Guarantees every region gets at least one probe.
        cell_size = min(1.0, cfg.connection_radius * 0.5)
        nx = max(1, int((b.x_max - b.x_min) / cell_size))
        ny = max(1, int((b.y_max - b.y_min) / cell_size))
        nz = max(1, int((b.z_max - b.z_min) / cell_size))
        # Cap total cells to keep cost manageable
        while nx * ny * nz > 600 and cell_size < 3.0:
            cell_size *= 1.3
            nx = max(1, int((b.x_max - b.x_min) / cell_size))
            ny = max(1, int((b.y_max - b.y_min) / cell_size))
            nz = max(1, int((b.z_max - b.z_min) / cell_size))
        probes: List[Point3D] = []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    cx = b.x_min + (ix + rng.random()) * cell_size
                    cy = b.y_min + (iy + rng.random()) * cell_size
                    cz = b.z_min + (iz + rng.random()) * cell_size
                    cx = min(cx, b.x_max - 0.01)
                    cy = min(cy, b.y_max - 0.01)
                    cz = min(cz, b.z_max - 0.01)
                    pt = (float(cx), float(cy), float(cz))
                    if self.env.is_free(pt, radius=cfg.collision_radius):
                        probes.append(pt)

        k_check = min(20, len(self.nodes))

        # Detect voids
        void_with_nearest: List[Tuple[Point3D, int, float]] = []
        for pt in probes:
            _, idxs = tree.query(np.asarray(pt), k=k_check)
            attached = any(
                self.env.segment_is_free(pt, self.nodes[int(j)],
                                         radius=cfg.collision_radius,
                                         step=cfg.collision_step)
                for j in np.atleast_1d(idxs))
            if not attached:
                nearest_d, nearest_i = tree.query(np.asarray(pt))
                void_with_nearest.append((pt, int(nearest_i), float(nearest_d)))

        if not void_with_nearest:
            return  # full coverage

        # Fill voids: mini-RRT from nearest node toward void + beam expand
        void_with_nearest.sort(key=lambda x: -x[2])  # most isolated first
        budget = min(len(void_with_nearest),
                     max(10, int(cfg.max_nodes * 0.10)),
                     max(0, self._full_budget - len(self.nodes)))
        if budget <= 0:
            return

        added = 0
        for void_pt, nearest_idx, _ in void_with_nearest[:budget]:
            origin = self.nodes[nearest_idx]
            new_pts = self._mini_rrt_toward(origin, void_pt, rng,
                                             max_iters=150, step=0.3)
            for p in new_pts[:6]:
                d_nn, _ = tree.query(np.asarray(p))
                if float(d_nn) < cfg.min_spacing * 0.4:
                    continue
                self.nodes.append(p)
                self._frontier.append(_Frontier(
                    explored=np.zeros(self._dirs.shape[0], dtype=bool)))
                added += 1
                tree = KDTree(np.asarray(self.nodes))

        if added > 0:
            # Beam-expand from stepping stones
            n0 = len(self.nodes) - added
            tree_holder[0] = tree
            for idx in range(n0, len(self.nodes)):
                if len(self.nodes) >= self._full_budget:
                    break
                self._expand_from(idx, tree_holder)
            # Wire new nodes into existing graph (append, don't rebuild)
            self._wire_new_nodes(n0)

    def _mini_rrt_toward(self, origin: Point3D, target: Point3D,
                          rng: np.random.Generator,
                          max_iters: int = 80,
                          step: float = 0.4) -> List[Point3D]:
        """
        Grow a small RRT from origin biased toward target.
        Returns list of new free-space nodes discovered.
        """
        cfg = self.config
        b = self.env.bounds
        tree_pts: List[Point3D] = [origin]
        tree_par: List[int] = [-1]

        target_arr = np.asarray(target)

        for it in range(max_iters):
            # 50% bias toward target, 50% random
            if rng.random() < 0.5:
                q = tuple(float(x) for x in target_arr)
            else:
                q = (float(rng.uniform(b.x_min, b.x_max)),
                     float(rng.uniform(b.y_min, b.y_max)),
                     float(rng.uniform(b.z_min, b.z_max)))

            pts_arr = np.asarray(tree_pts)
            dists = np.linalg.norm(pts_arr - np.asarray(q), axis=1)
            i_near = int(np.argmin(dists))
            near = tree_pts[i_near]

            nv = np.asarray(q) - np.asarray(near)
            dn = float(np.linalg.norm(nv))
            if dn < 1e-9:
                continue
            dirv = nv / dn
            new_p = tuple(float(x) for x in
                          (np.asarray(near) + dirv * min(step, dn)))

            if not self.env.is_free(new_p, radius=cfg.collision_radius):
                continue
            if not self.env.segment_is_free(near, new_p,
                                             radius=cfg.collision_radius,
                                             step=cfg.collision_step):
                continue
            tree_pts.append(new_p)
            tree_par.append(i_near)

            # Check if we can now see the target
            if self.env.segment_is_free(new_p, target,
                                         radius=cfg.collision_radius,
                                         step=cfg.collision_step):
                tree_pts.append(target)
                break

        return tree_pts[1:]  # exclude origin (already in roadmap)

    def _ensure_pre_sketch_connectivity(self,
                                          tree_holder: List) -> None:
        """
        Ensure the pre-sketch roadmap is connected by beam-driven
        inter-component expansion.

        Root cause: in complex mazes, beam expansion can produce a
        fragmented graph (8+ components). NMS can only prune, not add,
        so sketching a disconnected graph guarantees a disconnected result.

        Strategy: for each pair of nearest components, cast beams from
        boundary nodes toward the other component. Discovered nodes form
        stepping stones that close the gap. This is the same beam-driven
        philosophy as the initial exploration — not a random-sample patch.
        """
        cfg = self.config
        for _round in range(3):  # max 3 rounds
            comps = self._connected_components()
            if len(comps) <= 1:
                return

            arr = np.asarray(self.nodes, dtype=float)
            n_before = len(self.nodes)

            # Find the two closest components and expand between them
            from scipy.spatial import KDTree as KDT
            best_dist = float('inf')
            best_pair = (0, 1)
            comp_trees = {}
            for ci in range(len(comps)):
                comp_trees[ci] = KDT(arr[comps[ci]])

            for ci in range(len(comps)):
                for cj in range(ci + 1, len(comps)):
                    pts_j = arr[comps[cj]]
                    dists, _ = comp_trees[ci].query(pts_j)
                    d_min = float(np.min(dists))
                    if d_min < best_dist:
                        best_dist = d_min
                        best_pair = (ci, cj)

            ci, cj = best_pair
            pts_i = arr[comps[ci]]
            pts_j = arr[comps[cj]]

            # Find boundary nodes (closest to the other component)
            dists_i, _ = comp_trees[cj].query(pts_i)
            dists_j, _ = comp_trees[ci].query(pts_j)

            # Top-3 closest boundary nodes from each side
            n_boundary = min(3, len(comps[ci]), len(comps[cj]))
            boundary_i = [comps[ci][int(k)] for k in np.argsort(dists_i)[:n_boundary]]
            boundary_j = [comps[cj][int(k)] for k in np.argsort(dists_j)[:n_boundary]]

            # Beam-expand from boundary nodes
            added = 0
            for idx in boundary_i + boundary_j:
                if len(self.nodes) >= cfg.max_nodes:
                    break
                added += self._expand_from(idx, tree_holder)

            if added == 0:
                # If beams didn't help, inject midpoint seeds
                for bi in boundary_i[:1]:
                    for bj in boundary_j[:1]:
                        mid = tuple(float((arr[bi][d] + arr[bj][d]) / 2) for d in range(3))
                        if self.env.is_free(mid, radius=cfg.collision_radius):
                            self.nodes.append(mid)
                            self._frontier.append(_Frontier(
                                explored=np.zeros(self._dirs.shape[0], dtype=bool)))
                            added += 1

            if added > 0:
                tree_holder[0] = KDTree(np.asarray(self.nodes))
                # Rebuild adjacency with new nodes
                adaptive_k = max(30, min(80, int(len(self.nodes) * 0.02)))
                self._build_knn_adjacency(k=adaptive_k)
            else:
                break  # no progress

    def _gap_aware_pass2(self) -> None:
        """
        Gap-aware densification pass.

        Two strategies:
        1. **Targeted inter-component bridging**: When multiple components
           exist, sample free-space points in the neighbourhood of the
           closest pair between components. This directly addresses the
           gap rather than relying on random probes.
        2. **Random gap probing**: Identifies under-sampled regions near
           obstacles by casting beam probes at random free-space points.
        """
        cfg = self.config
        if len(self.nodes) < 3:
            return
        tree = KDTree(np.asarray(self.nodes))
        rng = np.random.default_rng(cfg.seed + 99 + len(self.nodes))

        # ── Strategy 1: targeted inter-component bridging ──
        comps = self._connected_components()
        if len(comps) >= 2:
            arr = np.asarray(self.nodes, dtype=float)
            budget_bridge = min(
                self._full_budget - len(self.nodes),
                int(cfg.max_nodes * 0.05))
            if budget_bridge > 0:
                injected_bridge = 0
                # For each pair of components, find closest boundary nodes
                # and sample free points in a sphere around the gap
                for ci in range(len(comps)):
                    for cj in range(ci + 1, len(comps)):
                        if injected_bridge >= budget_bridge:
                            break
                        pts_i = arr[comps[ci]]
                        pts_j = arr[comps[cj]]
                        tree_i = KDTree(pts_i)
                        # Find closest point in comp_i to any point in comp_j
                        dists_j, idxs_j = tree_i.query(pts_j)
                        best_local = int(np.argmin(dists_j))
                        p_j = pts_j[best_local]
                        p_i = pts_i[int(idxs_j[best_local])]
                        mid = (p_i + p_j) / 2.0
                        gap_dist = float(np.linalg.norm(p_i - p_j))
                        # Sample free points in a sphere around the midpoint
                        radius = max(gap_dist, cfg.connection_radius)
                        n_try = max(30, int(radius * 20))
                        for _ in range(n_try):
                            offset = rng.normal(0, radius * 0.4, size=3)
                            candidate = tuple(float(x) for x in mid + offset)
                            if not self.env.is_free(candidate,
                                                     radius=cfg.collision_radius):
                                continue
                            d_nn, _ = tree.query(np.asarray(candidate))
                            if float(d_nn) < cfg.min_spacing * 0.5:
                                continue
                            self.nodes.append(candidate)
                            self._frontier.append(_Frontier(
                                explored=np.zeros(self._dirs.shape[0],
                                                  dtype=bool)))
                            injected_bridge += 1
                            tree = KDTree(np.asarray(self.nodes))
                            if injected_bridge >= budget_bridge:
                                break
                if injected_bridge > 0:
                    # Expand from bridging seeds
                    n0 = len(self.nodes) - injected_bridge
                    th = [KDTree(np.asarray(self.nodes))]
                    for idx in range(n0, len(self.nodes)):
                        if len(self.nodes) >= self._full_budget:
                            break
                        self._expand_from(idx, th)

        # ── Strategy 2: random gap probing ──

        # Scale probe count with environment volume
        vol = (self.env.bounds.x_max - self.env.bounds.x_min) * \
              (self.env.bounds.y_max - self.env.bounds.y_min) * \
              (self.env.bounds.z_max - self.env.bounds.z_min)
        n_probes = max(150, min(500, int(vol / 20)))

        gap_candidates = []
        probes = list(self.env.sample_free(n_probes, rng=rng,
                                           radius=cfg.collision_radius))
        for pt in probes:
            dist_nn, _ = tree.query(np.asarray(pt))
            if float(dist_nn) < cfg.connection_radius * 0.5:
                continue
            dists = self._cast_fan(pt)
            active = dists[dists > 0]
            if len(active) < 3:
                continue
            if float(np.min(active)) < 1.5:
                gap_candidates.append((-float(dist_nn), pt))

        # Inject top gaps
        gap_candidates.sort()
        budget2 = self._full_budget - len(self.nodes)
        if budget2 <= 0:
            return
        budget2 = min(budget2, int(cfg.max_nodes * 0.15))
        injected = 0
        for _, pt in gap_candidates[:budget2 * 3]:
            d, _ = tree.query(np.asarray(pt))
            if float(d) >= cfg.min_spacing * 0.7:
                self.nodes.append(pt)
                self._frontier.append(_Frontier(
                    explored=np.zeros(self._dirs.shape[0], dtype=bool)))
                injected += 1
                tree = KDTree(np.asarray(self.nodes))
                if injected >= budget2:
                    break

        if injected == 0:
            return

        # Expand from injected seeds (with adaptive beam refinement)
        n0 = len(self.nodes) - injected
        th = [KDTree(np.asarray(self.nodes))]
        for idx in range(n0, len(self.nodes)):
            if len(self.nodes) >= self._full_budget:
                break
            self._expand_from(idx, th)

        # Connect new nodes (no sketch — that's Phase 2's job)
        self._connect_edges(radius=cfg.connection_radius)
        self._bridge_components()

    def _connected_components(self) -> List[List[int]]:
        """BFS partition of the current roadmap's nodes into components."""
        from collections import deque
        n = len(self.nodes)
        visited = [False] * n
        comps: List[List[int]] = []
        for s in range(n):
            if visited[s]:
                continue
            comp = []
            q = deque([s])
            visited[s] = True
            while q:
                u = q.popleft()
                comp.append(u)
                for v in self._adj.get(u, {}):
                    if v < n and not visited[v]:
                        visited[v] = True
                        q.append(v)
            comps.append(comp)
        return comps

    def _bridge_components(self) -> None:
        """
        Try to merge disconnected components after sketching.

        Strategy — iterative pairwise bridging with union-find.

        Every round, we rebuild connected components, pick the two
        largest components that are geometrically closest, and attempt:

          1. **Direct bridge**: shortest collision-free edge between their
             boundary nodes (within ``sketch_link_radius * 1.5``).

          2. **One-hop bridge**: sample free waypoints near the closest
             pair; if a waypoint sees nodes in both components with
             collision-free segments, add it and two edges.

        We loop until no further bridging is possible. This is much more
        effective than the previous "small-to-main only" strategy in
        maze-like environments where many medium-sized components exist.
        """
        cfg = self.config
        max_bridge = cfg.sketch_link_radius * 1.5
        from scipy.spatial import KDTree

        added_edges: List[Edge3D] = []
        added_nodes: List[Point3D] = []
        max_rounds = 8

        def _attempt_bridge(comp_a: List[int], comp_b: List[int]) -> bool:
            """Try direct and indirect bridging between two components."""
            arr = np.asarray(self.nodes, dtype=float)
            a_pts = arr[np.asarray(comp_a)]
            b_pts = arr[np.asarray(comp_b)]
            a_tree = KDTree(a_pts)

            # Direct: check all pairs within max_bridge
            K = min(8, len(comp_b))
            dists, nn_idx = a_tree.query(b_pts, k=K)
            dists = np.atleast_2d(dists)
            nn_idx = np.atleast_2d(nn_idx)
            # sort all candidate edges globally by distance
            flat = []
            for bi in range(len(comp_b)):
                for ki in range(dists.shape[1]):
                    d = float(dists[bi, ki])
                    if d > max_bridge:
                        continue
                    flat.append((d, bi, int(nn_idx[bi, ki])))
            flat.sort()
            for d, bi, ai_local in flat:
                a_node_i = comp_a[ai_local]
                b_node_i = comp_b[bi]
                if self.env.segment_is_free(
                        self.nodes[a_node_i], self.nodes[b_node_i],
                        radius=cfg.collision_radius,
                        step=cfg.collision_step):
                    added_edges.append((self.nodes[a_node_i],
                                        self.nodes[b_node_i]))
                    return True

            # Indirect: try 3 closest pairs with 40 waypoint offsets each
            if len(flat) == 0:
                return False
            pair_list = sorted(flat[:12])[:3] if flat else []
            # fallback: always try closest centroid-to-centroid pair even
            # if it's > max_bridge
            if not pair_list:
                a_centroid = a_pts.mean(axis=0)
                b_centroid = b_pts.mean(axis=0)
                ai = int(np.argmin(np.linalg.norm(a_pts - b_centroid, axis=1)))
                bi = int(np.argmin(np.linalg.norm(b_pts - a_centroid, axis=1)))
                pair_list = [(float(np.linalg.norm(a_pts[ai] - b_pts[bi])), bi, ai)]

            rng = self._rng
            for _, bi, ai_local in pair_list:
                a_pt = arr[comp_a[ai_local]]
                b_pt = arr[comp_b[bi]]
                mid = 0.5 * (a_pt + b_pt)
                span = float(np.linalg.norm(a_pt - b_pt))
                # generate waypoints
                waypoints = [mid]
                for _ in range(40):
                    u = rng.normal(size=3)
                    u /= (np.linalg.norm(u) + 1e-12)
                    r = rng.uniform(0.0, max(span * 0.8, max_bridge * 0.5))
                    waypoints.append(mid + u * r)
                for wp_arr in waypoints:
                    wp = tuple(float(x) for x in wp_arr)
                    if not self.env.is_free(wp, radius=cfg.collision_radius):
                        continue
                    a_d = np.linalg.norm(a_pts - wp_arr, axis=1)
                    b_d = np.linalg.norm(b_pts - wp_arr, axis=1)
                    a_order = np.argsort(a_d)[:6]
                    b_order = np.argsort(b_d)[:6]
                    chosen_a = None; chosen_b = None
                    for ai_ord in a_order:
                        if a_d[ai_ord] > max_bridge * 1.2:
                            break
                        if self.env.segment_is_free(
                                self.nodes[comp_a[int(ai_ord)]], wp,
                                radius=cfg.collision_radius,
                                step=cfg.collision_step):
                            chosen_a = comp_a[int(ai_ord)]
                            break
                    if chosen_a is None:
                        continue
                    for bi_ord in b_order:
                        if b_d[bi_ord] > max_bridge * 1.2:
                            break
                        if self.env.segment_is_free(
                                wp, self.nodes[comp_b[int(bi_ord)]],
                                radius=cfg.collision_radius,
                                step=cfg.collision_step):
                            chosen_b = comp_b[int(bi_ord)]
                            break
                    if chosen_a is not None and chosen_b is not None:
                        added_nodes.append(wp)
                        added_edges.append((self.nodes[chosen_a], wp))
                        added_edges.append((wp, self.nodes[chosen_b]))
                        return True
            return False

        for _round in range(max_rounds):
            comps = self._connected_components()
            if len(comps) <= 1:
                break
            comps.sort(key=len, reverse=True)

            # For this round, try to bridge every non-largest component to
            # any larger-or-equal one (prioritising the largest first).
            # This lets medium-sized components merge sequentially.
            progress = False
            for i, comp_small in enumerate(comps[1:], start=1):
                # try against each larger component in order
                for comp_large in comps[:i]:
                    if _attempt_bridge(comp_large, comp_small):
                        progress = True
                        # apply changes and restart round
                        if added_nodes:
                            self.nodes.extend(added_nodes)
                            added_nodes.clear()
                        if added_edges:
                            # incremental update avoids O(E) rebuild
                            self._incrementally_add_edges(added_edges)
                            added_edges.clear()
                        break
                if progress:
                    break
            if not progress:
                break

        # --- Heavy-artillery bridging: RRT-Connect -------------------
        # After exhausting direct + 1-hop bridges, any remaining component
        # splits are usually rooms/corridors that require a multi-hop
        # path through a non-trivial doorway or stairwell. We use
        # bounded RRT-Connect between boundary nodes of unmerged
        # components. Each successful path is injected as a chain of
        # waypoint nodes + edges. Only runs on components with >= 10
        # nodes to avoid wasting cycles on noise.
        self._rrtc_bridge_components()

        # Final cleanup: keep all sufficiently-large components.
        # We no longer throw away nodes aggressively; only true outliers
        # (singletons and components < 5 nodes) are dropped, because
        # attachment to a 1-2 node island almost always fails.
        comps = self._connected_components()
        if len(comps) > 1:
            comps.sort(key=len, reverse=True)
            # Keep every component with >= max(5, 5% of total) nodes.
            threshold = max(5, int(0.05 * len(self.nodes)))
            keep = set()
            for c in comps:
                if len(c) >= threshold:
                    keep.update(c)
            if len(keep) < len(self.nodes):
                index_map = {old: new for new, old in enumerate(sorted(keep))}
                new_nodes = [self.nodes[i] for i in sorted(keep)]
                old_set = {self.nodes[i] for i in keep}
                new_edges = [(a, b) for (a, b) in self.edges
                             if a in old_set and b in old_set]
                self.nodes = new_nodes
                self.edges = new_edges
                self._rebuild_adjacency()

    def _rrtc_bridge_components(self,
                                 max_rrtc_iters: int = 3000,
                                 max_pairs_per_round: int = 6,
                                 min_comp_size: int = 3) -> None:
        """
        Heavy-duty component merger: for components that couldn't be
        bridged by direct edge or 1-hop waypoint, try bidirectional
        RRT-Connect between a sample of boundary nodes.

        Only runs on components with >= ``min_comp_size`` nodes (skipping
        tiny outlier singletons). Budget-bounded: at most
        ``max_pairs_per_round`` (component_a, component_b) pairs are
        attempted per round. Each attempt runs a single
        ``max_rrtc_iters``-iteration bi-directional RRT-Connect.
        Successful path's intermediate waypoints are injected as
        roadmap nodes.

        Typical cost: 50-500 ms per successful bridge on maze_hard.
        Runs only once at build time, so the amortised query cost is
        zero. Critical for maze_hard where rooms on different floors
        cannot be joined by simple 1-hop waypoints.
        """
        cfg = self.config
        step_rrtc = max(cfg.collision_step * 3.0, 0.4)
        b = self.env.bounds

        def _rrtc(p_from: Point3D, p_to: Point3D,
                   max_iters: int) -> List[Point3D] | None:
            """Bidirectional RRT-Connect between two fixed points."""
            Ta: List[Point3D] = [p_from]; Ta_par: List[int] = [-1]
            Tb: List[Point3D] = [p_to];   Tb_par: List[int] = [-1]

            def extend(T_pts, T_par, target):
                pts = np.asarray(T_pts)
                d2 = np.sum((pts - np.asarray(target)) ** 2, axis=1)
                i_near = int(np.argmin(d2))
                near = T_pts[i_near]
                nv = np.asarray(target) - np.asarray(near)
                dn = float(np.linalg.norm(nv))
                if dn < 1e-9:
                    return "Trapped", None
                if dn <= step_rrtc:
                    new_p = tuple(float(x) for x in target)
                else:
                    dirv = nv / dn
                    new_p = tuple(float(x) for x in
                                  (np.asarray(near) + dirv * step_rrtc))
                if not self.env.is_free(new_p, radius=cfg.collision_radius):
                    return "Trapped", None
                if not self.env.segment_is_free(
                        near, new_p, radius=cfg.collision_radius,
                        step=cfg.collision_step):
                    return "Trapped", None
                T_pts.append(new_p); T_par.append(i_near)
                arrived = (new_p == tuple(target))
                return ("Reached" if arrived else "Advanced"), len(T_pts) - 1

            for _ in range(max_iters):
                if self._rng.random() < 0.4:
                    sample = p_to
                else:
                    sample = (
                        float(self._rng.uniform(b.x_min, b.x_max)),
                        float(self._rng.uniform(b.y_min, b.y_max)),
                        float(self._rng.uniform(b.z_min, b.z_max)),
                    )
                st, lasta = extend(Ta, Ta_par, sample)
                if st == "Trapped":
                    Ta, Ta_par, Tb, Tb_par = Tb, Tb_par, Ta, Ta_par
                    continue
                target = Ta[lasta]
                while True:
                    st2, lastb = extend(Tb, Tb_par, target)
                    if st2 == "Reached":
                        def trace(pts, par, i):
                            out = []
                            while i != -1:
                                out.append(pts[i]); i = par[i]
                            return out
                        pa = list(reversed(trace(Ta, Ta_par, lasta)))
                        pb = trace(Tb, Tb_par, lastb)
                        full = pa + pb[1:]
                        if full[0] != p_from:
                            full = list(reversed(full))
                        return full
                    if st2 == "Trapped":
                        break
                Ta, Ta_par, Tb, Tb_par = Tb, Tb_par, Ta, Ta_par
            return None

        for _round in range(3):
            comps = self._connected_components()
            if len(comps) <= 1:
                return
            comps.sort(key=len, reverse=True)
            # Only work on comps >= min_comp_size
            big_comps = [c for c in comps if len(c) >= min_comp_size]
            if len(big_comps) <= 1:
                return

            # attempt pairs: (main, comp_i) in order
            arr = np.asarray(self.nodes, dtype=float)
            main = big_comps[0]
            main_pts = arr[main]

            attempts_made = 0
            bridged_any = False
            for small in big_comps[1:]:
                if attempts_made >= max_pairs_per_round:
                    break
                small_pts = arr[small]

                # Try the closest 2 node-pairs first
                from scipy.spatial import KDTree
                main_tree = KDTree(main_pts)
                dists, nn_idx = main_tree.query(small_pts, k=1)
                dists = np.atleast_1d(dists)
                nn_idx = np.atleast_1d(nn_idx)
                order = np.argsort(dists)
                for pi in order[:2]:
                    attempts_made += 1
                    small_node = small[int(pi)]
                    main_node = main[int(nn_idx[int(pi)])]
                    path = _rrtc(
                        self.nodes[small_node],
                        self.nodes[main_node],
                        max_iters=max_rrtc_iters,
                    )
                    if path is None or len(path) < 2:
                        continue
                    # inject intermediate waypoints
                    new_indices = []
                    for wp in path[1:-1]:
                        self.nodes.append(wp)
                        new_indices.append(len(self.nodes) - 1)
                    # chain: small_node -> new_indices -> main_node
                    chain = [small_node] + new_indices + [main_node]
                    new_edges_batch: List[Edge3D] = []
                    for a_i, b_i in zip(chain[:-1], chain[1:]):
                        a_pt = self.nodes[a_i]; b_pt = self.nodes[b_i]
                        new_edges_batch.append((a_pt, b_pt))
                    self._incrementally_add_edges(new_edges_batch)
                    bridged_any = True
                    break  # one bridge per (main, small) is enough

            if not bridged_any:
                return  # can't make more progress

    # ----------------------------------------- diagnostics used by the viz --
    @property
    def directions(self) -> np.ndarray:
        return self._dirs