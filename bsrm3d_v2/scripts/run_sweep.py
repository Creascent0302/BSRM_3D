"""
run_sweep.py — Node-count sweep with dual-condition (w/ and w/o fallback).

This is the primary experiment: it shows roadmap quality at matched node
budgets under both pure A* (no fallback) and with-fallback conditions.

Outputs:
  results/sweep/sweep_raw.csv
  results/sweep/sweep_pure_astar.{pdf,png}
  results/sweep/sweep_with_fallback.{pdf,png}

Usage:
  python scripts/run_sweep.py --seeds 42 123 2026 --trials 10
  python scripts/run_sweep.py --seeds 42 123 2026 999 7777 --trials 20
"""
from __future__ import annotations
import argparse, math, os, sys, time
import heapq
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bsrm3d.config import Planner3DConfig, bsrm_config_for_env
from bsrm3d.environments.benchmark import build_benchmark_environment
from bsrm3d.planners.beam_bsrm3d import BeamBSRM3D
from bsrm3d.planners.baselines import DeltaPRM, HaltonPRM, PRMStar, SPARS2
from bsrm3d.planners.base_planner import BasePlanner3D
from bsrm3d.viz.paper_style import apply_paper_style, COLORS, MARKERS, PLANNER_ORDER, save_fig

import matplotlib.pyplot as plt


# ── BSRM factory (scales max_nodes to hit target post-sketch) ─────────
def _bsrm_factory(env, cfg, target_n, env_name):
    from bsrm3d.config import bsrm_config_for_env as _cfg
    c = _cfg(env_name, seed=cfg.seed)
    c.max_nodes = max(int(target_n * 3.5), 350)
    return BeamBSRM3D(env=env, config=c)


# ── Pure A* (no fallback) ─────────────────────────────────────────────
def _find_path_pure(self, start, goal):
    cfg = self.config
    if not self.env.is_free(start, radius=cfg.collision_radius):
        return [], math.inf
    if not self.env.is_free(goal, radius=cfg.collision_radius):
        return [], math.inf
    if not self.nodes:
        return [], math.inf
    arr, tree = self._ensure_tree()
    qr = max(cfg.connection_radius, 2.0)

    def _attach(q, k):
        kk = min(max(k, 4), len(self.nodes))
        d, idx = tree.query(np.asarray(q), k=kk)
        d, idx = np.atleast_1d(d), np.atleast_1d(idx)
        keep = d <= qr
        if not keep.any():
            return []
        ids, ds = idx[keep], d[keep]
        ok = np.array([self.env.segment_is_free(
            q, self.nodes[int(i)],
            radius=cfg.collision_radius, step=cfg.collision_step)
            for i in ids], dtype=bool)
        return [(int(i), float(dd)) for i, dd, o in zip(ids, ds, ok) if o]

    sc = _attach(start, cfg.neighbor_k)
    gc = _attach(goal, cfg.neighbor_k)
    if not sc or not gc:
        return [], math.inf

    S, G = -1, -2
    g_cost = {S: 0.0}; parent = {}
    heap = [(self.dist(start, goal), 0.0, S)]
    gm = {i: w for i, w in gc}
    while heap:
        _, gc2, cur = heapq.heappop(heap)
        if gc2 > g_cost.get(cur, math.inf):
            continue
        if cur == G:
            seq = [G]; x = G
            while x in parent:
                x = parent[x]; seq.append(x)
            seq.reverse()
            pts = [start if x == S else (goal if x == G else self.nodes[x])
                   for x in seq]
            L = float(np.sum(np.linalg.norm(
                np.asarray(pts)[1:] - np.asarray(pts)[:-1], axis=1)))
            return pts, L
        cand = sc if cur == S else list(self._adj.get(cur, {}).items())
        if cur != S and cur != G and cur in gm:
            cand = cand + [(G, gm[cur])]
        for nxt, w in cand:
            ng = gc2 + w
            if ng >= g_cost.get(nxt, math.inf):
                continue
            g_cost[nxt] = ng; parent[nxt] = cur
            h = 0.0 if nxt == G else self.dist(self.nodes[nxt], goal)
            heapq.heappush(heap, (ng + h, ng, nxt))
    return [], math.inf


# ── Sample start-goal pairs ───────────────────────────────────────────
def _pairs(env, n, seed):
    rng = np.random.default_rng(seed)
    starts = env.sample_free(n * 3, rng=rng, radius=0.15)
    goals = env.sample_free(n * 3, rng=rng, radius=0.15)
    md = 0.25 * env.bounds.diagonal
    return [(s, g) for s, g in zip(starts, goals)
            if np.linalg.norm(np.array(s) - np.array(g)) >= md][:n]


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--targets", type=int, nargs="+",
                        default=[100, 200, 400, 800, 1500])
    parser.add_argument("--benchmarks", nargs="+",
                        default=["maze3d", "maze_hard", "narrow", "narrow_tight"])
    parser.add_argument("--planners", nargs="+",
                        default=["BSRM3D", "SPARS2", "delta-PRM", "PRM*", "Halton-PRM"])
    parser.add_argument("--output", default="results/sweep")
    args = parser.parse_args()

    FACS = {
        "BSRM3D":     lambda e, c, N, en: _bsrm_factory(e, c, N, en),
        "delta-PRM":  lambda e, c, N, en: DeltaPRM(e, c, N),
        "PRM*":       lambda e, c, N, en: PRMStar(e, c, N),
        "Halton-PRM": lambda e, c, N, en: HaltonPRM(e, c, N),
        "SPARS2":     lambda e, c, N, en: SPARS2(e, c, N * 3),
    }
    orig_find = BasePlanner3D.find_path

    rows = []
    for cond, use_pure in [("pure_astar", True), ("with_fallback", False)]:
        BasePlanner3D.find_path = _find_path_pure if use_pure else orig_find
        for en in args.benchmarks:
            for seed in args.seeds:
                env = build_benchmark_environment(en); env.seed = seed
                pairs = _pairs(env, args.trials, seed + 7)
                for pn in args.planners:
                    if pn not in FACS:
                        continue
                    for tgt in args.targets:
                        cfg = Planner3DConfig(seed=seed)
                        t0 = time.perf_counter()
                        p = FACS[pn](env, cfg, tgt, en)
                        p.generate_roadmap()
                        bt = time.perf_counter() - t0
                        an = len(p.nodes)
                        for s, g in pairs:
                            tq = time.perf_counter()
                            path, L = p.find_path(s, g)
                            qt = time.perf_counter() - tq
                            rows.append(dict(
                                condition=cond, env=en, planner=pn,
                                target=tgt, seed=seed, actual_n=an,
                                success=int(bool(path)),
                                path_length=L if path else np.inf,
                                build_time=bt, query_time=qt))
            print(f"  {cond} {en} done", flush=True)
    BasePlanner3D.find_path = orig_find

    # Save CSV
    df = pd.DataFrame(rows)
    os.makedirs(args.output, exist_ok=True)
    csv_path = os.path.join(args.output, "sweep_raw.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} ({len(df)} rows)")

    # ── Plot ──────────────────────────────────────────────────────────
    apply_paper_style()
    agg = df.groupby(["condition", "env", "planner", "target"]).agg(
        actual_n=("actual_n", "mean"),
        succ=("success", "mean"),
    ).reset_index()
    agg["succ"] *= 100

    # Seed-level std for error bars
    seed_agg = df.groupby(["condition", "env", "planner", "target", "seed"]).agg(
        succ=("success", "mean")).reset_index()
    seed_agg["succ"] *= 100
    std_df = seed_agg.groupby(["condition", "env", "planner", "target"]).agg(
        succ_std=("succ", "std")).reset_index()
    agg = agg.merge(std_df, on=["condition", "env", "planner", "target"], how="left")
    agg["succ_std"] = agg["succ_std"].fillna(0)

    for cond, ylabel in [("pure_astar", "Success Rate (%) — Pure A*"),
                          ("with_fallback", "Success Rate (%) — With Fallback")]:
        sub = agg[agg.condition == cond]
        envs = [e for e in args.benchmarks if e in sub.env.unique()]
        fig, axes = plt.subplots(1, len(envs), figsize=(4.0 * len(envs), 3.2))
        if len(envs) == 1:
            axes = [axes]
        for ax, en in zip(axes, envs):
            esub = sub[sub.env == en]
            for pn in PLANNER_ORDER:
                r = esub[esub.planner == pn].sort_values("actual_n")
                if r.empty:
                    continue
                ax.errorbar(r.actual_n, r.succ, yerr=r.succ_std,
                            marker=MARKERS.get(pn, "o"),
                            color=COLORS.get(pn, "#333"),
                            label=pn, capsize=2, capthick=0.8)
            ax.set_xlabel("Node count (actual)")
            ax.set_title(en.replace("_", "\\_"))
            ax.set_xscale("log")
            if ax is axes[0]:
                ax.set_ylabel(ylabel)
                ax.legend()
        plt.tight_layout()
        out = os.path.join(args.output, f"sweep_{cond}.png")
        save_fig(fig, out)
        print(f"Saved: {out} + .pdf")


if __name__ == "__main__":
    main()
