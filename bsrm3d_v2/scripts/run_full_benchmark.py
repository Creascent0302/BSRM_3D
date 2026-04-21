#!/usr/bin/env python3
"""
BSRM-3D Full Benchmark Suite
=============================
Sweeps node budget across all environments and planners.
Produces CSV data + publication-ready figures.

Usage:
    python scripts/run_full_benchmark.py
    python scripts/run_full_benchmark.py --seeds 10 --trials 30  # heavier
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bsrm3d.config import Planner3DConfig
from bsrm3d.environments.benchmark import build_benchmark_environment
from bsrm3d.planners.beam_bsrm3d import BeamBSRM3D
from bsrm3d.planners.baselines import DeltaPRM, SPARS2, HaltonPRM
from bsrm3d.types import Point3D

# ─── Configuration ──────────────────────────────────────────────────

ENVIRONMENTS = [
    # (name, connection_radius)
    ("cluttered",       2.0),
    ("indoor",          2.0),
    ("maze_hard",       2.0),
    ("narrow_tight",    2.0),
    ("bugtrap",         2.0),
    ("maze3d",          2.0),
    ("forest",          2.0),
    ("building_3floor", 3.0),
]

NODE_BUDGETS = [200, 400, 600, 800, 1000, 1200, 1500, 2000]

PLANNERS = ["BSRM3D", "delta-PRM", "SPARS2", "Halton-PRM"]

# ─── Data collection ────────────────────────────────────────────────

@dataclass
class TrialRow:
    environment: str
    planner: str
    node_budget: int
    seed: int
    trial_id: int
    success: int
    build_time: float
    query_time: float
    path_length: float
    node_count: int
    edge_count: int


def _sample_pairs(env, n: int, seed: int, radius: float = 0.15):
    rng = np.random.default_rng(seed + 7777)
    S = list(env.sample_free(n * 3, rng=rng, radius=radius))
    G = list(env.sample_free(n * 3, rng=rng, radius=radius))
    diag = env.bounds.diagonal
    md = 0.25 * diag
    pairs = []
    for s, g in zip(S, G):
        if np.linalg.norm(np.array(s) - np.array(g)) >= md:
            pairs.append((s, g))
            if len(pairs) >= n:
                break
    return pairs


def _build_planner(planner_name, env, cfg, node_budget, rc):
    if planner_name == "BSRM3D":
        cfg_b = Planner3DConfig(
            seed=cfg.seed, max_nodes=node_budget,
            connection_radius=rc, beam_max_length=35.0)
        cfg_b.enable_two_pass = True
        p = BeamBSRM3D(env, cfg_b)
    elif planner_name == "delta-PRM":
        p = DeltaPRM(env, cfg, num_samples=node_budget,
                     connection_radius=rc)
    elif planner_name == "SPARS2":
        p = SPARS2(env, cfg, num_samples=node_budget)
    elif planner_name == "Halton-PRM":
        p = HaltonPRM(env, cfg, num_samples=node_budget,
                      connection_radius=rc)
    else:
        raise ValueError(f"Unknown planner: {planner_name}")
    return p


def run_benchmark(n_seeds=5, n_trials=20, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    rows: List[TrialRow] = []
    total_combos = len(ENVIRONMENTS) * len(PLANNERS) * len(NODE_BUDGETS) * n_seeds
    done = 0

    for env_name, rc in ENVIRONMENTS:
        for planner_name in PLANNERS:
            for nb in NODE_BUDGETS:
                for seed in range(n_seeds):
                    done += 1
                    env = build_benchmark_environment(env_name)
                    env.seed = seed
                    cfg = Planner3DConfig(seed=seed, connection_radius=rc)

                    # Build
                    t0 = time.perf_counter()
                    try:
                        p = _build_planner(planner_name, env, cfg, nb, rc)
                        p.generate_roadmap()
                        build_time = time.perf_counter() - t0
                    except Exception:
                        build_time = float("inf")
                        p = None

                    n_nodes = len(p.nodes) if p else 0
                    n_edges = len(p.edges) if p else 0

                    # Query
                    pairs = _sample_pairs(env, n_trials, seed)
                    for ti, (s, g) in enumerate(pairs):
                        if p is None or n_nodes < 2:
                            rows.append(TrialRow(
                                env_name, planner_name, nb, seed, ti,
                                0, build_time, 0.0, math.inf, n_nodes, n_edges))
                            continue
                        qt0 = time.perf_counter()
                        path, pl = p.find_path(s, g)
                        qt = time.perf_counter() - qt0
                        rows.append(TrialRow(
                            env_name, planner_name, nb, seed, ti,
                            int(bool(path)), build_time, qt,
                            pl if path else math.inf,
                            n_nodes, n_edges))

                    if done % 20 == 0 or done == total_combos:
                        pct = done / total_combos * 100
                        print(f"\r  [{pct:5.1f}%] {env_name}/{planner_name}/N={nb}/s={seed}"
                              f"  ({done}/{total_combos})", end="", flush=True)

    print()
    df = pd.DataFrame([asdict(r) for r in rows])
    csv_path = os.path.join(output_dir, "benchmark_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path} ({len(df)} rows)")
    return df


# ─── Plotting ───────────────────────────────────────────────────────

def plot_all(df: pd.DataFrame, output_dir="results"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from bsrm3d.viz.paper_style import apply_paper_style, COLORS, MARKERS, save_fig
        apply_paper_style()
    except Exception:
        COLORS = {"BSRM3D":"#d62728","delta-PRM":"#1f77b4",
                  "SPARS2":"#ff7f0e","Halton-PRM":"#2ca02c"}
        MARKERS = {"BSRM3D":"o","delta-PRM":"s","SPARS2":"v","Halton-PRM":"^"}
        def save_fig(fig, path):
            fig.savefig(path, dpi=220, bbox_inches="tight")
            fig.savefig(path.replace(".png",".pdf"), format="pdf", bbox_inches="tight")
            plt.close(fig)

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Replace inf path lengths for aggregation
    df = df.copy()
    df["finite_path"] = df["path_length"].replace([np.inf, -np.inf], np.nan)

    # Aggregate per (environment, planner, node_budget)
    agg = df.groupby(["environment", "planner", "node_budget"]).agg(
        success_rate=("success", "mean"),
        build_mean=("build_time", "mean"),
        query_mean=("query_time", "mean"),
        path_mean=("finite_path", "mean"),
        node_mean=("node_count", "mean"),
        edge_mean=("edge_count", "mean"),
        # std for shaded regions
        success_std=("success", "std"),
        build_std=("build_time", "std"),
    ).reset_index()
    agg["success_rate"] *= 100
    agg["success_std"] *= 100
    agg["query_ms"] = agg["query_mean"] * 1000

    envs = [e for e, _ in ENVIRONMENTS]
    planners = PLANNERS

    # ── Figure 1: Success rate sweep (multi-panel) ──────────────────
    _plot_metric_sweep(agg, envs, planners, "success_rate", "success_std",
                       "Success Rate (%)", "sweep_success_rate",
                       fig_dir, COLORS, MARKERS, ylim=(0, 105))

    # ── Figure 2: Build time sweep ──────────────────────────────────
    _plot_metric_sweep(agg, envs, planners, "build_mean", "build_std",
                       "Build Time (s)", "sweep_build_time",
                       fig_dir, COLORS, MARKERS)

    # ── Figure 3: Query time sweep ──────────────────────────────────
    _plot_metric_sweep(agg, envs, planners, "query_ms", None,
                       "Query Time (ms)", "sweep_query_time",
                       fig_dir, COLORS, MARKERS)

    # ── Figure 4: Node count sweep ──────────────────────────────────
    _plot_metric_sweep(agg, envs, planners, "node_mean", None,
                       "Node Count", "sweep_node_count",
                       fig_dir, COLORS, MARKERS)

    # ── Figure 5: Path length sweep ─────────────────────────────────
    _plot_metric_sweep(agg, envs, planners, "path_mean", None,
                       "Path Length", "sweep_path_length",
                       fig_dir, COLORS, MARKERS)

    # ── Figure 6: Combined 3-metric (success, build, query) ────────
    _plot_3metric(agg, envs, planners, fig_dir, COLORS, MARKERS)

    print(f"  Figures saved to {fig_dir}/")


def _plot_metric_sweep(agg, envs, planners, metric, std_col, ylabel,
                       fname, fig_dir, COLORS, MARKERS, ylim=None):
    import matplotlib.pyplot as plt
    n_envs = len(envs)
    cols = min(4, n_envs)
    rows_n = math.ceil(n_envs / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.8*cols, 3.0*rows_n),
                              squeeze=False)
    for idx, env in enumerate(envs):
        ax = axes[idx // cols][idx % cols]
        for pname in planners:
            sub = agg[(agg["environment"] == env) & (agg["planner"] == pname)]
            if sub.empty:
                continue
            sub = sub.sort_values("node_budget")
            x = sub["node_budget"].values
            y = sub[metric].values
            c = COLORS.get(pname, "#333")
            m = MARKERS.get(pname, "o")
            ax.plot(x, y, color=c, marker=m, markersize=4, label=pname)
            if std_col and std_col in sub.columns:
                s = sub[std_col].values
                ax.fill_between(x, y - s, y + s, color=c, alpha=0.12)
        ax.set_title(env.replace("_", " ").title(), fontsize=9)
        ax.set_xlabel("Node Budget")
        if idx % cols == 0:
            ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(ylim)
    # Hide unused axes
    for idx in range(n_envs, rows_n * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    # Single legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=len(planners), fontsize=8,
               bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, os.path.join(fig_dir, f"{fname}.png"))


def _plot_3metric(agg, envs, planners, fig_dir, COLORS, MARKERS):
    """3-row figure: success rate, build time, query time."""
    import matplotlib.pyplot as plt
    metrics = [
        ("success_rate", "success_std", "Success Rate (%)", (0, 105)),
        ("build_mean",   "build_std",   "Build Time (s)", None),
        ("query_ms",     None,          "Query Time (ms)", None),
    ]
    n_envs = len(envs)
    fig, axes = plt.subplots(3, n_envs, figsize=(2.8*n_envs, 7.5), squeeze=False)
    for row, (metric, std_col, ylabel, ylim) in enumerate(metrics):
        for col, env in enumerate(envs):
            ax = axes[row][col]
            for pname in planners:
                sub = agg[(agg["environment"] == env) & (agg["planner"] == pname)]
                if sub.empty: continue
                sub = sub.sort_values("node_budget")
                x = sub["node_budget"].values
                y = sub[metric].values
                c = COLORS.get(pname, "#333")
                m = MARKERS.get(pname, "o")
                ax.plot(x, y, color=c, marker=m, markersize=3, label=pname)
                if std_col and std_col in sub.columns:
                    s = sub[std_col].values
                    ax.fill_between(x, y - s, y + s, color=c, alpha=0.1)
            if row == 0:
                ax.set_title(env.replace("_", " ").title(), fontsize=8)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            if row == 2:
                ax.set_xlabel("Node Budget", fontsize=7)
            if ylim:
                ax.set_ylim(ylim)
            ax.tick_params(labelsize=6)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=len(planners), fontsize=7,
               bbox_to_anchor=(0.5, 1.01))
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, os.path.join(fig_dir, "sweep_3metric.png"))


# ─── Summary table ──────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, output_dir="results"):
    """Print and save a LaTeX-ready summary table at N=2000."""
    summary = df[df["node_budget"] == 2000].copy()
    summary["finite_path"] = summary["path_length"].replace(
        [np.inf, -np.inf], np.nan)
    tbl = summary.groupby(["environment", "planner"]).agg(
        succ=("success", "mean"),
        build=("build_time", "mean"),
        query=("query_time", "mean"),
        path=("finite_path", "mean"),
        nodes=("node_count", "mean"),
    ).reset_index()
    tbl["succ"] *= 100
    tbl["query_ms"] = tbl["query"] * 1000

    txt_path = os.path.join(output_dir, "summary_table.txt")
    with open(txt_path, "w") as f:
        f.write(f"{'Environment':17s} {'Planner':12s} {'Succ%':>6s} "
                f"{'Build':>7s} {'Query':>7s} {'Path':>7s} {'Nodes':>6s}\n")
        f.write("-" * 70 + "\n")
        for _, r in tbl.iterrows():
            p_str = f"{r['path']:6.1f}" if not np.isnan(r['path']) else "   inf"
            f.write(f"{r['environment']:17s} {r['planner']:12s} "
                    f"{r['succ']:5.0f}% {r['build']:6.2f}s "
                    f"{r['query_ms']:5.1f}ms {p_str} {r['nodes']:5.0f}\n")
    print(f"  Summary: {txt_path}")
    # Also print to stdout
    with open(txt_path) as f:
        print(f.read())


# ─── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    print(f"  Config: {args.seeds} seeds × {args.trials} trials/seed")
    print(f"  Planners: {PLANNERS}")
    print(f"  Budgets: {NODE_BUDGETS}")
    print(f"  Environments: {[e for e,_ in ENVIRONMENTS]}\n")

    df = run_benchmark(n_seeds=args.seeds, n_trials=args.trials,
                       output_dir=args.output)
    print()
    plot_all(df, output_dir=args.output)
    print()
    print_summary(df, output_dir=args.output)


if __name__ == "__main__":
    main()
