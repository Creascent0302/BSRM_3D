"""
Plotting helpers for BSRM-3D benchmarks (memory-efficient version).
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PLANNER_COLORS = {
    "BSRM3D":      "#e74c3c",
    "delta-PRM":   "#3498db",
    "PRM*":        "#9b59b6",
    "Lazy-PRM":    "#16a085",
    "Halton-PRM":  "#27ae60",
    "SPARS2":      "#e67e22",
    "RRT":         "#2ecc71",
    "RRT-Connect": "#f39c12",
}


def _agg(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["finite_len"] = d["path_length"].replace([np.inf, -np.inf], np.nan)
    g = (
        d.groupby(["benchmark", "planner"], as_index=False)
        .agg(
            success_rate=("success", "mean"),
            mean_build_time=("build_time", "mean"),
            mean_query_time=("query_time", "mean"),
            mean_total_time=("total_time", "mean"),
            mean_path_length=("finite_len", "mean"),
            mean_node_count=("node_count", "mean"),
            mean_edge_count=("edge_count", "mean"),
            n_trials=("trial_id", "count"),
        )
    )
    g["success_rate"] *= 100.0
    return g


def _ordered_planners(summary: pd.DataFrame) -> list:
    preferred = ["BSRM3D", "delta-PRM", "PRM*", "RRT", "RRT-Connect"]
    have = list(summary["planner"].drop_duplicates())
    ordered = [p for p in preferred if p in have]
    ordered += [p for p in have if p not in preferred]
    return ordered


def _bar_chart(summary: pd.DataFrame, metric: str, ylabel: str,
               out_path: str, log: bool = False, dpi: int = 130) -> str:
    envs = list(summary["benchmark"].drop_duplicates())
    planners = _ordered_planners(summary)
    fig, axes = plt.subplots(1, len(envs), figsize=(3.6 * len(envs), 3.4),
                             squeeze=False)
    axes = axes[0]
    for i, env in enumerate(envs):
        ax = axes[i]
        sub = summary[summary["benchmark"] == env]
        vals = []
        for p in planners:
            row = sub[sub["planner"] == p]
            v = float(row[metric].iloc[0]) if not row.empty else 0.0
            if not np.isfinite(v):
                v = 0.0
            vals.append(v)
        x = np.arange(len(planners))
        colors = [PLANNER_COLORS.get(p, "#7f7f7f") for p in planners]
        bars = ax.bar(x, vals, color=colors, edgecolor="black",
                      linewidth=0.4, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(planners, rotation=30, ha="right", fontsize=8)
        ax.set_title(env, fontsize=10)
        if i == 0:
            ax.set_ylabel(ylabel, fontsize=9)
        if log and any(v > 0 for v in vals):
            ax.set_yscale("log")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v,
                        f"{v:.2g}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    plt.close("all")
    return out_path


def plot_summary_charts(df: pd.DataFrame, output_dir: str = "results") -> List[str]:
    """Memory-efficient charts: one small file per metric."""
    os.makedirs(output_dir, exist_ok=True)
    summary = _agg(df)

    metrics = [
        ("success_rate",     "Success Rate (%)",       False),
        ("mean_build_time",  "Build Time (s)",         False),
        ("mean_query_time",  "Query Time (s)",         True),
        ("mean_node_count",  "Node Count",             False),
        ("mean_path_length", "Path Length (m)",        False),
        ("mean_total_time",  "Total Time (s)",         True),
    ]

    paths: List[str] = []
    for metric, ylabel, log in metrics:
        out = os.path.join(output_dir, f"comparison_{metric}.png")
        try:
            _bar_chart(summary, metric, ylabel, out, log=log, dpi=130)
            paths.append(out)
        except Exception as e:
            print(f"  [!] failed {metric}: {e}")
        plt.close("all")

    # 2x3 grid overview
    try:
        envs = list(summary["benchmark"].drop_duplicates())
        planners = _ordered_planners(summary)
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
        for mi, (metric, ylabel, log) in enumerate(metrics):
            ax = axes[mi // 3, mi % 3]
            width = 0.15
            x = np.arange(len(envs))
            for pi, p in enumerate(planners):
                vals = []
                for env in envs:
                    row = summary[(summary["benchmark"] == env)
                                  & (summary["planner"] == p)]
                    v = float(row[metric].iloc[0]) if not row.empty else 0.0
                    if not np.isfinite(v):
                        v = 0.0
                    vals.append(v)
                ax.bar(x + pi * width - (len(planners) - 1) * width / 2, vals,
                       width=width,
                       color=PLANNER_COLORS.get(p, "#7f7f7f"),
                       label=p, alpha=0.9, edgecolor="black", linewidth=0.3)
            ax.set_xticks(x)
            ax.set_xticklabels(envs, rotation=20, ha="right", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(ylabel, fontsize=10)
            if log:
                vals_all = [float(v) for v in summary[metric].values
                            if np.isfinite(v) and v > 0]
                if vals_all:
                    ax.set_yscale("log")
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)
            if mi == 0:
                ax.legend(fontsize=7, loc="upper right")
        plt.tight_layout()
        grid_out = os.path.join(output_dir, "comparison_grid.png")
        fig.savefig(grid_out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        paths.append(grid_out)
    except Exception as e:
        print(f"  [!] grid failed: {e}")
        plt.close("all")

    return paths


# --------------------------------------------------------------- 3D scene --
def plot_scene_3d(
    env,
    nodes: Sequence,
    edges: Sequence,
    path: Optional[Sequence],
    output_path: str,
    title: str,
    start=None, goal=None,
    max_edges: int = 3000,
    dpi: int = 130,
) -> str:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    legend = []
    if hasattr(env, "grid") and env.grid.any():
        xs, ys, zs = np.nonzero(env.grid)
        total = int(xs.size)
        max_items = 5000
        if total > max_items:
            step = max(1, total // max_items)
            xs, ys, zs = xs[::step], ys[::step], zs[::step]
        cx = env.bounds.x_min + (xs + 0.5) * env.voxel_size
        cy = env.bounds.y_min + (ys + 0.5) * env.voxel_size
        cz = env.bounds.z_min + (zs + 0.5) * env.voxel_size
        ax.scatter(cx, cy, cz, s=5, c=cz, cmap="Greys",
                   vmin=env.bounds.z_min - 1, vmax=env.bounds.z_max + 1,
                   alpha=0.25, marker="s", edgecolors="none")
        legend.append(Patch(facecolor="#7f7f7f", edgecolor="#7f7f7f",
                            alpha=0.3, label="obstacles"))

    if nodes:
        n = np.asarray(nodes, dtype=float)
        ax.scatter(n[:, 0], n[:, 1], n[:, 2], s=16, c="#1f77b4",
                   alpha=0.9, edgecolors="white", linewidths=0.3,
                   depthshade=False)
        legend.append(Line2D([0], [0], marker="o", color="w",
                             markerfacecolor="#1f77b4", markersize=7,
                             label="nodes"))

    if edges:
        edrawn = list(edges)
        if max_edges and len(edrawn) > max_edges:
            step = max(1, len(edrawn) // max_edges)
            edrawn = edrawn[::step]
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        segs = np.array([[a, b] for a, b in edrawn], dtype=float)
        lc = Line3DCollection(segs, colors="#4f79d3", linewidths=0.6, alpha=0.5)
        ax.add_collection3d(lc)
        legend.append(Line2D([0], [0], color="#4f79d3", lw=1.4, label="edges"))

    if path and len(path) > 1:
        p = np.asarray(path, dtype=float)
        ax.plot(p[:, 0], p[:, 1], p[:, 2],
                color="#e74c3c", linewidth=3.0, zorder=6)
        legend.append(Line2D([0], [0], color="#e74c3c", lw=2.6, label="path"))

    if start is not None:
        ax.scatter([start[0]], [start[1]], [start[2]], s=100, c="#2ecc71",
                   marker="o", edgecolors="black", depthshade=False, zorder=7)
        legend.append(Line2D([0], [0], marker="o", color="w",
                             markerfacecolor="#2ecc71", markersize=9,
                             label="start"))
    if goal is not None:
        ax.scatter([goal[0]], [goal[1]], [goal[2]], s=110, c="#ffbf00",
                   marker="^", edgecolors="black", depthshade=False, zorder=7)
        legend.append(Line2D([0], [0], marker="^", color="w",
                             markerfacecolor="#ffbf00", markersize=9,
                             label="goal"))

    if hasattr(env, "bounds"):
        ax.set_xlim(env.bounds.x_min, env.bounds.x_max)
        ax.set_ylim(env.bounds.y_min, env.bounds.y_max)
        ax.set_zlim(env.bounds.z_min, env.bounds.z_max)
        ax.set_box_aspect((
            env.bounds.x_max - env.bounds.x_min,
            env.bounds.y_max - env.bounds.y_min,
            env.bounds.z_max - env.bounds.z_min,
        ))
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.grid(True, alpha=0.25)
    if legend:
        ax.legend(handles=legend, loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    plt.close("all")
    return output_path
