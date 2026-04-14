from __future__ import annotations

import os
from typing import Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from bsrm3d.types import Point3D


def _agg(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["finite_path_length"] = data["path_length"].replace([np.inf, -np.inf], np.nan)
    grouped = (
        data.groupby(["benchmark", "planner"], as_index=False)
        .agg(
            success_rate=("success", "mean"),
            mean_solve_time=("solve_time", "mean"),
            mean_path_length=("finite_path_length", "mean"),
        )
    )
    grouped["success_rate"] *= 100.0
    return grouped


def plot_ompl_comparison(df: pd.DataFrame, output_dir: str = "results/ompl") -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    summary = _agg(df)

    planners = list(summary["planner"].drop_duplicates())
    envs = list(summary["benchmark"].drop_duplicates())

    def _draw(metric: str, ylabel: str, filename: str) -> str:
        fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 5), squeeze=False)
        axes = axes[0]

        for i, env_name in enumerate(envs):
            ax = axes[i]
            sub = summary[summary["benchmark"] == env_name]
            vals = []
            for p in planners:
                row = sub[sub["planner"] == p]
                vals.append(float(row[metric].iloc[0]) if not row.empty else 0.0)

            x = np.arange(len(planners))
            bars = ax.bar(x, vals, color="#4c78a8", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(planners, rotation=25, ha="right")
            ax.set_title(env_name)
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)

            for b in bars:
                h = b.get_height()
                if np.isfinite(h):
                    ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        out = os.path.join(output_dir, filename)
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return out

    success_path = _draw("success_rate", "Success Rate (%)", "comparison_success_rate.png")
    time_path = _draw("mean_solve_time", "Mean Solve Time (s)", "comparison_solve_time.png")
    return success_path, time_path


def plot_scene_3d(
    env,
    nodes: Sequence[Point3D],
    edges: Sequence[tuple[Point3D, Point3D]],
    path: Sequence[Point3D] | None,
    output_path: str,
    title: str,
    start: Point3D | None = None,
    goal: Point3D | None = None,
    obstacle_style: str = "solid",
    obstacle_max_items: int = 6000,
    max_edges_to_draw: int = 9000,
) -> str:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")

    legend_handles = []

    def _draw_obstacles_as_voxels() -> bool:
        if not (hasattr(env, "occupied_voxels") and hasattr(env, "voxel_size") and hasattr(env, "bounds")):
            return False

        voxels = list(env.occupied_voxels)
        if not voxels:
            return False

        # Keep rendering cost bounded for dense maps.
        if obstacle_max_items > 0 and len(voxels) > obstacle_max_items:
            step = max(1, len(voxels) // obstacle_max_items)
            voxels = voxels[::step]

        xs = [v[0] for v in voxels]
        ys = [v[1] for v in voxels]
        zs = [v[2] for v in voxels]

        x0, y0, z0 = min(xs), min(ys), min(zs)
        x1, y1, z1 = max(xs), max(ys), max(zs)
        shape = (x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1)

        grid = np.zeros(shape, dtype=bool)
        for vx, vy, vz in voxels:
            grid[vx - x0, vy - y0, vz - z0] = True

        # Use physical coordinates so voxel cubes align with world scale.
        idx = np.indices(np.array(shape) + 1)
        vs = float(env.voxel_size)
        x = env.bounds.x_min + (idx[0] + x0) * vs
        y = env.bounds.y_min + (idx[1] + y0) * vs
        z = env.bounds.z_min + (idx[2] + z0) * vs

        ax.voxels(
            x,
            y,
            z,
            grid,
            facecolors="#7f7f7f",
            edgecolor=(0.35, 0.35, 0.35, 0.25),
            linewidth=0.05,
            alpha=0.78,
        )
        legend_handles.append(Patch(facecolor="#7f7f7f", edgecolor="#7f7f7f", label="obstacles"))
        return True

    if obstacle_style == "solid":
        drawn = _draw_obstacles_as_voxels()
    else:
        drawn = False

    if not drawn and hasattr(env, "occupied_centers"):
        obstacle_points = env.occupied_centers(max_points=obstacle_max_items)
        if obstacle_points:
            arr = np.asarray(obstacle_points, dtype=float)
            ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=2, c="#555555", alpha=0.45)
            legend_handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555", markersize=6, label="obstacles"))

    if nodes:
        n = np.asarray(nodes, dtype=float)
        ax.scatter(n[:, 0], n[:, 1], n[:, 2], s=10, c="#1f77b4", alpha=0.9, depthshade=False)
        legend_handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=7, label="nodes"))

    if edges:
        edge_draw = list(edges)
        if max_edges_to_draw > 0 and len(edge_draw) > max_edges_to_draw:
            step = max(1, len(edge_draw) // max_edges_to_draw)
            edge_draw = edge_draw[::step]

        for a, b in edge_draw:
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="#4f79d3", alpha=0.6, linewidth=0.9)
        legend_handles.append(Line2D([0], [0], color="#4f79d3", lw=1.4, label="roadmap edges"))

    if path and len(path) > 1:
        p = np.asarray(path, dtype=float)
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color="#e74c3c", linewidth=3.2)
        legend_handles.append(Line2D([0], [0], color="#e74c3c", lw=2.6, label="path"))

    if start is not None:
        ax.scatter([start[0]], [start[1]], [start[2]], s=120, c="#2ca02c", marker="o", depthshade=False)
        legend_handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markersize=9, label="start"))
    if goal is not None:
        ax.scatter([goal[0]], [goal[1]], [goal[2]], s=130, c="#ffbf00", marker="^", depthshade=False)
        legend_handles.append(Line2D([0], [0], marker="^", color="w", markerfacecolor="#ffbf00", markersize=9, label="goal"))

    if hasattr(env, "bounds"):
        ax.set_xlim(env.bounds.x_min, env.bounds.x_max)
        ax.set_ylim(env.bounds.y_min, env.bounds.y_max)
        ax.set_zlim(env.bounds.z_min, env.bounds.z_max)
        ax.set_box_aspect((
            env.bounds.x_max - env.bounds.x_min,
            env.bounds.y_max - env.bounds.y_min,
            env.bounds.z_max - env.bounds.z_min,
        ))

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True, alpha=0.3)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path
