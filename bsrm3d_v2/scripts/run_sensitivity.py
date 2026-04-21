"""
run_sensitivity.py — Parameter sensitivity heatmaps.

Sweeps (r_min, angular_step_deg) and produces 2D heatmaps.

Outputs:
  results/sensitivity/sensitivity_raw.csv
  results/sensitivity/sensitivity_{env}.{pdf,png}

Usage:
  python scripts/run_sensitivity.py --seeds 42 123 --trials 5
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bsrm3d.config import Planner3DConfig
from bsrm3d.environments.benchmark import build_benchmark_environment
from bsrm3d.planners.beam_bsrm3d import BeamBSRM3D
from bsrm3d.viz.paper_style import apply_paper_style, save_fig


def _pairs(env, n, seed):
    rng = np.random.default_rng(seed)
    s = env.sample_free(n*3, rng=rng, radius=0.15)
    g = env.sample_free(n*3, rng=rng, radius=0.15)
    md = 0.25 * env.bounds.diagonal
    return [(a,b) for a,b in zip(s,g) if np.linalg.norm(np.array(a)-np.array(b))>=md][:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--benchmarks", nargs="+",
                        default=["cluttered", "maze3d", "narrow", "narrow_tight"])
    parser.add_argument("--r-min", type=float, nargs="+",
                        default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.8])
    parser.add_argument("--angular", type=float, nargs="+",
                        default=[5, 10, 15, 20, 25, 35])
    parser.add_argument("--output", default="results/sensitivity")
    args = parser.parse_args()

    rows = []
    for env_name in args.benchmarks:
        for rm in args.r_min:
            for ang in args.angular:
                for seed in args.seeds:
                    env = build_benchmark_environment(env_name); env.seed = seed
                    pairs = _pairs(env, args.trials, seed+7)
                    cfg = Planner3DConfig(seed=seed)
                    cfg.min_spacing = rm
                    cfg.angular_step_deg = ang
                    cfg.max_nodes = 2000
                    p = BeamBSRM3D(env, cfg)
                    t0 = time.perf_counter()
                    p.generate_roadmap()
                    bt = time.perf_counter()-t0
                    ok = sum(1 for s,g in pairs if p.find_path(s,g)[0])
                    rows.append(dict(env=env_name, r_min=rm, angular=ang, seed=seed,
                        success=ok, total=len(pairs),
                        rate=ok/max(1,len(pairs))*100,
                        nodes=len(p.nodes), build_time=bt))
        print(f"  {env_name} done", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(args.output, exist_ok=True)
    df.to_csv(os.path.join(args.output, "sensitivity_raw.csv"), index=False)
    print(f"Saved: {args.output}/sensitivity_raw.csv ({len(df)} rows)")

    # Heatmaps
    apply_paper_style()
    agg = df.groupby(["env","r_min","angular"]).agg(
        rate=("rate","mean"), N=("nodes","mean")).reset_index()

    for env_name in args.benchmarks:
        sub = agg[agg.env == env_name]
        rms = sorted(sub.r_min.unique())
        angs = sorted(sub.angular.unique())
        rate_grid = np.full((len(rms), len(angs)), np.nan)
        node_grid = np.full((len(rms), len(angs)), np.nan)
        for i, rm in enumerate(rms):
            for j, ang in enumerate(angs):
                r = sub[(sub.r_min==rm) & (sub.angular==ang)]
                if not r.empty:
                    rate_grid[i,j] = r.rate.iloc[0]
                    node_grid[i,j] = r.N.iloc[0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 2.8))
        im1 = ax1.imshow(rate_grid, aspect="auto", origin="lower",
                         vmin=0, vmax=100, cmap="RdYlGn")
        ax1.set_xticks(range(len(angs)))
        ax1.set_xticklabels([f"{a:.0f}" for a in angs])
        ax1.set_yticks(range(len(rms)))
        ax1.set_yticklabels([f"{r:.1f}" for r in rms])
        ax1.set_xlabel(r"$\Delta\theta$ (deg)")
        ax1.set_ylabel(r"$r_{\min}$ (m)")
        ax1.set_title("Success Rate (%)")
        for i in range(len(rms)):
            for j in range(len(angs)):
                v = rate_grid[i,j]
                if not np.isnan(v):
                    ax1.text(j, i, f"{v:.0f}", ha="center", va="center",
                             fontsize=7, color="white" if v<50 else "black")
        fig.colorbar(im1, ax=ax1, shrink=0.85)

        im2 = ax2.imshow(node_grid, aspect="auto", origin="lower", cmap="YlOrRd")
        ax2.set_xticks(range(len(angs)))
        ax2.set_xticklabels([f"{a:.0f}" for a in angs])
        ax2.set_yticks(range(len(rms)))
        ax2.set_yticklabels([f"{r:.1f}" for r in rms])
        ax2.set_xlabel(r"$\Delta\theta$ (deg)")
        ax2.set_ylabel(r"$r_{\min}$ (m)")
        ax2.set_title("Node Count")
        for i in range(len(rms)):
            for j in range(len(angs)):
                v = node_grid[i,j]
                if not np.isnan(v):
                    ax2.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im2, ax=ax2, shrink=0.85)

        plt.suptitle(env_name.replace("_","\\_"), fontsize=11, y=1.02)
        plt.tight_layout()
        out = os.path.join(args.output, f"sensitivity_{env_name}.png")
        save_fig(fig, out)
        print(f"Saved: {out} + .pdf")


if __name__ == "__main__":
    main()
