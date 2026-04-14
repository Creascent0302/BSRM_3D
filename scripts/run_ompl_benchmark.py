import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bsrm3d.config import Planner3DConfig
from bsrm3d.ompl_integration import run_unified_benchmark, plot_ompl_comparison, save_scene_snapshots


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified benchmark: BSRM3D + OMPL classic planners")
    parser.add_argument("--nodes", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument(
        "--ompl-planners",
        nargs="+",
        default=["RRT", "RRTConnect", "PRM", "BITstar"],
        help="OMPL planners to evaluate",
    )
    parser.add_argument("--output", type=str, default="results/ompl/unified_benchmark.csv")
    parser.add_argument("--save-scenes", action="store_true", help="Save one 3D roadmap image per environment and planner")
    parser.add_argument("--scene-output-dir", type=str, default="results/ompl/scenes")
    parser.add_argument("--scene-obstacle-style", type=str, choices=["solid", "points"], default="solid")
    parser.add_argument("--scene-obstacle-max", type=int, default=6000, help="Max obstacle items to draw in a scene")
    parser.add_argument("--scene-max-edges", type=int, default=9000, help="Max roadmap edges to draw per scene")
    args = parser.parse_args()

    cfg = Planner3DConfig(num_nodes=args.nodes, seed=args.seed)

    df = run_unified_benchmark(
        benchmarks=["cluttered", "tunnel", "rooms"],
        planner_cfg=cfg,
        ompl_planners=args.ompl_planners,
        trials_per_env=args.trials,
        ompl_timeout=args.timeout,
    )

    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved raw benchmark CSV: {args.output}")

    success_fig, time_fig = plot_ompl_comparison(df, output_dir=output_dir)
    print(f"Saved comparison chart: {success_fig}")
    print(f"Saved comparison chart: {time_fig}")

    if args.save_scenes:
        scene_paths = save_scene_snapshots(
            benchmarks=["cluttered", "tunnel", "rooms"],
            planner_cfg=cfg,
            ompl_planners=args.ompl_planners,
            ompl_timeout=args.timeout,
            output_dir=args.scene_output_dir,
            obstacle_style=args.scene_obstacle_style,
            obstacle_max_items=args.scene_obstacle_max,
            max_edges_to_draw=args.scene_max_edges,
        )
        print(f"Saved scene snapshots ({len(scene_paths)}):")
        for p in scene_paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
