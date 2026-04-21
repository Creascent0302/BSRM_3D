"""Save one 3D scene snapshot per (env, planner) for the paper figures."""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bsrm3d.config import Planner3DConfig, bsrm_config_for_env
from bsrm3d.environments.benchmark import available_benchmarks, build_benchmark_environment
from bsrm3d.planners.baselines import (DeltaPRM, PRMStar, LazyPRM, HaltonPRM,
                                        SPARS2, RRT, RRTConnect)
from bsrm3d.planners.beam_bsrm3d import BeamBSRM3D
from bsrm3d.viz.charts import plot_scene_3d


# Each factory: (env, cfg, N) -> planner instance.
# BSRM uses env-tuned config; others use the provided cfg.
def _bsrm_scene(env, cfg, N, env_name):
    cfg_tuned = bsrm_config_for_env(env_name, seed=cfg.seed)
    return BeamBSRM3D(env=env, config=cfg_tuned)

PLANNER_FACTORIES = {
    "BSRM3D":      _bsrm_scene,
    "delta-PRM":   lambda env, cfg, N, env_name: DeltaPRM(env=env, config=cfg, num_samples=N),
    "PRM*":        lambda env, cfg, N, env_name: PRMStar(env=env, config=cfg, num_samples=N),
    "Lazy-PRM":    lambda env, cfg, N, env_name: LazyPRM(env=env, config=cfg, num_samples=N),
    "Halton-PRM":  lambda env, cfg, N, env_name: HaltonPRM(env=env, config=cfg, num_samples=N),
    "SPARS2":      lambda env, cfg, N, env_name: SPARS2(env=env, config=cfg, num_samples=N*2),
    "RRT":         lambda env, cfg, N, env_name: RRT(env=env, config=cfg),
    "RRT-Connect": lambda env, cfg, N, env_name: RRTConnect(env=env, config=cfg),
}


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


def save_scene(env_name: str, planner_name: str, cfg: Planner3DConfig,
               output_dir: str, prm_samples: int = 1000) -> str:
    env = build_benchmark_environment(env_name)
    rng = np.random.default_rng(env.seed + 19)

    # For multi-floor envs, force the start/goal to span different floors
    # (different sign of z - 3.0) so the scene visualisation actually
    # exercises the stairwell.
    needs_cross_floor = env_name in ("maze3d", "maze_hard")

    min_d = 0.35 * env.bounds.diagonal
    best = None
    for _ in range(200):
        s_list = env.sample_free(1, rng=rng, radius=0.15)
        g_list = env.sample_free(1, rng=rng, radius=0.15)
        if not s_list or not g_list:
            continue
        s = s_list[0]; g = g_list[0]
        dist = float(np.linalg.norm(np.array(s) - np.array(g)))
        if dist < min_d:
            continue
        if needs_cross_floor:
            # require start and goal on different sides of z=3
            if (s[2] - 3.0) * (g[2] - 3.0) > 0:
                continue
        best = (s, g)
        break
    if best is None:
        s_list = env.sample_free(1, rng=rng, radius=0.15)
        g_list = env.sample_free(1, rng=rng, radius=0.15)
        best = (s_list[0], g_list[0])
    start, goal = best

    factory = PLANNER_FACTORIES.get(planner_name)
    if factory is None:
        raise ValueError(f"unknown planner {planner_name}")

    planner = factory(env, cfg, prm_samples, env_name)
    planner.generate_roadmap()
    path, L = planner.find_path(start, goal)

    if planner_name in ("RRT", "RRT-Connect"):
        nodes, edges = [], []
    else:
        nodes, edges = planner.nodes, planner.edges

    status = "solved" if path else "unsolved"
    title = f"{env_name} | {planner_name} | N={len(nodes)} E={len(edges)} {status} L={L:.2f}"
    fn = os.path.join(output_dir,
                      f"{_safe_name(env_name)}_{_safe_name(planner_name)}_scene.png")
    return plot_scene_3d(env, nodes, edges, path if path else None, fn, title,
                         start=start, goal=goal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=available_benchmarks())
    ap.add_argument("--planners", nargs="+",
                    default=["BSRM3D", "SPARS2", "delta-PRM",
                             "PRM*", "Lazy-PRM", "RRT-Connect"])
    ap.add_argument("--prm-samples", type=int, default=1000)
    ap.add_argument("--output-dir", default="results/scenes")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = Planner3DConfig()
    saved: List[str] = []
    for b in args.benchmarks:
        for p in args.planners:
            print(f"  scene: {b} | {p}")
            t0 = time.perf_counter()
            try:
                out = save_scene(b, p, cfg, args.output_dir,
                                 prm_samples=args.prm_samples)
                print(f"    -> {out}  ({time.perf_counter() - t0:.2f}s)")
                saved.append(out)
            except Exception as e:
                print(f"    !! failed: {e}")
    print(f"\nSaved {len(saved)} scenes.")


if __name__ == "__main__":
    main()
