from __future__ import annotations

from dataclasses import dataclass
import os
from time import perf_counter
from typing import Callable, Dict, Iterable, List, Sequence
import math

import numpy as np
import pandas as pd

from bsrm3d.config import Planner3DConfig
from bsrm3d.environments.benchmark import build_benchmark_environment
from bsrm3d.planners.beam_bsrm3d import BeamBSRM3D
from bsrm3d.types import Edge3D, Point3D


@dataclass
class TrialResult:
    benchmark: str
    planner: str
    trial_id: int
    success: int
    solve_time: float
    path_length: float
    node_count: float
    edge_count: float


def _path_length(points: Sequence[Point3D]) -> float:
    if len(points) < 2:
        return math.inf
    arr = np.asarray(points, dtype=float)
    seg = np.linalg.norm(arr[1:] - arr[:-1], axis=1)
    return float(np.sum(seg))


def _sample_pairs(env, count: int) -> List[tuple[Point3D, Point3D]]:
    starts = list(env.sample_free(count))
    goals = list(env.sample_free(count))
    if len(starts) < count or len(goals) < count:
        raise RuntimeError(
            f"Failed to sample enough free start-goal pairs: requested={count}, "
            f"starts={len(starts)}, goals={len(goals)}"
        )
    return list(zip(starts, goals))


def _ensure_ompl() -> None:
    try:
        import ompl.base as _ob  # type: ignore
        import ompl.geometric as _og  # type: ignore
        _ = (_ob, _og)
    except Exception as e:
        raise RuntimeError(
            "OMPL python bindings are not available. Install OMPL Python bindings first "
            "(for example with conda-forge: 'conda install -c conda-forge ompl')."
        ) from e


def _create_ompl_factory_map() -> Dict[str, Callable]:
    import ompl.geometric as og  # type: ignore

    return {
        "RRT": lambda ss: og.RRT(ss.getSpaceInformation()),
        "RRTConnect": lambda ss: og.RRTConnect(ss.getSpaceInformation()),
        "PRM": lambda ss: og.PRM(ss.getSpaceInformation()),
        "BITstar": lambda ss: og.BITstar(ss.getSpaceInformation()),
    }


def _solve_with_ompl(env, start: Point3D, goal: Point3D, planner_name: str, timeout: float) -> tuple[bool, float, float]:
    import ompl.base as ob  # type: ignore
    import ompl.geometric as og  # type: ignore

    space = ob.RealVectorStateSpace(3)
    bounds = ob.RealVectorBounds(3)
    bounds.setLow(0, env.bounds.x_min)
    bounds.setHigh(0, env.bounds.x_max)
    bounds.setLow(1, env.bounds.y_min)
    bounds.setHigh(1, env.bounds.y_max)
    bounds.setLow(2, env.bounds.z_min)
    bounds.setHigh(2, env.bounds.z_max)
    space.setBounds(bounds)

    ss = og.SimpleSetup(space)

    def is_valid(state: ob.State) -> bool:
        p = (float(state[0]), float(state[1]), float(state[2]))
        return env.is_free(p)

    if hasattr(ob, "StateValidityCheckerFn"):
        ss.setStateValidityChecker(ob.StateValidityCheckerFn(is_valid))
    else:
        class _StateValidityChecker(ob.StateValidityChecker):
            def __init__(self, si: ob.SpaceInformation):
                super().__init__(si)

            def isValid(self, state: ob.State) -> bool:
                return is_valid(state)

        ss.setStateValidityChecker(_StateValidityChecker(ss.getSpaceInformation()))

    s = space.allocState()
    s[0], s[1], s[2] = start
    g = space.allocState()
    g[0], g[1], g[2] = goal
    ss.setStartAndGoalStates(s, g)

    planner_map = _create_ompl_factory_map()
    if planner_name not in planner_map:
        raise ValueError(f"Unsupported OMPL planner: {planner_name}")
    ss.setPlanner(planner_map[planner_name](ss))

    t0 = perf_counter()
    solved = ss.solve(timeout)
    dt = perf_counter() - t0

    if not solved:
        return False, dt, math.inf

    path = ss.getSolutionPath()
    states = path.getStates()
    pts = [(float(st[0]), float(st[1]), float(st[2])) for st in states]
    return True, dt, _path_length(pts)


def _extract_planner_graph(ss) -> tuple[List[Point3D], List[Edge3D]]:
    import ompl.base as ob  # type: ignore

    planner_data = ob.PlannerData(ss.getSpaceInformation())
    ss.getPlannerData(planner_data)

    nodes: List[Point3D] = []
    for i in range(planner_data.numVertices()):
        st = planner_data.getVertex(i).getState()
        nodes.append((float(st[0]), float(st[1]), float(st[2])))

    edges: List[Edge3D] = []
    used = set()
    for i in range(planner_data.numVertices()):
        for j in planner_data.getEdges(i):
            a, b = (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
            if a == b or (a, b) in used:
                continue
            used.add((a, b))
            edges.append((nodes[a], nodes[b]))

    return nodes, edges


def _solve_with_ompl_graph(
    env,
    start: Point3D,
    goal: Point3D,
    planner_name: str,
    timeout: float,
) -> tuple[bool, float, float, List[Point3D], List[Edge3D], List[Point3D]]:
    import ompl.base as ob  # type: ignore
    import ompl.geometric as og  # type: ignore

    space = ob.RealVectorStateSpace(3)
    bounds = ob.RealVectorBounds(3)
    bounds.setLow(0, env.bounds.x_min)
    bounds.setHigh(0, env.bounds.x_max)
    bounds.setLow(1, env.bounds.y_min)
    bounds.setHigh(1, env.bounds.y_max)
    bounds.setLow(2, env.bounds.z_min)
    bounds.setHigh(2, env.bounds.z_max)
    space.setBounds(bounds)

    ss = og.SimpleSetup(space)

    def is_valid(state: ob.State) -> bool:
        p = (float(state[0]), float(state[1]), float(state[2]))
        return env.is_free(p)

    if hasattr(ob, "StateValidityCheckerFn"):
        ss.setStateValidityChecker(ob.StateValidityCheckerFn(is_valid))
    else:
        class _StateValidityChecker(ob.StateValidityChecker):
            def __init__(self, si: ob.SpaceInformation):
                super().__init__(si)

            def isValid(self, state: ob.State) -> bool:
                return is_valid(state)

        ss.setStateValidityChecker(_StateValidityChecker(ss.getSpaceInformation()))

    s = space.allocState()
    s[0], s[1], s[2] = start
    g = space.allocState()
    g[0], g[1], g[2] = goal
    ss.setStartAndGoalStates(s, g)

    planner_map = _create_ompl_factory_map()
    if planner_name not in planner_map:
        raise ValueError(f"Unsupported OMPL planner: {planner_name}")
    ss.setPlanner(planner_map[planner_name](ss))

    t0 = perf_counter()
    solved = ss.solve(timeout)
    dt = perf_counter() - t0

    nodes, edges = _extract_planner_graph(ss)

    if not solved:
        return False, dt, math.inf, nodes, edges, []

    path = ss.getSolutionPath()
    states = path.getStates()
    pts = [(float(st[0]), float(st[1]), float(st[2])) for st in states]
    return True, dt, _path_length(pts), nodes, edges, pts


def _run_bsrm_trials(
    benchmark_name: str,
    planner_cfg: Planner3DConfig,
    pairs: Sequence[tuple[Point3D, Point3D]],
) -> List[TrialResult]:
    env = build_benchmark_environment(benchmark_name)

    t0 = perf_counter()
    planner = BeamBSRM3D(env=env, config=planner_cfg)
    nodes, edges = planner.generate_roadmap()
    build_time = perf_counter() - t0
    amortized_build = build_time / max(1, len(pairs))

    results: List[TrialResult] = []
    for i, (s, g) in enumerate(pairs):
        q0 = perf_counter()
        path, plen = planner.find_path(s, g)
        qtime = perf_counter() - q0
        success = int(bool(path))
        total = amortized_build + qtime
        results.append(
            TrialResult(
                benchmark=benchmark_name,
                planner="BSRM3D",
                trial_id=i,
                success=success,
                solve_time=total,
                path_length=float(plen if success else math.inf),
                node_count=float(len(nodes)),
                edge_count=float(len(edges)),
            )
        )
    return results


def _run_ompl_trials(
    benchmark_name: str,
    planners: Iterable[str],
    pairs: Sequence[tuple[Point3D, Point3D]],
    timeout: float,
) -> List[TrialResult]:
    env = build_benchmark_environment(benchmark_name)
    rows: List[TrialResult] = []
    for name in planners:
        for i, (s, g) in enumerate(pairs):
            ok, t, plen = _solve_with_ompl(env, s, g, planner_name=name, timeout=timeout)
            rows.append(
                TrialResult(
                    benchmark=benchmark_name,
                    planner=name,
                    trial_id=i,
                    success=int(ok),
                    solve_time=float(t),
                    path_length=float(plen),
                    node_count=math.nan,
                    edge_count=math.nan,
                )
            )
    return rows


def run_unified_benchmark(
    benchmarks: Sequence[str],
    planner_cfg: Planner3DConfig | None = None,
    ompl_planners: Sequence[str] = ("RRT", "RRTConnect", "PRM", "BITstar"),
    trials_per_env: int = 20,
    ompl_timeout: float = 2.0,
) -> pd.DataFrame:
    """Run BSRM and OMPL planners under the same benchmark environments and start-goal pairs."""
    cfg = planner_cfg or Planner3DConfig()
    all_rows: List[TrialResult] = []

    _ensure_ompl()

    for env_name in benchmarks:
        env = build_benchmark_environment(env_name)
        pairs = _sample_pairs(env, trials_per_env)
        all_rows.extend(_run_bsrm_trials(env_name, cfg, pairs))
        all_rows.extend(_run_ompl_trials(env_name, ompl_planners, pairs, ompl_timeout))

    df = pd.DataFrame([r.__dict__ for r in all_rows])
    return df


def save_scene_snapshots(
    benchmarks: Sequence[str],
    planner_cfg: Planner3DConfig | None = None,
    ompl_planners: Sequence[str] = ("RRT", "RRTConnect", "PRM", "BITstar"),
    ompl_timeout: float = 2.0,
    output_dir: str = "results/ompl/scenes",
    obstacle_style: str = "solid",
    obstacle_max_items: int = 6000,
    max_edges_to_draw: int = 9000,
) -> List[str]:
    from .visualization import plot_scene_3d

    def _safe_name(name: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)

    _ensure_ompl()
    cfg = planner_cfg or Planner3DConfig()
    os.makedirs(output_dir, exist_ok=True)

    image_paths: List[str] = []
    for env_name in benchmarks:
        env = build_benchmark_environment(env_name)
        start, goal = _sample_pairs(env, 1)[0]

        bsrm = BeamBSRM3D(env=env, config=cfg)
        nodes, edges = bsrm.generate_roadmap()
        path, _ = bsrm.find_path(start, goal)
        bsrm_out = os.path.join(output_dir, f"{env_name}_BSRM3D_scene.png")
        image_paths.append(
            plot_scene_3d(
                env=env,
                nodes=nodes,
                edges=edges,
                path=path if path else None,
                start=start,
                goal=goal,
                title=f"{env_name} | BSRM3D",
                output_path=bsrm_out,
                obstacle_style=obstacle_style,
                obstacle_max_items=obstacle_max_items,
                max_edges_to_draw=max_edges_to_draw,
            )
        )

        for planner_name in ompl_planners:
            ok, _, _, ompl_nodes, ompl_edges, ompl_path = _solve_with_ompl_graph(
                env=env,
                start=start,
                goal=goal,
                planner_name=planner_name,
                timeout=ompl_timeout,
            )
            tag = "solved" if ok else "unsolved"
            file_name = f"{env_name}_{_safe_name(planner_name)}_scene.png"
            out = os.path.join(output_dir, file_name)
            image_paths.append(
                plot_scene_3d(
                    env=env,
                    nodes=ompl_nodes,
                    edges=ompl_edges,
                    path=ompl_path if ompl_path else None,
                    start=start,
                    goal=goal,
                    title=f"{env_name} | {planner_name} | {tag}",
                    output_path=out,
                    obstacle_style=obstacle_style,
                    obstacle_max_items=obstacle_max_items,
                    max_edges_to_draw=max_edges_to_draw,
                )
            )

    return image_paths
