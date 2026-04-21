"""
run_ablation.py — Enhancement ablation experiment.

Outputs:
  results/ablation/ablation_raw.csv
  results/ablation/ablation_table.{pdf,png}

Usage:
  python scripts/run_ablation.py --seeds 42 123 2026 --trials 10
"""
from __future__ import annotations
import argparse, math, os, sys, time, heapq
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bsrm3d.config import bsrm_config_for_env
from bsrm3d.environments.benchmark import build_benchmark_environment
from bsrm3d.planners.beam_bsrm3d import BeamBSRM3D
from bsrm3d.planners.base_planner import BasePlanner3D
from bsrm3d.viz.paper_style import apply_paper_style, save_fig
import matplotlib.pyplot as plt


def _pairs(env, n, seed):
    rng = np.random.default_rng(seed)
    s = env.sample_free(n*3, rng=rng, radius=0.15)
    g = env.sample_free(n*3, rng=rng, radius=0.15)
    md = 0.25 * env.bounds.diagonal
    return [(a,b) for a,b in zip(s,g) if np.linalg.norm(np.array(a)-np.array(b))>=md][:n]


def _find_path_pure(planner, start, goal):
    """Pure A* without any RRT-Connect fallback."""
    cfg = planner.config
    if not planner.env.is_free(start, radius=cfg.collision_radius): return [], math.inf
    if not planner.env.is_free(goal, radius=cfg.collision_radius): return [], math.inf
    if not planner.nodes: return [], math.inf
    arr, tree = planner._ensure_tree()
    qr = max(cfg.connection_radius, 2.0)
    def attach(q):
        kk = min(max(cfg.neighbor_k,4), len(planner.nodes))
        d,idx = tree.query(np.asarray(q), k=kk)
        d,idx = np.atleast_1d(d), np.atleast_1d(idx)
        keep = d <= qr
        if not keep.any(): return []
        ids,ds = idx[keep], d[keep]
        ok = np.array([planner.env.segment_is_free(q, planner.nodes[int(i)],
            radius=cfg.collision_radius, step=cfg.collision_step) for i in ids])
        return [(int(i),float(dd)) for i,dd,o in zip(ids,ds,ok) if o]
    sc, gc = attach(start), attach(goal)
    if not sc or not gc: return [], math.inf
    S,G = -1,-2; g_cost={S:0.0}; parent={}
    heap = [(planner.dist(start,goal), 0.0, S)]
    gm = {i:w for i,w in gc}
    while heap:
        _,gc2,cur = heapq.heappop(heap)
        if gc2 > g_cost.get(cur,math.inf): continue
        if cur == G:
            seq=[G]; x=G
            while x in parent: x=parent[x]; seq.append(x)
            seq.reverse()
            pts = [start if x==S else (goal if x==G else planner.nodes[x]) for x in seq]
            L = float(np.sum(np.linalg.norm(np.asarray(pts)[1:]-np.asarray(pts)[:-1],axis=1)))
            return pts, L
        cand = sc if cur==S else list(planner._adj.get(cur,{}).items())
        if cur!=S and cur!=G and cur in gm: cand = cand+[(G,gm[cur])]
        for nxt,w in cand:
            ng = gc2+w
            if ng >= g_cost.get(nxt,math.inf): continue
            g_cost[nxt]=ng; parent[nxt]=cur
            h = 0.0 if nxt==G else planner.dist(planner.nodes[nxt],goal)
            heapq.heappush(heap, (ng+h, ng, nxt))
    return [], math.inf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42,123,2026])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--benchmarks", nargs="+", default=["maze_hard","narrow_tight"])
    parser.add_argument("--output", default="results/ablation")
    args = parser.parse_args()

    # 3 configurations matching 3 paper contributions
    CONFIGS = [
        ("Base (2D-style NMS)",
         dict(disable_artic=True, disable_rrtc_bridge=True, use_pure=True)),
        ("+ Topology-Preserving NMS",
         dict(disable_artic=False, disable_rrtc_bridge=True, use_pure=True)),
        ("+ Hybrid Query (full)",
         dict(disable_artic=False, disable_rrtc_bridge=False, use_pure=False)),
    ]

    rows = []
    for env_name in args.benchmarks:
        for seed in args.seeds:
            env = build_benchmark_environment(env_name); env.seed = seed
            pairs = _pairs(env, args.trials, seed+7)
            for label, flags in CONFIGS:
                cfg = bsrm_config_for_env(env_name, seed=seed)
                p = BeamBSRM3D(env, cfg)
                if flags["disable_artic"]:
                    p._is_local_articulation = lambda *a,**kw: False
                if flags["disable_rrtc_bridge"]:
                    p._rrtc_bridge_components = lambda: None
                t0 = time.perf_counter()
                p.generate_roadmap()
                bt = time.perf_counter()-t0
                ok=0; plens=[]
                for s,g in pairs:
                    if flags["use_pure"]:
                        path, L = _find_path_pure(p, s, g)
                    else:
                        path, L = p.find_path(s, g)
                    if path: ok+=1; plens.append(L)
                rows.append(dict(env=env_name, seed=seed, config=label,
                    success=ok, total=len(pairs),
                    rate=ok/max(1,len(pairs))*100,
                    build_time=bt, nodes=len(p.nodes)))
        sub = pd.DataFrame([r for r in rows if r["env"]==env_name])
        g = sub.groupby("config").agg(succ=("success","sum"), total=("total","sum"),
            build=("build_time","mean"), N=("nodes","mean")).reset_index()
        g["rate"] = (g.succ/g.total*100).round(1)
        print(f"\n{env_name}:")
        for _,row in g.iterrows():
            print(f"  {row.config:30s}: {row.rate:5.1f}%  N={row.N:.0f}  build={row.build:.2f}s")

    df = pd.DataFrame(rows)
    os.makedirs(args.output, exist_ok=True)
    df.to_csv(os.path.join(args.output, "ablation_raw.csv"), index=False)
    print(f"\nSaved: {args.output}/ablation_raw.csv")

    # Plot
    apply_paper_style()
    agg = df.groupby(["env","config"]).agg(rate=("rate","mean"), build=("build_time","mean"),
        N=("nodes","mean")).reset_index()
    configs = [c[0] for c in CONFIGS]
    colors_bar = ["#bdc3c7","#3498db","#e74c3c"]
    envs = args.benchmarks
    fig, axes = plt.subplots(1, len(envs), figsize=(4.5*len(envs), 2.5))
    if len(envs)==1: axes=[axes]
    for ax, en in zip(axes, envs):
        sub = agg[agg.env==en]
        y = np.arange(len(configs))
        vals = [float(sub[sub.config==c].rate.iloc[0]) if not sub[sub.config==c].empty else 0
                for c in configs]
        bars = ax.barh(y, vals, color=colors_bar, edgecolor="white", height=0.55)
        for bar,v in zip(bars,vals):
            ax.text(min(bar.get_width()+1.5, 105), bar.get_y()+bar.get_height()/2,
                    f"{v:.0f}%", va="center", fontsize=8)
        ax.set_yticks(y); ax.set_yticklabels(configs, fontsize=7.5)
        ax.set_xlabel("Success Rate (%)")
        ax.set_title(en.replace("_","\\_"))
        ax.set_xlim(0, 115)
    plt.tight_layout()
    save_fig(fig, os.path.join(args.output, "ablation_table.png"))
    print(f"Saved: {args.output}/ablation_table.png + .pdf")


if __name__ == "__main__":
    main()
