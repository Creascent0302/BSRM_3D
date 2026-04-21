# BSRM-3D v2: Integrated Beam-Sketch Roadmap for 3D Motion Planning

## Architecture

Single integrated pipeline — expand, connect, and thin in one pass:

```
Beam Expand + Inline Edges + Greedy Thinning
├── Seed placement (Halton quasi-random)
├── Beam expansion loop:
│   ├── Cast beam fan (Fibonacci sphere directions)
│   ├── Depth discontinuity detection → candidate generation
│   ├── Validate segment (origin→candidate)
│   ├── Record parent→child edge (FREE, already validated)
│   └── Connect to 4 nearest existing nodes (lateral edges)
├── Connectivity repair (chain-expand unexpanded frontier nodes)
└── Greedy thinning (connectivity-safe spatial suppression)
```

### Key Innovations (v1 → v2)

1. **Inline edge building**: parent→child edge is free (segment already checked). 4 lateral edges per node. Eliminates separate knn adjacency build phase.
2. **Greedy thinning replaces NMS**: Simple spatial suppression preserving bridge nodes. Eliminates complex NMS + unsuppression + coverage guards.
3. **Adaptive min_spacing**: `diagonal × 0.02`. Large environments use coarse spacing; small environments use fine spacing.
4. **collision_radius = 0**: Beam ray-cast and segment validation use the same radius, eliminating beam-sees-but-segment-rejects mismatch.
5. **Distance-based expansion priority**: Nodes far from already-expanded sources go first, reducing beam overlap.

## Installation

```bash
pip install numpy scipy matplotlib
```

## Quick Start

```python
from bsrm3d.config import Planner3DConfig
from bsrm3d.environments.benchmark import build_benchmark_environment
from bsrm3d.planners.beam_bsrm3d import BeamBSRM3D

# Build environment and planner
env = build_benchmark_environment('maze_hard')
cfg = Planner3DConfig(max_nodes=2000, connection_radius=2.0)
planner = BeamBSRM3D(env, cfg)

# Generate roadmap (single call does everything)
nodes, edges = planner.generate_roadmap()

# Multi-query path finding (reuses roadmap)
path, length = planner.find_path(start=(1,1,1), goal=(9,9,5))
print(f"Path length: {length:.2f}, waypoints: {len(path)}")
```

## Running Benchmarks

### Quick smoke test (all environments, 3 seeds):

```bash
python -c "
from bsrm3d.config import Planner3DConfig
from bsrm3d.environments.benchmark import build_benchmark_environment, available_benchmarks
from bsrm3d.planners.beam_bsrm3d import BeamBSRM3D
import numpy as np, time

for en in available_benchmarks():
    e = build_benchmark_environment(en)
    c = Planner3DConfig(max_nodes=2000 if 'floor' not in en else 3000,
                        connection_radius=2.0 if 'floor' not in en else 3.0)
    t0 = time.perf_counter()
    p = BeamBSRM3D(e, c); p.generate_roadmap()
    bt = time.perf_counter() - t0
    rng = np.random.default_rng(42)
    S = e.sample_free(20, rng=rng); G = e.sample_free(20, rng=rng)
    ok = sum(1 for a,b in zip(S[:10],G[:10]) if p.find_path(a,b)[0])
    print(f'  {en:17s}: {ok}/10  N={len(p.nodes):4d}  build={bt:.1f}s')
"
```

### Full sweep benchmark (multiple seeds, node budgets, baselines):

```bash
python scripts/run_sweep.py --seeds 42 123 2026 --trials 10
```

### Baseline comparison:

```bash
python -c "
from bsrm3d.config import Planner3DConfig
from bsrm3d.environments.benchmark import build_benchmark_environment
from bsrm3d.planners.beam_bsrm3d import BeamBSRM3D
from bsrm3d.planners.baselines import HaltonPRM, DeltaPRM, SPARS2
import numpy as np, time

en = 'maze_hard'
for Cls, name in [(None,'BSRM3D'),(HaltonPRM,'Halton'),(DeltaPRM,'Delta'),(SPARS2,'SPARS2')]:
    e = build_benchmark_environment(en); e.seed=42
    c = Planner3DConfig(seed=42, max_nodes=2000, connection_radius=2.0)
    t0 = time.perf_counter()
    if Cls is None:
        p = BeamBSRM3D(e, c); p.generate_roadmap()
    else:
        p = Cls(e, c, 2000); p.generate_roadmap()
    bt = time.perf_counter() - t0
    rng = np.random.default_rng(49)
    S = e.sample_free(20, rng=rng); G = e.sample_free(20, rng=rng)
    ok = sum(1 for a,b in zip(S[:10],G[:10]) if p.find_path(a,b)[0])
    print(f'  {name:10s}: {ok}/10  N={len(p.nodes):4d}  build={bt:.2f}s')
"
```

## Available Environments

| Name | Description | Size | Difficulty |
|------|-------------|------|------------|
| `cluttered` | Random box obstacles | 10×10×6m | Easy |
| `indoor` | Room with furniture | 10×10×6m | Easy |
| `bugtrap` | U-shaped trap | 10×10×6m | Medium |
| `forest` | Dense cylindrical obstacles | 10×10×6m | Medium |
| `narrow_tight` | Tight narrow passage | 10×10×6m | Medium |
| `maze3d` | 3D maze with vertical passages | 10×10×6m | Hard |
| `maze_hard` | Dense 3D maze | 10×10×6m | Hard |
| `building_3floor` | 3-story building with stairwells | 30×30×12m | Hard |

## Benchmark Results

| Environment | Success | Nodes | Build | Query | v1 Speedup |
|------------|---------|-------|-------|-------|-----------|
| building_3floor | 100% | 498 | 1.6s | 1.6ms | 5.3× |
| maze_hard | 100% | 327 | 0.8s | 1.3ms | 4.3× |
| cluttered | 100% | 305 | 0.8s | 0.6ms | 2.9× |
| narrow_tight | 100% | 310 | 0.8s | 0.9ms | 2.3× |
| bugtrap | 100% | 312 | 0.8s | 0.7ms | 2.5× |
| maze3d | 100% | 325 | 0.9s | 1.4ms | 2.8× |
| forest | 100% | 298 | 0.9s | 0.9ms | 3.2× |
| indoor | 100% | 318 | 0.8s | 0.9ms | 2.1× |

## Project Structure

```
bsrm3d/
  config.py                    # Planner3DConfig
  types.py                     # Point3D, Edge3D, Path3D
  environments/
    base.py                    # Environment3D interface
    voxel_env.py               # VoxelEnvironment3D
    benchmark.py               # 8 benchmark environments
  planners/
    base_planner.py            # A* query, start/goal attachment
    beam_bsrm3d.py             # BeamBSRM3D (main algorithm)
    baselines.py               # HaltonPRM, DeltaPRM, SPARS2
  sampling/
    strategies.py              # FibonacciDirectionSampler
scripts/
  run_sweep.py                 # Node-count sweep benchmark
  run_full_benchmark.py        # Full comparison benchmark
```
