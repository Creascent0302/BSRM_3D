# BSRM 3D + OMPL (Python Bindings Only)

This project is now intentionally simplified to one experiment pipeline:

1. your BSRM3D planner implementation,
2. OMPL classic geometric planners through OMPL Python bindings,
3. unified benchmark CSV and OMPL-focused comparison plots.

All non-essential visualization/demo paths (Open3D and fallback rendering) were removed.

## Minimal Project Structure

```text
BSRM_3D/
	bsrm3d/
		config.py
		types.py
		environments/
			base.py
			voxel_env.py
			benchmark.py
		sampling/
			fibonacci_sphere.py
			strategies.py
		planners/
			base_planner.py
			beam_bsrm3d.py
		ompl_integration/
			runner.py
			visualization.py
	scripts/
		run_ompl_benchmark.py
	requirements.txt
```

## What Is Kept

1. BSRM3D core algorithm (frontier expansion + beam tracing + roadmap query).
2. 3D benchmark environment construction used by both BSRM3D and OMPL validity checks.
3. OMPL Python planner comparison (RRT, RRTConnect, PRM, BITstar).
4. Comparison visualization dedicated to OMPL experiment outputs.

## OMPL Python Installation

Use your preferred method for OMPL Python bindings. Recommended:

```bash
conda install -c conda-forge ompl
```

Then install Python dependencies in this repo:

```bash
pip install -r requirements.txt
```

## Run Unified Benchmark

```bash
python scripts/run_ompl_benchmark.py \
	--nodes 1200 \
	--seed 42 \
	--trials 20 \
	--timeout 2.0 \
	--ompl-planners RRT RRTConnect PRM BITstar
```

Save one 3D scene/roadmap image for each environment and planner:

```bash
python scripts/run_ompl_benchmark.py \
	--nodes 1200 \
	--seed 42 \
	--trials 20 \
	--timeout 2.0 \
	--ompl-planners RRT RRTConnect PRM BITstar \
	--save-scenes \
	--scene-output-dir results/ompl/scenes \
	--scene-obstacle-style solid \
	--scene-obstacle-max 6000 \
	--scene-max-edges 9000
```

Scene style options:

1. `--scene-obstacle-style solid`: render obstacles as solid voxels (default).
2. `--scene-obstacle-style points`: render obstacles as point cloud.
3. `--scene-obstacle-max`: cap solid voxels for faster rendering on dense maps.
4. `--scene-max-edges`: downsample very dense roadmap edges to keep structure readable.

## Outputs

```text
results/ompl/unified_benchmark.csv
results/ompl/comparison_success_rate.png
results/ompl/comparison_solve_time.png
results/ompl/scenes/*.png
```

## Notes on Timing

For BSRM3D, roadmap build time is amortized over the trial count and added to each query time.
This makes per-trial timing more comparable to OMPL planners that solve per trial.

## 补充
还没怎么仔细看，目前直接运行的终端命令是python scripts/run_ompl_benchmark.py --nodes 1200 --seed 42 --trials 20 --timeout 2.0 --ompl-planners RRT RRTConnect PRM BITstar --save-scenes --scene-output-dir results/ompl/scenes --scene-obstacle-style solid --scene-obstacle-max 6000 --scene-max-edges 9000