# Depth-Assisted Sparse Visual Odometry

This is a `uv`-managed Python project. The project dependencies, Python version
constraints, and development tooling are defined in `pyproject.toml`.

## Setup

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create/update the local environment from `pyproject.toml`:

```bash
uv sync
```

## Running the Pipeline

Run all project commands through `uv` so they use the locked project environment. The pipeline consists of four main steps.

### 1. Download Dataset

Download the TartanAir sequences defined in `configs/default.yaml`. By default, this downloads the sequences listed in `tartan_test.txt` to `assets/data/tartanair`.

```bash
uv run python scripts/download.py
```

Options:
- `--num_workers`: Number of parallel download workers (default: 4)

### 2. Process Sequences (Visual Odometry)

Run the VO pipeline (frontend + backend) on the downloaded sequences. This will generate estimated trajectories in `assets/outputs/`.

```bash
uv run python scripts/process.py
```

Options:
- `--num_workers`: Number of parallel processing workers (default: 1)
- `--max_virtual_frames`: Limit the number of frames processed per sequence (useful for quick testing)

### 3. Evaluate Trajectories

Evaluate the estimated trajectories against the ground truth to compute Absolute Trajectory Error (ATE) and Relative Pose Error (RPE). This writes JSON result files to `assets/tables/`.

```bash
uv run python scripts/evaluate.py
```

Options:
- `--num_workers`: Number of parallel evaluation workers (default: 1)

### 4. Generate Assets

Aggregate the evaluation results into a single CSV and Markdown table for publication.

```bash
uv run python scripts/assets.py
```

This writes:
- `assets/tables/aggregated_results.csv`
- `assets/tables/aggregated_results.md`

You can then inject these tables directly into your publication.

### Run All Steps

You can run the entire pipeline sequentially using `make`:

```bash
make all
```

### Revision Round 2 (Wave 1 post-processing)

After trajectories are already produced (i.e. `make process` has been run),
post-processing target re-aggregates per-sequence ATE/RPE
from the saved trajectories under the current `dasvo/evaluation.py` logic and
regenerates the detailed paper tables with normalised ATE (\% of GT trajectory
length), per-cell coverage (`n_finite/32`), and Wilcoxon p-values with
Holm--Bonferroni correction:

```bash
make revision
```

It writes/refreshes the following LaTeX assets in the paper tree:

- `tables/vo_metrics_summary_detailed_ate.tex` (absolute + normalised ATE)
- `tables/vo_metrics_summary_detailed_rpe.tex` (RPE)
- `tables/wave1_wilcoxon_holm.tex` (paired Wilcoxon with Holm correction)

This is fast (single-digit seconds) and safe to rerun; it does not touch the
visual odometry processing stage.

## Configuration

The pipeline is configured via `configs/default.yaml`. You can override the configuration file by setting the `DASVO_CONFIG_FILE` environment variable:

```bash
DASVO_CONFIG_FILE=configs/my_config.yaml uv run python scripts/process.py
```

## Tests

Run the test suite using `pytest`:

```bash
uv run pytest tests/
```
