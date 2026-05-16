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

Run all project commands through `uv` so they use the locked project environment.
The canonical order is `download` → `process` → `evaluate` → `assets` → `revision`,
all chained by `make all`. Each stage is described below.

### 1. Download Dataset

Download the TartanAir sequences defined in `configs/default.yaml`. By default,
this downloads the sequences listed in `tartan_dry.txt` to `assets/data/tartanair`.

```bash
make download   # or: uv run python scripts/download.py
```

Options:
- `--num_workers`: Number of parallel download workers (default: 4)

### 2. Process Sequences (Visual Odometry)

Run the VO pipeline (frontend + backend) on the downloaded sequences. For every
`(frontend, backend, degradation, frame_stride, sequence)` cell, this writes:

- `<seq>_traj.txt` — translation trajectory (Nx3)
- `<seq>_pose.txt` — full pose trajectory in TUM format (`Nx7`: `tx ty tz qx qy qz qw`)
- `<seq>_done.txt` — completion marker

A cell is re-processed only when either the marker or the pose sidecar is missing,
so adding a new derived metric that needs more pose data is a matter of removing
the sidecar (or bumping its name) — partial runs cannot leave the cell in a
half-state.

```bash
make process    # uses scripts/process.py
```

A multi-pipeline variant amortises image loading and degradation across all cells
sharing a physical frame and is much faster when several pipelines are pending:

```bash
uv run python scripts/process-fast.py
```

Options for both:
- `--num_workers`: Number of parallel processing workers (default: 1)
- `--max_virtual_frames`: Limit the number of frames processed per sequence
  (useful for quick testing)

### 3. Evaluate Trajectories

Evaluate the estimated trajectories against ground truth and write per-sequence
JSON result files to `assets/tables/<fe>/<be>/<deg>/<stride>/<seq>_results.json`.
Each JSON contains:

- `ate` — ATE under the alignment matching the backend (SE(3) for PnP, Sim(3) for Essential)
- `ate_sim3` — ATE under Sim(3) alignment for both backends (matches `ate` for Essential by design)
- `rpe` — translational RPE (m/frame)
- `rpe_rot` — rotational RPE (deg/frame), computed from `<seq>_pose.txt` when present
- `rpe_rot_available` — whether the pose sidecar was available for this cell
- alignment metadata, frame count, failure flag

```bash
make evaluate
```

The target is idempotent: per-cell `_eval_done_v2.txt` markers skip cells whose
JSON already matches the current schema. **If the per-sequence JSON schema
changes**, bump the marker filename in `scripts/evaluate.py` so the old JSONs
are regenerated on the next plain `make evaluate`. The escape hatch
`make evaluate-force` re-runs every cell regardless of marker state.

Options:
- `--num_workers`: Number of parallel evaluation workers (default: 1)

### 4. Generate Assets

Aggregate per-sequence JSONs into a single CSV plus compact LaTeX tables, render
publication figures, and run the Kaggle EDA notebook script.

```bash
make assets
```

This writes (in `assets/tables/` and mirrors into the paper tree):

- `aggregated_results.csv` / `.md` — per-cell mean/std/median/IQR/CI for ATE,
  ATE-Sim(3), translational RPE, rotational RPE
- `vo_metrics_summary.tex` — compact baseline ATE table (frontend × backend × stride)
- `vo_metrics_summary_ate_sim3.tex` — PnP-only ATE under Sim(3) alignment, isolating
  trajectory-shape error from metric-scale drift
- `vo_metrics_by_sequence.tex` — compact baseline translational RPE table
- `vo_metrics_rotational_rpe.tex` — compact baseline rotational RPE table
- `robustness_metrics.tex` — ATE at stride 4 across degradations
- figures under `assets/figures/`

### 5. Revision (peer-review detailed tables)

`make revision` re-aggregates per-sequence JSONs and emits the detailed
peer-review tables with median/IQR, 95% bootstrap CIs, per-cell coverage
(`n_finite/32`), normalised ATE (% of GT trajectory length), and paired Wilcoxon
tests with Holm–Bonferroni correction:

```bash
make revision
```

LaTeX assets written into the paper tree:

- `tables/vo_metrics_summary_detailed_ate.tex` — absolute + normalised ATE
- `tables/vo_metrics_summary_detailed_rpe.tex` — translational RPE
- `tables/vo_metrics_summary_detailed_ate_sim3.tex` — SE(3) vs Sim(3) PnP ATE side-by-side
- `tables/vo_metrics_summary_detailed_rpe_rot.tex` — rotational RPE
- `tables/wave1_wilcoxon_holm.tex` — paired Wilcoxon with Holm correction

Single-digit seconds to run, safe to rerun, and does not touch processing. The
target depends on `evaluate` only (not `evaluate-force`), because the evaluate
marker scheme already keeps JSONs in sync with the current schema.

### Run All Steps

```bash
make all   # download → process → evaluate → assets → revision
```

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
