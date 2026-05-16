.PHONY: download link process evaluate evaluate-force assets revision all help

# Canonical pipeline:
#   make download   # one-time, fetch TartanAir split
#   make process    # heavy, produces *_traj.txt + *_pose.txt + *_done.txt per (fe, be, deg, stride, seq)
#   make evaluate   # cheap, idempotent, writes per-sequence *_results.json
#   make assets     # aggregates and emits compact LaTeX tables + EDA + figures
#   make revision   # emits detailed peer-review tables (CIs, normalised ATE, Wilcoxon+Holm)
#   make all        # download + process + evaluate + assets + revision

help:
	@echo "Targets: download, link, process, evaluate, evaluate-force, assets, revision, all"
	@echo "Typical sequence: make process && make evaluate && make assets && make revision"

download:
	uv run python scripts/download.py

process:
	uv run python scripts/process.py

# evaluate is idempotent: per-cell marker files (_eval_done_v2.txt) skip already-evaluated cells.
# Bump the marker name in scripts/evaluate.py whenever the per-sequence JSON schema changes
# so existing JSONs are regenerated on the next plain `make evaluate`.
evaluate:
	uv run python scripts/evaluate.py

# Escape hatch: force every cell to be re-evaluated regardless of marker state.
evaluate-force:
	uv run python scripts/evaluate.py --overwrite=True

# Compact paper tables + EDA + figures + comment placeholders.
assets:
	uv run python scripts/assets.py
	uv run python scripts/publication/dasvo_tartanair_kaggle_eda.py

# Detailed peer-review tables: normalised ATE, Sim(3) ATE, rotational RPE, Wilcoxon+Holm.
# Depends only on evaluate (not evaluate-force) because evaluate is now idempotent and
# its marker scheme guarantees JSONs match the current schema.
revision: evaluate
	uv run python scripts/publication/wave1_revision_metrics.py

all: download process evaluate assets revision
