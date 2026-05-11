.PHONY: download link process evaluate evaluate-force assets revision all

download:
	uv run python scripts/download.py

process:
	uv run python scripts/process.py

evaluate:
	uv run python scripts/evaluate.py

evaluate-force:
	uv run python scripts/evaluate.py --overwrite=True

assets:
	uv run python scripts/assets.py
	uv run python scripts/publication/dasvo_tartanair_kaggle_eda.py

revision: evaluate-force
	uv run python scripts/publication/wave1_revision_metrics.py

test:
	uv run pytest tests/

all: download process evaluate assets
