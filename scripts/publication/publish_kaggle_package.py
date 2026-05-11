import argparse
import json
from pathlib import Path

import kagglehub

from create_kaggle_package import create_kaggle_package


def build_handle(handle: str | None, owner_slug: str | None, dataset_slug: str) -> str:
    if handle:
        return handle
    if not owner_slug:
        raise ValueError("Provide either --handle or both --owner-slug and --dataset-slug.")
    return f"{owner_slug}/{dataset_slug}"


def parse_ignore_patterns(patterns: list[str], patterns_file: Path | None) -> list[str]:
    parsed = list(patterns)
    if patterns_file is not None:
        loaded = json.loads(patterns_file.read_text(encoding="utf-8"))
        if not isinstance(loaded, list) or not all(isinstance(item, str) for item in loaded):
            raise ValueError(f"Ignore pattern file must contain a JSON list of strings: {patterns_file}")
        parsed.extend(loaded)
    return parsed


def publish_kaggle_package(
    handle: str,
    data_dir: Path,
    staging_dir: Path,
    output_file: Path | None,
    title: str,
    dataset_slug: str,
    owner_slug: str | None,
    notebook_file: Path | None,
    version_notes: str | None,
    ignore_patterns: list[str],
    force: bool,
    skip_archive: bool,
    skip_stage: bool,
    max_sequences: int | None,
) -> None:
    if not skip_stage:
        create_kaggle_package(
            data_dir=data_dir,
            staging_dir=staging_dir,
            output_file=None if skip_archive else output_file,
            title=title,
            dataset_slug=dataset_slug,
            owner_slug=owner_slug,
            notebook_file=notebook_file,
            dry_run=False,
            force=force,
            max_sequences=max_sequences,
        )

    if not staging_dir.exists():
        raise FileNotFoundError(f"Staging directory does not exist: {staging_dir}")

    upload_kwargs = {}
    if version_notes:
        upload_kwargs["version_notes"] = version_notes
    if ignore_patterns:
        upload_kwargs["ignore_patterns"] = ignore_patterns

    print(f"Uploading Kaggle dataset from {staging_dir} to {handle}")
    if ignore_patterns:
        print(f"Ignore patterns: {ignore_patterns}")
    if version_notes:
        print(f"Version notes: {version_notes}")

    kagglehub.dataset_upload(handle, str(staging_dir), **upload_kwargs)
    print("Upload complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Stage the DASVO Kaggle package and upload it with kagglehub.dataset_upload()."
        )
    )
    parser.add_argument(
        "--handle",
        default=None,
        help="Full Kaggle handle in the form <username>/<dataset-slug>.",
    )
    parser.add_argument(
        "--owner-slug",
        default=None,
        help="Kaggle username. Used together with --dataset-slug when --handle is omitted.",
    )
    parser.add_argument(
        "--dataset-slug",
        default="dasvo-tartanair-rgbd-validation-split",
        help="Kaggle dataset slug.",
    )
    parser.add_argument(
        "--title",
        default="DASVO TartanAir RGB-D Validation Split",
        help="Human-readable dataset title for staged metadata.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("assets/data/tartanair"),
        help="Root TartanAir data directory.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path("dist/kaggle/dasvo-tartanair-rgbd-validation-split"),
        help="Directory containing the Kaggle-ready dataset payload.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/kaggle/dasvo-tartanair-rgbd-validation-split.tar.gz"),
        help="Optional archive path passed to the staging step.",
    )
    parser.add_argument(
        "--skip-archive",
        action="store_true",
        help="Skip archive generation during staging.",
    )
    parser.add_argument(
        "--skip-stage",
        action="store_true",
        help="Upload the existing staging directory without rebuilding it first.",
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("scripts/publication/dasvo_tartanair_kaggle_eda.ipynb"),
        help="Notebook to copy into the staged package.",
    )
    parser.add_argument(
        "--version-notes",
        default=None,
        help="Optional Kaggle dataset version notes.",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=[],
        help="Extra ignore pattern to pass to kagglehub.dataset_upload(). Repeat as needed.",
    )
    parser.add_argument(
        "--ignore-patterns-file",
        type=Path,
        default=None,
        help="JSON file containing a list of extra ignore patterns.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the staging directory and archive during rebuild.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Optional limit for smoke tests.",
    )

    args = parser.parse_args()
    publish_kaggle_package(
        handle=build_handle(args.handle, args.owner_slug, args.dataset_slug),
        data_dir=args.data_dir,
        staging_dir=args.staging_dir,
        output_file=args.output,
        title=args.title,
        dataset_slug=args.dataset_slug,
        owner_slug=args.owner_slug,
        notebook_file=args.notebook,
        version_notes=args.version_notes,
        ignore_patterns=parse_ignore_patterns(args.ignore_pattern, args.ignore_patterns_file),
        force=args.force,
        skip_archive=args.skip_archive,
        skip_stage=args.skip_stage,
        max_sequences=args.max_sequences,
    )
