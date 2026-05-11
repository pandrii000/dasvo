import tarfile
from pathlib import Path
from typing import Iterator

import fire
from tqdm import tqdm


IMAGE_DIR_NAME = "image_left"
DEPTH_DIR_NAME = "depth_left"
POSE_FILE_NAME = "pose_left.txt"


def is_kaggle_payload_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name == POSE_FILE_NAME:
        return True
    if path.parent.name == IMAGE_DIR_NAME and path.suffix.lower() == ".png":
        return True
    if path.parent.name == DEPTH_DIR_NAME and path.suffix.lower() == ".npy":
        return True
    return False


def iter_payload_files(data_dir: Path) -> Iterator[Path]:
    for path in data_dir.rglob("*"):
        if is_kaggle_payload_file(path):
            yield path


def count_payload_files(data_dir: Path) -> int:
    return sum(1 for _ in iter_payload_files(data_dir))


def create_kaggle_package(
    data_dir: str | Path = "assets/data/tartanair",
    output: str | Path = "dist/kaggle/dasvo-tartanair-rgbd-validation-split.tar",
    force: bool = False,
) -> None:
    """Create an uncompressed Kaggle tar from TartanAir image/depth/pose files."""
    data_dir = Path(data_dir)
    output = Path(output)

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    if output.exists() and not force:
        raise FileExistsError(f"Output already exists: {output}. Pass --force=True to overwrite.")

    total_files = count_payload_files(data_dir)
    if total_files == 0:
        raise ValueError(f"No image/depth/pose payload files found under {data_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    root_name = data_dir.name

    with tarfile.open(output, "w") as archive:
        for source_file in tqdm(iter_payload_files(data_dir), total=total_files, desc="Archiving files"):
            archive_name = Path(root_name) / source_file.relative_to(data_dir)
            archive.add(source_file, arcname=archive_name)

    print(f"Archived {total_files} files to {output}")


if __name__ == "__main__":
    fire.Fire(create_kaggle_package)
