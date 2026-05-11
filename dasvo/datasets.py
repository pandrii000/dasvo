from __future__ import annotations
from typing import List

from dataclasses import dataclass
from pathlib import Path

from dasvo.settings import PROJECT_ROOT, SEQUENCES_FILE, DATA_ROOT


@dataclass(frozen=True)
class TartanAirSequence:
    root: Path
    env: str
    difficulty: str
    trajectory: str

    @property
    def sequence_id(self) -> str:
        return f"{self.env}_{self.difficulty.lower()}_{self.trajectory.lower()}"

    @property
    def data_path(self) -> Path:
        return self.root.joinpath(self.env, self.difficulty, self.trajectory)

    @property
    def images_path(self) -> Path:
        return self.data_path.joinpath("image_left")

    @property
    def depth_path(self) -> Path:
        return self.data_path.joinpath("depth_left")

    @property
    def pose_path(self) -> Path:
        return self.data_path.joinpath("pose_left.txt")

    @property
    def archive_dir(self) -> Path:
        return self.root.joinpath(self.env, self.difficulty)

    @property
    def image_left_zip_path(self) -> Path:
        return self.archive_dir.joinpath("image_left.zip")

    @property
    def depth_left_zip_path(self) -> Path:
        return self.archive_dir.joinpath("depth_left.zip")

    def with_root(self, root: Path) -> TartanAirSequence:
        return TartanAirSequence(Path(root), self.env, self.difficulty, self.trajectory)

    def images_exist(self) -> bool:
        return self.images_path.exists()

    def depth_exist(self) -> bool:
        return self.depth_path.exists()

    def pose_exist(self) -> bool:
        return self.pose_path.exists()

    def set_image_left_zip_downloaded(self) -> None:
        self._write_marker(self.archive_dir.joinpath("image_downloaded.txt"))

    def is_image_left_zip_downloaded(self) -> bool:
        return self.archive_dir.joinpath("image_downloaded.txt").is_file()

    def set_image_left_zip_extracted(self) -> None:
        self._write_marker(self.archive_dir.joinpath("image_extracted.txt"))

    def is_image_left_zip_extracted(self) -> bool:
        return self.archive_dir.joinpath("image_extracted.txt").is_file()

    def set_depth_left_zip_downloaded(self) -> None:
        self._write_marker(self.archive_dir.joinpath("depth_downloaded.txt"))

    def is_depth_left_zip_downloaded(self) -> bool:
        return self.archive_dir.joinpath("depth_downloaded.txt").is_file()

    def set_depth_left_zip_extracted(self) -> None:
        self._write_marker(self.archive_dir.joinpath("depth_extracted.txt"))

    def is_depth_left_zip_extracted(self) -> bool:
        return self.archive_dir.joinpath("depth_extracted.txt").is_file()

    def set_pose_left_downloaded(self) -> None:
        self._write_marker(self.data_path.joinpath("pose_downloaded.txt"))

    def is_pose_left_downloaded(self) -> bool:
        return self.data_path.joinpath("pose_downloaded.txt").is_file()

    def _write_marker(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    
    def __str__(self) -> str:
        return f"TartanAirSequence(sequence_id={self.sequence_id}, data_path={self.data_path}, images_path={self.images_path}, depth_path={self.depth_path}, pose_path={self.pose_path})"


def list_sequences() -> List[TartanAirSequence]:
    lines = SEQUENCES_FILE.read_text().splitlines()
    sequences: List[TartanAirSequence] = []
    for line in lines:
        env, difficulty, trajectory = line.strip().split("/")
        sequences.append(TartanAirSequence(DATA_ROOT, env, difficulty, trajectory))
    return sequences
