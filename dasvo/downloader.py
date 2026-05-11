from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

from tqdm import tqdm
from loguru import logger

from dasvo.datasets import TartanAirSequence


class TartanAirDownloader:
    HF_BASE_URL = "https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main"
    S3_ENDPOINT_URL = (
        "https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/"
        "AUTH_ac8533a83cff4d48bc8c608ad222d330"
    )
    S3_BUCKET = "tartanair"
    IMAGE_DIR = "image_left"
    DEPTH_DIR = "depth_left"
    POSE_FILE = "pose_left.txt"

    def __init__(
        self,
        source: str = "hf",
        root: Path | str | None = None,
        progress: bool = True,
        chunk_size: int = 16 * 1024 * 1024,
    ) -> None:
        source = source.lower()
        if source in {"huggingface", "hf"}:
            self.source = "hf"
        elif source in {"s3", "cmu"}:
            self.source = "s3"
        else:
            raise ValueError("source must be 'hf' or 's3'")

        self.root = Path(root).expanduser() if root is not None else None
        self.progress = progress
        self.chunk_size = chunk_size
        self._s3_client = None

    def download(
        self,
        sequences: TartanAirSequence | Iterable[TartanAirSequence],
        *,
        extract: bool = True,
        clean: bool = False,
        overwrite: bool = False,
    ) -> None:
        assert extract or not clean, "clean=True requires extract=True"

        logger.info(f"Normalizing sequences: {sequences}")
        seqs = self._normalize_sequences(sequences)
        root = self._target_root(seqs)
        seqs = [seq.with_root(root) for seq in seqs]
        env, difficulty = self._assert_same_environment(seqs)
        base_dir = root / env / difficulty
        batch_seq = seqs[0]
        trajectories = {seq.trajectory for seq in seqs}

        image_key = f"{env}/{difficulty}/{batch_seq.image_left_zip_path.name}"
        images_ready = self._data_dirs_ready(root, seqs, self.IMAGE_DIR)
        if images_ready and not batch_seq.is_image_left_zip_extracted():
            batch_seq.set_image_left_zip_extracted()

        if not (extract and images_ready and not overwrite):
            image_archive_ready = self._archive_ready(batch_seq.image_left_zip_path)
            if image_archive_ready and not batch_seq.is_image_left_zip_downloaded():
                batch_seq.set_image_left_zip_downloaded()

            if overwrite or not image_archive_ready:
                logger.info(f"Downloading archive: {image_key} to {batch_seq.image_left_zip_path}")
                self._download_file(
                    image_key,
                    batch_seq.image_left_zip_path,
                    overwrite=overwrite or batch_seq.image_left_zip_path.exists(),
                )
                if not self._archive_ready(batch_seq.image_left_zip_path):
                    raise ValueError(f"Downloaded archive is missing or corrupted: {batch_seq.image_left_zip_path}")
                batch_seq.set_image_left_zip_downloaded()

            if extract:
                logger.info(f"Extracting archive: {image_key} to {root}")
                self._extract_archive(
                    batch_seq.image_left_zip_path,
                    base_dir,
                    mark_extracted=batch_seq.set_image_left_zip_extracted,
                    env=env,
                    difficulty=difficulty,
                    trajectories=trajectories,
                    overwrite=overwrite or not images_ready,
                )
                if not self._data_dirs_ready(root, seqs, self.IMAGE_DIR):
                    raise ValueError(f"Archive did not produce non-empty {self.IMAGE_DIR} directories")
                if clean and batch_seq.image_left_zip_path.exists():
                    batch_seq.image_left_zip_path.unlink()

        depth_key = f"{env}/{difficulty}/{batch_seq.depth_left_zip_path.name}"
        depth_ready = self._data_dirs_ready(root, seqs, self.DEPTH_DIR)
        if depth_ready and not batch_seq.is_depth_left_zip_extracted():
            batch_seq.set_depth_left_zip_extracted()

        if not (extract and depth_ready and not overwrite):
            depth_archive_ready = self._archive_ready(batch_seq.depth_left_zip_path)
            if depth_archive_ready and not batch_seq.is_depth_left_zip_downloaded():
                batch_seq.set_depth_left_zip_downloaded()

            if overwrite or not depth_archive_ready:
                logger.info(f"Downloading archive: {depth_key} to {batch_seq.depth_left_zip_path}")
                self._download_file(
                    depth_key,
                    batch_seq.depth_left_zip_path,
                    overwrite=overwrite or batch_seq.depth_left_zip_path.exists(),
                )
                if not self._archive_ready(batch_seq.depth_left_zip_path):
                    raise ValueError(f"Downloaded archive is missing or corrupted: {batch_seq.depth_left_zip_path}")
                batch_seq.set_depth_left_zip_downloaded()

            if extract:
                logger.info(f"Extracting archive: {depth_key} to {root}")
                self._extract_archive(
                    batch_seq.depth_left_zip_path,
                    base_dir,
                    mark_extracted=batch_seq.set_depth_left_zip_extracted,
                    env=env,
                    difficulty=difficulty,
                    trajectories=trajectories,
                    overwrite=overwrite or not depth_ready,
                )
                if not self._data_dirs_ready(root, seqs, self.DEPTH_DIR):
                    raise ValueError(f"Archive did not produce non-empty {self.DEPTH_DIR} directories")
                if clean and batch_seq.depth_left_zip_path.exists():
                    batch_seq.depth_left_zip_path.unlink()

        logger.info(f"Downloading poses for sequences: {seqs}")
        for seq in seqs:
            key = f"{seq.env}/{seq.difficulty}/{seq.trajectory}/{self.POSE_FILE}"
            if self._file_ready(seq.pose_path) and not seq.is_pose_left_downloaded():
                seq.set_pose_left_downloaded()
            if overwrite or not (seq.is_pose_left_downloaded() and self._file_ready(seq.pose_path)):
                logger.info(f"Downloading pose: {key} to {seq.pose_path}")
                self._download_file(key, seq.pose_path, overwrite=overwrite)
                if not self._file_ready(seq.pose_path):
                    raise ValueError(f"Downloaded pose is missing or empty: {seq.pose_path}")
                seq.set_pose_left_downloaded()

    def _normalize_sequences(
        self, sequences: TartanAirSequence | Iterable[TartanAirSequence]
    ) -> list[TartanAirSequence]:
        if isinstance(sequences, TartanAirSequence):
            seqs = [sequences]
        else:
            seqs = list(sequences)

        if not seqs:
            raise ValueError("sequences must not be empty")
        if not all(isinstance(seq, TartanAirSequence) for seq in seqs):
            raise TypeError("sequences must contain TartanAirSequence objects")
        return seqs

    def _target_root(self, seqs: list[TartanAirSequence]) -> Path:
        if self.root is not None:
            return self.root

        roots = {seq.root for seq in seqs}
        if len(roots) != 1:
            raise ValueError("all sequences must share one root unless downloader.root is set")
        return seqs[0].root

    def _assert_same_environment(self, seqs: list[TartanAirSequence]) -> tuple[str, str]:
        env, difficulty = seqs[0].env, seqs[0].difficulty
        mismatches = [seq.sequence_id for seq in seqs if (seq.env, seq.difficulty) != (env, difficulty)]
        if mismatches:
            raise ValueError("all sequences must share the same env/difficulty")
        return env, difficulty

    def _sequence_path(self, root: Path, seq: TartanAirSequence) -> Path:
        return root / seq.env / seq.difficulty / seq.trajectory

    def _download_file(self, key: str, target: Path, overwrite: bool = False) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.source == "s3":
            self._download_s3(key, target, overwrite=overwrite)
        else:
            self._download_http(f"{self.HF_BASE_URL}/{key}", target, overwrite=overwrite)

    def _download_http(self, url: str, target: Path, overwrite: bool = False) -> None:
        total_size = self._http_content_length(url)
        if overwrite and target.exists():
            target.unlink()
        local_size = target.stat().st_size if target.exists() else 0
        local_size = self._prepare_partial_file(target, local_size, total_size)

        if total_size > 0 and local_size == total_size:
            return

        request = urllib.request.Request(url)
        if local_size:
            request.add_header("Range", f"bytes={local_size}-")

        with urllib.request.urlopen(request) as response:
            status = getattr(response, "status", 200)
            append = bool(local_size and status == 206)
            initial = local_size if append else 0
            mode = "ab" if append else "wb"

            with tqdm(
                total=total_size or None,
                initial=initial,
                unit="B",
                unit_scale=True,
                desc=target.name,
                disable=not self.progress,
                leave=False,
            ) as progress:
                with target.open(mode) as file:
                    while chunk := response.read(self.chunk_size):
                        file.write(chunk)
                        progress.update(len(chunk))

    def _download_s3(self, key: str, target: Path, overwrite: bool = False) -> None:
        client = self._client()
        response = client.head_object(Bucket=self.S3_BUCKET, Key=key)
        total_size = int(response.get("ContentLength", 0))
        if overwrite and target.exists():
            target.unlink()
        local_size = target.stat().st_size if target.exists() else 0
        local_size = self._prepare_partial_file(target, local_size, total_size)

        if total_size > 0 and local_size == total_size:
            return

        kwargs = {"Bucket": self.S3_BUCKET, "Key": key}
        if local_size:
            kwargs["Range"] = f"bytes={local_size}-"

        response = client.get_object(**kwargs)
        with tqdm(
            total=total_size or None,
            initial=local_size,
            unit="B",
            unit_scale=True,
            desc=target.name,
            disable=not self.progress,
            leave=False,
        ) as progress:
            with target.open("ab" if local_size else "wb") as file:
                for chunk in response["Body"].iter_chunks(chunk_size=self.chunk_size):
                    if chunk:
                        file.write(chunk)
                        progress.update(len(chunk))

    def _client(self):
        if self._s3_client is None:
            import boto3
            from botocore import UNSIGNED
            from botocore.client import Config

            self._s3_client = boto3.client(
                "s3",
                endpoint_url=self.S3_ENDPOINT_URL,
                config=Config(signature_version=UNSIGNED),
            )
        return self._s3_client

    def _http_content_length(self, url: str) -> int:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request) as response:
            return int(response.headers.get("Content-Length") or 0)

    def _prepare_partial_file(self, target: Path, local_size: int, total_size: int) -> int:
        if total_size > 0 and local_size > total_size:
            target.unlink()
            return 0
        return local_size

    def _archive_ready(self, archive_path: Path) -> bool:
        if not archive_path.is_file() or not zipfile.is_zipfile(archive_path):
            return False
        try:
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                return bool(zip_ref.namelist()) and zip_ref.testzip() is None
        except zipfile.BadZipFile:
            return False

    def _data_dirs_ready(
        self, root: Path, seqs: list[TartanAirSequence], data_dir: str
    ) -> bool:
        return all(self._dir_ready(self._sequence_path(root, seq) / data_dir) for seq in seqs)

    def _dir_ready(self, path: Path) -> bool:
        return path.is_dir() and any(path.iterdir())

    def _file_ready(self, path: Path) -> bool:
        return path.is_file() and path.stat().st_size > 0

    def _extract_archive(
        self,
        archive_path: Path,
        extract_dir: Path,
        *,
        mark_extracted,
        env: str,
        difficulty: str,
        trajectories: set[str],
        overwrite: bool,
    ) -> None:
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            members = [
                (member, relative_path)
                for member in zip_ref.infolist()
                if (relative_path := self._member_relative_path(
                    member.filename,
                    env=env,
                    difficulty=difficulty,
                    trajectories=trajectories,
                ))
                is not None
            ]

            with tqdm(
                members,
                desc=f"Extracting {archive_path.name}",
                unit="file",
                disable=not self.progress,
                leave=False,
            ) as progress:
                for member, relative_path in progress:
                    target = extract_dir / relative_path
                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue

                    if not overwrite and target.exists():
                        continue

                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zip_ref.open(member, "r") as src, target.open("wb") as dst:
                        while chunk := src.read(self.chunk_size):
                            dst.write(chunk)

            mark_extracted()

    def _member_relative_path(
        self,
        filename: str,
        *,
        env: str,
        difficulty: str,
        trajectories: set[str],
    ) -> Path | None:
        if not filename or filename.startswith("/"):
            return None
        parts = Path(filename).parts
        if len(parts) >= 2 and parts[0] in trajectories:
            return Path(*parts)
        if len(parts) >= 4 and parts[0] == env and parts[1] == difficulty and parts[2] in trajectories:
            return Path(*parts[2:])
        return None
