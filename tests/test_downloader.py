import zipfile

import pytest

from dasvo.datasets import TartanAirSequence
from dasvo.downloader import TartanAirDownloader


class RecordingDownloader(TartanAirDownloader):
    def __init__(self, **kwargs):
        super().__init__(progress=False, **kwargs)
        self.calls = []

    def _download_file(self, key, target, overwrite=False):
        self.calls.append((key, target, overwrite))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix == ".zip":
            with zipfile.ZipFile(target, "w") as archive:
                for trajectory in ["P001", "P002"]:
                    if target.name == "image_left.zip":
                        archive.writestr(f"{trajectory}/image_left/000000_left.png", b"png")
                    else:
                        archive.writestr(f"{trajectory}/depth_left/000000_left_depth.npy", b"npy")
        else:
            target.write_text("0 0 0 0 0 0 1\n", encoding="utf-8")


def write_zip(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)


def test_download_fetches_all_sequence_assets_and_cleans_archives(tmp_path):
    seqs = [
        TartanAirSequence(tmp_path, "forest", "Easy", "P001"),
        TartanAirSequence(tmp_path, "forest", "Easy", "P002"),
    ]
    downloader = RecordingDownloader()

    downloader.download(seqs, extract=True, clean=True)

    base_dir = tmp_path / "forest" / "Easy"
    assert downloader.calls == [
        ("forest/Easy/image_left.zip", base_dir / "image_left.zip", False),
        ("forest/Easy/depth_left.zip", base_dir / "depth_left.zip", False),
        ("forest/Easy/P001/pose_left.txt", base_dir / "P001" / "pose_left.txt", False),
        ("forest/Easy/P002/pose_left.txt", base_dir / "P002" / "pose_left.txt", False),
    ]
    assert not (base_dir / "image_left.zip").exists()
    assert not (base_dir / "depth_left.zip").exists()
    assert (base_dir / "image_downloaded.txt").exists()
    assert (base_dir / "image_extracted.txt").exists()
    assert (base_dir / "depth_downloaded.txt").exists()
    assert (base_dir / "depth_extracted.txt").exists()
    assert (base_dir / "P001" / "image_left" / "000000_left.png").exists()
    assert (base_dir / "P002" / "image_left" / "000000_left.png").exists()
    assert (base_dir / "P001" / "depth_left" / "000000_left_depth.npy").exists()
    assert (base_dir / "P002" / "depth_left" / "000000_left_depth.npy").exists()
    assert (base_dir / "P001" / "pose_downloaded.txt").exists()
    assert (base_dir / "P002" / "pose_downloaded.txt").exists()


def test_extract_false_downloads_archives_and_poses_without_unzipping(tmp_path):
    seqs = [
        TartanAirSequence(tmp_path, "forest", "Hard", "P001"),
        TartanAirSequence(tmp_path, "forest", "Hard", "P002"),
    ]
    downloader = RecordingDownloader()

    downloader.download(seqs, extract=False)

    base_dir = tmp_path / "forest" / "Hard"
    assert downloader.calls == [
        ("forest/Hard/image_left.zip", base_dir / "image_left.zip", False),
        ("forest/Hard/depth_left.zip", base_dir / "depth_left.zip", False),
        ("forest/Hard/P001/pose_left.txt", base_dir / "P001" / "pose_left.txt", False),
        ("forest/Hard/P002/pose_left.txt", base_dir / "P002" / "pose_left.txt", False),
    ]
    assert (base_dir / "image_left.zip").exists()
    assert (base_dir / "depth_left.zip").exists()
    assert (base_dir / "image_downloaded.txt").exists()
    assert (base_dir / "depth_downloaded.txt").exists()
    assert not (base_dir / "image_extracted.txt").exists()
    assert not (base_dir / "depth_extracted.txt").exists()
    assert not seqs[0].images_path.exists()
    assert not seqs[0].depth_path.exists()
    assert seqs[0].pose_path.exists()
    assert seqs[1].pose_path.exists()
    assert (seqs[0].pose_path.parent / "pose_downloaded.txt").exists()
    assert (seqs[1].pose_path.parent / "pose_downloaded.txt").exists()


def test_download_requires_shared_environment_and_difficulty(tmp_path):
    seqs = [
        TartanAirSequence(tmp_path, "forest", "Easy", "P001"),
        TartanAirSequence(tmp_path, "forest", "Hard", "P002"),
    ]

    with pytest.raises(ValueError, match="same env/difficulty"):
        RecordingDownloader().download(seqs)


def test_clean_requires_extract(tmp_path):
    seq = TartanAirSequence(tmp_path, "forest", "Easy", "P001")

    with pytest.raises(AssertionError, match="clean=True requires extract=True"):
        RecordingDownloader().download(seq, extract=False, clean=True)


def test_existing_complete_sequence_skips_download_and_marks_ready(tmp_path):
    seqs = [
        TartanAirSequence(tmp_path, "forest", "Easy", "P001"),
        TartanAirSequence(tmp_path, "forest", "Easy", "P002"),
    ]
    for seq in seqs:
        seq.images_path.mkdir(parents=True)
        (seq.images_path / "000000_left.png").write_bytes(b"png")
        seq.depth_path.mkdir(parents=True)
        (seq.depth_path / "000000_left_depth.npy").write_bytes(b"npy")
        seq.pose_path.parent.mkdir(parents=True, exist_ok=True)
        seq.pose_path.write_text("existing\n", encoding="utf-8")

    downloader = RecordingDownloader()
    downloader.download(seqs)

    assert downloader.calls == []
    assert (tmp_path / "forest" / "Easy" / "image_extracted.txt").exists()
    assert (tmp_path / "forest" / "Easy" / "depth_extracted.txt").exists()
    assert (seqs[0].pose_path.parent / "pose_downloaded.txt").exists()
    assert (seqs[1].pose_path.parent / "pose_downloaded.txt").exists()


def test_valid_archives_skip_download_when_extract_is_false(tmp_path):
    seq = TartanAirSequence(tmp_path, "forest", "Easy", "P001")
    write_zip(seq.image_left_zip_path, {"P001/image_left/000000_left.png": b"png"})
    write_zip(seq.depth_left_zip_path, {"P001/depth_left/000000_left_depth.npy": b"npy"})
    seq.pose_path.parent.mkdir(parents=True, exist_ok=True)
    seq.pose_path.write_text("existing\n", encoding="utf-8")

    downloader = RecordingDownloader()
    downloader.download(seq, extract=False)

    assert downloader.calls == []
    assert (seq.archive_dir / "image_downloaded.txt").exists()
    assert (seq.archive_dir / "depth_downloaded.txt").exists()
    assert (seq.pose_path.parent / "pose_downloaded.txt").exists()


def test_extract_accepts_env_difficulty_prefixed_tartanair_archives(tmp_path):
    seq = TartanAirSequence(tmp_path, "forest", "Easy", "P001")
    write_zip(
        seq.image_left_zip_path,
        {
            "forest/": b"",
            "forest/Easy/": b"",
            "forest/Easy/P001/image_left/000000_left.png": b"png",
            "forest/Easy/P999/image_left/ignored.png": b"png",
        },
    )
    write_zip(
        seq.depth_left_zip_path,
        {
            "forest/Easy/P001/depth_left/000000_left_depth.npy": b"npy",
            "forest/Easy/P999/depth_left/ignored.npy": b"npy",
        },
    )
    seq.pose_path.parent.mkdir(parents=True, exist_ok=True)
    seq.pose_path.write_text("existing\n", encoding="utf-8")

    downloader = RecordingDownloader()
    downloader.download(seq)

    assert downloader.calls == []
    assert (seq.images_path / "000000_left.png").exists()
    assert (seq.depth_path / "000000_left_depth.npy").exists()
    assert not (tmp_path / "forest" / "Easy" / "P999").exists()


def test_corrupted_image_archive_is_redownloaded(tmp_path):
    seq = TartanAirSequence(tmp_path, "forest", "Easy", "P001")
    seq.image_left_zip_path.parent.mkdir(parents=True)
    seq.image_left_zip_path.write_bytes(b"not a zip")
    write_zip(seq.depth_left_zip_path, {"P001/depth_left/000000_left_depth.npy": b"npy"})
    seq.pose_path.parent.mkdir(parents=True, exist_ok=True)
    seq.pose_path.write_text("existing\n", encoding="utf-8")

    downloader = RecordingDownloader()
    downloader.download(seq, extract=False)

    assert downloader.calls == [("forest/Easy/image_left.zip", seq.image_left_zip_path, True)]
    assert zipfile.is_zipfile(seq.image_left_zip_path)


def test_overwrite_redownloads_even_when_extracted_dirs_exist(tmp_path):
    seq = TartanAirSequence(tmp_path, "forest", "Easy", "P001")
    seq.images_path.mkdir(parents=True)
    (seq.images_path / "old.png").write_bytes(b"old")
    seq.depth_path.mkdir(parents=True)
    (seq.depth_path / "old.npy").write_bytes(b"old")
    seq.pose_path.parent.mkdir(parents=True, exist_ok=True)
    seq.pose_path.write_text("existing\n", encoding="utf-8")

    downloader = RecordingDownloader()
    downloader.download(seq, overwrite=True)

    assert downloader.calls == [
        ("forest/Easy/image_left.zip", seq.image_left_zip_path, True),
        ("forest/Easy/depth_left.zip", seq.depth_left_zip_path, True),
        ("forest/Easy/P001/pose_left.txt", seq.pose_path, True),
    ]
    assert (seq.images_path / "000000_left.png").exists()
    assert (seq.depth_path / "000000_left_depth.npy").exists()
    assert (seq.pose_path.parent / "pose_downloaded.txt").exists()


def test_source_aliases():
    assert TartanAirDownloader(source="hf", progress=False).source == "hf"
    assert TartanAirDownloader(source="huggingface", progress=False).source == "hf"
    assert TartanAirDownloader(source="s3", progress=False).source == "s3"
    assert TartanAirDownloader(source="cmu", progress=False).source == "s3"
