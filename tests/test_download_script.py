from importlib.util import module_from_spec, spec_from_file_location

import pytest

from dasvo.datasets import TartanAirSequence


spec = spec_from_file_location(
    "download_script", "scripts/download.py"
)
download_script = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(download_script)


class RecordingDownloader:
    calls = []
    instances = []

    def __init__(self, *, source, root, progress):
        self.source = source
        self.root = root
        self.progress = progress
        self.instances.append((source, root, progress))

    def download(self, sequences, **kwargs):
        self.calls.append(([seq.sequence_id for seq in sequences], kwargs))


def test_download_batches_sequences_by_env_and_difficulty(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "dasvo.datasets.list_sequences",
        lambda: [
            TartanAirSequence(tmp_path, "forest", "Easy", "P002"),
            TartanAirSequence(tmp_path, "forest", "Easy", "P001"),
            TartanAirSequence(tmp_path, "office", "Hard", "P003"),
        ],
    )
    monkeypatch.setattr(download_script, "TartanAirDownloader", RecordingDownloader)
    RecordingDownloader.calls = []
    RecordingDownloader.instances = []

    download_script.download(root=tmp_path, progress=False)

    assert sorted(RecordingDownloader.calls) == sorted([
        (
            ["forest_easy_p001", "forest_easy_p002"],
            {
                "extract": True,
                "clean": True,
                "overwrite": False,
            },
        ),
        (
            ["office_hard_p003"],
            {
                "extract": True,
                "clean": True,
                "overwrite": False,
            },
        ),
    ])
    assert RecordingDownloader.instances == [("hf", tmp_path, False), ("hf", tmp_path, False)]


def test_download_filters_by_env_and_difficulty(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "dasvo.datasets.list_sequences",
        lambda: [
            TartanAirSequence(tmp_path, "forest", "Easy", "P001"),
            TartanAirSequence(tmp_path, "forest", "Hard", "P002"),
        ],
    )
    monkeypatch.setattr(download_script, "TartanAirDownloader", RecordingDownloader)
    RecordingDownloader.calls = []
    RecordingDownloader.instances = []

    download_script.download(root=tmp_path, env="forest", difficulty="Hard")

    assert RecordingDownloader.calls == [
        (
            ["forest_hard_p002"],
            {
                "extract": True,
                "clean": True,
                "overwrite": False,
            },
        )
    ]
    assert RecordingDownloader.instances == [("hf", tmp_path, True)]


def test_download_errors_when_filter_matches_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr("dasvo.datasets.list_sequences", lambda: [])

    with pytest.raises(ValueError, match="no sequences matched"):
        download_script.download(root=tmp_path)


def test_download_uses_one_worker_per_env_difficulty_batch_when_parallel(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "dasvo.datasets.list_sequences",
        lambda: [
            TartanAirSequence(tmp_path, "forest", "Easy", "P002"),
            TartanAirSequence(tmp_path, "forest", "Easy", "P001"),
            TartanAirSequence(tmp_path, "office", "Hard", "P003"),
        ],
    )
    monkeypatch.setattr(download_script, "TartanAirDownloader", RecordingDownloader)
    RecordingDownloader.calls = []
    RecordingDownloader.instances = []

    download_script.download(root=tmp_path, progress=False, num_workers=2)

    assert sorted(RecordingDownloader.calls) == sorted(
        [
            (
                ["forest_easy_p001", "forest_easy_p002"],
                {
                    "extract": True,
                    "clean": True,
                    "overwrite": False,
                },
            ),
            (
                ["office_hard_p003"],
                {
                    "extract": True,
                    "clean": True,
                    "overwrite": False,
                },
            ),
        ]
    )
    assert RecordingDownloader.instances == [("hf", tmp_path, False), ("hf", tmp_path, False)]


def test_download_rejects_non_positive_num_workers(tmp_path):
    with pytest.raises(ValueError, match="num_workers must be at least 1"):
        download_script.download(root=tmp_path, num_workers=0)
