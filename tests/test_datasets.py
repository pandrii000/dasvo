# uv run pytest

from pathlib import Path

import dasvo.datasets as datasets_mod
from dasvo.datasets import TartanAirSequence, list_sequences
from dasvo.settings import DATA_ROOT, PROJECT_ROOT, SEQUENCES_FILE


def test_settings_paths_exist_and_resolve_under_project():
    assert PROJECT_ROOT.is_dir()
    assert SEQUENCES_FILE.is_file()
    assert DATA_ROOT == PROJECT_ROOT.joinpath("assets", "data", "tartanair")


def test_datasets_uses_same_paths_as_settings():
    """datasets imports DATA_ROOT and SEQUENCES_FILE from settings; tests patch dasvo.datasets.* for isolation."""
    assert datasets_mod.DATA_ROOT == DATA_ROOT
    assert datasets_mod.SEQUENCES_FILE == SEQUENCES_FILE


def test_list_sequences_nonempty():
    seqs = list_sequences()
    assert len(seqs) >= 1
    assert all(isinstance(s, TartanAirSequence) for s in seqs)


def test_list_sequences_parse_first_line_matches_file():
    seqs = list_sequences()
    first_line = SEQUENCES_FILE.read_text(encoding="utf-8").splitlines()[0].strip()
    env, difficulty, traj = first_line.split("/")
    assert seqs[0].env == env
    assert seqs[0].difficulty == difficulty
    assert seqs[0].trajectory == traj
    assert seqs[0].root == DATA_ROOT


def test_list_sequences_respects_monkeypatched_paths(monkeypatch, tmp_path):
    # Patch dasvo.datasets (not dasvo.settings): list_sequences uses DATA_ROOT / SEQUENCES_FILE bound in that module.
    seq_file = tmp_path / "sequences.txt"
    seq_file.write_text("customenv/Hard/P099\n", encoding="utf-8")
    data_root = tmp_path / "data_root"
    monkeypatch.setattr("dasvo.datasets.SEQUENCES_FILE", seq_file)
    monkeypatch.setattr("dasvo.datasets.DATA_ROOT", data_root)
    seqs = list_sequences()
    assert len(seqs) == 1
    s = seqs[0]
    assert s.root == data_root
    assert s.env == "customenv"
    assert s.difficulty == "Hard"
    assert s.trajectory == "P099"


def test_tartan_air_sequence_id_and_paths(tmp_path):
    root = tmp_path / "root"
    seq = TartanAirSequence(root, "abandonedfactory", "Easy", "P011")
    assert seq.sequence_id == "abandonedfactory_easy_p011"
    assert seq.data_path == root / "abandonedfactory" / "Easy" / "P011"
    assert seq.archive_dir == root / "abandonedfactory" / "Easy"
    assert seq.image_left_zip_path == seq.archive_dir / "image_left.zip"
    assert seq.depth_left_zip_path == seq.archive_dir / "depth_left.zip"
    assert seq.images_path == seq.data_path / "image_left"
    assert seq.depth_path == seq.data_path / "depth_left"
    assert seq.pose_path == seq.data_path / "pose_left.txt"


def test_status_marker_methods_create_expected_files(tmp_path):
    seq = TartanAirSequence(tmp_path, "forest", "Easy", "P001")

    assert not seq.is_image_left_zip_downloaded()
    assert not seq.is_image_left_zip_extracted()
    assert not seq.is_depth_left_zip_downloaded()
    assert not seq.is_depth_left_zip_extracted()
    assert not seq.is_pose_left_downloaded()

    seq.set_image_left_zip_downloaded()
    seq.set_image_left_zip_extracted()
    seq.set_depth_left_zip_downloaded()
    seq.set_depth_left_zip_extracted()
    seq.set_pose_left_downloaded()

    assert seq.is_image_left_zip_downloaded()
    assert seq.is_image_left_zip_extracted()
    assert seq.is_depth_left_zip_downloaded()
    assert seq.is_depth_left_zip_extracted()
    assert seq.is_pose_left_downloaded()
    assert (seq.archive_dir / "image_downloaded.txt").exists()
    assert (seq.archive_dir / "image_extracted.txt").exists()
    assert (seq.archive_dir / "depth_downloaded.txt").exists()
    assert (seq.archive_dir / "depth_extracted.txt").exists()
    assert (seq.data_path / "pose_downloaded.txt").exists()
    assert (seq.archive_dir / "image_downloaded.txt").read_text(encoding="utf-8") == ""


def test_exist_flags(tmp_path):
    root = tmp_path
    seq = TartanAirSequence(root, "e", "Easy", "P001")
    seq.data_path.mkdir(parents=True)
    seq.images_path.mkdir()
    seq.depth_path.mkdir()
    seq.pose_path.write_text("0 0 0 0 0 0 1\n", encoding="utf-8")
    assert seq.images_exist()
    assert seq.depth_exist()
    assert seq.pose_exist()


def test_exist_flags_false_when_missing(tmp_path):
    seq = TartanAirSequence(tmp_path, "missing", "Easy", "P001")
    assert not seq.images_exist()
    assert not seq.depth_exist()
    assert not seq.pose_exist()


def test_str_includes_sequence_id_and_paths(tmp_path):
    seq = TartanAirSequence(tmp_path, "e", "Hard", "P002")
    text = str(seq)
    assert "sequence_id=e_hard_p002" in text
    assert "image_left" in text and "depth_left" in text and "pose_left.txt" in text


def test_with_root_rebinds_sequence_paths(tmp_path):
    seq = TartanAirSequence(tmp_path / "old", "forest", "Hard", "P002")
    rebound = seq.with_root(tmp_path / "new")

    assert rebound.root == tmp_path / "new"
    assert rebound.env == seq.env
    assert rebound.difficulty == seq.difficulty
    assert rebound.trajectory == seq.trajectory
    assert rebound.data_path == tmp_path / "new" / "forest" / "Hard" / "P002"
