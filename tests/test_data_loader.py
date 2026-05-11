import numpy as np
import pytest

from dasvo.data_loader import TartanAirLoader
from dasvo.datasets import TartanAirSequence


def test_tartanair_loader_initialization_and_len(tmp_path):
    # Setup a mock sequence
    seq_dir = tmp_path / "env" / "Easy" / "P000"
    images_dir = seq_dir / "image_left"
    depth_dir = seq_dir / "depth_left"
    
    images_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    
    for i in range(5):
        (images_dir / f"{i:06d}_left.png").touch()
        (depth_dir / f"{i:06d}_left.npy").touch()
        
    pose_file = seq_dir / "pose_left.txt"
    pose_file.write_text("0 0 0 0 0 0 1\n" * 5)
    
    seq = TartanAirSequence(tmp_path, "env", "Easy", "P000")
    
    loader = TartanAirLoader(seq, frame_stride=2)
    assert len(loader) == 3  # indices 0, 2, 4
    
    loader_virtual = TartanAirLoader(seq, frame_stride=1, max_virtual_frames=2)
    assert len(loader_virtual) == 2  # indices 0, 1


def test_tartanair_loader_missing_files(tmp_path):
    seq = TartanAirSequence(tmp_path, "env", "Easy", "P000")
    
    with pytest.raises(ValueError, match="is missing or incomplete"):
        TartanAirLoader(seq)
