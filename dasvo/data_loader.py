import os
from pathlib import Path

import cv2
import numpy as np

from dasvo.datasets import TartanAirSequence
from dasvo.degradation import apply_visual_degradation
from dasvo.settings import SETTINGS


class TartanAirLoader:
    """
    Data loader for TartanAir dataset sequences.
    """

    def __init__(
        self,
        sequence: TartanAirSequence,
        frame_stride: int = 1,
        degradation: str | None = None,
        max_virtual_frames: int | None = None,
    ):
        self.sequence = sequence

        if not sequence.images_exist() or not sequence.depth_exist() or not sequence.pose_exist():
            raise ValueError(f"Sequence '{sequence.sequence_id}' is missing or incomplete.")

        self.image_dir = sequence.images_path
        self.depth_dir = sequence.depth_path
        self.pose_file = sequence.pose_path

        self.image_files = sorted(
            [f for f in os.listdir(self.image_dir) if f.endswith(".png")]
        )
        self.depth_files = sorted(
            [f for f in os.listdir(self.depth_dir) if f.endswith(".npy")]
        )

        if os.path.exists(self.pose_file):
            self.poses = np.loadtxt(self.pose_file)
        else:
            self.poses = None

        self.frame_stride = max(1, int(frame_stride))
        n = len(self.image_files)
        self.physical_indices = list(range(0, n, self.frame_stride))
        if max_virtual_frames is not None and int(max_virtual_frames) > 0:
            self.physical_indices = self.physical_indices[: int(max_virtual_frames)]
        self.degradation = degradation if degradation and degradation != "none" else None

        self.K = np.array(SETTINGS.camera.intrinsics)

    def __len__(self):
        return len(self.physical_indices)

    def get_frame(self, idx):
        """
        Returns RGB image, depth map in meters, ground-truth pose, and intrinsics.
        """
        phys = self.physical_indices[idx]
        img_path = os.path.join(self.image_dir, self.image_files[phys])
        depth_path = os.path.join(self.depth_dir, self.depth_files[phys])

        img = cv2.imread(img_path)
        img = apply_visual_degradation(img, self.degradation, seed=phys)
        depth = np.load(depth_path) if os.path.exists(depth_path) else None
        pose = self.poses[phys] if self.poses is not None else None

        return img, depth, pose, self.K
