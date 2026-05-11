"""
Unit tests for the dasvo.geometry module.
"""

import numpy as np
from dasvo.geometry import (
    EssentialMatrixBackend,
    PnPBackend,
)


def test_essential_matrix_backend_rejects_few_points():
    backend = EssentialMatrixBackend()
    pts1 = np.array([[10.0, 10.0], [20.0, 20.0]])
    pts2 = pts1.copy()
    K = np.array([[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]])

    rotation, translation = backend.estimate_relative_pose(pts1, pts2, K)

    assert rotation is None
    assert translation is None


def test_pnp_backend_rejects_few_points():
    backend = PnPBackend()
    pts1 = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    pts2 = pts1.copy()
    depth = np.ones((64, 64), dtype=np.float32)
    K = np.array([[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]])

    rotation, translation = backend.estimate_relative_pose(pts1, pts2, K, depth1=depth)

    assert rotation is None
    assert translation is None


def test_pnp_backend_rejects_implausible_translation(monkeypatch):
    def fake_solve_pnp_ransac(*args, **kwargs):
        return True, np.zeros((3, 1)), np.array([[1e6], [0.0], [0.0]]), np.arange(20)

    monkeypatch.setattr("cv2.solvePnPRansac", fake_solve_pnp_ransac)

    backend = PnPBackend()
    pts1 = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0], [50.0, 50.0]])
    pts2 = pts1.copy()
    depth = np.ones((64, 64), dtype=np.float32)
    K = np.array([[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]])

    rotation, translation = backend.estimate_relative_pose(pts1, pts2, K, depth1=depth)

    assert rotation is None
    assert translation is None
