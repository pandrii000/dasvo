"""
Unit tests for the dasvo.evaluation module.
"""

import numpy as np
from dasvo.evaluation import align_umeyama, bootstrap_ate_rpe, compute_rpe, load_gt_stride

def test_align_umeyama_identity():
    """
    Test Umeyama alignment with identical models (should return 0 RMSE).
    """
    model = np.random.rand(100, 3)
    data = model.copy()
    
    aligned_data, rmse = align_umeyama(model, data, with_scale=False)
    
    assert np.isclose(rmse, 0.0)
    assert np.allclose(model, aligned_data)

def test_align_umeyama_translation():
    """
    Test Umeyama alignment with simple translation.
    """
    model = np.random.rand(100, 3)
    data = model + np.array([1.0, -2.0, 3.0])
    
    aligned_data, rmse = align_umeyama(model, data, with_scale=False)
    
    assert np.isclose(rmse, 0.0, atol=1e-5)
    assert np.allclose(model, aligned_data, atol=1e-5)

def test_compute_rpe():
    """
    Test Relative Pose Error computation.
    """
    gt = np.zeros((10, 3))
    gt[:, 0] = np.arange(10) # Moving 1 unit along X axis per step
    
    est = np.zeros((10, 3))
    est[:, 0] = np.arange(10) * 0.5 # Moving 0.5 units along X axis per step

    # RPE is computed on trajectories that are already in a common frame/scale.
    rpe = compute_rpe(gt, est)
    assert np.isclose(rpe, 0.5)

    aligned_est, _ = align_umeyama(gt, est, with_scale=True)
    aligned_rpe = compute_rpe(gt, aligned_est)
    assert np.isclose(aligned_rpe, 0.0)


def test_load_gt_stride(tmp_path):
    pose = np.zeros((10, 7))
    pose[:, 0] = np.arange(10)
    path = tmp_path / "pose_left.txt"
    np.savetxt(path, pose)
    sub = load_gt_stride(path, frame_stride=3)
    assert sub.shape == (4, 3)
    assert np.allclose(sub[:, 0], [0, 3, 6, 9])


def test_subset_bootstrap_stable():
    rng = np.random.default_rng(0)
    gt = rng.normal(size=(200, 3)).cumsum(axis=0)
    est = gt + rng.normal(scale=0.05, size=gt.shape)

    am, ast, rm, rst = bootstrap_ate_rpe(
        gt, est, with_scale=False, n_pose_samples=80, n_segment_samples=80, n_boot=30, seed=1
    )
    assert np.isfinite(am) and np.isfinite(rm)
    assert ast >= 0 and rst >= 0
