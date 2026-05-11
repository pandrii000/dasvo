"""
Trajectory alignment and error metrics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_gt(pose_file):
    """
    Loads ground-truth trajectory from TartanAir format.
    """
    poses = np.loadtxt(pose_file)
    return poses[:, :3]


def load_gt_stride(pose_file: str | Path, frame_stride: int) -> np.ndarray:
    """
    Translation trajectory subsampled with the same uniform stride as TartanAirLoader.
    """
    full = load_gt(pose_file)
    step = max(1, int(frame_stride))
    return full[::step]


def load_est(traj_file):
    """
    Loads estimated trajectory.
    """
    return np.loadtxt(traj_file)


def align_umeyama(model, data, with_scale=False):
    """
    Aligns data to model using the Umeyama algorithm.
    """
    mu_m = model.mean(0)
    mu_d = data.mean(0)

    m_zero = model - mu_m
    d_zero = data - mu_d

    var_d = np.var(data, axis=0).sum()
    if var_d == 0:
        return data, np.inf

    covariance = d_zero.T @ m_zero / len(model)
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T

    if np.linalg.det(rotation) < 0:
        vt[2, :] *= -1
        rotation = vt.T @ u.T
        singular_values[-1] *= -1

    if with_scale:
        scale = np.sum(singular_values) / var_d
    else:
        scale = 1.0

    translation = mu_m - scale * rotation @ mu_d
    aligned_data = (scale * rotation @ data.T).T + translation

    error = model - aligned_data
    rmse = np.sqrt(np.mean(np.sum(error**2, axis=1)))

    return aligned_data, rmse


def compute_rpe(gt, est, delta=1):
    """
    Computes translational relative pose error between aligned trajectories.

    The trajectory inputs must already be in the same coordinate frame and scale.
    The returned value is the RMSE of the vector error between local translation
    increments over ``delta`` frames.
    """
    errors = compute_rpe_errors(gt, est, delta=delta)
    if len(errors) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(errors**2)))


def compute_rpe_errors(gt, est, delta=1):
    """
    Computes per-frame translational relative pose errors.
    """
    if delta <= 0:
        raise ValueError("delta must be positive")
    if len(gt) <= delta or len(est) <= delta:
        return np.array([], dtype=float)

    gt_diff = gt[delta:] - gt[:-delta]
    est_diff = est[delta:] - est[:-delta]

    return np.linalg.norm(gt_diff - est_diff, axis=1)


def compute_error_stats(errors: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    """
    Computes statistical metrics for an array of errors.
    """
    if len(errors) == 0:
        return {
            "mean": float("nan"), "std": float("nan"), "median": float("nan"), 
            "iqr": float("nan"), "se": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")
        }
    
    errors = np.asarray(errors)
    mean = float(np.mean(errors))
    std = float(np.std(errors))
    median = float(np.median(errors))
    iqr = float(np.percentile(errors, 75) - np.percentile(errors, 25))
    se = std / np.sqrt(len(errors))
    
    if n_boot > 0 and len(errors) > 1:
        rng = np.random.default_rng(seed)
        boot_means = rng.choice(errors, size=(n_boot, len(errors)), replace=True).mean(axis=1)
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))
    else:
        ci_lower = float("nan")
        ci_upper = float("nan")
        
    return {
        "mean": mean,
        "std": std,
        "median": median,
        "iqr": iqr,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }


def compute_rpe_subset(gt: np.ndarray, est: np.ndarray, segment_indices: np.ndarray, delta: int = 1) -> float:
    """
    RPE on a random subset of consecutive frame pairs (segment index i uses motion i->i+delta).
    """
    if len(segment_indices) == 0:
        return float("nan")
    gt_diff = gt[delta:] - gt[:-delta]
    est_diff = est[delta:] - est[:-delta]
    errors = np.linalg.norm(gt_diff[segment_indices] - est_diff[segment_indices], axis=1)
    return float(np.sqrt(np.mean(errors**2)))


def bootstrap_ate_rpe(
    gt: np.ndarray,
    est: np.ndarray,
    *,
    with_scale: bool,
    n_pose_samples: int,
    n_segment_samples: int,
    n_boot: int,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """
    Monte Carlo: repeatedly subsample pose indices (memoryless across draws) for ATE and
    segment indices for RPE. Returns (ate_mean, ate_std, rpe_mean, rpe_std).
    """
    rng = np.random.default_rng(seed)
    L = min(len(gt), len(est))
    if L < 3:
        return (float("nan"),) * 4

    est_aligned, _ = align_umeyama(gt[:L], est[:L], with_scale=with_scale)
    n_pose = max(3, min(int(n_pose_samples), L))
    n_seg_total = max(1, L - 1)
    n_seg = max(1, min(int(n_segment_samples), n_seg_total))

    ate_stats: list[float] = []
    rpe_stats: list[float] = []
    for _ in range(int(n_boot)):
        pose_idx = np.sort(rng.choice(L, size=n_pose, replace=False))
        _, ate = align_umeyama(gt[pose_idx], est[pose_idx], with_scale=with_scale)
        ate_stats.append(float(ate))

        seg_idx = np.sort(rng.choice(n_seg_total, size=n_seg, replace=False))
        rpe_stats.append(compute_rpe_subset(gt[:L], est_aligned[:L], seg_idx, delta=1))

    return (
        float(np.mean(ate_stats)),
        float(np.std(ate_stats)),
        float(np.mean(rpe_stats)),
        float(np.std(rpe_stats)),
    )


def bootstrap_convergence_curve(
    gt: np.ndarray,
    est: np.ndarray,
    *,
    with_scale: bool,
    pose_sample_sizes: list[int],
    segment_sample_sizes: list[int] | None,
    n_boot: int,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """
    For each paired (n_pose, n_seg) in zip(pose_sample_sizes, segment_sample_sizes or pose),
    run bootstrap_ate_rpe and return arrays of means/stds.
    """
    if segment_sample_sizes is None:
        segment_sample_sizes = pose_sample_sizes
    if len(segment_sample_sizes) != len(pose_sample_sizes):
        raise ValueError("pose_sample_sizes and segment_sample_sizes must match")

    ate_m = []
    ate_s = []
    rpe_m = []
    rpe_s = []
    for n_p, n_s in zip(pose_sample_sizes, segment_sample_sizes, strict=True):
        am, ast, rm, rst = bootstrap_ate_rpe(
            gt,
            est,
            with_scale=with_scale,
            n_pose_samples=n_p,
            n_segment_samples=n_s,
            n_boot=n_boot,
            seed=seed + n_p,
        )
        ate_m.append(am)
        ate_s.append(ast)
        rpe_m.append(rm)
        rpe_s.append(rst)

    return {
        "pose_sizes": np.array(pose_sample_sizes),
        "ate_mean": np.array(ate_m),
        "ate_std": np.array(ate_s),
        "rpe_mean": np.array(rpe_m),
        "rpe_std": np.array(rpe_s),
    }
