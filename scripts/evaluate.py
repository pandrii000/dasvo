import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fire
import numpy as np
from loguru import logger
from tqdm import tqdm

from dasvo.datasets import TartanAirSequence, list_sequences
from dasvo.evaluation import (
    align_umeyama,
    compute_rpe,
    compute_rpe_rotational,
    load_est,
    load_est_pose,
    load_gt_pose_stride,
    load_gt_stride,
)
from dasvo.settings import SETTINGS


def evaluate_sequence(
    sequence: TartanAirSequence,
    frontend_name: str,
    backend_name: str,
    degradation: str,
    frame_stride: int,
    overwrite: bool = False,
) -> None:
    output_dir = SETTINGS.paths.outputs_root / frontend_name / backend_name / degradation / str(frame_stride)
    traj_file = output_dir / f"{sequence.sequence_id}_traj.txt"
    pose_file = output_dir / f"{sequence.sequence_id}_pose.txt"

    if not traj_file.exists():
        return

    eval_dir = SETTINGS.paths.tables_root / frontend_name / backend_name / degradation / str(frame_stride)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Bump the marker name so existing evaluations that predate the rotational
    # RPE wiring are recomputed automatically. Old markers stay on disk
    # harmlessly; the new field is filled the first time evaluate runs.
    marker_file = eval_dir / f"{sequence.sequence_id}_eval_done_v2.txt"
    if marker_file.exists() and not overwrite:
        return

    try:
        gt = load_gt_stride(sequence.pose_path, frame_stride)
        est = load_est(traj_file)

        min_len = min(len(gt), len(est))
        if min_len < 3:
            return

        gt = gt[:min_len]
        est = est[:min_len]

        with_scale = (backend_name in SETTINGS.evaluation.monocular_scale_align_backends)

        est_aligned, ate = align_umeyama(gt, est, with_scale=with_scale)
        rpe = compute_rpe(gt, est_aligned, delta=SETTINGS.evaluation.rpe_delta)

        # PnP ATE under Sim(3) alignment isolates trajectory-shape error from
        # metric-scale drift. Essential is already Sim(3)-aligned, so the same
        # value is reused for the column.
        ate_sim3 = float("nan")
        if backend_name == "pnp":
            _, ate_sim3_val = align_umeyama(gt, est, with_scale=True)
            ate_sim3 = float(ate_sim3_val)
        elif backend_name == "essential":
            ate_sim3 = float(ate)

        rpe_rot = float("nan")
        if pose_file.exists():
            try:
                gt_pose = load_gt_pose_stride(sequence.pose_path, frame_stride)
                est_pose = load_est_pose(pose_file)
                pose_len = min(len(gt_pose), len(est_pose), min_len)
                if pose_len > SETTINGS.evaluation.rpe_delta:
                    rpe_rot = compute_rpe_rotational(
                        gt_pose[:pose_len, 3:7],
                        est_pose[:pose_len, 3:7],
                        delta=SETTINGS.evaluation.rpe_delta,
                    )
            except Exception as pose_exc:
                logger.warning(
                    f"Rotational RPE skipped for {sequence.sequence_id} "
                    f"[{frontend_name}-{backend_name}-{degradation}-s{frame_stride}]: {pose_exc}"
                )

        failed = not (np.isfinite(ate) and np.isfinite(rpe))

        results = {
            "ate": float(ate),
            "ate_sim3": ate_sim3,
            "rpe": float(rpe),
            "rpe_rot": rpe_rot,
            "sequence": sequence.sequence_id,
            "frontend": frontend_name,
            "backend": backend_name,
            "degradation": degradation,
            "frame_stride": frame_stride,
            "frames": min_len,
            "alignment": "sim3_umeyama" if with_scale else "se3_umeyama",
            "alignment_with_scale": bool(with_scale),
            "rpe_delta": SETTINGS.evaluation.rpe_delta,
            "rpe_rot_available": bool(pose_file.exists() and np.isfinite(rpe_rot)),
            "failed": bool(failed),
        }

        result_file = eval_dir / f"{sequence.sequence_id}_results.json"
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)

        marker_file.touch()

    except Exception as e:
        logger.error(f"Error evaluating {sequence.sequence_id} with {frontend_name}-{backend_name}-{degradation}: {e}")
        raise


def main(
    num_workers: int = SETTINGS.experiment.num_workers,
    overwrite: bool = False,
) -> None:
    sequences = list_sequences()
    
    jobs = []
    for seq in sequences:
        if not seq.pose_exist():
            continue
            
        for fe in SETTINGS.experiment.frontends:
            for be in SETTINGS.experiment.backends:
                for deg in SETTINGS.experiment.degradations:
                    for stride in SETTINGS.experiment.frame_strides:
                        jobs.append((seq, fe, be, deg, stride, overwrite))

    logger.info(f"Total evaluation jobs: {len(jobs)}")

    if num_workers <= 1:
        for job in tqdm(jobs, desc="Evaluating sequences"):
            evaluate_sequence(*job)
    else:
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            futures = [executor.submit(evaluate_sequence, *job) for job in jobs]
            for future in tqdm(as_completed(futures), total=len(jobs), desc="Evaluating sequences"):
                future.result()


if __name__ == "__main__":
    fire.Fire(main)
