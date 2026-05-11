import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fire
import numpy as np
from loguru import logger
from tqdm import tqdm

from dasvo.datasets import TartanAirSequence, list_sequences
from dasvo.evaluation import align_umeyama, compute_rpe, load_est, load_gt_stride
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
    
    if not traj_file.exists():
        return

    eval_dir = SETTINGS.paths.tables_root / frontend_name / backend_name / degradation / str(frame_stride)
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    marker_file = eval_dir / f"{sequence.sequence_id}_eval_done.txt"
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
        failed = not (np.isfinite(ate) and np.isfinite(rpe))

        results = {
            "ate": float(ate),
            "rpe": float(rpe),
            "sequence": sequence.sequence_id,
            "frontend": frontend_name,
            "backend": backend_name,
            "degradation": degradation,
            "frame_stride": frame_stride,
            "frames": min_len,
            "alignment": "sim3_umeyama" if with_scale else "se3_umeyama",
            "alignment_with_scale": bool(with_scale),
            "rpe_delta": SETTINGS.evaluation.rpe_delta,
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
