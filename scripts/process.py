import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fire
import numpy as np
from loguru import logger
from tqdm import tqdm

from dasvo.data_loader import TartanAirLoader
from dasvo.datasets import TartanAirSequence, list_sequences
from dasvo.front_end import KLTFrontEnd, ORBFrontEnd
from dasvo.geometry import EssentialMatrixBackend, PnPBackend
from dasvo.settings import SETTINGS


def process_sequence(
    sequence: TartanAirSequence,
    frontend_name: str,
    backend_name: str,
    degradation: str,
    frame_stride: int,
    max_virtual_frames: int | None,
) -> None:
    output_dir = SETTINGS.paths.outputs_root / frontend_name / backend_name / degradation / str(frame_stride)
    output_dir.mkdir(parents=True, exist_ok=True)

    marker_file = output_dir / f"{sequence.sequence_id}_done.txt"
    if marker_file.exists():
        return

    try:
        loader = TartanAirLoader(
            sequence=sequence,
            frame_stride=frame_stride,
            degradation=degradation,
            max_virtual_frames=max_virtual_frames,
        )

        if frontend_name == "orb":
            frontend = ORBFrontEnd()
        elif frontend_name == "klt":
            frontend = KLTFrontEnd()
        else:
            raise ValueError(f"Unknown frontend: {frontend_name}")

        if backend_name == "essential":
            backend = EssentialMatrixBackend()
        elif backend_name == "pnp":
            backend = PnPBackend()
        else:
            raise ValueError(f"Unknown backend: {backend_name}")

        trajectory = []
        current_pose = np.eye(4)
        prev_depth = None

        n_frames = len(loader)
        for i in range(n_frames):
            if i % 100 == 0:
                logger.info(f"[{frontend_name}-{backend_name}-{degradation}] {sequence.sequence_id}: frame {i}/{n_frames}")
                
            img, depth, _, K = loader.get_frame(i)
            pts1, pts2 = frontend.process_frame(img)

            if pts1 is None or len(pts1) == 0:
                trajectory.append(current_pose[:3, 3])
                prev_depth = depth
                continue

            R, t = backend.estimate_relative_pose(pts1, pts2, K, prev_depth)

            if R is not None and t is not None:
                T_rel = np.eye(4)
                T_rel[:3, :3] = R
                T_rel[:3, 3] = t.flatten()
                current_pose = current_pose @ np.linalg.inv(T_rel)

            trajectory.append(current_pose[:3, 3])
            prev_depth = depth

        traj_file = output_dir / f"{sequence.sequence_id}_traj.txt"
        np.savetxt(traj_file, np.array(trajectory))
        
        marker_file.touch()

    except Exception as e:
        logger.error(f"Error processing {sequence.sequence_id} with {frontend_name}-{backend_name}-{degradation}: {e}")


def main(
    num_workers: int = SETTINGS.experiment.num_workers,
    max_virtual_frames: int | None = SETTINGS.experiment.max_virtual_frames,
) -> None:
    sequences = list_sequences()
    
    jobs = []
    for seq in sequences:
        if not seq.images_exist() or not seq.depth_exist() or not seq.pose_exist():
            continue
            
        for fe in SETTINGS.experiment.frontends:
            for be in SETTINGS.experiment.backends:
                for deg in SETTINGS.experiment.degradations:
                    for stride in SETTINGS.experiment.frame_strides:
                        jobs.append((seq, fe, be, deg, stride, max_virtual_frames))

    logger.info(f"Total jobs to process: {len(jobs)}")

    if num_workers <= 1:
        for job in tqdm(jobs, desc="Processing sequences"):
            process_sequence(*job)
    else:
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            futures = [executor.submit(process_sequence, *job) for job in jobs]
            for future in tqdm(as_completed(futures), total=len(jobs), desc="Processing sequences"):
                future.result()


if __name__ == "__main__":
    fire.Fire(main)
