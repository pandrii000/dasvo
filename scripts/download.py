from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from random import shuffle

import fire
from loguru import logger

import dasvo.datasets as datasets
from dasvo.downloader import TartanAirDownloader
from dasvo.settings import DATA_ROOT, DOWNLOAD_SOURCE, SEQUENCES_FILE


def _download_group(
    batch: list,
    *,
    source: str,
    root: Path,
    progress: bool,
    extract: bool,
    clean: bool,
    overwrite: bool,
) -> None:
    downloader = TartanAirDownloader(source=source, root=root, progress=progress)
    downloader.download(
        sorted(batch, key=lambda seq: seq.trajectory),
        extract=extract,
        clean=clean,
        overwrite=overwrite,
    )


def download(
    source: str = DOWNLOAD_SOURCE,
    root: str | Path = DATA_ROOT,
    sequences_file: str | Path = SEQUENCES_FILE,
    env: str | None = None,
    difficulty: str | None = None,
    extract: bool = True,
    clean: bool = True,
    overwrite: bool = False,
    progress: bool = True,
    num_workers: int = 4,
) -> None:
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")

    root = Path(root)
    datasets.DATA_ROOT = root
    datasets.SEQUENCES_FILE = Path(sequences_file)

    groups = defaultdict(list)
    for seq in datasets.list_sequences():
        if env and seq.env != env:
            continue
        if difficulty and seq.difficulty != difficulty:
            continue
        groups[(seq.env, seq.difficulty)].append(seq)

    if not groups:
        raise ValueError("no sequences matched the requested filters")

    batches = [batch for _, batch in sorted(groups.items())]
    shuffle(batches)
    for batch in batches:
        logger.info(f"Downloading sequences: {batch}")

    if num_workers == 1:
        for batch in batches:
            _download_group(
                batch,
                source=source,
                root=root,
                progress=progress,
                extract=extract,
                clean=clean,
                overwrite=overwrite,
            )
        return

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _download_group,
                batch,
                source=source,
                root=root,
                progress=progress,
                extract=extract,
                clean=clean,
                overwrite=overwrite,
            )
            for batch in batches
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    fire.Fire(download)
