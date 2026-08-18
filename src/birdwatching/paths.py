

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
PREDICTIONS_DIR = PROJECT_ROOT / "artifacts" / "predictions"

_LEGACY_NESTED_ROOT = (
    PROJECT_ROOT / "birds_dataset" / "birds_dataset" / "birds_dataset"
)

_DATA_ROOT_CANDIDATES: tuple[Path, ...] = (
    PROJECT_ROOT / "birds_dataset",
    PROJECT_ROOT / "birds_dataset" / "birds_dataset",
    _LEGACY_NESTED_ROOT,
)


def resolve_data_root(explicit: Union[None, str, Path, os.PathLike] = None) -> Path:

    if explicit is not None:
        p = Path(explicit).expanduser().resolve()   #resolve the path
        if not (p / "classes.txt").is_file():
            raise FileNotFoundError(
                f"Data root has no classes.txt: {p}",
            )   #raise an error if the classes.txt file is not found
        return p

    env = os.environ.get("BIRDWATCHING_DATA_ROOT")   #get the environment variable
    if env:
        return resolve_data_root(env)   #resolve the path

    for cand in _DATA_ROOT_CANDIDATES:
        p = cand.resolve()
        if (p / "classes.txt").is_file():
            return p

    return _LEGACY_NESTED_ROOT.resolve()   



DATA_ROOT: Path = resolve_data_root()  #default data root
