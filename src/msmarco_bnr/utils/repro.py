from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReproState:
    seed: int
    deterministic: bool
    torch_available: bool


def set_reproducibility(seed: int, deterministic: bool) -> ReproState:
    # Python
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch (optional; smoke runs can skip installing torch).
    try:
        import torch  # type: ignore
    except Exception:  # noqa: BLE001
        return ReproState(seed=seed, deterministic=deterministic, torch_available=False)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuBLAS deterministic (if CUDA). Harmless on CPU.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Some ops may not have deterministic implementations; surface errors early.
        torch.use_deterministic_algorithms(True)

    return ReproState(seed=seed, deterministic=deterministic, torch_available=True)
