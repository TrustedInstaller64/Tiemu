"""Explicit PyTorch device selection without changing training precision."""

import logging
import random

import numpy as np
import torch

logger = logging.getLogger("OmphalosLogger")


def resolve_torch_device(requested="auto") -> torch.device:
    """Resolve auto to MPS, CUDA, then CPU; reject unavailable explicit choices."""
    requested = str(requested).lower()
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(f"未知 PyTorch 设备: {requested}; 可选 auto/cpu/mps/cuda")
    if requested == "cpu":
        logger.info("PyTorch 设备: cpu (显式配置, float32)")
        return torch.device("cpu")
    mps_available = torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()
    if requested == "auto":
        selected = "mps" if mps_available else "cuda" if cuda_available else "cpu"
    else:
        selected = requested
        if (selected == "mps" and not mps_available) or (selected == "cuda" and not cuda_available):
            raise RuntimeError(f"显式指定的 PyTorch 设备 {selected} 不可用，请检查硬件和 PyTorch 构建。")
    logger.info("PyTorch 设备: %s (配置: %s, float32)", selected, requested)
    return torch.device(selected)


def seed_everything(seed):
    """Seed existing RNG families; None retains the original unseeded behavior."""
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    # torch.manual_seed also visits unavailable accelerator backends. Seed the
    # CPU generator directly, then only the accelerators that can be used.
    torch.random.default_generator.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
