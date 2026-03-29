from __future__ import annotations

from typing import Any

import torch


DeviceLike = Any


def _get_directml_device() -> DeviceLike | None:
    try:
        import torch_directml
    except ImportError:
        return None

    try:
        return torch_directml.device()
    except Exception:
        return None


def resolve_device(preferred: str = "auto") -> tuple[DeviceLike, str, str]:
    """解析运行设备，默认优先 DirectML。"""
    normalized = preferred.lower()
    valid = {"auto", "directml", "cuda", "cpu"}
    if normalized not in valid:
        raise RuntimeError(f"不支持的设备选项: {preferred}")

    directml_device = _get_directml_device()

    if normalized == "directml":
        if directml_device is None:
            raise RuntimeError(
                "请求使用 DirectML，但当前环境不可用；请安装 torch-directml 并在受支持的平台上运行，或改用 --device cuda/cpu"
            )
        return directml_device, "directml", "显式指定设备: directml"

    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("请求使用 CUDA，但当前环境未检测到可用 CUDA 设备")
        return torch.device("cuda"), "cuda", "显式指定设备: cuda"

    if normalized == "cpu":
        return torch.device("cpu"), "cpu", "显式指定设备: cpu"

    if directml_device is not None:
        return directml_device, "directml", "自动选择设备: directml"

    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda", "自动选择设备: cuda（DirectML 不可用）"

    return torch.device("cpu"), "cpu", "自动选择设备: cpu（DirectML 和 CUDA 均不可用）"