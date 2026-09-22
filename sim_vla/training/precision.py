"""Online autocast without changing checkpoint parameter dtypes."""

import contextlib

import torch


def autocast(device, precision: str):
    if precision not in ("float32", "bfloat16"):
        raise ValueError("online precision must be float32 or bfloat16")
    device = torch.device(device)
    if precision == "float32":
        return contextlib.nullcontext()
    if device.type not in ("cpu", "cuda"):
        raise ValueError(f"bfloat16 autocast is unsupported on {device.type}")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
