"""Deterministic golden-tensor fingerprints and numerical comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from heretic_nx.hashing import sha256_bytes, sha256_json


def tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    metadata = sha256_json({"shape": list(value.shape), "dtype": str(value.dtype)})
    return sha256_bytes(metadata.encode("ascii") + value.view(torch.uint8).numpy().tobytes())


@dataclass(frozen=True)
class GoldenTensor:
    name: str
    sha256: str
    shape: tuple[int, ...]
    dtype: str

    @classmethod
    def capture(cls, name: str, tensor: Tensor) -> "GoldenTensor":
        return cls(name, tensor_sha256(tensor), tuple(tensor.shape), str(tensor.dtype))


@dataclass(frozen=True)
class GoldenComparison:
    maximum_absolute_error: float
    maximum_relative_error: float
    within_tolerance: bool


def compare_golden(
    actual: Tensor,
    expected: Tensor,
    *,
    atol: float,
    rtol: float,
) -> GoldenComparison:
    if actual.shape != expected.shape:
        raise ValueError("golden tensors must have the same shape")
    difference = (actual.float() - expected.float()).abs()
    relative = difference / expected.float().abs().clamp_min(torch.finfo(torch.float32).eps)
    return GoldenComparison(
        maximum_absolute_error=float(difference.max().item()) if difference.numel() else 0.0,
        maximum_relative_error=float(relative.max().item()) if relative.numel() else 0.0,
        within_tolerance=bool(torch.all(difference <= atol + rtol * expected.float().abs()).item()),
    )
