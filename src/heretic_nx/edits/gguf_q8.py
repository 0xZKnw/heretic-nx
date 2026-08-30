"""Legacy Q8 plan API routed through the hardened mixed-GGUF backend."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from heretic_nx.hashing import canonical_json


SHA256_PATTERN = r"^[0-9a-f]{64}$"


class GGUFQ8TensorEdit(BaseModel):
    """One low-rank output-space ablation applied to a Q8_0 matrix."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    tensor_name: str = Field(min_length=1)
    a_key: str = Field(min_length=1)
    b_key: str | None = None
    right_key: str | None = None
    strength: float = Field(ge=0)
    preserve_row_norms: bool = True

    @model_validator(mode="after")
    def validate_factor_mode(self) -> "GGUFQ8TensorEdit":
        if self.right_key is not None and self.b_key is not None:
            raise ValueError("right_key and b_key are mutually exclusive")
        if self.right_key is not None and self.preserve_row_norms:
            raise ValueError("direct right-factor edits cannot preserve row norms")
        return self

    @property
    def resolved_b_key(self) -> str:
        return self.b_key or self.a_key


class GGUFQ8AblationPlan(BaseModel):
    """Backward-compatible, hash-bound Q8_0 ablation plan."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["gguf-q8-ablation-v1"] = "gguf-q8-ablation-v1"
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    tensor_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    edits: tuple[GGUFQ8TensorEdit, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_targets(self) -> "GGUFQ8AblationPlan":
        targets = [edit.tensor_name for edit in self.edits]
        if len(targets) != len(set(targets)):
            raise ValueError("each GGUF tensor may be edited at most once")
        return self

    @classmethod
    def read(cls, path: str | Path) -> "GGUFQ8AblationPlan":
        return cls.model_validate_json(Path(path).read_bytes())

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical_json(self.model_dump()) + b"\n")


def _gguf_api() -> tuple[Any, Any, Any, Any]:
    try:
        from gguf import GGMLQuantizationType, GGUFReader
        from gguf.quants import dequantize, quantize
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "direct GGUF editing requires the 'gguf' extra: "
            "python -m pip install -e '.[gguf]'"
        ) from error
    return GGUFReader, GGMLQuantizationType, dequantize, quantize


GGUFSnapshotCopyMode = Literal["clone", "copy"]


def _copy_source(
    source: Path,
    target: Path,
    *,
    minimum_free_after_copy: int = 0,
) -> GGUFSnapshotCopyMode:
    """Clone when available, otherwise budget and perform a full byte copy.

    ``minimum_free_after_copy`` lets multi-output transactions reserve space
    for later copy-on-write payloads before starting a potentially model-sized
    fallback copy. Existing callers keep the same behavior with the default.
    """

    if (
        isinstance(minimum_free_after_copy, bool)
        or not isinstance(minimum_free_after_copy, int)
        or minimum_free_after_copy < 0
    ):
        raise ValueError("minimum_free_after_copy must be a non-negative integer")

    if sys.platform == "darwin" and Path("/bin/cp").is_file():
        try:
            subprocess.run(
                ("/bin/cp", "-c", str(source), str(target)),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return "clone"
        except subprocess.CalledProcessError:
            pass
    source_size = source.stat().st_size
    required_free = source_size + minimum_free_after_copy
    free = shutil.disk_usage(target.parent).free
    if free < required_free:
        raise RuntimeError(
            "insufficient free space for full GGUF snapshot copy fallback: "
            f"need {required_free} bytes, have {free}"
        )
    shutil.copyfile(source, target)
    return "copy"


def inspect_q8_gguf(path: str | Path) -> dict[str, Any]:
    """Return edit-relevant Q8_0 metadata without dequantizing weights."""

    GGUFReader, GGMLQuantizationType, _dequantize, _quantize = _gguf_api()
    source = Path(path).expanduser().resolve(strict=True)
    reader = GGUFReader(source)
    q8 = [
        {
            "name": tensor.name,
            "shape": [int(value) for value in reversed(tensor.shape.tolist())],
            "quantized_bytes": int(tensor.n_bytes),
            "data_offset": int(tensor.data_offset),
        }
        for tensor in reader.tensors
        if tensor.tensor_type == GGMLQuantizationType.Q8_0
    ]
    return {
        "schema_version": "gguf-q8-inspection-v1",
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "tensor_count": len(reader.tensors),
        "q8_0_tensor_count": len(q8),
        "q8_0_tensors": q8,
    }


def apply_q8_gguf_ablation(
    source_path: str | Path,
    output_path: str | Path | None,
    plan_path: str | Path,
    tensor_path: str | Path,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Apply a v1 Q8 plan through the hardened generic backend."""

    # Imported lazily to avoid the compatibility module cycle: gguf_quant uses
    # the v1 plan models and clone helper defined above.
    from .gguf_quant import apply_quantized_gguf_ablation

    return apply_quantized_gguf_ablation(
        source_path,
        output_path,
        plan_path,
        tensor_path,
        dry_run=dry_run,
        force=force,
    )
