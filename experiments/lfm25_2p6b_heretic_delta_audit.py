#!/usr/bin/env python3
"""Audit the disclosed Heretic Q8 delta against the exact official Q8 base."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import re
import sys

import numpy as np
from safetensors.torch import load_file
import torch
import torch.nn.functional as F

from experiments.lfm25_2p6b_residual_stream import FOLDS, RESIDUAL_CACHE
from heretic_nx.geometry.residual import fit_residual_stream_axes
from heretic_nx.hashing import canonical_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
GGUF_PY = PROJECT_ROOT / "references" / "llama.cpp-b10603" / "gguf-py"
sys.path.insert(0, str(GGUF_PY))

from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402


BASE_GGUF = (
    Path.home()
    / ".lmstudio"
    / "models"
    / "LiquidAI"
    / "LFM2.5-2.6B-GGUF"
    / "LFM2.5-2.6B-Q8_0.gguf"
)
HERETIC_GGUF = (
    ROOT
    / "references"
    / "abiray-lfm25-2p6b-q8"
    / "LFM2.5-2.6B-heretic-Q8_0.gguf"
)
RUN_DIR = ROOT / "runs" / "lfm25-2p6b-heretic-delta-audit"
REPORT = RUN_DIR / "report.json"
OUTPUT_PATTERN = re.compile(
    r"^blk\.(?P<layer>\d+)\."
    r"(?P<kind>attn_output|shortconv\.out_proj|ffn_down)\.weight$"
)


def family(kind: str) -> str:
    return {
        "attn_output": "gqa",
        "shortconv.out_proj": "liv",
        "ffn_down": "ffn",
    }[kind]


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    base_reader = GGUFReader(BASE_GGUF)
    heretic_reader = GGUFReader(HERETIC_GGUF)
    base_tensors = {tensor.name: tensor for tensor in base_reader.tensors}
    heretic_tensors = {tensor.name: tensor for tensor in heretic_reader.tensors}
    if set(base_tensors) != set(heretic_tensors):
        raise RuntimeError("GGUF tensor registries differ")
    changed = [
        name
        for name, tensor in base_tensors.items()
        if not np.array_equal(tensor.data, heretic_tensors[name].data)
    ]
    unexpected = [name for name in changed if OUTPUT_PATTERN.match(name) is None]
    if unexpected:
        raise RuntimeError(f"unexpected changed GGUF tensors: {unexpected}")

    cached = load_file(RESIDUAL_CACHE)
    axes = fit_residual_stream_axes(
        cached["safe"],
        cached["target"],
        folds=FOLDS,
        remove_safe_mean=True,
    )
    raw_axes = torch.stack([axis.axis.float() for axis in axes])

    rows = []
    left_vectors = []
    reference = None
    for index, name in enumerate(changed):
        match = OUTPUT_PATTERN.match(name)
        if match is None:
            raise AssertionError(name)
        base_tensor = base_tensors[name]
        heretic_tensor = heretic_tensors[name]
        base = dequantize(base_tensor.data, base_tensor.tensor_type)
        edited = dequantize(heretic_tensor.data, heretic_tensor.tensor_type)
        delta = torch.from_numpy(edited - base).to("cuda")
        base_gpu = torch.from_numpy(base).to("cuda")
        torch.manual_seed(3100 + index)
        u, singular, _v = torch.svd_lowrank(delta, q=6, niter=3)
        vector = u[:, 0].float().cpu()
        if reference is None:
            reference = vector
        elif float(torch.dot(vector, reference)) < 0:
            vector = -vector
        left_vectors.append(vector)
        total_energy = float(delta.square().sum())
        singular_energy = singular.float().square().cumsum(0).cpu()
        layer = int(match.group("layer"))
        row_norm_base = torch.linalg.vector_norm(base_gpu.float(), dim=1)
        row_norm_edited = torch.linalg.vector_norm(
            torch.from_numpy(edited).to("cuda").float(),
            dim=1,
        )
        rows.append(
            {
                "name": name,
                "layer": layer,
                "family": family(match.group("kind")),
                "shape": list(delta.shape),
                "relative_l2": float(
                    torch.linalg.vector_norm(delta)
                    / torch.linalg.vector_norm(base_gpu).clamp_min(1e-12)
                ),
                "rank1_energy_fraction": float(singular_energy[0] / total_energy),
                "rank3_energy_fraction": float(singular_energy[2] / total_energy),
                "rank6_energy_fraction": float(singular_energy[5] / total_energy),
                "row_norm_mean_relative_change": float(
                    (
                        (row_norm_edited - row_norm_base).abs()
                        / row_norm_base.clamp_min(1e-12)
                    ).mean()
                ),
                "left_vector_cosine_own_residual_axis": abs(
                    float(torch.dot(vector, raw_axes[layer]))
                ),
            }
        )
        print(
            json.dumps(
                {
                    "audited": index + 1,
                    "total": len(changed),
                    "name": name,
                    "relative_l2": rows[-1]["relative_l2"],
                    "rank3_energy": rows[-1]["rank3_energy_fraction"],
                }
            ),
            flush=True,
        )
        del base, edited, delta, base_gpu, u, singular, _v
        torch.cuda.empty_cache()

    consensus = F.normalize(torch.stack(left_vectors).mean(dim=0), dim=0)
    consensus_cosines = (raw_axes @ consensus).abs()
    consensus_ranking = [
        {"layer": int(layer), "cosine": float(consensus_cosines[layer])}
        for layer in torch.argsort(consensus_cosines, descending=True)
    ]
    report = {
        "schema_version": "lfm25-2p6b-heretic-q8-delta-audit-v1",
        "base": {
            "path": str(BASE_GGUF),
            "sha256": sha256_file(BASE_GGUF),
        },
        "heretic": {
            "path": str(HERETIC_GGUF),
            "sha256": sha256_file(HERETIC_GGUF),
        },
        "tensor_count": len(base_tensors),
        "changed_tensor_count": len(changed),
        "unchanged_tensor_count": len(base_tensors) - len(changed),
        "changed_tensors": rows,
        "families": {
            family_name: sum(row["family"] == family_name for row in rows)
            for family_name in ("liv", "gqa", "ffn")
        },
        "global_left_vector_residual_axis_ranking": consensus_ranking,
        "interpretation_guard": (
            "Deltas are reconstructed from two Q8_0 files. Exact raw equality "
            "identifies edited tensors, while low-rank diagnostics are approximate."
        ),
    }
    REPORT.write_bytes(canonical_json(report) + b"\n")
    print(
        json.dumps(
            {
                "changed": len(changed),
                "families": report["families"],
                "top_residual_layers": consensus_ranking[:10],
                "report": str(REPORT),
            },
            indent=2,
        ),
        flush=True,
    )
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
