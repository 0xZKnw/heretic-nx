#!/usr/bin/env python3
"""Materialize the holdout-passing residual-stream candidate in BF16."""

from __future__ import annotations

import gc
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.lfm25_prime_uncensor import (
    EXPECTED_MODEL_SHA256,
    FRESH_PATH,
    MODEL_REVISION,
)
from experiments.lfm25_residual_stream_select import (
    REPORT,
    RESIDUALS,
    apply_scale,
    build_residual_editors,
)
from heretic_nx.edits.residual_stream import snapshot_residual_stream_weights
from heretic_nx.hashing import canonical_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
RUN_DIR = ROOT / "runs" / "lfm25-residual-stream-build"
BUILD_REPORT = RUN_DIR / "report.json"
OUTPUT_PATH = (
    PROJECT_ROOT / "outputs" / "LFM2.5-1.2B-Thinking-Heretic-NX-Residual-Stream"
)


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {OUTPUT_PATH}")
    source_report = json.loads(REPORT.read_text(encoding="utf-8"))
    if not source_report.get("passed") or source_report.get("selected") is None:
        raise RuntimeError("residual-stream candidate has not passed its locked holdout")
    scale = float(source_report["selected"]["scale"])
    cached = load_file(RESIDUALS)
    tokenizer = AutoTokenizer.from_pretrained(FRESH_PATH)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        FRESH_PATH, dtype=torch.bfloat16, device_map=0
    ).eval()
    editors = build_residual_editors(model, cached["safe"], cached["target"])
    originals = snapshot_residual_stream_weights(model, editors)
    apply_scale(model, editors, originals, scale)
    model.save_pretrained(
        OUTPUT_PATH,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(OUTPUT_PATH)
    for filename in ("LICENSE", "chat_template.jinja"):
        source = FRESH_PATH / filename
        if source.exists():
            shutil.copy2(source, OUTPUT_PATH / filename)

    build = {
        "schema_version": "lfm25-residual-stream-build-v1",
        "engine": "Heretic NX",
        "algorithm_profile": "Residual-Stream",
        "validation_protocol": "PRIME",
        "source_model": "LiquidAI/LFM2.5-1.2B-Thinking",
        "source_revision": MODEL_REVISION,
        "source_checkpoint_sha256": sha256_file(FRESH_PATH / "model.safetensors"),
        "expected_source_checkpoint_sha256": EXPECTED_MODEL_SHA256.lower(),
        "selection_report": "runs/lfm25-residual-stream-selection/report.json",
        "selection_report_sha256": sha256_file(REPORT),
        "residual_cache_sha256": sha256_file(RESIDUALS),
        "scale": scale,
        "active_sites": len(editors),
        "dtype": "bfloat16",
        "quantized": False,
        "output_checkpoint_sha256": sha256_file(OUTPUT_PATH / "model.safetensors"),
    }
    metadata_path = OUTPUT_PATH / "HERETIC_NX_BUILD.json"
    metadata_path.write_bytes(canonical_json(build) + b"\n")
    build["metadata_sha256"] = sha256_file(metadata_path)
    BUILD_REPORT.write_bytes(canonical_json(build) + b"\n")
    print(json.dumps(build, indent=2))
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
