#!/usr/bin/env python3
"""Collect semantic projection activations for a PRIME Q8 search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from datasets import load_dataset
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
import numpy as np
from safetensors.torch import save_file
import torch
from transformers import AutoTokenizer

from experiments.lfm25_2p6b_residual_stream import (
    BAD_DATASET,
    BAD_REVISION,
    GOOD_DATASET,
    GOOD_REVISION,
    render,
)
from experiments.lfm25_8b_a1b_q8_build import SOURCE
from heretic_nx.data.research_splits import (
    build_research_split,
    verify_manifest_texts,
)
from heretic_nx.hashing import (
    canonical_json,
    sha256_directory,
    sha256_file,
    sha256_json,
)


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT / "checkpoints" / "lfm25-8b-a1b"
MODEL_PATH = ROOT / "checkpoints" / "lfm25-8b-a1b-mlx-8bit"
RUN_DIR = ROOT / "runs" / "lfm25-8b-a1b-q8-direct"
OUTPUT = RUN_DIR / "mlx-site-activations.safetensors"
PROGRESS = RUN_DIR / "mlx-site-activations.progress.json"
SAFE_ARRAY = RUN_DIR / "mlx-site-activations.safe.npy"
TARGET_ARRAY = RUN_DIR / "mlx-site-activations.target.npy"
COUNT = 400
POOL_COUNT = 1024
SPLIT_SEED = 20260830
LAYERS = 24
SITES = 48
WIDTH = 2048
MAX_LENGTH = 512


def site_specs(layer_types: list[str]) -> list[dict[str, object]]:
    if len(layer_types) != LAYERS:
        raise RuntimeError(f"expected {LAYERS} layer types, got {len(layer_types)}")
    rows = []
    for layer, layer_type in enumerate(layer_types):
        if layer_type == "full_attention":
            family, kind = "gqa", "attention_out"
        elif layer_type == "conv":
            family, kind = "liv", "liv_mix_out"
        else:
            raise RuntimeError(f"unsupported layer type at {layer}: {layer_type}")
        rows.append(
            {
                "index": len(rows),
                "site_id": f"L{layer:02d}:{kind}",
                "layer": layer,
                "family": family,
                "kind": kind,
            }
        )
        rows.append(
            {
                "index": len(rows),
                "site_id": f"L{layer:02d}:ffn_out",
                "layer": layer,
                "family": "ffn",
                "kind": "ffn_out",
            }
        )
    return rows


def semantic_stack(model: object, tokens: list[int]) -> np.ndarray:
    inputs = mx.array([tokens], dtype=mx.int32)
    core = model.model
    h = core.embed_tokens(inputs)
    cache = [None] * len(core.layers)
    attention_mask = create_attention_mask(h, cache[core.fa_idx])
    convolution_mask = create_ssm_mask(h, cache[core.conv_idx])
    final_states = []
    for layer, layer_cache in zip(core.layers, cache):
        mask = attention_mask if layer.is_attention_layer else convolution_mask
        normalized = layer.operator_norm(h)
        if layer.is_attention_layer:
            operator = layer.self_attn(normalized, mask=mask, cache=layer_cache)
        else:
            operator = layer.conv(normalized, mask=mask, cache=layer_cache)
        middle = h + operator
        ffn = layer.feed_forward(layer.ffn_norm(middle))
        h = middle + ffn
        final_states.extend(
            (operator[0, -1].astype(mx.float32), ffn[0, -1].astype(mx.float32))
        )
    mx.eval(*final_states)
    value = np.stack([np.asarray(state) for state in final_states]).astype(
        np.float16,
        copy=False,
    )
    if value.shape != (SITES, WIDTH) or not np.isfinite(value).all():
        raise RuntimeError(f"invalid semantic activation stack: {value.shape}")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


def encode_rows(tokenizer: object, values: list[str]) -> list[list[int]]:
    rows = []
    for value in values:
        tokens = tokenizer.encode(value, add_special_tokens=False)
        if len(tokens) > MAX_LENGTH:
            tokens = [tokens[0], *tokens[-(MAX_LENGTH - 1) :]]
        rows.append(tokens)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--count", type=int, default=COUNT)
    parser.add_argument("--pool-count", type=int, default=POOL_COUNT)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()
    if not 1 <= args.count <= COUNT:
        raise ValueError(f"count must be between 1 and {COUNT}")
    if not args.model.is_dir():
        raise RuntimeError(f"MLX teacher is missing: {args.model}")
    if args.pool_count < args.count:
        raise ValueError("pool-count must be at least count")
    if not args.source.is_file():
        raise RuntimeError(f"target Q8 source is missing: {args.source}")

    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    specs = site_specs([str(value) for value in config["layer_types"]])
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    safe_rows = load_dataset(
        GOOD_DATASET,
        revision=GOOD_REVISION,
        split=f"train[:{args.pool_count}]",
    )
    target_rows = load_dataset(
        BAD_DATASET,
        revision=BAD_REVISION,
        split=f"train[:{args.pool_count}]",
    )
    safe_pool = [str(row["text"]) for row in safe_rows]
    target_pool = [str(row["text"]) for row in target_rows]
    safe_manifest = build_research_split(
        safe_pool,
        purpose="geometry",
        dataset_id=GOOD_DATASET,
        revision=GOOD_REVISION,
        source_split="train",
        seed=args.split_seed,
        count=args.count,
    )
    target_manifest = build_research_split(
        target_pool,
        purpose="geometry",
        dataset_id=BAD_DATASET,
        revision=BAD_REVISION,
        source_split="train",
        seed=args.split_seed,
        count=args.count,
    )
    prompts = {
        "safe": render(
            tokenizer,
            verify_manifest_texts(safe_manifest, safe_pool),
            close_think=False,
        ),
        "target": render(
            tokenizer,
            verify_manifest_texts(target_manifest, target_pool),
            close_think=False,
        ),
    }
    token_rows = {
        label: encode_rows(tokenizer, values) for label, values in prompts.items()
    }
    expected = {
        "schema_version": "lfm25-8b-a1b-mlx-sites-progress-v2",
        "model": str(args.model.resolve()),
        "model_sha256": sha256_directory(args.model),
        "source": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "count": args.count,
        "pool_count": args.pool_count,
        "split_seed": args.split_seed,
        "sites": specs,
        "sites_sha256": sha256_json(specs),
        "width": WIDTH,
        "max_length": MAX_LENGTH,
        "close_think": False,
        "safe_split_manifest": safe_manifest.to_dict(),
        "safe_split_manifest_sha256": safe_manifest.sha256,
        "target_split_manifest": target_manifest.to_dict(),
        "target_split_manifest_sha256": target_manifest.sha256,
        "safe_tokens_sha256": sha256_json(token_rows["safe"]),
        "target_tokens_sha256": sha256_json(token_rows["target"]),
    }
    progress = {
        **expected,
        "label": "safe",
        "completed": 0,
        "seconds": 0.0,
        "complete": False,
        "output_sha256": None,
    }
    mode = "w+"
    if PROGRESS.is_file():
        loaded = json.loads(PROGRESS.read_text(encoding="utf-8"))
        if not all(loaded.get(key) == value for key, value in expected.items()):
            raise RuntimeError(f"stale site checkpoint: {PROGRESS}")
        label = loaded.get("label")
        completed = loaded.get("completed")
        if (
            label not in {"safe", "target"}
            or isinstance(completed, bool)
            or not isinstance(completed, int)
            or not 0 <= completed <= args.count
            or not isinstance(loaded.get("complete"), bool)
        ):
            raise RuntimeError(f"invalid site checkpoint: {PROGRESS}")
        if loaded["complete"]:
            if (
                label != "target"
                or completed != args.count
                or not OUTPUT.is_file()
                or loaded.get("output_sha256") != sha256_file(OUTPUT)
            ):
                raise RuntimeError(f"corrupt completed site checkpoint: {PROGRESS}")
            print(
                json.dumps(
                    {
                        "output": str(OUTPUT),
                        "output_sha256": loaded["output_sha256"],
                        "reused": True,
                    },
                    indent=2,
                ),
                flush=True,
            )
            return
        if loaded.get("output_sha256") is not None:
            raise RuntimeError(f"partial site checkpoint claims an output: {PROGRESS}")
        progress = loaded
        mode = "r+"
    safe = np.lib.format.open_memmap(
        SAFE_ARRAY,
        mode=mode,
        dtype=np.float16,
        shape=(args.count, SITES, WIDTH),
    )
    target = np.lib.format.open_memmap(
        TARGET_ARRAY,
        mode=mode,
        dtype=np.float16,
        shape=(args.count, SITES, WIDTH),
    )
    print(json.dumps({"load": str(args.model), "sites": len(specs)}), flush=True)
    model, _ = load(args.model, lazy=True)
    labels = ("safe", "target")
    start_label = labels.index(str(progress["label"]))
    for label_index in range(start_label, len(labels)):
        label = labels[label_index]
        matrix = safe if label == "safe" else target
        completed = int(progress["completed"]) if label_index == start_label else 0
        for index in range(completed, args.count):
            started = time.time()
            matrix[index] = semantic_stack(model, token_rows[label][index])
            matrix.flush()
            progress["label"] = label
            progress["completed"] = index + 1
            progress["seconds"] = float(progress["seconds"]) + time.time() - started
            write_json(PROGRESS, progress)
            if (index + 1) % 8 == 0 or index + 1 == args.count:
                print(
                    json.dumps(
                        {
                            "collect": label,
                            "completed": index + 1,
                            "total": args.count,
                            "seconds": round(float(progress["seconds"]), 3),
                        }
                    ),
                    flush=True,
                )
        if label_index + 1 < len(labels):
            progress["label"] = labels[label_index + 1]
            progress["completed"] = 0
            write_json(PROGRESS, progress)

    save_file(
        {
            "safe": torch.from_numpy(np.asarray(safe).copy()),
            "target": torch.from_numpy(np.asarray(target).copy()),
        },
        OUTPUT,
        metadata={
            "manifest_sha256": sha256_json(expected),
            "sites_sha256": sha256_json(specs),
            "safe_split_manifest_sha256": safe_manifest.sha256,
            "target_split_manifest_sha256": target_manifest.sha256,
            "model_sha256": str(expected["model_sha256"]),
            "source_sha256": str(expected["source_sha256"]),
        },
    )
    progress["label"] = "target"
    progress["completed"] = args.count
    progress["complete"] = True
    progress["output_sha256"] = sha256_file(OUTPUT)
    write_json(PROGRESS, progress)
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "shape": [args.count, SITES, WIDTH],
                "seconds": progress["seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
