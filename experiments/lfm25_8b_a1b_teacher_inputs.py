#!/usr/bin/env python3
"""Collect input states for distilling the compact eight-site Q8 teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from datasets import load_dataset
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
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
from experiments.lfm25_8b_a1b_mlx_sites import MODEL_PATH, TOKENIZER_PATH
from experiments.lfm25_8b_a1b_q8_build import RUN_DIR
from heretic_nx.hashing import canonical_json, sha256_json


TEACHER_MERGE = RUN_DIR / "prime-ops8-b2.merge.json"
BASE_REPORT = RUN_DIR / "base-full.json"
TEACHER_REPORT = RUN_DIR / "prime-ops8-b2-full.json"
OUTPUT = RUN_DIR / "teacher-op-inputs.safetensors"
SAFE_ARRAY = RUN_DIR / "teacher-op-inputs.safe.npy"
HARMFUL_ARRAY = RUN_DIR / "teacher-op-inputs.harmful.npy"
PROGRESS = RUN_DIR / "teacher-op-inputs.progress.json"
SAFE_COUNT = 1024
PAIR_COUNT = 32
WIDTH = 2048
MAX_LENGTH = 512


class CaptureLinear(nn.Module):
    def __init__(self, linear: nn.Module):
        super().__init__()
        self.linear = linear
        self.value: mx.array | None = None

    def __call__(self, value: mx.array) -> mx.array:
        self.value = value
        return self.linear(value)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


def selected_sites() -> list[dict[str, object]]:
    report = json.loads(TEACHER_MERGE.read_text(encoding="utf-8"))
    rows = list(report["candidate"]["selected"])
    if len(rows) != 8 or any(row["family"] not in {"gqa", "liv"} for row in rows):
        raise RuntimeError("the pinned teacher must contain eight operator sites")
    return rows


def install_captures(model: object, sites: list[dict[str, object]]) -> list[CaptureLinear]:
    captures = []
    for row in sites:
        layer = model.model.layers[int(row["layer"])]
        operator = layer.self_attn if row["family"] == "gqa" else layer.conv
        capture = CaptureLinear(operator.out_proj)
        operator.out_proj = capture
        captures.append(capture)
    return captures


def captured_stack(captures: list[CaptureLinear], start: int | None) -> np.ndarray:
    values = []
    for capture in captures:
        if capture.value is None:
            raise RuntimeError("an operator input capture did not fire")
        value = capture.value[0]
        value = value[-1:] if start is None else value[start:]
        values.append(value.astype(mx.float32))
    mx.eval(*values)
    stacked = np.stack([np.asarray(value) for value in values], axis=1)
    if stacked.ndim != 3 or stacked.shape[1:] != (len(captures), WIDTH):
        raise RuntimeError(f"invalid captured operator inputs: {stacked.shape}")
    if not np.isfinite(stacked).all():
        raise RuntimeError("captured operator inputs contain non-finite values")
    return stacked.astype(np.float16, copy=False)


def encode(tokenizer: object, value: str) -> list[int]:
    tokens = tokenizer.encode(value, add_special_tokens=False)
    if len(tokens) > MAX_LENGTH:
        tokens = [tokens[0], *tokens[-(MAX_LENGTH - 1) :]]
    return [int(token) for token in tokens]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--safe-count", type=int, default=SAFE_COUNT)
    parser.add_argument("--pair-count", type=int, default=PAIR_COUNT)
    args = parser.parse_args()
    if not 1 <= args.safe_count <= SAFE_COUNT:
        raise ValueError(f"safe-count must be between 1 and {SAFE_COUNT}")
    if not 1 <= args.pair_count <= PAIR_COUNT:
        raise ValueError(f"pair-count must be between 1 and {PAIR_COUNT}")
    required = (args.model, TEACHER_MERGE, BASE_REPORT, TEACHER_REPORT)
    if not all(path.exists() for path in required):
        raise RuntimeError("the model or pinned teacher evidence is missing")

    sites = selected_sites()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    safe_rows = load_dataset(
        GOOD_DATASET,
        revision=GOOD_REVISION,
        split=f"train[:{args.safe_count}]",
    )
    safe_rendered = render(
        tokenizer,
        [str(row["text"]) for row in safe_rows],
        close_think=True,
    )
    safe_tokens = [encode(tokenizer, value) for value in safe_rendered]

    base = json.loads(BASE_REPORT.read_text(encoding="utf-8"))
    teacher = json.loads(TEACHER_REPORT.read_text(encoding="utf-8"))
    target_rows = load_dataset(BAD_DATASET, revision=BAD_REVISION, split="test")
    target_rendered = render(
        tokenizer,
        [str(target_rows[index]["text"]) for index in range(104)],
        close_think=True,
    )
    eligible = [
        index
        for index, (base_hit, teacher_hit) in enumerate(
            zip(base["marker_hits"], teacher["marker_hits"])
        )
        if int(base_hit) == 1 and int(teacher_hit) == 0
    ]
    if len(eligible) < args.pair_count:
        raise RuntimeError("too few base-refusal/teacher-success trajectories")
    # Spread the limited trajectory budget across the full 104-row suite.
    positions = np.linspace(0, len(eligible) - 1, args.pair_count).round().astype(int)
    pair_indices = [eligible[int(position)] for position in positions]
    harmful_rows = []
    response_starts = []
    for index in pair_indices:
        prompt = encode(tokenizer, target_rendered[index])
        response = tokenizer.encode(
            str(base["responses"][index]), add_special_tokens=False
        )
        combined = prompt + [int(token) for token in response]
        if len(combined) > MAX_LENGTH:
            response = response[: max(MAX_LENGTH - len(prompt), 1)]
            combined = prompt + [int(token) for token in response]
        harmful_rows.append(combined)
        response_starts.append(len(prompt))
    harmful_tokens = sum(len(row) - start for row, start in zip(harmful_rows, response_starts))

    expected = {
        "schema_version": "lfm25-8b-a1b-teacher-op-inputs-v1",
        "model": str(args.model.resolve()),
        "safe_count": args.safe_count,
        "pair_count": args.pair_count,
        "pair_indices": pair_indices,
        "site_ids": [str(row["site_id"]) for row in sites],
        "safe_tokens_sha256": sha256_json(safe_tokens),
        "harmful_tokens_sha256": sha256_json(harmful_rows),
        "harmful_response_tokens": harmful_tokens,
    }
    progress = {**expected, "phase": "safe", "completed": 0, "offset": 0, "seconds": 0.0}
    mode = "w+"
    if PROGRESS.is_file():
        loaded = json.loads(PROGRESS.read_text(encoding="utf-8"))
        if not all(loaded.get(key) == value for key, value in expected.items()):
            raise RuntimeError(f"stale teacher-input checkpoint: {PROGRESS}")
        progress = loaded
        mode = "r+"
    safe = np.lib.format.open_memmap(
        SAFE_ARRAY,
        mode=mode,
        dtype=np.float16,
        shape=(args.safe_count, len(sites), WIDTH),
    )
    harmful = np.lib.format.open_memmap(
        HARMFUL_ARRAY,
        mode=mode,
        dtype=np.float16,
        shape=(harmful_tokens, len(sites), WIDTH),
    )

    print(json.dumps({"load": str(args.model), "sites": len(sites)}), flush=True)
    model, _ = load(args.model, lazy=True)
    captures = install_captures(model, sites)
    if progress["phase"] == "safe":
        for index in range(int(progress["completed"]), len(safe_tokens)):
            started = time.time()
            inputs = mx.array([safe_tokens[index]], dtype=mx.int32)
            output = model.model(inputs)
            mx.eval(output)
            safe[index] = captured_stack(captures, None)[0]
            safe.flush()
            progress.update(
                {
                    "completed": index + 1,
                    "seconds": float(progress["seconds"]) + time.time() - started,
                }
            )
            write_json(PROGRESS, progress)
            if (index + 1) % 64 == 0 or index + 1 == len(safe_tokens):
                print(
                    json.dumps(
                        {
                            "collect": "safe",
                            "completed": index + 1,
                            "total": len(safe_tokens),
                            "seconds": round(float(progress["seconds"]), 3),
                        }
                    ),
                    flush=True,
                )
        progress.update({"phase": "harmful", "completed": 0, "offset": 0})
        write_json(PROGRESS, progress)

    offset = int(progress["offset"])
    for index in range(int(progress["completed"]), len(harmful_rows)):
        started = time.time()
        inputs = mx.array([harmful_rows[index]], dtype=mx.int32)
        output = model.model(inputs)
        mx.eval(output)
        values = captured_stack(captures, response_starts[index])
        stop = offset + len(values)
        harmful[offset:stop] = values
        harmful.flush()
        offset = stop
        progress.update(
            {
                "completed": index + 1,
                "offset": offset,
                "seconds": float(progress["seconds"]) + time.time() - started,
            }
        )
        write_json(PROGRESS, progress)
        print(
            json.dumps(
                {
                    "collect": "harmful",
                    "completed": index + 1,
                    "total": len(harmful_rows),
                    "tokens": offset,
                    "seconds": round(float(progress["seconds"]), 3),
                }
            ),
            flush=True,
        )
    if offset != harmful_tokens:
        raise RuntimeError(f"harmful token count mismatch: {offset} != {harmful_tokens}")
    save_file(
        {
            "safe": torch.from_numpy(np.asarray(safe).copy()),
            "harmful": torch.from_numpy(np.asarray(harmful).copy()),
        },
        OUTPUT,
        metadata={"manifest_sha256": sha256_json(expected)},
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "safe_shape": list(safe.shape),
                "harmful_shape": list(harmful.shape),
                "seconds": progress["seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
