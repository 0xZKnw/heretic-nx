#!/usr/bin/env python3
"""Refusal-first evaluation for direct-Q8 Ling-3.0-tiny candidates."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

from experiments.lfm25_2p6b_residual_stream import (
    BAD_DATASET,
    BAD_REVISION,
    MAX_NEW_TOKENS,
    REFUSAL_MARKERS,
    normalized_response,
)
from heretic_nx.eval.gguf_runtime import native_completion
from heretic_nx.hashing import canonical_json, sha256_json


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT / "checkpoints" / "ling3-tiny"
RUN_DIR = ROOT / "runs" / "ling3-tiny-q8-direct"
BATCH_SIZE = 8
SYSTEM_PROMPT = "You are a helpful assistant."


def render(tokenizer: Any, prompts: list[str]) -> list[str]:
    chats = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        for prompt in prompts
    ]
    return list(
        tokenizer.apply_chat_template(
            chats,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(value) + b"\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:1236")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=104)
    parser.add_argument(
        "--indices",
        help="comma-separated one-based row indices; overrides start/stop",
    )
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.parallel <= 0:
        raise ValueError("parallel must be positive")
    if not 0 <= args.start < args.stop <= 104:
        raise ValueError("evaluation bounds must satisfy 0 <= start < stop <= 104")
    selected_indices = None
    if args.indices is not None:
        try:
            selected_indices = [int(value) - 1 for value in args.indices.split(",")]
        except ValueError as error:
            raise ValueError("indices must be comma-separated one-based integers") from error
        if (
            not selected_indices
            or len(selected_indices) != len(set(selected_indices))
            or min(selected_indices) < 0
            or max(selected_indices) >= 104
        ):
            raise ValueError("indices must be unique values between 1 and 104")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    rows = load_dataset(BAD_DATASET, revision=BAD_REVISION, split="test")
    row_indices = (
        selected_indices
        if selected_indices is not None
        else list(range(args.start, args.stop))
    )
    rendered = render(tokenizer, [str(rows[index]["text"]) for index in row_indices])
    prompt_tokens = [
        [int(token) for token in tokenizer.encode(prompt, add_special_tokens=False)]
        for prompt in rendered
    ]
    expected = {
        "schema_version": "ling3-tiny-q8-refusal-partial-v1",
        "model": args.model,
        "label": args.label,
        "bounds": [args.start, args.stop] if selected_indices is None else None,
        "row_indices_one_based": [index + 1 for index in row_indices],
        "prompt_tokens_sha256": sha256_json(prompt_tokens),
        "max_new_tokens": args.max_new_tokens,
        "temperature": -1,
        "thinking": "off",
        "api": "native pre-tokenized /completion",
    }
    partial = RUN_DIR / f"{args.label}.partial.json"
    checkpoint: dict[str, Any] = {
        **expected,
        "completed": 0,
        "responses": [],
        "seconds": 0.0,
    }
    if partial.is_file():
        loaded = json.loads(partial.read_text(encoding="utf-8"))
        if not all(loaded.get(key) == value for key, value in expected.items()):
            raise RuntimeError(f"stale evaluation checkpoint: {partial}")
        checkpoint = loaded
    responses = [str(value) for value in checkpoint["responses"]]
    completed = int(checkpoint["completed"])
    if completed != len(responses) or completed > len(prompt_tokens):
        raise RuntimeError(f"invalid evaluation checkpoint: {partial}")

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for start in range(completed, len(prompt_tokens), BATCH_SIZE):
            batch = prompt_tokens[start : start + BATCH_SIZE]
            started = time.time()
            produced = list(
                pool.map(
                    lambda tokens: native_completion(
                        args.endpoint,
                        tokens,
                        max_tokens=args.max_new_tokens,
                    ),
                    batch,
                )
            )
            checkpoint["seconds"] = float(checkpoint["seconds"]) + (
                time.time() - started
            )
            responses.extend(produced)
            checkpoint["responses"] = responses
            checkpoint["completed"] = len(responses)
            write_json(partial, checkpoint)
            print(
                json.dumps(
                    {
                        "evaluation": args.label,
                        "completed": len(responses),
                        "total": len(prompt_tokens),
                        "seconds": round(float(checkpoint["seconds"]), 3),
                    }
                ),
                flush=True,
            )

    hits = [
        int(any(marker in normalized_response(text) for marker in REFUSAL_MARKERS))
        for text in responses
    ]
    report = {
        "schema_version": "ling3-tiny-q8-refusal-v1",
        "model": args.model,
        "label": args.label,
        "dataset": {"id": BAD_DATASET, "revision": BAD_REVISION},
        "protocol": expected,
        "count": len(responses),
        "refusal_markers": sum(hits),
        "marker_hits": hits,
        "hit_rows_one_based": [
            index + 1 for index, hit in zip(row_indices, hits) if hit
        ],
        "response_sha256": sha256_json(responses),
        "seconds": float(checkpoint["seconds"]),
        "responses": responses,
    }
    report_path = args.report or RUN_DIR / f"{args.label}.json"
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "label": args.label,
                "count": len(responses),
                "refusal_markers": sum(hits),
                "hit_rows_one_based": report["hit_rows_one_based"],
                "seconds": report["seconds"],
                "report": str(report_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
