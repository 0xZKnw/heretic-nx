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
from heretic_nx.data.research_splits import (
    REFUSAL_NORMALIZER_V1,
    build_research_split,
    refusal_marker_rule_sha256,
    subset_research_split,
    verify_manifest_texts,
)
from heretic_nx.eval.gguf_runtime import (
    attest_native_model,
    NativeRuntimeClient,
    require_native_model_identity,
)
from heretic_nx.hashing import canonical_json, sha256_json


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT / "checkpoints" / "ling3-tiny"
RUN_DIR = ROOT / "runs" / "ling3-tiny-q8-direct"
BATCH_SIZE = 8
SYSTEM_PROMPT = "You are a helpful assistant."
POOL_ROWS = 400
SPLIT_SEED = 20260830
MARKER_RULE_SHA256 = refusal_marker_rule_sha256(REFUSAL_MARKERS)


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
    parser.add_argument(
        "--phase",
        choices=("geometry", "selection", "public-report"),
        default="public-report",
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--artifact-sha256")
    parser.add_argument("--pool-rows", type=int, default=POOL_ROWS)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument(
        "--indices",
        help="comma-separated one-based row indices; overrides start/stop",
    )
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--refusal-cap",
        type=int,
        help="stop screening as soon as cap + 1 refusal markers are observed",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.parallel <= 0:
        raise ValueError("parallel must be positive")
    if args.pool_rows <= 0:
        raise ValueError("pool-rows must be positive")
    if args.refusal_cap is not None and args.refusal_cap < 0:
        raise ValueError("refusal-cap must be non-negative")
    selected_indices = None
    if args.indices is not None:
        if args.phase != "public-report":
            raise ValueError("explicit official row indices are public-report only")
        try:
            selected_indices = [int(value) - 1 for value in args.indices.split(",")]
        except ValueError as error:
            raise ValueError(
                "indices must be comma-separated one-based integers"
            ) from error
        if (
            not selected_indices
            or len(selected_indices) != len(set(selected_indices))
            or min(selected_indices) < 0
        ):
            raise ValueError("indices must be unique positive one-based values")

    runtime_model = attest_native_model(
        args.endpoint,
        args.artifact,
        expected_model=args.model,
    )
    if (
        args.artifact_sha256 is not None
        and args.artifact_sha256 != runtime_model["artifact_sha256"]
    ):
        raise RuntimeError("attested artifact does not match --artifact-sha256")
    artifact_sha256 = str(runtime_model["artifact_sha256"])
    client = NativeRuntimeClient(args.endpoint)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    source_split = "test" if args.phase == "public-report" else "train"
    dataset_split = (
        "test" if source_split == "test" else f"train[:{args.pool_rows}]"
    )
    rows = load_dataset(BAD_DATASET, revision=BAD_REVISION, split=dataset_split)
    pool = [str(row["text"]) for row in rows]
    full_manifest = build_research_split(
        pool,
        purpose=args.phase,
        dataset_id=BAD_DATASET,
        revision=BAD_REVISION,
        source_split=source_split,
        seed=args.split_seed,
    )
    stop = len(full_manifest.rows) if args.stop is None else args.stop
    if selected_indices is not None:
        positions = selected_indices
        bounds = None
    else:
        if not 0 <= args.start < stop <= len(full_manifest.rows):
            raise ValueError(
                "evaluation bounds must select rows inside the frozen split"
            )
        positions = list(range(args.start, stop))
        bounds = [args.start, stop]
    manifest = subset_research_split(full_manifest, positions)
    raw_prompts = verify_manifest_texts(manifest, pool)
    rendered = render(tokenizer, raw_prompts)
    prompt_tokens = [
        [int(token) for token in tokenizer.encode(prompt, add_special_tokens=False)]
        for prompt in rendered
    ]
    expected = {
        "schema_version": "ling3-tiny-q8-refusal-partial-v2",
        "model": args.model,
        "artifact_sha256": artifact_sha256,
        "artifact_attested": True,
        "runtime_model": runtime_model,
        "label": args.label,
        "purpose": args.phase,
        "bounds": bounds,
        "source_indices_one_based": [row.source_index + 1 for row in manifest.rows],
        "split_manifest_sha256": manifest.sha256,
        "full_split_manifest_sha256": full_manifest.sha256,
        "full_split_count": len(full_manifest.rows),
        "prompt_tokens_sha256": sha256_json(prompt_tokens),
        "refusal_markers": list(REFUSAL_MARKERS),
        "refusal_normalizer": REFUSAL_NORMALIZER_V1,
        "marker_rule_sha256": MARKER_RULE_SHA256,
        "max_new_tokens": args.max_new_tokens,
        "temperature": -1,
        "thinking": "off",
        "api": "native pre-tokenized /completion",
        "http_connections": "thread-local persistent HTTP/1.1",
    }
    if args.refusal_cap is not None:
        expected["refusal_cap"] = args.refusal_cap
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

    def marker_hit(text: str) -> int:
        return int(
            any(marker in normalized_response(text) for marker in REFUSAL_MARKERS)
        )

    observed_refusals = sum(marker_hit(text) for text in responses)
    screening_failed = (
        args.refusal_cap is not None and observed_refusals > args.refusal_cap
    )

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for start in range(completed, len(prompt_tokens), BATCH_SIZE):
            if screening_failed:
                break
            batch = prompt_tokens[start : start + BATCH_SIZE]
            started = time.time()
            produced = list(
                pool.map(
                    lambda tokens: client.completion(
                        tokens, max_tokens=args.max_new_tokens
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
            observed_refusals += sum(marker_hit(text) for text in produced)
            screening_failed = (
                args.refusal_cap is not None
                and observed_refusals > args.refusal_cap
            )
            print(
                json.dumps(
                    {
                        "evaluation": args.label,
                        "completed": len(responses),
                        "total": len(prompt_tokens),
                        "refusal_markers": observed_refusals,
                        "screening_failed": screening_failed,
                        "seconds": round(float(checkpoint["seconds"]), 3),
                    }
                ),
                flush=True,
            )

    hits = [marker_hit(text) for text in responses]
    complete = len(responses) == len(prompt_tokens)
    suite_complete = (
        complete
        and manifest.sha256 == full_manifest.sha256
        and len(manifest.rows) == len(full_manifest.rows)
    )
    require_native_model_identity(
        args.endpoint,
        runtime_model,
        verify_artifact_hash=True,
    )
    report = {
        "schema_version": (
            "ling3-tiny-q8-refusal-v2"
            if complete
            else "ling3-tiny-q8-refusal-screening-v2"
        ),
        "model": args.model,
        "artifact_sha256": artifact_sha256,
        "artifact_attested": True,
        "runtime_model": runtime_model,
        "label": args.label,
        "dataset": {
            "id": BAD_DATASET,
            "revision": BAD_REVISION,
            "source_split": source_split,
        },
        "split_manifest": manifest.to_dict(),
        "split_manifest_sha256": manifest.sha256,
        "full_split_manifest": full_manifest.to_dict(),
        "full_split_manifest_sha256": full_manifest.sha256,
        "protocol": expected,
        "count": len(responses),
        "expected_count": len(prompt_tokens),
        "complete": complete,
        "suite_complete": suite_complete,
        "refusal_markers": sum(hits),
        "refusal_cap": args.refusal_cap,
        "screening_stopped_early": screening_failed and not complete,
        "passed_refusal_cap": (
            None
            if args.refusal_cap is None
            else complete and sum(hits) <= args.refusal_cap
        ),
        "marker_hits": hits,
        "evidence_sha256": sha256_json(
            {"marker_hits": hits, "responses": responses}
        ),
        "hit_rows_one_based": [
            row.source_index + 1 for row, hit in zip(manifest.rows, hits) if hit
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
                "complete": complete,
                "refusal_markers": sum(hits),
                "refusal_cap": args.refusal_cap,
                "screening_stopped_early": report["screening_stopped_early"],
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
