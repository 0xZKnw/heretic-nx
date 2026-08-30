#!/usr/bin/env python3
"""Refusal-first evaluation for direct-Q8 LFM2.5-8B-A1B candidates."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
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
    render,
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
    lm_studio_completion,
    NativeRuntimeClient,
    require_native_model_identity,
)
from heretic_nx.hashing import canonical_json, sha256_file, sha256_json


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT / "checkpoints" / "lfm25-8b-a1b"
RUN_DIR = ROOT / "runs" / "lfm25-8b-a1b-q8-direct"
BATCH_SIZE = 8
POOL_ROWS = 400
SPLIT_SEED = 20260830
MARKER_RULE_SHA256 = refusal_marker_rule_sha256(REFUSAL_MARKERS)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:1236")
    parser.add_argument("--api", choices=("native", "openai"), default="native")
    parser.add_argument(
        "--phase",
        choices=("geometry", "selection", "public-report"),
        default="public-report",
    )
    parser.add_argument("--artifact", type=Path)
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
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.parallel <= 0:
        raise ValueError("parallel must be positive")
    if args.pool_rows <= 0:
        raise ValueError("pool-rows must be positive")
    if (
        args.artifact_sha256 is not None
        and re.fullmatch(r"[0-9a-f]{64}", args.artifact_sha256) is None
    ):
        raise ValueError("artifact-sha256 must be a lowercase SHA-256")
    if args.api == "native" and args.artifact is None:
        raise ValueError("native evaluation requires --artifact for attestation")
    if args.api == "openai" and args.phase != "public-report":
        raise ValueError(
            "unattested OpenAI-compatible evaluation is public-report only"
        )
    if args.api == "openai" and args.artifact is None and args.artifact_sha256 is None:
        raise ValueError("openai evaluation requires --artifact or --artifact-sha256")
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

    runtime_model = None
    if args.api == "native":
        assert args.artifact is not None
        runtime_model = attest_native_model(
            args.endpoint,
            args.artifact,
            expected_model=args.model,
        )
        artifact_sha256 = str(runtime_model["artifact_sha256"])
        if (
            args.artifact_sha256 is not None
            and args.artifact_sha256 != artifact_sha256
        ):
            raise RuntimeError("attested artifact does not match --artifact-sha256")
        artifact_attested = True
        native_client = NativeRuntimeClient(args.endpoint)
    else:
        artifact_sha256 = (
            sha256_file(args.artifact)
            if args.artifact is not None
            else str(args.artifact_sha256)
        )
        if (
            args.artifact_sha256 is not None
            and args.artifact_sha256 != artifact_sha256
        ):
            raise RuntimeError("local artifact does not match --artifact-sha256")
        artifact_attested = False
        native_client = None

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
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
    rendered = render(tokenizer, raw_prompts, close_think=True)
    prompt_tokens = [
        tokenizer.encode(prompt, add_special_tokens=False) for prompt in rendered
    ]
    expected = {
        "schema_version": "lfm25-8b-a1b-q8-refusal-partial-v2",
        "model": args.model,
        "artifact_sha256": artifact_sha256,
        "artifact_attested": artifact_attested,
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
        "api": args.api,
    }
    if native_client is not None:
        expected["http_connections"] = "thread-local persistent HTTP/1.1"
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
    if completed != len(responses) or completed > len(rendered):
        raise RuntimeError(f"invalid evaluation checkpoint: {partial}")

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for start in range(completed, len(rendered), BATCH_SIZE):
            batch = (
                prompt_tokens[start : start + BATCH_SIZE]
                if args.api == "native"
                else rendered[start : start + BATCH_SIZE]
            )
            started = time.time()
            if args.api == "native":
                assert native_client is not None
                produced = list(
                    pool.map(
                        lambda tokens: native_client.completion(
                            tokens, max_tokens=args.max_new_tokens
                        ),
                        batch,
                    )
                )
            else:
                produced = list(
                    pool.map(
                        lambda prompt: lm_studio_completion(
                            args.endpoint,
                            args.model,
                            prompt,
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
                        "total": len(rendered),
                        "seconds": round(float(checkpoint["seconds"]), 3),
                    }
                ),
                flush=True,
            )

    hits = [
        int(any(marker in normalized_response(text) for marker in REFUSAL_MARKERS))
        for text in responses
    ]
    if runtime_model is not None:
        require_native_model_identity(
            args.endpoint,
            runtime_model,
            verify_artifact_hash=True,
        )
    complete = len(responses) == len(manifest.rows)
    suite_complete = (
        complete
        and manifest.sha256 == full_manifest.sha256
        and len(manifest.rows) == len(full_manifest.rows)
    )
    report = {
        "schema_version": "lfm25-8b-a1b-q8-refusal-v2",
        "model": args.model,
        "artifact_sha256": artifact_sha256,
        "artifact_attested": artifact_attested,
        "runtime_model": runtime_model,
        "label": args.label,
        "endpoint": args.endpoint,
        "dataset": {
            "id": BAD_DATASET,
            "revision": BAD_REVISION,
            "source_split": source_split,
        },
        "split_manifest": manifest.to_dict(),
        "split_manifest_sha256": manifest.sha256,
        "full_split_manifest": full_manifest.to_dict(),
        "full_split_manifest_sha256": full_manifest.sha256,
        "protocol": {
            "schema_version": expected["schema_version"],
            "model": args.model,
            "artifact_sha256": artifact_sha256,
            "artifact_attested": artifact_attested,
            "purpose": args.phase,
            "bounds": bounds,
            "source_indices_one_based": [
                row.source_index + 1 for row in manifest.rows
            ],
            "split_manifest_sha256": manifest.sha256,
            "full_split_manifest_sha256": full_manifest.sha256,
            "full_split_count": len(full_manifest.rows),
            "close_think": True,
            "max_new_tokens": args.max_new_tokens,
            "temperature": -1,
            "api": args.api,
            **(
                {"http_connections": expected["http_connections"]}
                if native_client is not None
                else {}
            ),
            "prompt_tokens_sha256": expected["prompt_tokens_sha256"],
            "refusal_markers": list(REFUSAL_MARKERS),
            "refusal_normalizer": REFUSAL_NORMALIZER_V1,
            "marker_rule_sha256": MARKER_RULE_SHA256,
        },
        "count": len(responses),
        "expected_count": len(manifest.rows),
        "complete": complete,
        "suite_complete": suite_complete,
        "refusal_markers": sum(hits),
        "marker_hits": hits,
        "evidence_sha256": sha256_json(
            {"marker_hits": hits, "responses": responses}
        ),
        "response_sha256": sha256_json(responses),
        "seconds": float(checkpoint["seconds"]),
        "responses_per_second": len(responses)
        / max(float(checkpoint["seconds"]), 1e-9),
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
                "seconds": report["seconds"],
                "report": str(report_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
