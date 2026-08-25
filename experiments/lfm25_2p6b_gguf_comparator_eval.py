#!/usr/bin/env python3
"""Matched llama.cpp evaluation for the pinned LFM2.5-2.6B Heretic Q8."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Callable

from datasets import load_dataset
from transformers import AutoTokenizer

from experiments.lfm25_2p6b_eval import (
    COMPARISON_ALPHA,
    NONINFERIORITY_MARGIN,
    SEED,
)
from experiments.lfm25_2p6b_residual_stream import (
    MAX_NEW_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    REFUSAL_MARKERS,
    base_path,
    normalized_response,
    render,
)
from experiments.lfm25_closed_track_eval import expanded_capability_rows
from experiments.lfm25_residual_stream_capability import LETTERS, task_scores
from experiments.lfm25_xstest_retest import XSTEST_ID, XSTEST_REVISION
from heretic_nx.eval.capability import paired_bootstrap_interval
from heretic_nx.eval.gguf_runtime import (
    RESTRICTED_GRAMMAR,
    native_completion,
    native_restricted_choice,
)
from heretic_nx.hashing import (
    canonical_json,
    sha256_file,
    sha256_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = (
    ROOT
    / "references"
    / "abiray-lfm25-2p6b-q8"
    / "LFM2.5-2.6B-heretic-Q8_0.gguf"
)
DEFAULT_RUN_DIR = ROOT / "runs" / "lfm25-2p6b-gguf-comparator"
PINNED_REPOSITORY = "Abiray/LFM2.5-2.6B-Heretic-Abliterated-GGUF"
PINNED_REPOSITORY_REVISION = "1eaf992a33529fc839cbeca32109a9c4c43b57c4"
PINNED_ARTIFACT_SHA256 = (
    "027f0a8308879a21163dd0c981b7397d1b8828dc06ce01e72250d3adf2f87f9b"
)
BATCH_SIZE = 32
LLAMA_CPP_RUNTIME = "official llama.cpp Windows CUDA 12.4 x64"
LLAMA_CPP_BUILD = "0.3.0-dev build 10621 commit c1d0e7a00"


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(value) + b"\n")
    temporary.replace(path)


def resume_map(
    *,
    prompts: list[Any],
    worker: Callable[[Any], Any],
    partial_path: Path,
    expected: dict[str, Any],
    parallel: int,
    encode: Callable[[Any], Any],
) -> tuple[list[Any], float]:
    checkpoint: dict[str, Any] = {
        **expected,
        "completed": 0,
        "values": [],
        "seconds": 0.0,
    }
    if partial_path.is_file():
        loaded = json.loads(partial_path.read_text(encoding="utf-8"))
        if not all(loaded.get(key) == value for key, value in expected.items()):
            raise RuntimeError(f"stale comparator checkpoint: {partial_path}")
        checkpoint = loaded
    completed = int(checkpoint["completed"])
    values = list(checkpoint["values"])
    if completed != len(values) or completed > len(prompts):
        raise RuntimeError(f"invalid comparator checkpoint: {partial_path}")
    if completed == len(prompts):
        return values, float(checkpoint["seconds"])

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        for start in range(completed, len(prompts), BATCH_SIZE):
            batch = prompts[start : start + BATCH_SIZE]
            started = time.time()
            produced = list(pool.map(worker, batch))
            checkpoint["seconds"] = float(checkpoint["seconds"]) + (
                time.time() - started
            )
            values.extend(encode(value) for value in produced)
            checkpoint["values"] = values
            checkpoint["completed"] = len(values)
            write_json(partial_path, checkpoint)
            print(
                json.dumps(
                    {
                        "checkpoint": partial_path.name,
                        "completed": len(values),
                        "total": len(prompts),
                        "seconds": round(float(checkpoint["seconds"]), 3),
                    }
                ),
                flush=True,
            )
    return values, float(checkpoint["seconds"])


def common_evidence(
    args: argparse.Namespace,
    artifact_hash: str,
    *,
    include_lm_studio: bool,
) -> dict[str, Any]:
    evidence = {
        "repository": PINNED_REPOSITORY,
        "repository_revision": PINNED_REPOSITORY_REVISION,
        "artifact": args.artifact.name,
        "artifact_sha256": artifact_hash,
        "base_tokenizer": {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
        },
    }
    if include_lm_studio:
        evidence["lm_studio_model_identifier"] = args.model
        evidence["endpoint"] = args.endpoint
    return evidence


def run_xstest(args: argparse.Namespace, artifact_hash: str) -> None:
    dataset = load_dataset(
        XSTEST_ID,
        revision=XSTEST_REVISION,
        split="prompts",
    )
    rows = [dict(row) for row in dataset]
    row_payload = [
        {"id": row["id"], "type": row["type"], "prompt": row["prompt"]}
        for row in rows
    ]
    rows_hash = sha256_json(row_payload)
    unsafe = [str(row["type"]).startswith("contrast_") for row in rows]
    tokenizer = AutoTokenizer.from_pretrained(base_path())
    rendered = render(
        tokenizer,
        [str(row["prompt"]) for row in rows],
        close_think=True,
    )
    prompt_tokens = [
        tokenizer.encode(value, add_special_tokens=False) for value in rendered
    ]
    prompt_tokens_hash = sha256_json(prompt_tokens)
    expected = {
        "schema_version": "lfm25-2p6b-gguf-xstest-native-partial-v1",
        "artifact_sha256": artifact_hash,
        "rows_sha256": rows_hash,
        "prompt_tokens_sha256": prompt_tokens_hash,
        "max_new_tokens": MAX_NEW_TOKENS,
        "close_think": True,
        "temperature": -1,
        "runtime_build": LLAMA_CPP_BUILD,
    }
    responses, seconds = resume_map(
        prompts=prompt_tokens,
        worker=lambda tokens: native_completion(
            args.native_endpoint,
            tokens,
            max_tokens=MAX_NEW_TOKENS,
        ),
        partial_path=args.run_dir / "xstest-native.partial.json",
        expected=expected,
        parallel=args.parallel,
        encode=str,
    )
    hits = [
        int(any(marker in normalized_response(text) for marker in REFUSAL_MARKERS))
        for text in responses
    ]
    safe_hits = [hit for hit, is_unsafe in zip(hits, unsafe) if not is_unsafe]
    unsafe_hits = [hit for hit, is_unsafe in zip(hits, unsafe) if is_unsafe]
    report = {
        "schema_version": "lfm25-2p6b-gguf-xstest-native-v1",
        "comparator": common_evidence(
            args,
            artifact_hash,
            include_lm_studio=False,
        ),
        "runtime": {
            "engine": "llama.cpp",
            "distribution": LLAMA_CPP_RUNTIME,
            "build": LLAMA_CPP_BUILD,
            "endpoint": args.native_endpoint,
            "api": "native /completion",
        },
        "dataset": {
            "id": XSTEST_ID,
            "revision": XSTEST_REVISION,
            "rows_sha256": rows_hash,
        },
        "protocol": {
            "close_think": True,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": -1,
            "prompt_tokens_sha256": prompt_tokens_hash,
            "refusal_markers": REFUSAL_MARKERS,
            "metric": "lexical refusal-marker proxy",
        },
        "results": {
            "all_count": len(hits),
            "all_refusal_markers": sum(hits),
            "safe_count": len(safe_hits),
            "safe_refusal_markers": sum(safe_hits),
            "unsafe_count": len(unsafe_hits),
            "unsafe_refusal_markers": sum(unsafe_hits),
            "safe_marker_hits": safe_hits,
            "unsafe_marker_hits": unsafe_hits,
            "response_sha256": sha256_json(responses),
            "seconds": seconds,
            "responses_per_second": len(responses) / max(seconds, 1e-9),
        },
        "interpretation_guard": (
            "Lexical refusal markers are not semantic task-success or safety "
            "judgments. This report evaluates the pinned Q8 GGUF runtime artifact."
        ),
    }
    write_json(args.run_dir / "xstest.json", report)
    print(json.dumps(report, indent=2), flush=True)


def run_capability(args: argparse.Namespace, artifact_hash: str) -> None:
    rows = expanded_capability_rows()
    rows_hash = sha256_json(rows)
    tokenizer = AutoTokenizer.from_pretrained(base_path())
    letter_ids = [
        tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS
    ]
    if any(len(ids) != 1 for ids in letter_ids):
        raise RuntimeError(f"answer labels are not single tokens: {letter_ids}")
    answer_ids = [ids[0] for ids in letter_ids]
    rendered = render(
        tokenizer,
        [str(row["prompt"]) for row in rows],
        close_think=True,
    )
    prompt_tokens = [
        tokenizer.encode(value, add_special_tokens=False) for value in rendered
    ]
    prompt_tokens_hash = sha256_json(prompt_tokens)
    expected = {
        "schema_version": "lfm25-2p6b-gguf-capability-native-partial-v1",
        "artifact_sha256": artifact_hash,
        "rows_sha256": rows_hash,
        "prompt_tokens_sha256": prompt_tokens_hash,
        "answer_token_ids": answer_ids,
        "grammar": RESTRICTED_GRAMMAR,
        "max_new_tokens": 1,
        "close_think": True,
        "temperature": -1,
        "runtime_build": LLAMA_CPP_BUILD,
    }
    raw_choices, seconds = resume_map(
        prompts=prompt_tokens,
        worker=lambda tokens: native_restricted_choice(
            args.native_endpoint,
            tokens,
        ),
        partial_path=args.run_dir / "capability-native.partial.json",
        expected=expected,
        parallel=args.parallel,
        encode=str,
    )
    choices = [str(choice).strip() for choice in raw_choices]
    invalid = [
        {"index": index, "value": value}
        for index, value in enumerate(choices)
        if value not in LETTERS
    ]
    if invalid:
        raise RuntimeError(f"restricted answer generation escaped labels: {invalid[:5]}")
    predictions = [LETTERS.index(choice) for choice in choices]
    correctness = [
        int(prediction == int(row["answer"]))
        for prediction, row in zip(predictions, rows)
    ]
    native_report = json.loads(
        (
            ROOT
            / "runs"
            / "lfm25-2p6b-eval-prime-v8"
            / "capability.json"
        ).read_text(encoding="utf-8")
    )
    base_correctness = native_report["results"]["base"]["correctness"]
    candidate_correctness = native_report["results"]["residual_stream"][
        "correctness"
    ]
    comparator_minus_base = asdict(
        paired_bootstrap_interval(
            base_correctness,
            correctness,
            margin=NONINFERIORITY_MARGIN,
            alpha=COMPARISON_ALPHA,
            resamples=10_000,
            seed=SEED + 10,
        )
    )
    candidate_minus_comparator = asdict(
        paired_bootstrap_interval(
            correctness,
            candidate_correctness,
            margin=NONINFERIORITY_MARGIN,
            alpha=COMPARISON_ALPHA,
            resamples=10_000,
            seed=SEED + 11,
        )
    )
    report = {
        "schema_version": "lfm25-2p6b-gguf-capability-native-v1",
        "comparator": common_evidence(
            args,
            artifact_hash,
            include_lm_studio=False,
        ),
        "runtime": {
            "engine": "llama.cpp",
            "distribution": LLAMA_CPP_RUNTIME,
            "build": LLAMA_CPP_BUILD,
            "endpoint": args.native_endpoint,
            "api": "native /completion",
        },
        "rows": len(rows),
        "rows_sha256": rows_hash,
        "protocol": {
            "scoring": "restricted first-token A/B/C/D raw-logit argmax",
            "answer_token_ids": answer_ids,
            "prompt_tokens_sha256": prompt_tokens_hash,
            "grammar": RESTRICTED_GRAMMAR,
            "close_think": True,
            "temperature": -1,
            "max_new_tokens": 1,
            "explanation": (
                "Pre-tokenized prompts preserve exactly one explicit BOS. The "
                "llama.cpp native grammar masks every token except the four "
                "answer labels before deterministic greedy selection. No logit "
                "bias, top-k, top-p, or stochastic sampler is used."
            ),
        },
        "results": {
            "count": len(rows),
            "predictions": predictions,
            "correctness": correctness,
            "accuracy": sum(correctness) / len(correctness),
            "tasks": task_scores(rows, correctness),
            "seconds": seconds,
            "rows_per_second": len(rows) / max(seconds, 1e-9),
        },
        "paired_comparisons": {
            "comparator_minus_base": comparator_minus_base,
            "candidate_minus_comparator": candidate_minus_comparator,
        },
        "interpretation_guard": (
            "The comparator is a Q8 GGUF runtime artifact while base and PRIME "
            "were evaluated in native BF16. The grammar-restricted decision rule "
            "is validated on the PRIME BF16 GGUF, but precision/runtime differ."
        ),
    }
    write_json(args.run_dir / "capability.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("xstest", "capability", "all"))
    parser.add_argument("--model", default="abiray-heretic")
    parser.add_argument("--endpoint", default="http://127.0.0.1:1234")
    parser.add_argument("--native-endpoint", default="http://127.0.0.1:1235")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    args.artifact = args.artifact.resolve()
    args.run_dir = args.run_dir.resolve()
    if args.parallel <= 0:
        raise ValueError("parallel must be positive")
    if not args.artifact.is_file():
        raise RuntimeError(f"missing pinned comparator artifact: {args.artifact}")
    artifact_hash = sha256_file(args.artifact)
    if artifact_hash != PINNED_ARTIFACT_SHA256:
        raise RuntimeError(
            f"comparator hash mismatch: {artifact_hash} != {PINNED_ARTIFACT_SHA256}"
        )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in {"xstest", "all"}:
        run_xstest(args, artifact_hash)
    if args.mode in {"capability", "all"}:
        run_capability(args, artifact_hash)


if __name__ == "__main__":
    main()
