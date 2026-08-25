#!/usr/bin/env python3
"""Validate the LM Studio restricted-choice bridge on the PRIME BF16 GGUF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from experiments.lfm25_2p6b_gguf_comparator_eval import (
    LLAMA_CPP_BUILD,
    LLAMA_CPP_RUNTIME,
    RESTRICTED_GRAMMAR,
    native_restricted_choice,
    resume_map,
    write_json,
)
from experiments.lfm25_2p6b_residual_stream import base_path, render
from experiments.lfm25_closed_track_eval import expanded_capability_rows
from experiments.lfm25_residual_stream_capability import LETTERS, task_scores
from heretic_nx.hashing import sha256_file, sha256_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = (
    Path.home()
    / ".lmstudio"
    / "models"
    / "0xzknw"
    / "LFM2.5-2.6B-Residual-Stream-PRIME-v8-GGUF"
    / "LFM2.5-2.6B-Residual-Stream-PRIME-v8-BF16.gguf"
)
DEFAULT_RUN_DIR = ROOT / "runs" / "lfm25-2p6b-gguf-comparator"
PINNED_ARTIFACT_SHA256 = (
    "83b7312291673ba9da17cf3022f7021dc9a7cffb4c5427cbeed81f8ee070eb77"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:1235")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    args.artifact = args.artifact.resolve()
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.parallel <= 0:
        raise ValueError("parallel must be positive")
    if not args.artifact.is_file():
        raise RuntimeError(f"missing PRIME GGUF artifact: {args.artifact}")
    artifact_hash = sha256_file(args.artifact)
    if artifact_hash != PINNED_ARTIFACT_SHA256:
        raise RuntimeError(
            f"PRIME GGUF hash mismatch: {artifact_hash} != "
            f"{PINNED_ARTIFACT_SHA256}"
        )

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
        "schema_version": "lfm25-2p6b-prime-gguf-capability-native-partial-v1",
        "artifact_sha256": artifact_hash,
        "rows_sha256": rows_hash,
        "prompt_tokens_sha256": prompt_tokens_hash,
        "answer_token_ids": answer_ids,
        "grammar": RESTRICTED_GRAMMAR,
        "max_new_tokens": 1,
        "close_think": True,
        "temperature": -1,
    }
    raw_choices, seconds = resume_map(
        prompts=prompt_tokens,
        worker=lambda tokens: native_restricted_choice(
            args.endpoint,
            tokens,
        ),
        partial_path=args.run_dir / "prime-native-tokens.partial.json",
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

    native_report_path = (
        ROOT / "runs" / "lfm25-2p6b-eval-prime-v8" / "capability.json"
    )
    native_report = json.loads(native_report_path.read_text(encoding="utf-8"))
    native = native_report["results"]["residual_stream"]
    native_predictions = [int(value) for value in native["predictions"]]
    native_correctness = [int(value) for value in native["correctness"]]
    if len(native_predictions) != len(predictions):
        raise RuntimeError("native and GGUF prediction counts differ")
    prediction_matches = [
        int(left == right)
        for left, right in zip(native_predictions, predictions)
    ]
    correctness_matches = [
        int(left == right)
        for left, right in zip(native_correctness, correctness)
    ]
    gguf_accuracy = sum(correctness) / len(correctness)
    native_accuracy = sum(native_correctness) / len(native_correctness)
    report = {
        "schema_version": "lfm25-2p6b-prime-gguf-capability-native-v1",
        "artifact": {
            "filename": args.artifact.name,
            "sha256": artifact_hash,
            "format": "GGUF BF16",
        },
        "runtime": {
            "engine": "llama.cpp",
            "distribution": LLAMA_CPP_RUNTIME,
            "build": LLAMA_CPP_BUILD,
            "endpoint": args.endpoint,
            "api": "native /completion",
        },
        "native_reference": {
            "report": str(native_report_path.relative_to(ROOT)),
            "report_sha256": sha256_file(native_report_path),
            "format": "Transformers BF16",
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
        },
        "results": {
            "gguf": {
                "accuracy": gguf_accuracy,
                "tasks": task_scores(rows, correctness),
                "predictions": predictions,
                "correctness": correctness,
            },
            "native": {
                "accuracy": native_accuracy,
                "tasks": native["tasks"],
            },
            "gguf_minus_native_accuracy": gguf_accuracy - native_accuracy,
            "prediction_agreement": sum(prediction_matches) / len(prediction_matches),
            "correctness_agreement": sum(correctness_matches) / len(correctness_matches),
            "seconds": seconds,
            "rows_per_second": len(rows) / max(seconds, 1e-9),
        },
        "interpretation_guard": (
            "This validates the llama.cpp grammar-restricted argmax against the "
            "same PRIME weights evaluated through the native Transformers scorer. "
            "It does not imply bit-identical kernels or runtimes."
        ),
    }
    write_json(args.run_dir / "prime-native-validation.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
