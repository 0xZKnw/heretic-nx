#!/usr/bin/env python3
"""Exact full-vocabulary first-token KL for Ling-3.0-tiny Q8 candidates."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from typing import Any
from urllib.request import Request, urlopen

from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer

from experiments.lfm25_2p6b_residual_stream import GOOD_DATASET, GOOD_REVISION
from experiments.ling3_tiny_q8_eval import render
from heretic_nx.hashing import canonical_json, sha256_json


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT / "checkpoints" / "ling3-tiny"
RUN_DIR = ROOT / "runs" / "ling3-tiny-q8-direct" / "kl"
VOCAB_SIZE = 157_184
BATCH_SIZE = 4


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


def log_probs(endpoint: str, tokens: list[int]) -> np.ndarray:
    result = None
    for attempt in range(3):
        request = Request(
            endpoint.rstrip("/") + "/completion",
            data=canonical_json(
                {
                    "prompt": tokens,
                    "n_predict": 1,
                    "temperature": -1,
                    "n_probs": VOCAB_SIZE,
                    "stream": False,
                }
            ),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=300) as response:
            result = json.loads(response.read())
        if "error" in result:
            raise RuntimeError(str(result["error"]))
        if result.get("completion_probabilities"):
            break
        if attempt < 2:
            time.sleep(0.25 * (attempt + 1))
    if result is None or not result.get("completion_probabilities"):
        raise RuntimeError(
            "llama.cpp omitted completion probabilities after retries: "
            f"keys={sorted(result or {})}"
        )
    rows = result["completion_probabilities"][0]["top_logprobs"]
    if len(rows) != VOCAB_SIZE:
        raise RuntimeError(f"llama.cpp returned {len(rows)} of {VOCAB_SIZE} logits")
    values = np.empty(VOCAB_SIZE, dtype=np.float32)
    seen = np.zeros(VOCAB_SIZE, dtype=np.bool_)
    for row in rows:
        token_id = int(row["id"])
        if not 0 <= token_id < VOCAB_SIZE or seen[token_id]:
            raise RuntimeError(f"invalid or duplicate token id: {token_id}")
        values[token_id] = float(row["logprob"])
        seen[token_id] = True
    if not bool(seen.all()) or not np.isfinite(values).all():
        raise RuntimeError("full-vocabulary log probabilities are incomplete")
    mass = float(np.exp(values.astype(np.float64)).sum())
    if abs(mass - 1.0) > 1e-3:
        raise RuntimeError(f"probability mass is not normalized: {mass}")
    return values


def prompts() -> tuple[list[list[int]], str]:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    rows = load_dataset(GOOD_DATASET, revision=GOOD_REVISION, split="test")
    rendered = render(tokenizer, [str(rows[index]["text"]) for index in range(104)])
    token_rows = [
        [int(token) for token in tokenizer.encode(value, add_special_tokens=False)]
        for value in rendered
    ]
    return token_rows, sha256_json(token_rows)


def export_tokens(args: argparse.Namespace) -> None:
    token_rows, prompts_sha256 = prompts()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(" ".join(str(token) for token in row) + "\n" for row in token_rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "count": len(token_rows),
                "prompt_tokens_sha256": prompts_sha256,
                "maximum_prompt_tokens": max(map(len, token_rows)),
            },
            indent=2,
        ),
        flush=True,
    )


def raw_logits(path: Path) -> np.memmap:
    expected_values = 104 * VOCAB_SIZE
    expected_bytes = expected_values * np.dtype(np.float32).itemsize
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"invalid raw logits size for {path}: {path.stat().st_size} != {expected_bytes}"
        )
    matrix = np.memmap(path, mode="r", dtype=np.float32, shape=(104, VOCAB_SIZE))
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"non-finite raw logits in {path}")
    return matrix


def normalized_log_probs(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    maximum = float(np.max(values))
    return values - (maximum + np.log(np.exp(values - maximum).sum()))


def compare_raw(args: argparse.Namespace) -> None:
    base_path = Path(args.base)
    candidate_path = Path(args.candidate)
    base = raw_logits(base_path)
    candidate = raw_logits(candidate_path)
    values = []
    for row in range(base.shape[0]):
        log_p = normalized_log_probs(base[row])
        log_q = normalized_log_probs(candidate[row])
        probability = np.exp(log_p)
        values.append(float(np.sum(probability * (log_p - log_q))))
    mean = float(np.mean(values))
    result = {
        "schema_version": "ling3-tiny-q8-first-token-kl-raw-v1",
        "base": str(base_path),
        "candidate": str(candidate_path),
        "count": len(values),
        "vocab_size": VOCAB_SIZE,
        "mean_first_token_kl": mean,
        "maximum_first_token_kl": max(values),
        "median_first_token_kl": float(np.median(values)),
        "p95_first_token_kl": float(np.quantile(values, 0.95)),
        "per_row": values,
        "hard_cap": 0.05,
        "passed": mean <= 0.05,
    }
    report = Path(args.report)
    write_json(report, result)
    print(json.dumps({**result, "report": str(report)}, indent=2), flush=True)


def collect(args: argparse.Namespace) -> None:
    token_rows, prompts_sha256 = prompts()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    array_path = RUN_DIR / f"{args.label}.npy"
    progress_path = RUN_DIR / f"{args.label}.progress.json"
    expected = {
        "schema_version": "ling3-tiny-q8-first-token-logprobs-v1",
        "label": args.label,
        "model": args.model,
        "prompt_tokens_sha256": prompts_sha256,
        "vocab_size": VOCAB_SIZE,
        "count": len(token_rows),
        "artifact_sha256": args.artifact_sha256,
        "thinking": "off",
    }
    progress = {**expected, "completed": 0, "seconds": 0.0}
    if progress_path.is_file():
        loaded = json.loads(progress_path.read_text(encoding="utf-8"))
        if not all(loaded.get(key) == value for key, value in expected.items()):
            raise RuntimeError(f"stale KL checkpoint: {progress_path}")
        progress = loaded
        matrix = np.lib.format.open_memmap(array_path, mode="r+")
        if matrix.shape != (len(token_rows), VOCAB_SIZE):
            raise RuntimeError(f"invalid KL matrix shape: {matrix.shape}")
    else:
        matrix = np.lib.format.open_memmap(
            array_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(token_rows), VOCAB_SIZE),
        )
        write_json(progress_path, progress)
    completed = int(progress["completed"])
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for start in range(completed, len(token_rows), BATCH_SIZE):
            batch = token_rows[start : start + BATCH_SIZE]
            started = time.time()
            produced = list(pool.map(lambda row: log_probs(args.endpoint, row), batch))
            matrix[start : start + len(produced)] = np.stack(produced)
            matrix.flush()
            progress["seconds"] = float(progress["seconds"]) + time.time() - started
            progress["completed"] = start + len(produced)
            write_json(progress_path, progress)
            print(
                json.dumps(
                    {
                        "log_probs": args.label,
                        "completed": progress["completed"],
                        "total": len(token_rows),
                        "seconds": round(float(progress["seconds"]), 3),
                    }
                ),
                flush=True,
            )
    print(
        json.dumps(
            {
                "label": args.label,
                "matrix": str(array_path),
                "shape": list(matrix.shape),
                "seconds": progress["seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


def compare(args: argparse.Namespace) -> None:
    base_path = RUN_DIR / f"{args.base_label}.npy"
    candidate_path = RUN_DIR / f"{args.candidate_label}.npy"
    base = np.load(base_path, mmap_mode="r")
    candidate = np.load(candidate_path, mmap_mode="r")
    if base.shape != candidate.shape or base.shape != (104, VOCAB_SIZE):
        raise RuntimeError("base and candidate KL matrices are not aligned")
    values = []
    for row in range(base.shape[0]):
        log_p = np.asarray(base[row], dtype=np.float64)
        log_q = np.asarray(candidate[row], dtype=np.float64)
        probability = np.exp(log_p)
        values.append(float(np.sum(probability * (log_p - log_q))))
    mean = float(np.mean(values))
    result = {
        "schema_version": "ling3-tiny-q8-first-token-kl-v1",
        "base_label": args.base_label,
        "candidate_label": args.candidate_label,
        "count": len(values),
        "mean_first_token_kl": mean,
        "maximum_first_token_kl": max(values),
        "median_first_token_kl": float(np.median(values)),
        "p95_first_token_kl": float(np.quantile(values, 0.95)),
        "per_row": values,
        "hard_cap": 0.05,
        "passed": mean <= 0.05,
    }
    report = RUN_DIR / f"{args.candidate_label}-vs-{args.base_label}.json"
    write_json(report, result)
    print(json.dumps({**result, "report": str(report)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--label", required=True)
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--artifact-sha256", required=True)
    collect_parser.add_argument("--endpoint", default="http://127.0.0.1:1236")
    collect_parser.add_argument("--parallel", type=int, default=4)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--base-label", required=True)
    compare_parser.add_argument("--candidate-label", required=True)
    export_parser = subparsers.add_parser("export-tokens")
    export_parser.add_argument("--output", required=True)
    raw_parser = subparsers.add_parser("compare-raw")
    raw_parser.add_argument("--base", required=True)
    raw_parser.add_argument("--candidate", required=True)
    raw_parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if args.command == "collect":
        if args.parallel <= 0:
            raise ValueError("parallel must be positive")
        collect(args)
    elif args.command == "compare":
        compare(args)
    elif args.command == "export-tokens":
        export_tokens(args)
    else:
        compare_raw(args)


if __name__ == "__main__":
    main()
