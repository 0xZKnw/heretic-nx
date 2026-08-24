#!/usr/bin/env python3
"""Paired, pinned capability check for the Residual-Stream candidate."""

from __future__ import annotations

import gc
import json
import random
from dataclasses import asdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.lfm25_prime_uncensor import FRESH_PATH, SYSTEM_PROMPT
from experiments.lfm25_residual_stream_build import OUTPUT_PATH
from experiments.lfm25_residual_stream_select import HERETIC_PATH
from heretic_nx.eval.capability import paired_bootstrap_interval
from heretic_nx.hashing import canonical_json, sha256_file, sha256_json


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "lfm25-residual-stream-capability"
REPORT = RUN_DIR / "report.json"
SEED = 1307
BATCH_SIZE = 64
LETTERS = ("A", "B", "C", "D")
NONINFERIORITY_MARGIN = 0.03
DATASETS = {
    "arc_challenge": {
        "id": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "split": "test",
        "count": 64,
    },
    "hellaswag": {
        "id": "Rowan/hellaswag",
        "config": None,
        "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "split": "validation",
        "count": 64,
    },
    "mmlu": {
        "id": "cais/mmlu",
        "config": "all",
        "revision": "c30699e8356da336a370243923dbaf21066bb9fe",
        "split": "test",
        "per_subject": 2,
    },
}


def prompt_text(question: str, choices: list[str], *, task: str) -> str:
    instruction = (
        "Choose the most plausible continuation."
        if task == "hellaswag"
        else "Answer the multiple-choice question."
    )
    options = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(LETTERS, choices)
    )
    return (
        f"{instruction}\n\n{question}\n\n{options}\n\n"
        "Reply with only A, B, C, or D."
    )


def selected_rows() -> list[dict]:
    generator = random.Random(SEED)
    rows = []

    arc_spec = DATASETS["arc_challenge"]
    arc = load_dataset(
        arc_spec["id"],
        arc_spec["config"],
        revision=arc_spec["revision"],
        split=arc_spec["split"],
    )
    for index in sorted(generator.sample(range(len(arc)), arc_spec["count"])):
        row = arc[index]
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        if len(labels) != 4 or set(labels) != set(LETTERS):
            continue
        ordered = [texts[labels.index(letter)] for letter in LETTERS]
        rows.append(
            {
                "id": f"arc:{row['id']}",
                "task": "arc_challenge",
                "prompt": prompt_text(str(row["question"]), ordered, task="arc_challenge"),
                "answer": LETTERS.index(str(row["answerKey"])),
            }
        )

    hella_spec = DATASETS["hellaswag"]
    hella = load_dataset(
        hella_spec["id"],
        revision=hella_spec["revision"],
        split=hella_spec["split"],
    )
    for index in sorted(generator.sample(range(len(hella)), hella_spec["count"])):
        row = hella[index]
        rows.append(
            {
                "id": f"hellaswag:{row['ind']}",
                "task": "hellaswag",
                "prompt": prompt_text(
                    str(row["ctx"]), list(row["endings"]), task="hellaswag"
                ),
                "answer": int(row["label"]),
            }
        )

    mmlu_spec = DATASETS["mmlu"]
    mmlu = load_dataset(
        mmlu_spec["id"],
        mmlu_spec["config"],
        revision=mmlu_spec["revision"],
        split=mmlu_spec["split"],
    )
    by_subject: dict[str, list[int]] = {}
    for index, subject in enumerate(mmlu["subject"]):
        by_subject.setdefault(str(subject), []).append(index)
    for subject in sorted(by_subject):
        choices = by_subject[subject]
        count = min(mmlu_spec["per_subject"], len(choices))
        for index in sorted(generator.sample(choices, count)):
            row = mmlu[index]
            rows.append(
                {
                    "id": f"mmlu:{subject}:{index}",
                    "task": "mmlu",
                    "prompt": prompt_text(
                        str(row["question"]), list(row["choices"]), task="mmlu"
                    ),
                    "answer": int(row["answer"]),
                }
            )
    return rows


@torch.inference_mode()
def evaluate(path: Path, rows: list[dict]) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.padding_side = "left"
    letter_ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    if any(len(ids) != 1 for ids in letter_ids):
        raise RuntimeError(f"answer labels are not single tokens: {letter_ids}")
    answer_token_ids = torch.tensor([ids[0] for ids in letter_ids], device="cuda")
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map=0
    ).eval()
    rendered = tokenizer.apply_chat_template(
        [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["prompt"]},
            ]
            for row in rows
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    rendered = [text + "<think></think>\n" for text in rendered]
    predictions = []
    margins = []
    torch.cuda.reset_peak_memory_stats()
    for start in range(0, len(rows), BATCH_SIZE):
        batch = tokenizer(
            rendered[start : start + BATCH_SIZE],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
            return_token_type_ids=False,
        ).to(model.device)
        logits = model(**batch, use_cache=False).logits[:, -1].float()
        choice_logits = logits.index_select(1, answer_token_ids)
        values, indices = choice_logits.topk(k=2, dim=1)
        predictions.extend(indices[:, 0].cpu().tolist())
        margins.extend((values[:, 0] - values[:, 1]).cpu().tolist())
        print(
            json.dumps(
                {"model": path.name, "completed": min(start + BATCH_SIZE, len(rows))}
            ),
            flush=True,
        )
    correctness = [
        int(prediction == row["answer"])
        for prediction, row in zip(predictions, rows)
    ]
    result = {
        "predictions": predictions,
        "correctness": correctness,
        "accuracy": sum(correctness) / len(correctness),
        "mean_choice_margin": sum(margins) / len(margins),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def task_scores(rows: list[dict], correctness: list[int]) -> dict[str, dict]:
    result = {}
    for task in sorted({row["task"] for row in rows}):
        scores = [
            value
            for row, value in zip(rows, correctness)
            if row["task"] == task
        ]
        result[task] = {"count": len(scores), "accuracy": sum(scores) / len(scores)}
    return result


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    rows = selected_rows()
    paths = {
        "base": FRESH_PATH,
        "heretic": HERETIC_PATH,
        "heretic_nx_residual_stream": OUTPUT_PATH,
    }
    results = {name: evaluate(path, rows) for name, path in paths.items()}
    for result in results.values():
        result["tasks"] = task_scores(rows, result["correctness"])
    comparisons = {
        "candidate_minus_base": asdict(
            paired_bootstrap_interval(
                results["base"]["correctness"],
                results["heretic_nx_residual_stream"]["correctness"],
                margin=NONINFERIORITY_MARGIN,
                resamples=10_000,
                seed=SEED,
            )
        ),
        "candidate_minus_heretic": asdict(
            paired_bootstrap_interval(
                results["heretic"]["correctness"],
                results["heretic_nx_residual_stream"]["correctness"],
                margin=NONINFERIORITY_MARGIN,
                resamples=10_000,
                seed=SEED + 1,
            )
        ),
    }
    passed = all(value["noninferiority_passed"] for value in comparisons.values())
    report = {
        "schema_version": "lfm25-residual-stream-capability-v1",
        "seed": SEED,
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "datasets": DATASETS,
        "selected_rows_sha256": sha256_json(rows),
        "model_sha256": {
            name: sha256_file(path / "model.safetensors")
            for name, path in paths.items()
        },
        "results": results,
        "comparisons": comparisons,
        "passed": passed,
        "interpretation_guard": (
            "This is a deterministic, post-selection capability check using first-token "
            "letter scoring. It is broader than KL but not a comprehensive benchmark suite."
        ),
    }
    REPORT.write_bytes(canonical_json(report) + b"\n")
    print(
        canonical_json(
            {
                "accuracy": {name: value["accuracy"] for name, value in results.items()},
                "tasks": {name: value["tasks"] for name, value in results.items()},
                "comparisons": comparisons,
                "passed": passed,
                "report": str(REPORT),
            }
        ).decode(),
        flush=True,
    )


if __name__ == "__main__":
    main()
