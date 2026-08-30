"""Reproduce masked/chunked teacher-forced KL speed and working-set gains."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from statistics import median
import time

import torch
from torch import Tensor

from heretic_nx.eval.capability import SequenceDrift, teacher_forced_sequence_kl


def legacy_teacher_forced_sequence_kl(
    baseline_logits: Tensor,
    candidate_logits: Tensor,
    token_mask: Tensor,
    *,
    top_k: int,
) -> SequenceDrift:
    """Pre-optimization full-tensor reference implementation."""

    mask = token_mask.bool()
    base_log_probs = torch.log_softmax(baseline_logits.float(), dim=-1)
    candidate_log_probs = torch.log_softmax(candidate_logits.float(), dim=-1)
    base_probs = base_log_probs.exp()
    per_token = torch.sum(
        base_probs * (base_log_probs - candidate_log_probs),
        dim=-1,
    )
    selected = per_token[mask]
    counts = mask.sum(dim=1)
    valid_sequences = counts > 0
    sequence_means = (
        (per_token * mask).sum(dim=1)[valid_sequences] / counts[valid_sequences]
    )
    top_indices = torch.topk(base_log_probs, k=top_k, dim=-1).indices
    top_mass = torch.gather(base_probs, -1, top_indices).sum(dim=-1)[mask]
    return SequenceDrift(
        token_count=int(mask.sum().item()),
        mean_token_kl=float(selected.mean().item()),
        maximum_sequence_kl=float(sequence_means.max().item()),
        mean_topk_mass_coverage=float(top_mass.mean().item()),
    )


def _time(call, *, repeats: int) -> tuple[float, SequenceDrift]:
    samples = []
    result = call()
    for _ in range(repeats):
        started = time.perf_counter()
        result = call()
        samples.append(time.perf_counter() - started)
    return median(samples), result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--vocab", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if min(
        args.batch,
        args.tokens,
        args.vocab,
        args.top_k,
        args.chunk_size,
        args.repeats,
        args.threads,
    ) < 1:
        parser.error("all numeric arguments must be positive")
    if args.top_k > args.vocab:
        parser.error("--top-k cannot exceed --vocab")

    torch.set_num_threads(args.threads)
    generator = torch.Generator().manual_seed(20260830)
    baseline = torch.randn(
        args.batch,
        args.tokens,
        args.vocab,
        generator=generator,
    )
    candidate = baseline + 0.05 * torch.randn(
        baseline.shape,
        generator=generator,
    )
    mask = torch.zeros(args.batch, args.tokens, dtype=torch.bool)
    # Deliberately heterogeneous lengths model ordinary padded eval batches.
    for row in range(args.batch):
        fraction = (row + 1) / args.batch
        length = max(1, round(args.tokens * fraction))
        mask[row, :length] = True

    legacy_call = lambda: legacy_teacher_forced_sequence_kl(
        baseline,
        candidate,
        mask,
        top_k=args.top_k,
    )
    optimized_call = lambda: teacher_forced_sequence_kl(
        baseline,
        candidate,
        mask,
        top_k=args.top_k,
        token_chunk_size=args.chunk_size,
    )
    # Warm both paths before collecting medians.
    legacy_call()
    optimized_call()
    legacy_seconds, legacy_result = _time(legacy_call, repeats=args.repeats)
    optimized_seconds, optimized_result = _time(
        optimized_call,
        repeats=args.repeats,
    )

    selected_tokens = int(mask.sum())
    total_tokens = mask.numel()
    element_bytes = torch.tensor([], dtype=torch.float32).element_size()
    legacy_probability_bytes = 3 * total_tokens * args.vocab * element_bytes
    legacy_top_index_bytes = total_tokens * args.top_k * 8
    chunk_tokens = min(args.chunk_size, selected_tokens)
    optimized_row_buffer_bytes = 3 * chunk_tokens * args.vocab * element_bytes
    optimized_metric_bytes = 2 * selected_tokens * element_bytes
    optimized_position_bytes = 2 * selected_tokens * 8
    optimized_topk_bytes = chunk_tokens * args.top_k * (element_bytes + 8)
    optimized_working_bytes = (
        optimized_row_buffer_bytes
        + optimized_metric_bytes
        + optimized_position_bytes
        + optimized_topk_bytes
    )
    differences = {
        key: abs(asdict(legacy_result)[key] - asdict(optimized_result)[key])
        for key in (
            "mean_token_kl",
            "maximum_sequence_kl",
            "mean_topk_mass_coverage",
        )
    }
    print(
        json.dumps(
            {
                "shape": [args.batch, args.tokens, args.vocab],
                "selected_tokens": selected_tokens,
                "selected_fraction": selected_tokens / total_tokens,
                "legacy_seconds_median": legacy_seconds,
                "optimized_seconds_median": optimized_seconds,
                "speedup": legacy_seconds / optimized_seconds,
                "softmax_row_reduction": total_tokens / selected_tokens,
                "legacy_working_bytes_lower_bound": (
                    legacy_probability_bytes + legacy_top_index_bytes
                ),
                "optimized_working_bytes_upper_bound": (
                    optimized_working_bytes
                ),
                "working_set_reduction_lower_bound": (
                    (legacy_probability_bytes + legacy_top_index_bytes)
                    / optimized_working_bytes
                ),
                "maximum_metric_difference": max(differences.values()),
                "metric_differences": differences,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
