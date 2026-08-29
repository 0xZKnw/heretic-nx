#!/usr/bin/env python3
"""Distil the compact PRIME teacher behind benign-penalized input detectors."""

from __future__ import annotations

import json
from pathlib import Path

from gguf import GGUFReader
from gguf.quants import dequantize
from safetensors.torch import load_file, save_file
import torch

from experiments.lfm25_8b_a1b_q8_build import RUN_DIR, SOURCE
from experiments.lfm25_8b_a1b_teacher_inputs import OUTPUT as INPUTS
from experiments.lfm25_8b_a1b_teacher_inputs import TEACHER_MERGE
from heretic_nx.hashing import canonical_json, sha256_file


OPERATORS = RUN_DIR / "prime-site-operators.safetensors"
OUTPUT = RUN_DIR / "teacher-op-distilled.safetensors"
REPORT = RUN_DIR / "teacher-op-distilled.json"
SAFE_LAMBDAS = (3.0, 10.0, 30.0, 100.0, 300.0)
CG_STEPS = 40
CG_TOLERANCE = 1e-5
RIDGE_FRACTION = 1e-4


def conjugate_gradient(
    harmful: torch.Tensor,
    safe: torch.Tensor,
    exact: torch.Tensor,
    *,
    safe_lambda: float,
) -> tuple[torch.Tensor, int, float]:
    harmful_count = float(len(harmful))
    safe_count = float(len(safe))
    ridge = float(
        RIDGE_FRACTION
        * (harmful.square().mean() + safe_lambda * safe.square().mean())
    )

    def covariance(values: torch.Tensor, vector: torch.Tensor, count: float) -> torch.Tensor:
        return values.T @ (values @ vector) / count

    def system(vector: torch.Tensor) -> torch.Tensor:
        return (
            covariance(harmful, vector, harmful_count)
            + safe_lambda * covariance(safe, vector, safe_count)
            + ridge * vector
        )

    right = covariance(harmful, exact, harmful_count)
    solution = torch.zeros_like(right)
    residual = right.clone()
    direction = residual.clone()
    initial_norm = torch.linalg.vector_norm(residual).clamp_min(1e-12)
    residual_square = torch.dot(residual, residual)
    residual_ratio = 1.0
    iterations = 0
    for iteration in range(1, CG_STEPS + 1):
        product = system(direction)
        alpha = residual_square / torch.dot(direction, product).clamp_min(1e-20)
        solution = solution + alpha * direction
        new_residual = residual - alpha * product
        residual_ratio = float(torch.linalg.vector_norm(new_residual) / initial_norm)
        iterations = iteration
        if residual_ratio <= CG_TOLERANCE:
            break
        new_square = torch.dot(new_residual, new_residual)
        direction = new_residual + (new_square / residual_square) * direction
        residual = new_residual
        residual_square = new_square
    return solution.detach().cpu(), iterations, residual_ratio


def correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.float() - left.float().mean()
    b = right.float() - right.float().mean()
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denominator) <= 1e-12:
        return 0.0
    return float(torch.dot(a, b) / denominator)


def main() -> None:
    required = (SOURCE, INPUTS, OPERATORS, TEACHER_MERGE)
    if not all(path.is_file() for path in required):
        raise RuntimeError("the base Q8, teacher inputs, or PRIME factors are missing")
    teacher = json.loads(TEACHER_MERGE.read_text(encoding="utf-8"))
    sites = list(teacher["candidate"]["selected"])
    if len(sites) != 8:
        raise RuntimeError("the distillation teacher must contain eight sites")
    inputs = load_file(INPUTS)
    safe_all = inputs["safe"].float()
    harmful_all = inputs["harmful"].float()
    if safe_all.shape[1:] != (len(sites), 2048):
        raise RuntimeError(f"invalid safe teacher inputs: {safe_all.shape}")
    if harmful_all.shape[1:] != (len(sites), 2048):
        raise RuntimeError(f"invalid harmful teacher inputs: {harmful_all.shape}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    operator_factors = load_file(OPERATORS)
    reader = GGUFReader(SOURCE)
    q8_tensors = {tensor.name: tensor for tensor in reader.tensors}
    payload: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, object] = {
        "schema_version": "lfm25-8b-a1b-teacher-op-distillation-v1",
        "teacher": {
            "merge": str(TEACHER_MERGE),
            "merge_sha256": sha256_file(TEACHER_MERGE),
            "beta": teacher["candidate"]["beta"],
            "sites": [row["site_id"] for row in sites],
        },
        "inputs": {
            "path": str(INPUTS),
            "sha256": sha256_file(INPUTS),
            "safe_shape": list(safe_all.shape),
            "harmful_shape": list(harmful_all.shape),
        },
        "safe_lambdas": list(SAFE_LAMBDAS),
        "cg_steps": CG_STEPS,
        "ridge_fraction": RIDGE_FRACTION,
        "device": str(device),
        "fits": {},
    }
    for site_index, row in enumerate(sites):
        site_id = str(row["site_id"])
        key = f"site{site_index:02d}"
        axis = operator_factors[str(row["factor_a_key"])].float()
        metric_right = operator_factors[str(row["factor_b_key"])].float()
        if axis.shape != (2048, 1) or metric_right.shape != (2048, 1):
            raise RuntimeError(f"teacher site {site_id} is not rank one")
        tensor = q8_tensors[str(row["tensor_name"])]
        weight = torch.from_numpy(
            dequantize(tensor.data, tensor.tensor_type).copy()
        ).float()
        exact = weight.T @ metric_right[:, 0]
        safe = safe_all[:, site_index].to(device)
        harmful = harmful_all[:, site_index].to(device)
        exact_device = exact.to(device)
        payload[f"{key}.axis"] = axis.contiguous()
        site_rows = []
        harmful_exact_cpu = harmful_all[:, site_index] @ exact
        safe_exact_cpu = safe_all[:, site_index] @ exact
        for safe_lambda in SAFE_LAMBDAS:
            detector, iterations, residual_ratio = conjugate_gradient(
                harmful,
                safe,
                exact_device,
                safe_lambda=safe_lambda,
            )
            harmful_fit = harmful_all[:, site_index] @ detector
            safe_fit = safe_all[:, site_index] @ detector
            row_diagnostics = {
                "site_id": site_id,
                "safe_lambda": safe_lambda,
                "harmful_retention": float(
                    harmful_fit.square().mean().sqrt()
                    / harmful_exact_cpu.square().mean().sqrt().clamp_min(1e-12)
                ),
                "harmful_correlation": correlation(harmful_fit, harmful_exact_cpu),
                "safe_retention": float(
                    safe_fit.square().mean().sqrt()
                    / safe_exact_cpu.square().mean().sqrt().clamp_min(1e-12)
                ),
                "safe_absolute_ratio": float(
                    safe_fit.square().mean().sqrt()
                    / harmful_exact_cpu.square().mean().sqrt().clamp_min(1e-12)
                ),
                "cg_iterations": iterations,
                "cg_residual_ratio": residual_ratio,
            }
            payload[f"lambda{safe_lambda:g}.{key}.right"] = detector[:, None].contiguous()
            site_rows.append(row_diagnostics)
            print(json.dumps({"distill": row_diagnostics}), flush=True)
        diagnostics["fits"][site_id] = site_rows
        del weight, exact, safe, harmful, exact_device
        if device.type == "mps":
            torch.mps.empty_cache()
    save_file(
        payload,
        OUTPUT,
        metadata={
            "inputs_sha256": sha256_file(INPUTS),
            "operators_sha256": sha256_file(OPERATORS),
        },
    )
    diagnostics["output"] = {"path": str(OUTPUT), "sha256": sha256_file(OUTPUT)}
    REPORT.write_bytes(canonical_json(diagnostics) + b"\n")
    print(
        json.dumps(
            {"output": diagnostics["output"], "report": str(REPORT)},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
