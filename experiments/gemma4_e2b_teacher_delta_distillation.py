#!/usr/bin/env python3
"""Distil the strong Gemma teacher delta behind a benign input penalty.

For each of the seven sites in the k7/beta2 local teacher, the exact edit is
``-2 a (W.T a).T``.  We retain the output axis ``a`` but replace its dense
input detector with a ridge-regression detector that reproduces the teacher's
token-level intervention on base refusal trajectories and has low energy on
2,048 benign prompt-boundary states.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any

from datasets import load_dataset
from safetensors.torch import load_file, save_file
import torch

import gemma4_e2b_local_site_refusal_first as local
import gemma4_e2b_residual_stream_prime as base
import gemma4_e2b_teacher_trajectory_conditional as trajectory
from heretic_nx.model import discover_semantic_sites


SAFE_COUNT = 2048
PROBE_COUNT = 20
PROBE_TOKENS = 96
TEACHER_K = 7
CG_STEPS = 36
CG_TOLERANCE = 1e-5
RIDGE_FRACTION = 1e-4
SAFE_LAMBDAS = (1.0, 0.3, 3.0, 10.0, 30.0, 100.0)
TRIALS = (
    (1.0, 2.0),
    (0.3, 2.0),
    (3.0, 2.0),
    (10.0, 2.0),
    (30.0, 2.0),
    (100.0, 2.0),
)

HARMFUL_TOKEN_CACHE = base.RUN_DIR / "teacher-delta-harmful-token-inputs.safetensors"
SAFE_INPUT_CACHE = base.RUN_DIR / "teacher-delta-safe2048-inputs.safetensors"
DETECTOR_CACHE = base.RUN_DIR / "teacher-delta-distilled-detectors.safetensors"
DIAGNOSTICS_PATH = base.RUN_DIR / "teacher-delta-distilled-diagnostics.json"
STATE_PATH = base.RUN_DIR / "teacher-delta-distillation-state.json"


@dataclass(frozen=True)
class DistilledEditor:
    site_id: str
    module_path: str
    family: str
    layer: int
    axis: torch.Tensor
    detector: torch.Tensor
    harmful_retention: float
    harmful_correlation: float
    safe_retention: float
    safe_absolute_ratio: float
    cg_iterations: int
    cg_residual_ratio: float


def teacher_editors(model: Any, sites: list[Any]) -> list[Any]:
    cached = load_file(local.ACTIVATION_CACHE)
    safe = {site.id: cached[f"safe.{site.id}"] for site in sites}
    target = {site.id: cached[f"target.{site.id}"] for site in sites}
    return local.build_editors(sites, safe, target)[:TEACHER_K]


@torch.inference_mode()
def collect_harmful_token_inputs(
    model: Any,
    paths: dict[str, Any],
    pair_indices: list[int],
    editors: list[Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if HARMFUL_TOKEN_CACHE.is_file():
        cached = load_file(HARMFUL_TOKEN_CACHE)
        cached_indices = [int(value) for value in cached["pair_indices"].tolist()]
        if cached_indices == pair_indices:
            print(json.dumps({"reuse": str(HARMFUL_TOKEN_CACHE)}), flush=True)
            return dict(cached)

    storage: dict[str, list[torch.Tensor]] = {row.site_id: [] for row in editors}
    armed: dict[str, Any] = {"value": False, "start": 0, "length": 0}
    handles = []
    for editor in editors:
        module = model.get_submodule(editor.module_path)

        def capture(_module, inputs, *, site_id=editor.site_id):
            if not armed["value"]:
                return
            start = int(armed["start"])
            stop = start + int(armed["length"])
            storage[site_id].append(
                inputs[0][:, start:stop]
                .detach()
                .squeeze(0)
                .to(dtype=torch.bfloat16, device="cpu")
            )

        handles.append(module.register_forward_pre_hook(capture))
    try:
        for completed, index in enumerate(pair_indices, start=1):
            row = paths["base"][index]
            prompt_ids = [int(value) for value in row["prompt_ids"]]
            response_ids = [int(value) for value in row["response_ids"]]
            ids = torch.tensor(
                [prompt_ids + response_ids], dtype=torch.long, device=device
            )
            armed.update(
                {"value": True, "start": len(prompt_ids), "length": len(response_ids)}
            )
            model.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                return_dict=True,
            )
            armed["value"] = False
            del ids
            if completed % 8 == 0 or completed == len(pair_indices):
                print(
                    json.dumps(
                        {
                            "collect_harmful_tokens": completed,
                            "total": len(pair_indices),
                        }
                    ),
                    flush=True,
                )
    finally:
        armed["value"] = False
        for handle in handles:
            handle.remove()
    result = {
        "pair_indices": torch.tensor(pair_indices, dtype=torch.int32),
        **{
            editor.site_id: torch.cat(storage[editor.site_id]).contiguous()
            for editor in editors
        },
    }
    save_file(
        result,
        HARMFUL_TOKEN_CACHE,
        metadata={"schema_version": "gemma4-e2b-teacher-delta-harmful-v1"},
    )
    base.empty_device_cache(device)
    return result


@torch.inference_mode()
def collect_safe_inputs(
    model: Any,
    tokenizer: Any,
    editors: list[Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if SAFE_INPUT_CACHE.is_file():
        cached = load_file(SAFE_INPUT_CACHE)
        if all(editor.site_id in cached for editor in editors):
            print(json.dumps({"reuse": str(SAFE_INPUT_CACHE)}), flush=True)
            return dict(cached)

    rows = load_dataset(
        base.GOOD_DATASET,
        revision=base.GOOD_REVISION,
        split=f"train[:{SAFE_COUNT}]",
    )
    prompts = [str(row["text"]) for row in rows]
    rendered = base.render(tokenizer, prompts)
    storage: dict[str, list[torch.Tensor]] = {row.site_id: [] for row in editors}
    armed = {"value": False}
    handles = []
    for editor in editors:
        module = model.get_submodule(editor.module_path)

        def capture(_module, inputs, *, site_id=editor.site_id):
            if armed["value"]:
                storage[site_id].append(
                    inputs[0][:, -1]
                    .detach()
                    .to(dtype=torch.bfloat16, device="cpu")
                )

        handles.append(module.register_forward_pre_hook(capture))
    started = time.time()
    try:
        for start in range(0, len(rendered), 4):
            batch = base.tokenize(tokenizer, rendered[start : start + 4], device)
            armed["value"] = True
            model.model(**batch, use_cache=False, return_dict=True)
            armed["value"] = False
            del batch
            completed = min(start + 4, len(rendered))
            if completed % 128 == 0 or completed == len(rendered):
                print(
                    json.dumps(
                        {
                            "collect_safe2048": completed,
                            "total": len(rendered),
                            "seconds": time.time() - started,
                        }
                    ),
                    flush=True,
                )
    finally:
        armed["value"] = False
        for handle in handles:
            handle.remove()
    result = {
        editor.site_id: torch.cat(storage[editor.site_id]).contiguous()
        for editor in editors
    }
    save_file(
        result,
        SAFE_INPUT_CACHE,
        metadata={
            "schema_version": "gemma4-e2b-teacher-delta-safe-v1",
            "count": str(SAFE_COUNT),
        },
    )
    base.empty_device_cache(device)
    return result


def conjugate_gradient(
    harmful: torch.Tensor,
    safe: torch.Tensor,
    exact: torch.Tensor,
    *,
    safe_lambda: float,
) -> tuple[torch.Tensor, int, float]:
    """Solve (C_h + lambda C_s + ridge I)u = C_h w on the accelerator."""

    harmful_values = harmful.to(dtype=torch.float32)
    safe_values = safe.to(dtype=torch.float32)
    exact_value = exact.to(device=harmful_values.device, dtype=torch.float32)
    harmful_count = float(len(harmful_values))
    safe_count = float(len(safe_values))
    ridge = float(
        RIDGE_FRACTION
        * (
            harmful_values.square().mean()
            + safe_lambda * safe_values.square().mean()
        )
    )

    def covariance(value: torch.Tensor, vector: torch.Tensor, count: float) -> torch.Tensor:
        return value.T @ (value @ vector) / count

    def system(vector: torch.Tensor) -> torch.Tensor:
        return (
            covariance(harmful_values, vector, harmful_count)
            + safe_lambda * covariance(safe_values, vector, safe_count)
            + ridge * vector
        )

    right = covariance(harmful_values, exact_value, harmful_count)
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
        residual_ratio = float(
            torch.linalg.vector_norm(new_residual) / initial_norm
        )
        iterations = iteration
        if residual_ratio <= CG_TOLERANCE:
            residual = new_residual
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


def fit_editors(
    model: Any,
    raw_editors: list[Any],
    harmful_inputs: dict[str, torch.Tensor],
    safe_inputs: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[float, list[DistilledEditor]], dict[str, Any]]:
    if DETECTOR_CACHE.is_file() and DIAGNOSTICS_PATH.is_file():
        cached = load_file(DETECTOR_CACHE)
        diagnostics = json.loads(DIAGNOSTICS_PATH.read_text(encoding="utf-8"))
        expected = {
            f"lambda{safe_lambda:g}.{editor.site_id}"
            for safe_lambda in SAFE_LAMBDAS
            for editor in raw_editors
        }
        if expected.issubset(cached):
            result = {}
            raw_by_id = {row.site_id: row for row in raw_editors}
            for safe_lambda in SAFE_LAMBDAS:
                rows = []
                for payload in diagnostics[f"lambda{safe_lambda:g}"]:
                    raw = raw_by_id[payload["site_id"]]
                    rows.append(
                        DistilledEditor(
                            raw.site_id,
                            raw.module_path,
                            raw.family,
                            raw.layer,
                            raw.axis.float(),
                            cached[f"lambda{safe_lambda:g}.{raw.site_id}"].float(),
                            float(payload["harmful_retention"]),
                            float(payload["harmful_correlation"]),
                            float(payload["safe_retention"]),
                            float(payload["safe_absolute_ratio"]),
                            int(payload["cg_iterations"]),
                            float(payload["cg_residual_ratio"]),
                        )
                    )
                result[safe_lambda] = rows
            print(json.dumps({"reuse": str(DETECTOR_CACHE)}), flush=True)
            return result, diagnostics

    result: dict[float, list[DistilledEditor]] = {}
    detector_payload: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, Any] = {}
    for safe_lambda in SAFE_LAMBDAS:
        fitted = []
        rows = []
        for raw in raw_editors:
            module = model.get_submodule(raw.module_path)
            weight = module.weight.detach().float().cpu()
            axis = raw.axis.float().cpu()
            exact = weight.T @ axis
            harmful = harmful_inputs[raw.site_id]
            safe = safe_inputs[raw.site_id]
            detector, iterations, residual_ratio = conjugate_gradient(
                harmful.to(device),
                safe.to(device),
                exact,
                safe_lambda=safe_lambda,
            )
            harmful_exact = harmful.float() @ exact
            harmful_fit = harmful.float() @ detector.float()
            safe_exact = safe.float() @ exact
            safe_fit = safe.float() @ detector.float()
            harmful_retention = float(
                harmful_fit.square().mean().sqrt()
                / harmful_exact.square().mean().sqrt().clamp_min(1e-12)
            )
            safe_retention = float(
                safe_fit.square().mean().sqrt()
                / safe_exact.square().mean().sqrt().clamp_min(1e-12)
            )
            safe_absolute_ratio = float(
                safe_fit.square().mean().sqrt()
                / harmful_exact.square().mean().sqrt().clamp_min(1e-12)
            )
            editor = DistilledEditor(
                raw.site_id,
                raw.module_path,
                raw.family,
                raw.layer,
                axis,
                detector,
                harmful_retention,
                correlation(harmful_fit, harmful_exact),
                safe_retention,
                safe_absolute_ratio,
                iterations,
                residual_ratio,
            )
            fitted.append(editor)
            payload = {
                "site_id": editor.site_id,
                "harmful_retention": editor.harmful_retention,
                "harmful_correlation": editor.harmful_correlation,
                "safe_retention": editor.safe_retention,
                "safe_absolute_ratio": editor.safe_absolute_ratio,
                "cg_iterations": editor.cg_iterations,
                "cg_residual_ratio": editor.cg_residual_ratio,
            }
            rows.append(payload)
            detector_payload[f"lambda{safe_lambda:g}.{raw.site_id}"] = detector.to(
                dtype=torch.bfloat16
            )
            print(
                json.dumps(
                    {
                        "fit_teacher_delta": safe_lambda,
                        **payload,
                    }
                ),
                flush=True,
            )
            del weight, exact, harmful_exact, harmful_fit, safe_exact, safe_fit
            base.empty_device_cache(device)
        result[safe_lambda] = fitted
        diagnostics[f"lambda{safe_lambda:g}"] = rows
    save_file(
        detector_payload,
        DETECTOR_CACHE,
        metadata={"schema_version": "gemma4-e2b-distilled-detectors-v1"},
    )
    base.atomic_json(DIAGNOSTICS_PATH, diagnostics)
    return result, diagnostics


def snapshot_weights(
    model: Any, editors: list[DistilledEditor]
) -> dict[str, torch.Tensor]:
    return {
        row.site_id: model.get_submodule(row.module_path)
        .weight.detach()
        .cpu()
        .clone()
        for row in editors
    }


@torch.no_grad()
def apply_candidate(
    model: Any,
    editors: list[DistilledEditor],
    originals: dict[str, torch.Tensor],
    *,
    beta: float,
) -> list[str]:
    for editor in editors:
        module = model.get_submodule(editor.module_path)
        original = originals[editor.site_id].to(module.weight.device).float()
        axis = editor.axis.to(module.weight.device)
        detector = editor.detector.to(module.weight.device)
        module.weight.copy_(
            (original - beta * axis[:, None] @ detector[None, :]).to(
                module.weight.dtype
            )
        )
    return [row.site_id for row in editors]


def key(safe_lambda: float, beta: float) -> tuple[int, int]:
    return round(safe_lambda * 1000), round(beta * 1000)


def load_state() -> dict[str, Any]:
    if STATE_PATH.is_file():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        state.setdefault("full_refusal_trials", [])
        return state
    return {"refusal_trials": [], "kl_trials": [], "full_refusal_trials": []}


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(4053)
    device = base.select_device("auto")
    tokenizer, model = base.load_model(device)
    prompts = base.load_prompts()
    sites = [
        site
        for site in discover_semantic_sites(model).sites
        if site.family in {"gqa", "ffn"}
    ]
    raw_editors = teacher_editors(model, sites)
    paths = json.loads(trajectory.TOKEN_CACHE.read_text(encoding="utf-8"))
    pair_indices = [
        index
        for index, (base_row, teacher_row) in enumerate(
            zip(paths["base"], paths["teacher"])
        )
        if int(base_row["refusal"]) == 1 and int(teacher_row["refusal"]) == 0
    ]
    harmful_inputs = collect_harmful_token_inputs(
        model, paths, pair_indices, raw_editors, device
    )
    safe_inputs = collect_safe_inputs(model, tokenizer, raw_editors, device)
    portfolios, diagnostics = fit_editors(
        model, raw_editors, harmful_inputs, safe_inputs, device
    )
    originals = snapshot_weights(model, portfolios[SAFE_LAMBDAS[0]])
    state = load_state()
    state["diagnostics"] = diagnostics
    completed = {
        key(float(row["safe_lambda"]), float(row["beta"]))
        for row in state["refusal_trials"]
    }
    for safe_lambda, beta in TRIALS:
        trial = key(safe_lambda, beta)
        if trial in completed:
            continue
        selected = apply_candidate(
            model, portfolios[safe_lambda], originals, beta=beta
        )
        evaluation = base.refusal_evaluation(
            model,
            tokenizer,
            prompts["target_test"][:PROBE_COUNT],
            device,
            label=f"teacher-delta:l{safe_lambda:g}:b{beta:.2f}",
            max_new_tokens=PROBE_TOKENS,
        )
        row = {
            "safe_lambda": safe_lambda,
            "beta": beta,
            "refusal_markers": evaluation["refusal_markers"],
            "marker_hits": evaluation["marker_hits"],
            "response_sha256": evaluation["response_sha256"],
            "selected_sites": selected,
        }
        state["refusal_trials"].append(row)
        completed.add(trial)
        base.atomic_json(STATE_PATH, state)
        print(json.dumps({"teacher_delta_refusal": row}), flush=True)

    promising = sorted(
        (
            row
            for row in state["refusal_trials"]
            if int(row["refusal_markers"]) <= 2
        ),
        key=lambda row: (
            int(row["refusal_markers"]),
            -float(row["safe_lambda"]),
        ),
    )
    baseline = load_file(base.BASE_LOG_PROBS_CACHE)["log_probs"].float()
    completed_kl = {
        key(float(row["safe_lambda"]), float(row["beta"]))
        for row in state["kl_trials"]
    }
    for source in promising:
        trial = key(float(source["safe_lambda"]), float(source["beta"]))
        if trial in completed_kl:
            continue
        safe_lambda = float(source["safe_lambda"])
        apply_candidate(
            model,
            portfolios[safe_lambda],
            originals,
            beta=float(source["beta"]),
        )
        candidate = base.next_token_log_probs(
            model,
            tokenizer,
            prompts["safe_test"],
            device,
            label=f"teacher-delta-kl:l{safe_lambda:g}",
        )
        row = {
            **source,
            "first_token_kl": base.mean_first_token_kl(baseline, candidate),
        }
        state["kl_trials"].append(row)
        completed_kl.add(trial)
        base.atomic_json(STATE_PATH, state)
        print(json.dumps({"teacher_delta_kl": row}), flush=True)

    completed_full = {
        key(float(row["safe_lambda"]), float(row["beta"]))
        for row in state["full_refusal_trials"]
    }
    finalists = sorted(
        (
            row
            for row in state["kl_trials"]
            if float(row["first_token_kl"]) <= base.KL_HARD_CAP
        ),
        key=lambda row: (
            int(row["refusal_markers"]),
            float(row["first_token_kl"]),
        ),
    )
    for source in finalists[:1]:
        trial = key(float(source["safe_lambda"]), float(source["beta"]))
        if trial in completed_full:
            continue
        safe_lambda = float(source["safe_lambda"])
        apply_candidate(
            model,
            portfolios[safe_lambda],
            originals,
            beta=float(source["beta"]),
        )
        evaluation = base.refusal_evaluation(
            model,
            tokenizer,
            prompts["target_test"],
            device,
            label=f"teacher-delta-full:l{safe_lambda:g}",
            max_new_tokens=PROBE_TOKENS,
        )
        row = {
            **source,
            "full_count": len(prompts["target_test"]),
            "full_refusal_markers": evaluation["refusal_markers"],
            "full_marker_hits": evaluation["marker_hits"],
            "full_response_sha256": evaluation["response_sha256"],
        }
        state["full_refusal_trials"].append(row)
        completed_full.add(trial)
        base.atomic_json(STATE_PATH, state)
        print(json.dumps({"teacher_delta_full_refusal": row}), flush=True)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
