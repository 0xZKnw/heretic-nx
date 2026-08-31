#!/usr/bin/env python3
"""Add a residual-only repair operator on top of the lambda-100 parent."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any

from safetensors.torch import load_file, save_file
import torch

import gemma4_e2b_residual_stream_prime as base
import gemma4_e2b_teacher_delta_distillation as distilled
import gemma4_e2b_teacher_delta_residual_repair as repair
import gemma4_e2b_teacher_trajectory_conditional as trajectory
from heretic_nx.model import discover_semantic_sites


SAFE_LAMBDA = 100.0
PARENT_BETA = 3.0
REPAIR_GAMMAS = (0.5, 1.0, 1.5, 2.0, 3.0)

DETECTOR_CACHE = base.RUN_DIR / "teacher-delta-additive-repair-detectors.safetensors"
DIAGNOSTICS_PATH = base.RUN_DIR / "teacher-delta-additive-repair-diagnostics.json"
STATE_PATH = base.RUN_DIR / "teacher-delta-additive-repair-state.json"


@dataclass(frozen=True)
class AdditiveRepair:
    site_id: str
    module_path: str
    axis: torch.Tensor
    detector: torch.Tensor


def fit_repair_only(
    model: Any,
    raw_editors: list[Any],
    repair_inputs: dict[str, torch.Tensor],
    safe_inputs: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[list[AdditiveRepair], list[dict[str, Any]]]:
    if DETECTOR_CACHE.is_file() and DIAGNOSTICS_PATH.is_file():
        cached = load_file(DETECTOR_CACHE)
        if all(row.site_id in cached for row in raw_editors):
            diagnostics = json.loads(
                DIAGNOSTICS_PATH.read_text(encoding="utf-8")
            )
            return (
                [
                    AdditiveRepair(
                        row.site_id,
                        row.module_path,
                        row.axis.float(),
                        cached[row.site_id].float(),
                    )
                    for row in raw_editors
                ],
                diagnostics,
            )
    editors = []
    diagnostics = []
    detector_payload = {}
    for raw in raw_editors:
        module = model.get_submodule(raw.module_path)
        weight = module.weight.detach().float().cpu()
        axis = raw.axis.float().cpu()
        exact = weight.T @ axis
        harmful = repair_inputs[raw.site_id]
        safe = safe_inputs[raw.site_id]
        detector, iterations, residual_ratio = distilled.conjugate_gradient(
            harmful.to(device),
            safe.to(device),
            exact,
            safe_lambda=SAFE_LAMBDA,
        )
        harmful_exact = harmful.float() @ exact
        harmful_fit = harmful.float() @ detector.float()
        safe_fit = safe.float() @ detector.float()
        row = {
            "site_id": raw.site_id,
            "harmful_retention": float(
                harmful_fit.square().mean().sqrt()
                / harmful_exact.square().mean().sqrt().clamp_min(1e-12)
            ),
            "safe_absolute_ratio": float(
                safe_fit.square().mean().sqrt()
                / harmful_exact.square().mean().sqrt().clamp_min(1e-12)
            ),
            "cg_iterations": iterations,
            "cg_residual_ratio": residual_ratio,
        }
        print(json.dumps({"fit_additive_repair": row}), flush=True)
        diagnostics.append(row)
        editors.append(
            AdditiveRepair(raw.site_id, raw.module_path, axis, detector)
        )
        detector_payload[raw.site_id] = detector.to(dtype=torch.bfloat16)
        del weight, exact, harmful_exact, harmful_fit, safe_fit
        base.empty_device_cache(device)
    save_file(
        detector_payload,
        DETECTOR_CACHE,
        metadata={"schema_version": "gemma4-e2b-additive-repair-v1"},
    )
    base.atomic_json(DIAGNOSTICS_PATH, diagnostics)
    return editors, diagnostics


@torch.no_grad()
def apply_composed(
    model: Any,
    parent_editors: list[Any],
    repair_editors: list[AdditiveRepair],
    originals: dict[str, torch.Tensor],
    *,
    gamma: float,
) -> None:
    repair_by_id = {row.site_id: row for row in repair_editors}
    for parent in parent_editors:
        extra = repair_by_id[parent.site_id]
        module = model.get_submodule(parent.module_path)
        original = originals[parent.site_id].to(module.weight.device).float()
        axis = parent.axis.to(module.weight.device)
        parent_detector = parent.detector.to(module.weight.device)
        repair_detector = extra.detector.to(module.weight.device)
        detector = PARENT_BETA * parent_detector + gamma * repair_detector
        module.weight.copy_(
            (original - axis[:, None] @ detector[None, :]).to(
                module.weight.dtype
            )
        )


def load_state() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"residual_trials": [], "kl_trials": [], "full_trials": []}


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(4058)
    device = base.select_device("auto")
    tokenizer, model = base.load_model(device)
    prompts = base.load_prompts()
    sites = [
        site
        for site in discover_semantic_sites(model).sites
        if site.family in {"gqa", "ffn"}
    ]
    raw_editors = distilled.teacher_editors(model, sites)
    paths = json.loads(trajectory.TOKEN_CACHE.read_text(encoding="utf-8"))
    pair_indices = [
        index
        for index, (base_row, teacher_row) in enumerate(
            zip(paths["base"], paths["teacher"])
        )
        if int(base_row["refusal"]) == 1 and int(teacher_row["refusal"]) == 0
    ]
    harmful_inputs = distilled.collect_harmful_token_inputs(
        model, paths, pair_indices, raw_editors, device
    )
    safe_inputs = distilled.collect_safe_inputs(
        model, tokenizer, raw_editors, device
    )
    parent_portfolios, _ = distilled.fit_editors(
        model, raw_editors, harmful_inputs, safe_inputs, device
    )
    parent_editors = parent_portfolios[SAFE_LAMBDA]
    originals = distilled.snapshot_weights(model, parent_editors)
    repair_inputs = dict(load_file(repair.INPUT_CACHE))
    repair_editors, diagnostics = fit_repair_only(
        model, raw_editors, repair_inputs, safe_inputs, device
    )

    parent_state = json.loads(distilled.STATE_PATH.read_text(encoding="utf-8"))
    parent_full = next(
        row
        for row in parent_state["full_refusal_trials"]
        if float(row["safe_lambda"]) == SAFE_LAMBDA
    )
    residual_indices = [
        index
        for index, hit in enumerate(parent_full["full_marker_hits"])
        if int(hit) == 1
    ]
    residual_prompts = [prompts["target_test"][index] for index in residual_indices]
    state = load_state()
    state["residual_indices"] = residual_indices
    state["diagnostics"] = diagnostics
    completed = {
        round(float(row["gamma"]) * 1000) for row in state["residual_trials"]
    }
    for gamma in REPAIR_GAMMAS:
        key = round(gamma * 1000)
        if key in completed:
            continue
        apply_composed(
            model,
            parent_editors,
            repair_editors,
            originals,
            gamma=gamma,
        )
        evaluation = base.refusal_evaluation(
            model,
            tokenizer,
            residual_prompts,
            device,
            label=f"teacher-delta-additive:g{gamma:.2f}",
            max_new_tokens=base.MAX_NEW_TOKENS,
        )
        row = {
            "parent_beta": PARENT_BETA,
            "gamma": gamma,
            "refusal_markers": evaluation["refusal_markers"],
            "marker_hits": evaluation["marker_hits"],
            "response_sha256": evaluation["response_sha256"],
        }
        state["residual_trials"].append(row)
        completed.add(key)
        base.atomic_json(STATE_PATH, state)
        print(json.dumps({"additive_residual": row}), flush=True)
        if int(row["refusal_markers"]) <= 3:
            break

    promising = sorted(
        (row for row in state["residual_trials"] if row["refusal_markers"] <= 3),
        key=lambda row: (float(row["gamma"]), int(row["refusal_markers"])),
    )
    baseline = load_file(base.BASE_LOG_PROBS_CACHE)["log_probs"].float()
    completed_kl = {
        round(float(row["gamma"]) * 1000) for row in state["kl_trials"]
    }
    for source in promising[:1]:
        gamma = float(source["gamma"])
        key = round(gamma * 1000)
        if key in completed_kl:
            continue
        apply_composed(
            model,
            parent_editors,
            repair_editors,
            originals,
            gamma=gamma,
        )
        candidate = base.next_token_log_probs(
            model,
            tokenizer,
            prompts["safe_test"],
            device,
            label=f"teacher-delta-additive-kl:g{gamma:.2f}",
        )
        row = {
            **source,
            "first_token_kl": base.mean_first_token_kl(baseline, candidate),
        }
        state["kl_trials"].append(row)
        completed_kl.add(key)
        base.atomic_json(STATE_PATH, state)
        print(json.dumps({"additive_kl": row}), flush=True)

    finalists = [
        row
        for row in state["kl_trials"]
        if float(row["first_token_kl"]) <= base.KL_HARD_CAP
    ]
    completed_full = {
        round(float(row["gamma"]) * 1000) for row in state["full_trials"]
    }
    for source in finalists[:1]:
        gamma = float(source["gamma"])
        key = round(gamma * 1000)
        if key in completed_full:
            continue
        apply_composed(
            model,
            parent_editors,
            repair_editors,
            originals,
            gamma=gamma,
        )
        evaluation = base.refusal_evaluation(
            model,
            tokenizer,
            prompts["target_test"],
            device,
            label=f"teacher-delta-additive-full:g{gamma:.2f}",
            max_new_tokens=base.MAX_NEW_TOKENS,
        )
        row = {
            **source,
            "full_count": len(prompts["target_test"]),
            "full_refusal_markers": evaluation["refusal_markers"],
            "full_marker_hits": evaluation["marker_hits"],
            "full_response_sha256": evaluation["response_sha256"],
        }
        state["full_trials"].append(row)
        completed_full.add(key)
        base.atomic_json(STATE_PATH, state)
        print(json.dumps({"additive_full": row}), flush=True)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
