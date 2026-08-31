#!/usr/bin/env python3
"""Second-pass residual repair for the distilled Gemma teacher delta."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any

from safetensors.torch import load_file, save_file
import torch

import gemma4_e2b_residual_stream_prime as base
import gemma4_e2b_teacher_delta_distillation as distilled
import gemma4_e2b_teacher_trajectory_conditional as trajectory
from heretic_nx.model import discover_semantic_sites


SAFE_LAMBDA = 100.0
PARENT_BETA = 3.0
REPAIR_START = 32
REPAIR_COUNT = 64
REPAIR_REPEAT = 4
TRIAL_BETAS = (2.0, 2.5, 3.0, 3.5)

PATH_CACHE = base.RUN_DIR / "teacher-delta-repair-train-paths.json"
INPUT_CACHE = base.RUN_DIR / "teacher-delta-repair-token-inputs.safetensors"
DETECTOR_CACHE = base.RUN_DIR / "teacher-delta-repair-detectors.safetensors"
DIAGNOSTICS_PATH = base.RUN_DIR / "teacher-delta-repair-diagnostics.json"
STATE_PATH = base.RUN_DIR / "teacher-delta-residual-repair-state.json"


@dataclass(frozen=True)
class RepairEditor:
    site_id: str
    module_path: str
    family: str
    layer: int
    axis: torch.Tensor
    detector: torch.Tensor


@torch.no_grad()
def restore_weights(
    model: Any, editors: list[Any], originals: dict[str, torch.Tensor]
) -> None:
    for editor in editors:
        module = model.get_submodule(editor.module_path)
        module.weight.copy_(
            originals[editor.site_id].to(
                device=module.weight.device, dtype=module.weight.dtype
            )
        )


def load_or_generate_repair_paths(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    parent_editors: list[Any],
    originals: dict[str, torch.Tensor],
    device: torch.device,
) -> list[dict[str, Any]]:
    if PATH_CACHE.is_file():
        payload = json.loads(PATH_CACHE.read_text(encoding="utf-8"))
        if len(payload.get("paths", [])) == len(prompts):
            print(json.dumps({"reuse": str(PATH_CACHE)}), flush=True)
            return payload["paths"]
    distilled.apply_candidate(
        model, parent_editors, originals, beta=PARENT_BETA
    )
    paths = trajectory.generate_paths(
        model,
        tokenizer,
        prompts,
        device,
        label=f"repair-parent-b{PARENT_BETA:.1f}",
    )
    base.atomic_json(
        PATH_CACHE,
        {
            "schema_version": "gemma4-e2b-repair-paths-v1",
            "parent_beta": PARENT_BETA,
            "start": REPAIR_START,
            "count": len(prompts),
            "paths": paths,
        },
    )
    return paths


@torch.inference_mode()
def load_or_collect_repair_inputs(
    model: Any,
    paths: list[dict[str, Any]],
    editors: list[Any],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    refusal_indices = [
        index for index, row in enumerate(paths) if int(row["refusal"]) == 1
    ]
    if len(refusal_indices) < 2:
        raise RuntimeError(
            f"only {len(refusal_indices)} residual repair rows; expand calibration"
        )
    if INPUT_CACHE.is_file():
        cached = load_file(INPUT_CACHE)
        cached_indices = [int(value) for value in cached["indices"].tolist()]
        if cached_indices == refusal_indices:
            print(json.dumps({"reuse": str(INPUT_CACHE)}), flush=True)
            return dict(cached), refusal_indices

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
        for completed, index in enumerate(refusal_indices, start=1):
            row = paths[index]
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
            print(
                json.dumps(
                    {
                        "collect_repair_inputs": completed,
                        "total": len(refusal_indices),
                    }
                ),
                flush=True,
            )
    finally:
        armed["value"] = False
        for handle in handles:
            handle.remove()
    result = {
        "indices": torch.tensor(refusal_indices, dtype=torch.int32),
        **{
            editor.site_id: torch.cat(storage[editor.site_id]).contiguous()
            for editor in editors
        },
    }
    save_file(
        result,
        INPUT_CACHE,
        metadata={"schema_version": "gemma4-e2b-repair-inputs-v1"},
    )
    base.empty_device_cache(device)
    return result, refusal_indices


def fit_repair_editors(
    model: Any,
    raw_editors: list[Any],
    original_harmful: dict[str, torch.Tensor],
    repair_inputs: dict[str, torch.Tensor],
    safe_inputs: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[list[RepairEditor], list[dict[str, Any]]]:
    if DETECTOR_CACHE.is_file() and DIAGNOSTICS_PATH.is_file():
        cached = load_file(DETECTOR_CACHE)
        if all(row.site_id in cached for row in raw_editors):
            diagnostics = json.loads(
                DIAGNOSTICS_PATH.read_text(encoding="utf-8")
            )
            return (
                [
                    RepairEditor(
                        row.site_id,
                        row.module_path,
                        row.family,
                        row.layer,
                        row.axis.float(),
                        cached[row.site_id].float(),
                    )
                    for row in raw_editors
                ],
                diagnostics,
            )

    fitted = []
    diagnostics = []
    detector_payload = {}
    for raw in raw_editors:
        module = model.get_submodule(raw.module_path)
        weight = module.weight.detach().float().cpu()
        axis = raw.axis.float().cpu()
        exact = weight.T @ axis
        original = original_harmful[raw.site_id]
        repair = repair_inputs[raw.site_id]
        augmented = torch.cat(
            (original, *([repair] * REPAIR_REPEAT)), dim=0
        ).contiguous()
        detector, iterations, residual_ratio = distilled.conjugate_gradient(
            augmented.to(device),
            safe_inputs[raw.site_id].to(device),
            exact,
            safe_lambda=SAFE_LAMBDA,
        )
        original_exact = original.float() @ exact
        original_fit = original.float() @ detector.float()
        repair_exact = repair.float() @ exact
        repair_fit = repair.float() @ detector.float()
        safe_fit = safe_inputs[raw.site_id].float() @ detector.float()
        row = {
            "site_id": raw.site_id,
            "original_retention": float(
                original_fit.square().mean().sqrt()
                / original_exact.square().mean().sqrt().clamp_min(1e-12)
            ),
            "repair_retention": float(
                repair_fit.square().mean().sqrt()
                / repair_exact.square().mean().sqrt().clamp_min(1e-12)
            ),
            "safe_absolute_ratio": float(
                safe_fit.square().mean().sqrt()
                / augmented.float()
                .matmul(exact)
                .square()
                .mean()
                .sqrt()
                .clamp_min(1e-12)
            ),
            "cg_iterations": iterations,
            "cg_residual_ratio": residual_ratio,
        }
        diagnostics.append(row)
        print(json.dumps({"fit_residual_repair": row}), flush=True)
        fitted.append(
            RepairEditor(
                raw.site_id,
                raw.module_path,
                raw.family,
                raw.layer,
                axis,
                detector,
            )
        )
        detector_payload[raw.site_id] = detector.to(dtype=torch.bfloat16)
        del weight, exact, augmented, original_exact, original_fit
        del repair_exact, repair_fit, safe_fit
        base.empty_device_cache(device)
    save_file(
        detector_payload,
        DETECTOR_CACHE,
        metadata={"schema_version": "gemma4-e2b-repair-detectors-v1"},
    )
    base.atomic_json(DIAGNOSTICS_PATH, diagnostics)
    return fitted, diagnostics


def load_state() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"residual_trials": [], "kl_trials": [], "full_trials": []}


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(4057)
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
    repair_prompts = prompts["target_train"][
        REPAIR_START : REPAIR_START + REPAIR_COUNT
    ]
    repair_paths = load_or_generate_repair_paths(
        model,
        tokenizer,
        repair_prompts,
        parent_editors,
        originals,
        device,
    )
    # The paths and their teacher-forced states must both come from the parent.
    distilled.apply_candidate(
        model, parent_editors, originals, beta=PARENT_BETA
    )
    repair_inputs, repair_indices = load_or_collect_repair_inputs(
        model, repair_paths, parent_editors, device
    )
    restore_weights(model, parent_editors, originals)
    editors, diagnostics = fit_repair_editors(
        model,
        raw_editors,
        harmful_inputs,
        repair_inputs,
        safe_inputs,
        device,
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
    state["repair_train_indices"] = repair_indices
    state["test_residual_indices"] = residual_indices
    state["diagnostics"] = diagnostics
    completed = {
        round(float(row["beta"]) * 1000) for row in state["residual_trials"]
    }
    for beta in TRIAL_BETAS:
        key = round(beta * 1000)
        if key in completed:
            continue
        distilled.apply_candidate(model, editors, originals, beta=beta)
        evaluation = base.refusal_evaluation(
            model,
            tokenizer,
            residual_prompts,
            device,
            label=f"teacher-delta-repair:b{beta:.2f}",
            max_new_tokens=base.MAX_NEW_TOKENS,
        )
        row = {
            "beta": beta,
            "refusal_markers": evaluation["refusal_markers"],
            "marker_hits": evaluation["marker_hits"],
            "response_sha256": evaluation["response_sha256"],
        }
        state["residual_trials"].append(row)
        completed.add(key)
        base.atomic_json(STATE_PATH, state)
        print(json.dumps({"repair_residual_trial": row}), flush=True)
        if int(row["refusal_markers"]) <= 3:
            break

    promising = sorted(
        (row for row in state["residual_trials"] if row["refusal_markers"] <= 3),
        key=lambda row: (float(row["beta"]), int(row["refusal_markers"])),
    )
    baseline = load_file(base.BASE_LOG_PROBS_CACHE)["log_probs"].float()
    completed_kl = {
        round(float(row["beta"]) * 1000) for row in state["kl_trials"]
    }
    for source in promising[:1]:
        beta = float(source["beta"])
        key = round(beta * 1000)
        if key in completed_kl:
            continue
        distilled.apply_candidate(model, editors, originals, beta=beta)
        candidate = base.next_token_log_probs(
            model,
            tokenizer,
            prompts["safe_test"],
            device,
            label=f"teacher-delta-repair-kl:b{beta:.2f}",
        )
        row = {
            **source,
            "first_token_kl": base.mean_first_token_kl(baseline, candidate),
        }
        state["kl_trials"].append(row)
        completed_kl.add(key)
        base.atomic_json(STATE_PATH, state)
        print(json.dumps({"repair_kl": row}), flush=True)

    finalists = [
        row
        for row in state["kl_trials"]
        if float(row["first_token_kl"]) <= base.KL_HARD_CAP
    ]
    completed_full = {
        round(float(row["beta"]) * 1000) for row in state["full_trials"]
    }
    for source in finalists[:1]:
        beta = float(source["beta"])
        key = round(beta * 1000)
        if key in completed_full:
            continue
        distilled.apply_candidate(model, editors, originals, beta=beta)
        evaluation = base.refusal_evaluation(
            model,
            tokenizer,
            prompts["target_test"],
            device,
            label=f"teacher-delta-repair-full:b{beta:.2f}",
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
        print(json.dumps({"repair_full": row}), flush=True)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
