#!/usr/bin/env python3
"""Teacher-trajectory conditional repair for Gemma 4 E2B.

The strong local-site candidate is used only as a behavioural teacher.  We
contrast base refusal trajectories with non-refusing teacher trajectories,
then learn rank-one edits whose input detector has minimum energy on benign
prompt-boundary states.  This targets the late refusal states missed by the
prompt-only conditional experiment while explicitly protecting first-token KL.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

from safetensors.torch import load_file, save_file
import torch

import gemma4_e2b_conditional_rank_one as conditional
import gemma4_e2b_local_site_refusal_first as local
import gemma4_e2b_residual_stream_prime as base
from heretic_nx.geometry.contrastive import fit_contrastive_axis
from heretic_nx.hashing import sha256_json
from heretic_nx.model import discover_semantic_sites


TRAIN_COUNT = 32
RESPONSE_TOKENS = 96
EARLY_TOKENS = 16
PROBE_COUNT = 20
TEACHER_K = 7
TEACHER_BETA = 2.0

TOKEN_CACHE = base.RUN_DIR / "teacher-trajectory-token-paths.json"
TRAJECTORY_CACHE = base.RUN_DIR / "teacher-trajectory-site-states.safetensors"
SAFE_INPUT_CACHE = base.RUN_DIR / "all-site-safe-inputs-256.safetensors"
STATE_PATH = base.RUN_DIR / "teacher-trajectory-conditional-state.json"

# Refusal is always screened at the real 96-token length.  KL is only measured
# for candidates with at most two markers on this 20-row development screen.
TRIALS = (
    ("late", 8, 1.00),
    ("late", 8, 1.50),
    ("late", 8, 2.00),
    ("late", 4, 1.50),
    ("all", 8, 1.50),
    ("early", 8, 1.50),
)


@dataclass(frozen=True)
class TrajectoryEditor:
    site_id: str
    module_path: str
    family: str
    layer: int
    axis: torch.Tensor
    detector: torch.Tensor
    target_projection: float
    score: float
    fold_cosine_minimum: float
    safe_gate_rms: float
    target_gate_mean: float
    safe_relative_effect: float
    target_relative_effect: float


def has_refusal(text: str) -> bool:
    normalized = base.normalized_response(text)
    return any(marker in normalized for marker in base.REFUSAL_MARKERS)


def termination_ids(model: Any, tokenizer: Any) -> set[int]:
    values = model.generation_config.eos_token_id
    if isinstance(values, int):
        result = {values}
    else:
        result = {int(value) for value in (values or [])}
    if tokenizer.pad_token_id is not None:
        result.add(int(tokenizer.pad_token_id))
    return result


def trim_response(ids: list[int], stops: set[int]) -> list[int]:
    for index, token_id in enumerate(ids):
        if token_id in stops:
            return ids[: max(index, 1)]
    return ids


@torch.inference_mode()
def generate_paths(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    device: torch.device,
    *,
    label: str,
) -> list[dict[str, Any]]:
    rendered = base.render(tokenizer, prompts)
    rows: list[dict[str, Any]] = []
    stops = termination_ids(model, tokenizer)
    started = time.time()
    for start in range(0, len(rendered), base.EVAL_BATCH):
        batch = base.tokenize(
            tokenizer, rendered[start : start + base.EVAL_BATCH], device
        )
        output = model.generate(
            **batch,
            max_new_tokens=RESPONSE_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated = output[:, batch["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=False)
        for offset in range(generated.shape[0]):
            mask = batch["attention_mask"][offset].bool()
            prompt_ids = batch["input_ids"][offset][mask].detach().cpu().tolist()
            response_ids = trim_response(
                generated[offset].detach().cpu().tolist(), stops
            )
            rows.append(
                {
                    "prompt_ids": prompt_ids,
                    "response_ids": response_ids,
                    "refusal": int(has_refusal(decoded[offset])),
                    "response_sha256": sha256_json(decoded[offset]),
                }
            )
        del output, generated, batch
        completed = min(start + base.EVAL_BATCH, len(rendered))
        if completed % 8 == 0 or completed == len(rendered):
            print(
                json.dumps(
                    {
                        "generate_paths": label,
                        "completed": completed,
                        "total": len(rendered),
                    }
                ),
                flush=True,
            )
    print(
        json.dumps(
            {
                "generated_paths": label,
                "refusals": sum(int(row["refusal"]) for row in rows),
                "seconds": time.time() - started,
            }
        ),
        flush=True,
    )
    base.empty_device_cache(device)
    return rows


def snapshot_selected(model: Any, editors: list[Any]) -> dict[str, torch.Tensor]:
    return {
        editor.site_id: model.get_submodule(editor.module_path)
        .weight.detach()
        .cpu()
        .clone()
        for editor in editors
    }


@torch.no_grad()
def apply_local_teacher(
    model: Any,
    editors: list[Any],
    originals: dict[str, torch.Tensor],
    beta: float,
) -> None:
    for editor in editors:
        module = model.get_submodule(editor.module_path)
        original = originals[editor.site_id].to(module.weight.device)
        axis = editor.axis.to(device=original.device, dtype=torch.float32)
        weight = original.float()
        edited = weight - beta * axis[:, None] @ (axis[None, :] @ weight)
        module.weight.copy_(edited.to(module.weight.dtype))


@torch.no_grad()
def restore_selected(
    model: Any, editors: list[Any], originals: dict[str, torch.Tensor]
) -> None:
    for editor in editors:
        module = model.get_submodule(editor.module_path)
        module.weight.copy_(
            originals[editor.site_id].to(
                device=module.weight.device, dtype=module.weight.dtype
            )
        )


def load_or_generate_paths(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    sites: list[Any],
    device: torch.device,
) -> dict[str, Any]:
    prompt_digest = sha256_json(prompts)
    if TOKEN_CACHE.is_file():
        cached = json.loads(TOKEN_CACHE.read_text(encoding="utf-8"))
        if (
            cached.get("prompt_sha256") == prompt_digest
            and int(cached.get("response_tokens", -1)) == RESPONSE_TOKENS
            and len(cached.get("base", [])) == len(prompts)
            and len(cached.get("teacher", [])) == len(prompts)
        ):
            print(json.dumps({"reuse": str(TOKEN_CACHE)}), flush=True)
            return cached

    payload: dict[str, Any] = {
        "schema_version": "gemma4-e2b-teacher-trajectories-v1",
        "prompt_sha256": prompt_digest,
        "response_tokens": RESPONSE_TOKENS,
        "teacher_k": TEACHER_K,
        "teacher_beta": TEACHER_BETA,
    }
    payload["base"] = generate_paths(
        model, tokenizer, prompts, device, label="base"
    )
    base.atomic_json(TOKEN_CACHE, payload)

    cached_outputs = load_file(local.ACTIVATION_CACHE)
    safe_outputs = {
        site.id: cached_outputs[f"safe.{site.id}"] for site in sites
    }
    target_outputs = {
        site.id: cached_outputs[f"target.{site.id}"] for site in sites
    }
    local_editors = local.build_editors(sites, safe_outputs, target_outputs)
    teacher_editors = local_editors[:TEACHER_K]
    teacher_originals = snapshot_selected(model, teacher_editors)
    try:
        apply_local_teacher(model, teacher_editors, teacher_originals, TEACHER_BETA)
        payload["teacher"] = generate_paths(
            model, tokenizer, prompts, device, label="teacher-k7-b2"
        )
    finally:
        restore_selected(model, teacher_editors, teacher_originals)
    payload["teacher_sites"] = [row.site_id for row in teacher_editors]
    base.atomic_json(TOKEN_CACHE, payload)
    return payload


def window_bounds(start: int, length: int, window: str) -> tuple[int, int]:
    stop = start + max(length, 1)
    if window == "early":
        return start, min(stop, start + EARLY_TOKENS)
    if window == "late":
        late_start = min(stop - 1, start + EARLY_TOKENS)
        return late_start, stop
    if window == "all":
        return start, stop
    raise ValueError(window)


@torch.inference_mode()
def collect_trajectory_states(
    model: Any,
    paths: list[dict[str, Any]],
    indices: list[int],
    sites: list[Any],
    device: torch.device,
    *,
    label: str,
) -> dict[str, torch.Tensor]:
    windows = ("early", "late", "all")
    input_rows: dict[tuple[str, str], list[torch.Tensor]] = {
        (window, site.id): [] for window in windows for site in sites
    }
    output_rows: dict[tuple[str, str], list[torch.Tensor]] = {
        (window, site.id): [] for window in windows for site in sites
    }
    armed: dict[str, Any] = {"value": False, "start": 0, "length": 0}
    handles = []
    for site in sites:
        module = model.get_submodule(site.module_path)

        def capture_input(_module, inputs, *, site_id=site.id):
            if not armed["value"]:
                return
            value = inputs[0]
            for window in windows:
                left, right = window_bounds(
                    int(armed["start"]), int(armed["length"]), window
                )
                input_rows[(window, site_id)].append(
                    value[:, left:right]
                    .detach()
                    .float()
                    .mean(dim=1)
                    .to(dtype=torch.bfloat16, device="cpu")
                )

        def capture_output(_module, _inputs, output, *, site_id=site.id):
            if not armed["value"]:
                return
            value = output[0] if isinstance(output, tuple) else output
            for window in windows:
                left, right = window_bounds(
                    int(armed["start"]), int(armed["length"]), window
                )
                output_rows[(window, site_id)].append(
                    value[:, left:right]
                    .detach()
                    .float()
                    .mean(dim=1)
                    .to(dtype=torch.bfloat16, device="cpu")
                )

        handles.append(module.register_forward_pre_hook(capture_input))
        handles.append(module.register_forward_hook(capture_output))

    try:
        for completed, index in enumerate(indices, start=1):
            row = paths[index]
            prompt_ids = [int(value) for value in row["prompt_ids"]]
            response_ids = [int(value) for value in row["response_ids"]]
            ids = torch.tensor(
                [prompt_ids + response_ids], dtype=torch.long, device=device
            )
            mask = torch.ones_like(ids)
            armed.update(
                {"value": True, "start": len(prompt_ids), "length": len(response_ids)}
            )
            model.model(
                input_ids=ids,
                attention_mask=mask,
                use_cache=False,
                return_dict=True,
            )
            armed["value"] = False
            del ids, mask
            if completed % 8 == 0 or completed == len(indices):
                print(
                    json.dumps(
                        {
                            "collect_trajectories": label,
                            "completed": completed,
                            "total": len(indices),
                        }
                    ),
                    flush=True,
                )
    finally:
        armed["value"] = False
        for handle in handles:
            handle.remove()
    base.empty_device_cache(device)
    return {
        **{
            f"{window}.input.{site.id}": torch.cat(input_rows[(window, site.id)]).contiguous()
            for window in windows
            for site in sites
        },
        **{
            f"{window}.output.{site.id}": torch.cat(output_rows[(window, site.id)]).contiguous()
            for window in windows
            for site in sites
        },
    }


def load_or_collect_trajectories(
    model: Any,
    paths: dict[str, Any],
    sites: list[Any],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    indices = [
        index
        for index, (base_row, teacher_row) in enumerate(
            zip(paths["base"], paths["teacher"])
        )
        if int(base_row["refusal"]) == 1 and int(teacher_row["refusal"]) == 0
    ]
    if len(indices) < 6:
        raise RuntimeError(
            f"only {len(indices)} base-refusal/teacher-success trajectory pairs"
        )
    if TRAJECTORY_CACHE.is_file():
        cached = load_file(TRAJECTORY_CACHE)
        cached_indices = [int(value) for value in cached["pair_indices"].tolist()]
        if cached_indices == indices:
            print(json.dumps({"reuse": str(TRAJECTORY_CACHE)}), flush=True)
            return dict(cached), indices

    baseline = collect_trajectory_states(
        model, paths["base"], indices, sites, device, label="base-refusal"
    )
    teacher = collect_trajectory_states(
        model, paths["teacher"], indices, sites, device, label="teacher-success"
    )
    payload = {
        "pair_indices": torch.tensor(indices, dtype=torch.int32),
        **{f"base.{key}": value for key, value in baseline.items()},
        **{f"teacher.{key}": value for key, value in teacher.items()},
    }
    save_file(
        payload,
        TRAJECTORY_CACHE,
        metadata={
            "schema_version": "gemma4-e2b-teacher-trajectory-states-v1",
            "model_revision": base.MODEL_REVISION,
        },
    )
    return payload, indices


@torch.inference_mode()
def collect_safe_inputs(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    sites: list[Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if SAFE_INPUT_CACHE.is_file():
        print(json.dumps({"reuse": str(SAFE_INPUT_CACHE)}), flush=True)
        return dict(load_file(SAFE_INPUT_CACHE))

    storage: dict[str, list[torch.Tensor]] = {site.id: [] for site in sites}
    armed = {"value": False}
    handles = []
    for site in sites:
        module = model.get_submodule(site.module_path)

        def capture(_module, inputs, *, site_id=site.id):
            if armed["value"]:
                storage[site_id].append(
                    inputs[0][:, -1]
                    .detach()
                    .to(dtype=torch.bfloat16, device="cpu")
                )

        handles.append(module.register_forward_pre_hook(capture))
    rendered = base.render(tokenizer, prompts)
    try:
        for start in range(0, len(rendered), 4):
            batch = base.tokenize(tokenizer, rendered[start : start + 4], device)
            armed["value"] = True
            model.model(**batch, use_cache=False, return_dict=True)
            armed["value"] = False
            del batch
            completed = min(start + 4, len(rendered))
            if completed % 32 == 0 or completed == len(rendered):
                print(
                    json.dumps(
                        {
                            "collect_safe_inputs": completed,
                            "total": len(rendered),
                        }
                    ),
                    flush=True,
                )
    finally:
        armed["value"] = False
        for handle in handles:
            handle.remove()
    result = {key: torch.cat(rows).contiguous() for key, rows in storage.items()}
    save_file(
        result,
        SAFE_INPUT_CACHE,
        metadata={
            "schema_version": "gemma4-e2b-all-site-safe-inputs-v1",
            "count": str(len(prompts)),
        },
    )
    base.empty_device_cache(device)
    return result


def build_editors(
    sites: list[Any],
    trajectories: dict[str, torch.Tensor],
    safe_inputs: dict[str, torch.Tensor],
    safe_outputs: dict[str, torch.Tensor],
    *,
    window: str,
) -> list[TrajectoryEditor]:
    editors = []
    for site in sites:
        teacher_output = trajectories[f"teacher.{window}.output.{site.id}"].float()
        base_output = trajectories[f"base.{window}.output.{site.id}"].float()
        base_input = trajectories[f"base.{window}.input.{site.id}"].float()
        evidence = fit_contrastive_axis(
            teacher_output,
            base_output,
            folds=3,
            remove_safe_mean=False,
        )
        axis = evidence.axis.float()
        target_projection = float(
            ((base_output - teacher_output) @ axis).mean()
        )
        detector, _ridge = conditional.minimum_safe_energy_detector(
            safe_inputs[site.id], base_input
        )
        safe_gate = conditional.gate_stats(safe_inputs[site.id], detector)
        target_gate = conditional.gate_stats(base_input, detector)
        safe_output_rms = float(
            safe_outputs[site.id].float().square().sum(dim=1).mean().sqrt()
        )
        target_output_rms = float(
            base_output.square().sum(dim=1).mean().sqrt()
        )
        safe_relative_effect = (
            abs(target_projection) * safe_gate["rms"] / max(safe_output_rms, 1e-8)
        )
        target_relative_effect = (
            abs(target_projection)
            * abs(target_gate["mean"])
            / max(target_output_rms, 1e-8)
        )
        score = (
            target_relative_effect
            * max(evidence.fold_cosine_minimum, 1e-3)
            / max(safe_relative_effect, 1e-7)
        )
        editors.append(
            TrajectoryEditor(
                site.id,
                site.module_path,
                site.family,
                site.layer,
                axis.cpu(),
                detector.cpu(),
                target_projection,
                score,
                evidence.fold_cosine_minimum,
                safe_gate["rms"],
                target_gate["mean"],
                safe_relative_effect,
                target_relative_effect,
            )
        )
    return sorted(editors, key=lambda row: (-row.score, row.site_id))


def editor_payload(editor: TrajectoryEditor) -> dict[str, Any]:
    return {
        "site_id": editor.site_id,
        "family": editor.family,
        "layer": editor.layer,
        "score": editor.score,
        "fold_cosine_minimum": editor.fold_cosine_minimum,
        "target_projection": editor.target_projection,
        "safe_gate_rms": editor.safe_gate_rms,
        "target_gate_mean": editor.target_gate_mean,
        "safe_relative_effect": editor.safe_relative_effect,
        "target_relative_effect": editor.target_relative_effect,
    }


@torch.no_grad()
def apply_candidate(
    model: Any,
    editors: list[TrajectoryEditor],
    originals: dict[str, torch.Tensor],
    *,
    k: int,
    beta: float,
) -> list[str]:
    selected = editors[:k]
    for editor in selected:
        module = model.get_submodule(editor.module_path)
        original = originals[editor.site_id].to(module.weight.device).float()
        axis = editor.axis.to(module.weight.device)
        detector = editor.detector.to(module.weight.device)
        delta = (
            beta
            * editor.target_projection
            * axis[:, None]
            @ detector[None, :]
        )
        module.weight.copy_((original - delta).to(module.weight.dtype))
    return [row.site_id for row in selected]


def trial_key(window: str, k: int, beta: float) -> tuple[str, int, int]:
    return window, int(k), round(float(beta) * 1000)


def load_state() -> dict[str, Any]:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"refusal_trials": [], "kl_trials": []}


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(4051)
    device = base.select_device("auto")
    tokenizer, model = base.load_model(device)
    prompts = base.load_prompts()
    sites = [
        site
        for site in discover_semantic_sites(model).sites
        if site.family in {"gqa", "ffn"}
    ]
    if len(sites) != 70:
        raise RuntimeError(f"expected 70 semantic sites, got {len(sites)}")

    paths = load_or_generate_paths(
        model,
        tokenizer,
        prompts["target_train"][:TRAIN_COUNT],
        sites,
        device,
    )
    trajectories, pair_indices = load_or_collect_trajectories(
        model, paths, sites, device
    )
    safe_inputs = collect_safe_inputs(
        model, tokenizer, prompts["safe_train"], sites, device
    )
    cached_outputs = load_file(local.ACTIVATION_CACHE)
    safe_outputs = {
        site.id: cached_outputs[f"safe.{site.id}"] for site in sites
    }
    rankings = {
        window: build_editors(
            sites,
            trajectories,
            safe_inputs,
            safe_outputs,
            window=window,
        )
        for window in ("early", "late", "all")
    }
    print(
        json.dumps(
            {
                "trajectory_pairs": len(pair_indices),
                "pair_indices": pair_indices,
                "ranking_heads": {
                    window: [editor_payload(row) for row in editors[:12]]
                    for window, editors in rankings.items()
                },
            },
            indent=2,
        ),
        flush=True,
    )

    maximum_k = max(k for _window, k, _beta in TRIALS)
    unique_editors = {
        editor.site_id: editor
        for editors in rankings.values()
        for editor in editors[:maximum_k]
    }
    candidate_editors = list(unique_editors.values())
    originals = snapshot_selected(model, candidate_editors)
    state = load_state()
    state["trajectory_pairs"] = len(pair_indices)
    state["ranking_heads"] = {
        window: [editor_payload(row) for row in editors[:16]]
        for window, editors in rankings.items()
    }
    completed = {
        trial_key(str(row["window"]), int(row["k"]), float(row["beta"]))
        for row in state["refusal_trials"]
    }
    for window, k, beta in TRIALS:
        key = trial_key(window, k, beta)
        if key in completed:
            continue
        restore_selected(model, candidate_editors, originals)
        selected = apply_candidate(
            model,
            rankings[window],
            originals,
            k=k,
            beta=beta,
        )
        evaluation = base.refusal_evaluation(
            model,
            tokenizer,
            prompts["target_test"][:PROBE_COUNT],
            device,
            label=f"trajectory-{window}:k{k}:b{beta:.2f}",
            max_new_tokens=RESPONSE_TOKENS,
        )
        row = {
            "window": window,
            "k": k,
            "beta": beta,
            "refusal_markers": evaluation["refusal_markers"],
            "marker_hits": evaluation["marker_hits"],
            "response_sha256": evaluation["response_sha256"],
            "selected_sites": selected,
        }
        state["refusal_trials"].append(row)
        completed.add(key)
        base.atomic_json(STATE_PATH, state)
        print(json.dumps({"trajectory_refusal": row}), flush=True)

    promising = sorted(
        (
            row
            for row in state["refusal_trials"]
            if int(row["refusal_markers"]) <= 2
        ),
        key=lambda row: (
            int(row["refusal_markers"]),
            int(row["k"]),
            float(row["beta"]),
        ),
    )
    completed_kl = {
        trial_key(str(row["window"]), int(row["k"]), float(row["beta"]))
        for row in state["kl_trials"]
    }
    baseline = load_file(base.BASE_LOG_PROBS_CACHE)["log_probs"].float()
    for source in promising:
        key = trial_key(
            str(source["window"]), int(source["k"]), float(source["beta"])
        )
        if key in completed_kl:
            continue
        window = str(source["window"])
        restore_selected(model, candidate_editors, originals)
        apply_candidate(
            model,
            rankings[window],
            originals,
            k=int(source["k"]),
            beta=float(source["beta"]),
        )
        candidate = base.next_token_log_probs(
            model,
            tokenizer,
            prompts["safe_test"],
            device,
            label=(
                f"trajectory-kl-{window}:k{source['k']}:b{source['beta']:.2f}"
            ),
        )
        row = {
            **source,
            "first_token_kl": base.mean_first_token_kl(baseline, candidate),
        }
        state["kl_trials"].append(row)
        completed_kl.add(key)
        base.atomic_json(STATE_PATH, state)
        print(json.dumps({"trajectory_kl": row}), flush=True)

    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
