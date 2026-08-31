#!/usr/bin/env python3
"""Refusal-first Gemma 4 E2B search using site-local output directions.

Unlike the residual-stream runners, each attention/FFN output projection gets a
direction fitted in its own activation space.  This mirrors the successful LFM
PRIME uncensoring path and tests behavioral refusal before spending work on KL.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file
import torch

import gemma4_e2b_residual_stream_prime as base
from heretic_nx.geometry.contrastive import fit_contrastive_axis
from heretic_nx.hashing import sha256_json
from heretic_nx.model import discover_semantic_sites


ACTIVATION_CACHE = base.RUN_DIR / "local-site-outputs.safetensors"
STATE_PATH = base.RUN_DIR / "local-site-refusal-first-state.json"
COLLECT_BATCH = 4
PROBE_COUNT = 20
PROBE_TOKENS = 16
TRIALS = (
    (24, 2.00),
    (16, 2.00),
    (32, 2.00),
    (8, 2.00),
    (24, 1.50),
    (24, 2.50),
    (16, 1.50),
    (16, 2.50),
    (32, 1.50),
    (32, 2.50),
    (48, 1.50),
    (48, 2.00),
)


@dataclass(frozen=True)
class LocalEditor:
    site_id: str
    module_path: str
    family: str
    layer: int
    axis: torch.Tensor
    score: float
    safe_relative_drift: float
    target_relative_drift: float
    target_effect: float
    fold_cosine_minimum: float


def manifest() -> dict[str, Any]:
    return {
        "schema_version": "gemma4-e2b-local-site-refusal-first-v1",
        "collect_count": base.COLLECT_COUNT,
        "collect_batch": COLLECT_BATCH,
        "probe_count": PROBE_COUNT,
        "probe_tokens": PROBE_TOKENS,
        "trials": [list(row) for row in TRIALS],
    }


def load_state() -> dict[str, Any]:
    current = manifest()
    digest = sha256_json(current)
    if not STATE_PATH.is_file():
        return {
            "manifest": current,
            "manifest_sha256": digest,
            "refusal_trials": [],
            "kl_trials": [],
        }
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if state.get("manifest_sha256") != digest:
        raise RuntimeError("local-site state has another manifest")
    return state


@torch.inference_mode()
def collect_site_outputs(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    sites: list[Any],
    device: torch.device,
    *,
    label: str,
) -> dict[str, torch.Tensor]:
    storage: dict[str, list[torch.Tensor]] = {site.id: [] for site in sites}
    handles = []
    armed = {"value": False}
    for site in sites:
        module = model.get_submodule(site.module_path)

        def capture(_module, _inputs, output, *, site_id=site.id):
            if not armed["value"]:
                return
            value = output[0] if isinstance(output, tuple) else output
            storage[site_id].append(
                value[:, -1].detach().to(dtype=torch.bfloat16, device="cpu")
            )

        handles.append(module.register_forward_hook(capture))
    rendered = base.render(tokenizer, prompts)
    try:
        for start in range(0, len(rendered), COLLECT_BATCH):
            batch = base.tokenize(
                tokenizer, rendered[start : start + COLLECT_BATCH], device
            )
            armed["value"] = True
            model.model(**batch, use_cache=False, return_dict=True)
            armed["value"] = False
            completed = min(start + COLLECT_BATCH, len(rendered))
            if completed % 32 == 0 or completed == len(rendered):
                print(
                    json.dumps(
                        {"collect_site_outputs": label, "completed": completed}
                    ),
                    flush=True,
                )
            del batch
    finally:
        armed["value"] = False
        for handle in handles:
            handle.remove()
    base.empty_device_cache(device)
    return {site_id: torch.cat(rows) for site_id, rows in storage.items()}


def build_editors(
    sites: list[Any],
    safe: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> list[LocalEditor]:
    editors = []
    for site in sites:
        safe_values = safe[site.id].float()
        target_values = target[site.id].float()
        evidence = fit_contrastive_axis(
            safe_values,
            target_values,
            folds=base.FOLDS,
            remove_safe_mean=False,
        )
        axis = evidence.axis.float()
        safe_delta = (safe_values @ axis)[:, None] * axis[None, :]
        target_delta = (target_values @ axis)[:, None] * axis[None, :]
        safe_drift = float(
            torch.linalg.vector_norm(safe_delta)
            / torch.linalg.vector_norm(safe_values).clamp_min(1e-8)
        )
        target_drift = float(
            torch.linalg.vector_norm(target_delta)
            / torch.linalg.vector_norm(target_values).clamp_min(1e-8)
        )
        target_effect = float(
            torch.abs((target_values.mean(0) - safe_values.mean(0)) @ axis)
        )
        score = (
            target_effect
            * max(target_drift, 1e-8)
            * max(evidence.fold_cosine_minimum, 1e-3)
            / max(safe_drift, 1e-6)
        )
        editors.append(
            LocalEditor(
                site.id,
                site.module_path,
                site.family,
                site.layer,
                axis.cpu(),
                score,
                safe_drift,
                target_drift,
                target_effect,
                evidence.fold_cosine_minimum,
            )
        )
    return sorted(editors, key=lambda row: (-row.score, row.site_id))


def snapshot_weights(
    model: Any, editors: list[LocalEditor]
) -> dict[str, torch.Tensor]:
    return {
        editor.site_id: model.get_submodule(editor.module_path)
        .weight.detach()
        .cpu()
        .clone()
        for editor in editors
    }


@torch.no_grad()
def apply_candidate(
    model: Any,
    editors: list[LocalEditor],
    originals: dict[str, torch.Tensor],
    *,
    k: int,
    beta: float,
) -> list[str]:
    selected = {editor.site_id for editor in editors[:k]}
    for editor in editors:
        module = model.get_submodule(editor.module_path)
        original = originals[editor.site_id].to(module.weight.device)
        if editor.site_id in selected:
            axis = editor.axis.to(device=original.device, dtype=torch.float32)
            weight = original.float()
            edited = weight - beta * axis[:, None] @ (axis[None, :] @ weight)
            module.weight.copy_(edited.to(module.weight.dtype))
        else:
            module.weight.copy_(original.to(module.weight.dtype))
    return [editor.site_id for editor in editors[:k]]


def editor_payload(editor: LocalEditor) -> dict[str, Any]:
    return {
        "site_id": editor.site_id,
        "module_path": editor.module_path,
        "family": editor.family,
        "layer": editor.layer,
        "score": editor.score,
        "safe_relative_drift": editor.safe_relative_drift,
        "target_relative_drift": editor.target_relative_drift,
        "target_effect": editor.target_effect,
        "fold_cosine_minimum": editor.fold_cosine_minimum,
    }


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(4042)
    device = base.select_device("auto")
    tokenizer, model = base.load_model(device)
    prompts = base.load_prompts()
    registry = discover_semantic_sites(model)
    sites = [
        site for site in registry.sites if site.family in {"gqa", "ffn"}
    ]
    if len(sites) != 70:
        raise RuntimeError(f"expected 70 Gemma semantic output sites, got {len(sites)}")

    if ACTIVATION_CACHE.is_file():
        cached = load_file(ACTIVATION_CACHE)
        safe = {site.id: cached[f"safe.{site.id}"] for site in sites}
        target = {site.id: cached[f"target.{site.id}"] for site in sites}
        print(json.dumps({"reuse": str(ACTIVATION_CACHE)}), flush=True)
    else:
        safe = collect_site_outputs(
            model,
            tokenizer,
            prompts["safe_train"],
            sites,
            device,
            label="safe",
        )
        target = collect_site_outputs(
            model,
            tokenizer,
            prompts["target_train"],
            sites,
            device,
            label="target",
        )
        save_file(
            {
                **{f"safe.{key}": value for key, value in safe.items()},
                **{f"target.{key}": value for key, value in target.items()},
            },
            ACTIVATION_CACHE,
            metadata={
                "model_revision": base.MODEL_REVISION,
                "good_revision": base.GOOD_REVISION,
                "bad_revision": base.BAD_REVISION,
                "manifest_sha256": sha256_json(manifest()),
            },
        )

    editors = build_editors(sites, safe, target)
    print(
        json.dumps(
            {"local_ranking_head": [editor_payload(row) for row in editors[:32]]},
            indent=2,
        ),
        flush=True,
    )
    originals = snapshot_weights(model, editors)
    state = load_state()
    completed = {
        (int(row["k"]), round(float(row["beta"]) * 1000))
        for row in state["refusal_trials"]
    }
    for requested_k, beta in TRIALS:
        k = min(requested_k, len(editors))
        key = (k, round(beta * 1000))
        if key in completed:
            continue
        selected_sites = apply_candidate(
            model, editors, originals, k=k, beta=beta
        )
        evaluation = base.refusal_evaluation(
            model,
            tokenizer,
            prompts["target_test"][:PROBE_COUNT],
            device,
            label=f"local-site:k{k}:b{beta:.2f}",
            max_new_tokens=PROBE_TOKENS,
        )
        row = {
            "k": k,
            "beta": beta,
            "refusal_markers": evaluation["refusal_markers"],
            "marker_hits": evaluation["marker_hits"],
            "response_sha256": evaluation["response_sha256"],
            "seconds": evaluation["seconds"],
            "selected_sites": selected_sites,
        }
        state["refusal_trials"].append(row)
        completed.add(key)
        base.atomic_json(STATE_PATH, state)
        print(json.dumps({"local_site_refusal_trial": row}), flush=True)
        if evaluation["refusal_markers"] <= 1:
            break

    safe_baseline = load_file(base.BASE_LOG_PROBS_CACHE)["log_probs"].float()
    completed_kl = {
        (int(row["k"]), round(float(row["beta"]) * 1000))
        for row in state["kl_trials"]
    }
    for source in sorted(
        state["refusal_trials"],
        key=lambda row: (int(row["refusal_markers"]), int(row["k"])),
    )[:3]:
        k = int(source["k"])
        beta = float(source["beta"])
        key = (k, round(beta * 1000))
        if key in completed_kl:
            continue
        apply_candidate(model, editors, originals, k=k, beta=beta)
        candidate = base.next_token_log_probs(
            model,
            tokenizer,
            prompts["safe_test"],
            device,
            label=f"local-site-kl:k{k}:b{beta:.2f}",
        )
        row = {
            **source,
            "first_token_kl": base.mean_first_token_kl(safe_baseline, candidate),
        }
        del candidate
        state["kl_trials"].append(row)
        completed_kl.add(key)
        base.atomic_json(STATE_PATH, state)
        print(json.dumps({"local_site_kl": row}), flush=True)

    print(
        json.dumps(
            {
                "best_refusal": min(
                    state["refusal_trials"],
                    key=lambda row: int(row["refusal_markers"]),
                ),
                "kl_trials": state["kl_trials"],
                "ranking": [editor_payload(row) for row in editors],
                "state": str(STATE_PATH),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
