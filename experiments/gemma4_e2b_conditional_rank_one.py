#!/usr/bin/env python3
"""Conditional rank-one Gemma edit keyed by harmful-specific FFN inputs."""

from __future__ import annotations

import json
import os

from safetensors.torch import load_file, save_file
import torch

import gemma4_e2b_residual_stream_prime as base
from heretic_nx.geometry.contrastive import fit_contrastive_axis


SITE_ID = "L23:ffn_out"
MODULE_PATH = "model.layers.23.mlp.down_proj"
INPUT_CACHE = base.RUN_DIR / "l23-ffn-inputs.safetensors"
STATE_PATH = base.RUN_DIR / "conditional-rank-one-state.json"
PROBE_COUNT = 20
PROBE_TOKENS = 16
TRIALS = (2.00, 1.75, 2.25, 1.50, 2.50, 1.25, 3.00)


@torch.inference_mode()
def collect_inputs(model, tokenizer, prompts, device, *, label):
    storage = []
    armed = {"value": False}
    module = model.get_submodule(MODULE_PATH)

    def capture(_module, inputs):
        if armed["value"]:
            storage.append(
                inputs[0][:, -1].detach().to(dtype=torch.bfloat16, device="cpu")
            )

    handle = module.register_forward_pre_hook(capture)
    rendered = base.render(tokenizer, prompts)
    try:
        for start in range(0, len(rendered), 4):
            batch = base.tokenize(tokenizer, rendered[start : start + 4], device)
            armed["value"] = True
            model.model(**batch, use_cache=False, return_dict=True)
            armed["value"] = False
            completed = min(start + 4, len(rendered))
            if completed % 32 == 0 or completed == len(rendered):
                print(
                    json.dumps({"collect_inputs": label, "completed": completed}),
                    flush=True,
                )
    finally:
        armed["value"] = False
        handle.remove()
    base.empty_device_cache(device)
    return torch.cat(storage)


def minimum_safe_energy_detector(safe, target):
    """Minimize E_safe[(x @ u)^2] with target mean 1 and safe mean 0."""
    safe = safe.float()
    target = target.float()
    count = safe.shape[0]
    mean_safe = safe.mean(0)
    mean_target = target.mean(0)
    constraints = torch.stack((mean_target, mean_safe), dim=1)
    mean_second_moment = safe.square().mean().clamp_min(1e-8)
    ridge = float(mean_second_moment * 1e-3)
    sample_gram = safe @ safe.T
    system = sample_gram + (count * ridge) * torch.eye(count)
    sample_rhs = safe @ constraints
    solved = torch.linalg.solve(system, sample_rhs)
    inverse_constraints = (constraints - safe.T @ solved) / ridge
    gram = constraints.T @ inverse_constraints
    desired = torch.tensor([1.0, 0.0])
    detector = inverse_constraints @ torch.linalg.solve(gram, desired)
    return detector, ridge


def gate_stats(values, detector):
    scores = values.float() @ detector
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "rms": float(scores.square().mean().sqrt()),
        "minimum": float(scores.min()),
        "maximum": float(scores.max()),
    }


@torch.no_grad()
def apply_candidate(module, original, axis, detector, target_projection, beta):
    weight = original.to(module.weight.device).float()
    a = axis.to(weight.device)
    u = detector.to(weight.device)
    edited = weight - beta * target_projection * a[:, None] @ u[None, :]
    module.weight.copy_(edited.to(module.weight.dtype))


def load_state():
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"refusal_trials": [], "kl_trials": []}


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(4047)
    device = base.select_device("auto")
    tokenizer, model = base.load_model(device)
    prompts = base.load_prompts()
    if INPUT_CACHE.is_file():
        cached_inputs = load_file(INPUT_CACHE)
        safe_inputs = cached_inputs["safe"]
        target_inputs = cached_inputs["target"]
        print(json.dumps({"reuse": str(INPUT_CACHE)}), flush=True)
    else:
        safe_inputs = collect_inputs(
            model, tokenizer, prompts["safe_train"], device, label="safe"
        )
        target_inputs = collect_inputs(
            model, tokenizer, prompts["target_train"], device, label="target"
        )
        save_file(
            {"safe": safe_inputs, "target": target_inputs}, INPUT_CACHE
        )

    outputs = load_file(
        base.RUN_DIR / "local-site-outputs.safetensors"
    )
    safe_outputs = outputs[f"safe.{SITE_ID}"].float()
    target_outputs = outputs[f"target.{SITE_ID}"].float()
    evidence = fit_contrastive_axis(
        safe_outputs,
        target_outputs,
        folds=base.FOLDS,
        remove_safe_mean=False,
    )
    axis = evidence.axis.float()
    target_projection = float((target_outputs @ axis).mean())
    detector, ridge = minimum_safe_energy_detector(safe_inputs, target_inputs)
    diagnostics = {
        "ridge": ridge,
        "target_projection": target_projection,
        "safe_gate": gate_stats(safe_inputs, detector),
        "target_gate": gate_stats(target_inputs, detector),
        "fold_cosine_minimum": evidence.fold_cosine_minimum,
    }
    print(json.dumps({"conditional_detector": diagnostics}, indent=2), flush=True)

    module = model.get_submodule(MODULE_PATH)
    original = module.weight.detach().cpu().clone()
    state = load_state()
    state["diagnostics"] = diagnostics
    completed = {
        round(float(row["beta"]) * 1000) for row in state["refusal_trials"]
    }
    for beta in TRIALS:
        key = round(beta * 1000)
        if key in completed:
            continue
        apply_candidate(
            module, original, axis, detector, target_projection, beta
        )
        evaluation = base.refusal_evaluation(
            model,
            tokenizer,
            prompts["target_test"][:PROBE_COUNT],
            device,
            label=f"conditional-r1:b{beta:.2f}",
            max_new_tokens=PROBE_TOKENS,
        )
        row = {
            "beta": beta,
            "refusal_markers": evaluation["refusal_markers"],
            "marker_hits": evaluation["marker_hits"],
            "response_sha256": evaluation["response_sha256"],
        }
        state["refusal_trials"].append(row)
        completed.add(key)
        base.atomic_json(STATE_PATH, state)
        print(json.dumps({"conditional_refusal": row}), flush=True)
        if int(row["refusal_markers"]) <= 1:
            break

    promising = sorted(
        state["refusal_trials"],
        key=lambda row: (int(row["refusal_markers"]), float(row["beta"])),
    )[:3]
    baseline = load_file(base.BASE_LOG_PROBS_CACHE)["log_probs"].float()
    completed_kl = {
        round(float(row["beta"]) * 1000) for row in state["kl_trials"]
    }
    for source in promising:
        key = round(float(source["beta"]) * 1000)
        if key in completed_kl:
            continue
        apply_candidate(
            module,
            original,
            axis,
            detector,
            target_projection,
            float(source["beta"]),
        )
        candidate = base.next_token_log_probs(
            model,
            tokenizer,
            prompts["safe_test"],
            device,
            label=f"conditional-r1-kl:b{source['beta']:.2f}",
        )
        row = {
            **source,
            "first_token_kl": base.mean_first_token_kl(baseline, candidate),
        }
        state["kl_trials"].append(row)
        completed_kl.add(key)
        base.atomic_json(STATE_PATH, state)
        print(json.dumps({"conditional_kl": row}), flush=True)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
