from __future__ import annotations

import json
from pathlib import Path

import pytest
from safetensors.torch import save_file
import torch

from experiments.lfm25_8b_a1b_q8_prime_build import (
    load_verified_prime_merge,
    load_verified_ranking,
)
from experiments import lfm25_8b_a1b_prime_sites as prime_sites
from experiments.lfm25_2p6b_residual_stream import (
    BAD_DATASET,
    BAD_REVISION,
    GOOD_DATASET,
    GOOD_REVISION,
)
from heretic_nx.data.research_splits import build_research_split
from heretic_nx.edits import GGUFQ8AblationPlan, GGUFQ8TensorEdit
from heretic_nx.hashing import (
    canonical_json,
    sha256_directory,
    sha256_file,
    sha256_json,
)


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json(value) + b"\n")


def _provenance_chain(tmp_path: Path) -> dict[str, Path]:
    source = tmp_path / "source.gguf"
    operators = tmp_path / "operators.safetensors"
    ranking_path = tmp_path / "ranking.json"
    plan_path = tmp_path / "teacher.plan.json"
    teacher_output = tmp_path / "teacher.gguf"
    merge_path = tmp_path / "teacher.merge.json"
    source.write_bytes(b"q8-source")
    teacher_output.write_bytes(b"q8-teacher")
    source_sha256 = sha256_file(source)
    activations_sha256 = sha256_json({"activations": 1})
    provenance = {
        "activation_manifest_sha256": sha256_json({"manifest": 1}),
        "safe_split_manifest": {"frozen": "safe"},
        "safe_split_manifest_sha256": sha256_json({"split": "safe"}),
        "target_split_manifest": {"frozen": "target"},
        "target_split_manifest_sha256": sha256_json({"split": "target"}),
        "model": str(tmp_path / "mlx-model"),
        "model_sha256": sha256_json({"model": 1}),
        "source": str(source.resolve()),
        "source_sha256": source_sha256,
        "sites_sha256": sha256_json(["L00:attention_out"]),
    }
    save_file(
        {
            "site00.a": torch.ones((4, 1)),
            "site00.b": torch.full((4, 1), 0.5),
        },
        operators,
        metadata={
            "activation_manifest_sha256": provenance[
                "activation_manifest_sha256"
            ],
            "safe_split_manifest_sha256": provenance[
                "safe_split_manifest_sha256"
            ],
            "target_split_manifest_sha256": provenance[
                "target_split_manifest_sha256"
            ],
            "source_sha256": source_sha256,
            "sites_sha256": provenance["sites_sha256"],
            "activations_sha256": activations_sha256,
        },
    )
    accepted = {
        "site_id": "L00:attention_out",
        "layer": 0,
        "family": "gqa",
        "factor_a_key": "site00.a",
        "factor_b_key": "site00.b",
        "rank": 1,
        "score": 1.0,
        "safe_proxy_drift": 0.01,
    }
    ranking = {
        "schema_version": "lfm25-8b-a1b-prime-sites-v2",
        "activations": {
            "path": str(tmp_path / "activations.safetensors"),
            "sha256": activations_sha256,
            "manifest_sha256": provenance["activation_manifest_sha256"],
            "shape": [1, 1, 4],
        },
        "geometry_provenance": provenance,
        "accepted": [accepted],
        "rejected": [],
        "operator_artifact": {
            "path": str(operators.resolve()),
            "sha256": sha256_file(operators),
        },
    }
    _write_json(ranking_path, ranking)
    selected = {**accepted, "tensor_name": "blk.0.attn_output.weight"}
    GGUFQ8AblationPlan(
        source_sha256=source_sha256,
        tensor_artifact_sha256=sha256_file(operators),
        edits=(
            GGUFQ8TensorEdit(
                tensor_name=selected["tensor_name"],
                a_key=selected["factor_a_key"],
                b_key=selected["factor_b_key"],
                strength=2.0,
                preserve_row_norms=False,
            ),
        ),
    ).write(plan_path)
    merge = {
        "schema_version": "gguf-quantized-static-merge-report-v3",
        "source": {
            "path": str(source.resolve()),
            "sha256": source_sha256,
        },
        "tensor_artifact": {
            "path": str(operators.resolve()),
            "sha256": sha256_file(operators),
        },
        "plan": {
            "path": str(plan_path.resolve()),
            "sha256": sha256_file(plan_path),
        },
        "output": {
            "path": str(teacher_output.resolve()),
            "sha256": sha256_file(teacher_output),
        },
        "candidate": {
            "beta": 2.0,
            "ranking_report": str(ranking_path.resolve()),
            "ranking_report_sha256": sha256_file(ranking_path),
            "operator_artifact_sha256": sha256_file(operators),
            "geometry_provenance_sha256": sha256_json(provenance),
            "selected": [selected],
            "selected_sites_sha256": sha256_json([selected]),
        },
    }
    _write_json(merge_path, merge)
    return {
        "source": source,
        "operators": operators,
        "ranking": ranking_path,
        "plan": plan_path,
        "teacher": teacher_output,
        "merge": merge_path,
    }


def _activation_chain(tmp_path: Path) -> dict[str, Path]:
    model = tmp_path / "mlx-model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    source = tmp_path / "source.gguf"
    source.write_bytes(b"q8-source")
    activations = tmp_path / "activations.safetensors"
    progress_path = tmp_path / "activations.progress.json"
    safe_pool = [f"safe row {index}" for index in range(256)]
    target_pool = [f"target row {index}" for index in range(256)]
    safe_manifest = build_research_split(
        safe_pool,
        purpose="geometry",
        dataset_id=GOOD_DATASET,
        revision=GOOD_REVISION,
        source_split="train",
        seed=20260830,
        count=2,
    )
    target_manifest = build_research_split(
        target_pool,
        purpose="geometry",
        dataset_id=BAD_DATASET,
        revision=BAD_REVISION,
        source_split="train",
        seed=20260830,
        count=2,
    )
    sites = [{"site_id": "L00:attention_out", "index": 0}]
    immutable = {
        "schema_version": "lfm25-8b-a1b-mlx-sites-progress-v2",
        "model": str(model.resolve()),
        "model_sha256": sha256_directory(model),
        "source": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "count": 2,
        "pool_count": 256,
        "split_seed": 20260830,
        "sites": sites,
        "sites_sha256": sha256_json(sites),
        "width": 4,
        "max_length": 16,
        "close_think": False,
        "safe_split_manifest": safe_manifest.to_dict(),
        "safe_split_manifest_sha256": safe_manifest.sha256,
        "target_split_manifest": target_manifest.to_dict(),
        "target_split_manifest_sha256": target_manifest.sha256,
        "safe_tokens_sha256": sha256_json([[1], [2]]),
        "target_tokens_sha256": sha256_json([[3], [4]]),
    }
    save_file(
        {
            "safe": torch.ones((2, 1, 4)),
            "target": torch.zeros((2, 1, 4)),
        },
        activations,
        metadata={
            "manifest_sha256": sha256_json(immutable),
            "sites_sha256": immutable["sites_sha256"],
            "safe_split_manifest_sha256": safe_manifest.sha256,
            "target_split_manifest_sha256": target_manifest.sha256,
            "model_sha256": immutable["model_sha256"],
            "source_sha256": immutable["source_sha256"],
        },
    )
    _write_json(
        progress_path,
        {
            **immutable,
            "label": "target",
            "completed": 2,
            "seconds": 1.0,
            "complete": True,
            "output_sha256": sha256_file(activations),
        },
    )
    return {
        "model": model,
        "source": source,
        "activations": activations,
        "progress": progress_path,
    }


def test_prime_provenance_accepts_one_fully_bound_chain(tmp_path: Path) -> None:
    paths = _provenance_chain(tmp_path)

    ranking = load_verified_ranking(
        source_path=paths["source"],
        operators_path=paths["operators"],
        ranking_path=paths["ranking"],
    )
    merge, merge_ranking = load_verified_prime_merge(
        paths["merge"],
        source_path=paths["source"],
        operators_path=paths["operators"],
        ranking_path=paths["ranking"],
    )

    assert ranking == merge_ranking
    assert merge["candidate"]["selected"][0]["site_id"] == "L00:attention_out"


def test_prime_provenance_rejects_v1_geometry(tmp_path: Path) -> None:
    paths = _provenance_chain(tmp_path)
    ranking = json.loads(paths["ranking"].read_text(encoding="utf-8"))
    ranking["schema_version"] = "lfm25-8b-a1b-prime-sites-v1"
    _write_json(paths["ranking"], ranking)

    with pytest.raises(RuntimeError, match="predates leak-safe"):
        load_verified_ranking(
            source_path=paths["source"],
            operators_path=paths["operators"],
            ranking_path=paths["ranking"],
        )


@pytest.mark.parametrize("target", ["source", "operators", "teacher"])
def test_prime_provenance_rejects_cross_wired_artifact(
    tmp_path: Path,
    target: str,
) -> None:
    paths = _provenance_chain(tmp_path)
    with paths[target].open("ab") as stream:
        stream.write(b"-changed")

    with pytest.raises(RuntimeError, match="hash|source|artifact"):
        load_verified_prime_merge(
            paths["merge"],
            source_path=paths["source"],
            operators_path=paths["operators"],
            ranking_path=paths["ranking"],
        )


def test_prime_provenance_rejects_rebound_selected_site(tmp_path: Path) -> None:
    paths = _provenance_chain(tmp_path)
    merge = json.loads(paths["merge"].read_text(encoding="utf-8"))
    merge["candidate"]["selected"][0]["factor_b_key"] = "site00.a"
    merge["candidate"]["selected_sites_sha256"] = sha256_json(
        merge["candidate"]["selected"]
    )
    _write_json(paths["merge"], merge)

    with pytest.raises(RuntimeError, match="site no longer matches"):
        load_verified_prime_merge(
            paths["merge"],
            source_path=paths["source"],
            operators_path=paths["operators"],
            ranking_path=paths["ranking"],
        )


def test_activation_provenance_rejects_v1_and_current_file_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _activation_chain(tmp_path)
    monkeypatch.setattr(prime_sites, "ACTIVATIONS", paths["activations"])
    monkeypatch.setattr(prime_sites, "PROGRESS", paths["progress"])

    progress, provenance = prime_sites.load_activation_provenance()
    assert progress["complete"] is True
    assert provenance["source_sha256"] == sha256_file(paths["source"])

    document = json.loads(paths["progress"].read_text(encoding="utf-8"))
    document["schema_version"] = "lfm25-8b-a1b-mlx-sites-progress-v1"
    _write_json(paths["progress"], document)
    with pytest.raises(RuntimeError, match="predate"):
        prime_sites.load_activation_provenance()

    document["schema_version"] = "lfm25-8b-a1b-mlx-sites-progress-v2"
    _write_json(paths["progress"], document)
    (paths["model"] / "weights.bin").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="model provenance"):
        prime_sites.load_activation_provenance()
