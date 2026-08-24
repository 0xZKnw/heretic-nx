from __future__ import annotations

from heretic_nx.edits.nx_ir2 import SemanticSiteRef
from heretic_nx.edits.nx_ir3 import (
    ArtifactReferenceIR,
    EditIR3,
    GateEvidenceIR,
    NXIR3,
)
from heretic_nx.hashing import sha256_file


def _site() -> SemanticSiteRef:
    return SemanticSiteRef(
        id="L02:attention_out",
        layer=2,
        family="gqa",
        kind="attention_out",
        module_path="model.layers.2.self_attn.out_proj",
        module_type="Linear",
        stream_dim=8,
        structure_hash="1" * 64,
    )


def test_nxir3_derives_claim_binds_edit_gates_and_verifies_all_files(tmp_path) -> None:
    names = (
        "base-model",
        "tokenizer",
        "chat-template",
        "config",
        "semantic-registry",
        "frozen-manifest",
        "tensors",
        "promotion-report",
    )
    files = {}
    artifacts = []
    for name in names:
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode())
        files[name] = path
        artifacts.append(ArtifactReferenceIR(name=name, sha256=sha256_file(path)))
    geometry_sha = "a" * 64
    causal_sha = "b" * 64
    evidence = (
        GateEvidenceIR(gate="geometry", status="pass", artifact_sha256=geometry_sha, reason="safe"),
        GateEvidenceIR(gate="causal", status="pass", artifact_sha256=causal_sha, reason="causal"),
        GateEvidenceIR(gate="behavior", status="pass", artifact_sha256="c" * 64, reason="held out"),
    )
    document = NXIR3(
        export_track="static-merge",
        base_model_id="example/model",
        base_model_revision="abc",
        artifacts=tuple(artifacts),
        split_manifest_sha256="d" * 64,
        evidence=evidence,
        claim="PRIME-candidate",
        edits=(
            EditIR3(
                id="edit-1",
                site=_site(),
                family="signed-spectral",
                rank=1,
                beta=0.5,
                tensor_keys=("edit-1.a", "edit-1.b"),
                geometry_evidence_sha256=geometry_sha,
                causal_evidence_sha256=causal_sha,
                merge_targets=("model.layers.2.self_attn.out_proj.weight",),
            ),
        ),
    )
    document.verify_files(files)
    path = tmp_path / "nx-ir3.json"
    document.write(path)
    assert NXIR3.read(path) == document
    files["tensors"].write_bytes(b"tampered")
    try:
        document.verify_files(files)
    except RuntimeError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("tampered NX-IR3 artifacts must fail closed")


def test_nxir3_rejects_self_asserted_prime_claim(tmp_path) -> None:
    artifacts = tuple(
        ArtifactReferenceIR(name=name, sha256="e" * 64)
        for name in (
            "base-model",
            "tokenizer",
            "chat-template",
            "config",
            "semantic-registry",
            "frozen-manifest",
            "tensors",
            "promotion-report",
        )
    )
    try:
        NXIR3(
            export_track="static-merge",
            base_model_id="example/model",
            base_model_revision="abc",
            artifacts=artifacts,
            split_manifest_sha256="f" * 64,
            evidence=(),
            claim="PRIME-validated",
        )
    except ValueError as error:
        assert "unsupported" in str(error)
    else:
        raise AssertionError("a PRIME claim without evidence must be rejected")
