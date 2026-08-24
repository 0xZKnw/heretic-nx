from __future__ import annotations

from heretic_nx.hashing import sha256_directory, sha256_file
from heretic_nx.provenance import PrimeProvenanceManifest, ProvenanceNode


def test_prime_provenance_is_a_verified_rooted_dag(tmp_path) -> None:
    dependencies = {
        "base-model": (),
        "tokenizer": (),
        "chat-template": (),
        "config": (),
        "semantic-registry": ("base-model", "config"),
        "split-manifest": (),
        "engine-source": (),
        "output-model": ("base-model", "engine-source", "semantic-registry"),
        "response-artifact": (
            "output-model",
            "tokenizer",
            "chat-template",
            "split-manifest",
        ),
        "judge-evidence": ("response-artifact",),
        "capability-report": ("response-artifact", "config"),
        "promotion-report": ("judge-evidence", "capability-report"),
    }
    paths = {}
    nodes = []
    for name, parents in dependencies.items():
        if name == "engine-source":
            path = tmp_path / name
            path.mkdir()
            (path / "engine.py").write_text("version = 3", encoding="utf-8")
            digest = sha256_directory(path)
            kind = "directory"
        else:
            path = tmp_path / f"{name}.bin"
            path.write_bytes(name.encode())
            digest = sha256_file(path)
            kind = "file"
        paths[name] = path
        nodes.append(ProvenanceNode(name=name, kind=kind, sha256=digest, depends_on=parents))
    manifest = PrimeProvenanceManifest(run_id="run-1", nodes=tuple(nodes))
    manifest.verify_paths(paths)
    path = tmp_path / "provenance.json"
    manifest.write(path)
    assert PrimeProvenanceManifest.read(path) == manifest
    paths["capability-report"].write_bytes(b"tampered")
    try:
        manifest.verify_paths(paths)
    except RuntimeError as error:
        assert "capability-report" in str(error)
    else:
        raise AssertionError("tampered provenance must fail closed")


def test_prime_provenance_rejects_unrooted_nodes(tmp_path) -> None:
    names = (
        "base-model",
        "tokenizer",
        "chat-template",
        "config",
        "semantic-registry",
        "split-manifest",
        "engine-source",
        "response-artifact",
        "judge-evidence",
        "capability-report",
        "output-model",
        "promotion-report",
    )
    nodes = tuple(
        ProvenanceNode(
            name=name,
            kind="file",
            sha256="a" * 64,
            depends_on=() if name != "promotion-report" else ("capability-report",),
        )
        for name in names
    )
    try:
        PrimeProvenanceManifest(run_id="bad", nodes=nodes)
    except ValueError as error:
        assert "not rooted" in str(error)
    else:
        raise AssertionError("unrooted evidence must not form a PRIME provenance chain")
