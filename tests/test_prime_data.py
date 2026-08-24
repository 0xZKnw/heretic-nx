from __future__ import annotations

import torch

from heretic_nx.data.boundary_mining import delta_debug_benign_refusal
from heretic_nx.data.oracles import OracleVerdict, benign_consensus
from heretic_nx.data.pairs import BenignPromptPair, paired_differences
from heretic_nx.data.splits import SplitAssignment, assign_split, validate_no_leakage


def test_benign_pair_requires_multiple_confident_oracles() -> None:
    good = (OracleVerdict("o1", True, 0.9), OracleVerdict("o2", True, 0.95))
    assert benign_consensus(good)
    pair = BenignPromptPair("p", "code", "kill a process", "terminate a process", "synonym", good, good)
    assert pair.validated()
    ambiguous = (OracleVerdict("o1", True, 0.9), OracleVerdict("o2", None, 0.9))
    assert not benign_consensus(ambiguous)


def test_paired_differences_preserve_pairing() -> None:
    anchor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    contrast = torch.tensor([[1.5, 1.0], [4.0, 6.0]])
    torch.testing.assert_close(paired_differences(anchor, contrast), torch.tensor([[0.5, -1.0], [1.0, 2.0]]))


def test_delta_debugging_never_accepts_unsafe_candidate() -> None:
    prompt = "please explain how to kill a python process safely"
    result = delta_debug_benign_refusal(
        prompt,
        refusal_predicate=lambda value: "kill" in value and "python" in value,
        benign_oracle=lambda value: "python" in value,
        minimum_tokens=2,
    )
    assert "kill" in result.minimized and "python" in result.minimized
    assert len(result.minimized.split()) < len(prompt.split())


def test_split_assignment_is_deterministic_and_leakage_fails() -> None:
    assert assign_split("x", seed=17) == assign_split("x", seed=17)
    assert assign_split("h", seed=17, risk_canary=True).split == "secret-h"
    try:
        validate_no_leakage(
            [SplitAssignment("x", "train-geometry", "a"), SplitAssignment("x", "secret-b", "b")]
        )
    except ValueError:
        pass
    else:
        raise AssertionError("cross-split duplication must fail")
