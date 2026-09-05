import pytest

from experiments.gemma4_e4b_q8_release import release_kind, public_paths, ROOT


def test_failed_capability_is_blocked_by_default():
    with pytest.raises(ValueError, match='explicit research'):
        release_kind(False, None)


def test_authorized_research_does_not_turn_failure_into_prime_success():
    assert release_kind(False, 'Owner approved the disclosed capability tradeoff.') == 'research_with_disclosed_tradeoffs'
    assert release_kind(True, None) == 'prime_gates_passed'


def test_blank_decision_is_rejected():
    with pytest.raises(ValueError, match='explicit decision'):
        release_kind(False, '  ')


def test_public_paths_remove_workspace_prefix_recursively():
    assert public_paths({'paths': [str(ROOT / 'outputs/model.gguf')]}) == {'paths': ['outputs/model.gguf']}
