import numpy as np
import pytest

from heretic_nx.eval.capability import paired_bootstrap_interval
from heretic_nx.eval.comparison import validate_capability_pair
from heretic_nx.hashing import sha256_json
from experiments.gemma4_e4b_q8_restore_norms import restore, row_norms
from experiments.gemma4_e4b_q8_screen_candidate import refusal_passed


def blocks(scale):
    value = np.zeros((4, 3, 34), dtype=np.uint8)
    value[..., :2] = np.full((4, 3), scale, dtype='<f2').view(np.uint8).reshape(4, 3, 2)
    value[..., 2:] = np.arange(-16, 16, dtype=np.int8).view(np.uint8)
    return value


def test_restore_norms_preserves_codes_and_matches_exact_power_of_two():
    base, edited = blocks(.125), blocks(.25)
    codes = edited[..., 2:].copy()
    restore(base, edited, 1)
    assert np.array_equal(codes, edited[..., 2:])
    assert np.array_equal(base, edited)


def test_zero_restoration_is_identity():
    base, edited = blocks(.125), blocks(.3)
    before = edited.copy()
    restore(base, edited, 0)
    assert np.array_equal(before, edited)


def test_partial_restoration_is_measured_after_scale_rounding():
    base, edited = blocks(.125), blocks(.3)
    expected = (row_norms(base) + row_norms(edited)) / 2
    restore(base, edited, .5)
    np.testing.assert_allclose(row_norms(edited), expected, rtol=5e-4)


def test_zero_row_cannot_restore_nonzero_norm():
    with pytest.raises(ValueError, match='zero row'):
        restore(blocks(.125), blocks(0), 1)


def passing_report():
    return {'complete': True, 'count': 104, 'row_indices': list(range(104)),
            'marker_hits': [1]*6 + [0]*98, 'refusal_markers': 6}


def test_full_gate_accepts_reordered_complete_set():
    report = passing_report()
    report['row_indices'].reverse()
    assert refusal_passed(report)


@pytest.mark.parametrize('change', [
    {'complete': False}, {'count': 16}, {'row_indices': [0]*104},
    {'marker_hits': [0]*104}, {'refusal_markers': 7},
])
def test_partial_or_inconsistent_evidence_cannot_start_kl(change):
    assert not refusal_passed({**passing_report(), **change})


def test_no_discordances_does_not_imply_zero_uncertainty():
    interval = paired_bootstrap_interval([1]*20, [1]*20, margin=.001)
    assert interval.lower < 0 < interval.upper
    assert not interval.noninferiority_passed
    assert interval.method == 'paired-binomial-bonferroni'


def test_capability_pair_recomputes_correctness_and_checks_runtime():
    rows = [{'task': 'tiny', 'answer': 0}]
    arm = {'schema_version': 'test', 'protocol': {'mode': 'restricted'},
           'datasets': {'rows_sha256': sha256_json(rows), 'rows': 1},
           'results': {'count': 1, 'predictions': [0], 'correctness': [1], 'accuracy': 1.,
                       'tasks': {'tiny': {'count': 1, 'accuracy': 1.}}},
           'artifact': {'sha256': 'a'*64},
           'runtime': {'artifact_sha256': 'a'*64, 'build_info': 'build', 'model_ftype': 'Q8'}}
    validate_capability_pair(arm, arm, rows)
    wrong = {**arm, 'results': {**arm['results'], 'correctness': [0]}}
    with pytest.raises(ValueError, match='correctness'):
        validate_capability_pair(arm, wrong, rows)
    wrong = {**arm, 'runtime': {**arm['runtime'], 'build_info': 'different'}}
    with pytest.raises(ValueError, match='build_info'):
        validate_capability_pair(arm, wrong, rows)
