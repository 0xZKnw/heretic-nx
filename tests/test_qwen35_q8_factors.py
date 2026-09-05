import pytest
import torch
from experiments.qwen35_9b_q8_factors import fit_detector


@pytest.mark.parametrize("weight", [0, 1, 10, 100])
def test_dual_detector_matches_feature_space_reference(weight):
    g = torch.Generator().manual_seed(42)
    t, s, v = torch.randn(12, 7, generator=g), torch.randn(10, 7, generator=g), torch.randn(7, generator=g)
    actual, report = fit_detector(t, s, v, weight)
    t, s, v = t.double(), s.double(), v.double()
    matrix = t.T @ t / len(t) + weight * s.T @ s / len(s)
    matrix.diagonal().add_(report["ridge"])
    expected = torch.linalg.solve(matrix, t.T @ (t @ v) / len(t))
    torch.testing.assert_close(actual.double(), expected, atol=1e-6, rtol=1e-5)
    assert report["converged"] and report["residual_ratio"] <= 1e-5


def test_detector_rejects_nonfinite_and_preserves_zero():
    t, s, v = torch.ones(8, 12), torch.zeros(8, 12), torch.zeros(12)
    actual, _ = fit_detector(t, s, v, 10)
    assert torch.equal(actual, v)
    with pytest.raises(ValueError):
        fit_detector(t, s, v, float("nan"))
