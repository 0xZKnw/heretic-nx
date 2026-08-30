from __future__ import annotations

import torch

from heretic_nx.edits.activation_op import ActivationOperator
from heretic_nx.edits.matrix_opt import fit_low_rank_matrix_operator


def _legacy_sparse_apply(
    operator: ActivationOperator,
    hidden: torch.Tensor,
    gate: float | torch.Tensor,
) -> torch.Tensor:
    delta = (hidden @ operator.b.to(hidden)) @ operator.a.to(hidden).T
    mask = torch.zeros(operator.dimension, dtype=hidden.dtype, device=hidden.device)
    assert operator.sparse_index is not None
    mask[operator.sparse_index.to(hidden.device)] = 1
    gate_tensor = torch.as_tensor(gate, device=hidden.device, dtype=hidden.dtype)
    while gate_tensor.ndim < delta.ndim:
        gate_tensor = gate_tensor.unsqueeze(-1)
    return hidden - operator.beta * gate_tensor * delta * mask


def test_indexed_sparse_apply_matches_legacy_mask_and_gradients() -> None:
    generator = torch.Generator().manual_seed(307)
    a = torch.randn(17, 3, generator=generator, requires_grad=True)
    b = torch.randn(17, 3, generator=generator, requires_grad=True)
    hidden = torch.randn(2, 5, 17, generator=generator, requires_grad=True)
    gate = torch.tensor([0.25, 0.8])
    operator = ActivationOperator(
        a=a,
        b=b,
        beta=0.7,
        sparse_index=torch.tensor([13, 2, 7, 2]),
    )

    expected = _legacy_sparse_apply(operator, hidden, gate)
    actual = operator.apply(hidden, gate)
    # The narrower GEMM is algebraically identical; BLAS may choose a different
    # kernel and change the last float32 bit for selected coordinates.
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    assert operator.sparse_index is not None
    torch.testing.assert_close(operator.sparse_index, torch.tensor([2, 7, 13]))
    untouched = torch.ones(17, dtype=torch.bool)
    untouched[operator.sparse_index] = False
    torch.testing.assert_close(
        actual[..., untouched], hidden[..., untouched], atol=0, rtol=0
    )

    expected.square().sum().backward(retain_graph=True)
    expected_grads = (hidden.grad.clone(), a.grad.clone(), b.grad.clone())
    hidden.grad = None
    a.grad = None
    b.grad = None
    actual.square().sum().backward()
    for actual_grad, expected_grad in zip((hidden.grad, a.grad, b.grad), expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-5, rtol=2e-5)


def test_sparse_apply_dense_fallback_and_edge_cases_match_semantics() -> None:
    generator = torch.Generator().manual_seed(311)
    a = torch.randn(12, 2, generator=generator)
    b = torch.randn(12, 2, generator=generator)
    hidden = torch.randn(4, 12, generator=generator)

    dense_selection = ActivationOperator(a, b, 0.4, torch.arange(9))
    torch.testing.assert_close(
        dense_selection.apply(hidden),
        _legacy_sparse_apply(dense_selection, hidden, 1.0),
        atol=0,
        rtol=0,
    )
    full_selection = ActivationOperator(a, b, 0.4, torch.arange(12))
    dense_operator = ActivationOperator(a, b, 0.4)
    torch.testing.assert_close(
        full_selection.apply(hidden), dense_operator.apply(hidden), atol=0, rtol=0
    )
    empty_selection = ActivationOperator(a, b, 0.4, torch.empty(0, dtype=torch.long))
    empty_output = empty_selection.apply(hidden)
    torch.testing.assert_close(empty_output, hidden, atol=0, rtol=0)
    assert empty_output.data_ptr() != hidden.data_ptr()


def test_sparse_index_rejects_non_integral_coordinates() -> None:
    factors = torch.ones(4, 1)
    try:
        ActivationOperator(factors, factors, 0.5, torch.tensor([1.0]))
    except ValueError as error:
        assert "integer" in str(error)
    else:
        raise AssertionError("floating-point sparse coordinates must be rejected")


def test_metric_projector_reuses_the_metric_application() -> None:
    class CountingMetric:
        dimension = 8
        calls = 0

        def apply(self, value: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return torch.arange(1, 9, dtype=value.dtype)[:, None] * value

    from heretic_nx.edits.activation_op import metric_projector_operator

    generator = torch.Generator().manual_seed(313)
    metric = CountingMetric()
    operator = metric_projector_operator(
        torch.randn(8, 3, generator=generator), metric, beta=0.5
    )
    assert metric.calls == 1
    assert torch.isfinite(operator.a).all()
    assert torch.isfinite(operator.b).all()


def _regression_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(19)
    protected = torch.randn(64, 32, generator=generator)
    target = torch.randn(64, 32, generator=generator)
    target[:, :3] += torch.tensor([3.0, 2.0, 1.0])
    return target, protected


def test_matrix_optimizer_restores_best_objective_instead_of_terminal_state() -> None:
    target, protected = _regression_fixture()
    short = fit_low_rank_matrix_operator(
        target,
        protected,
        rank=4,
        beta=0.8,
        steps=5,
        seed=7,
    )
    long = fit_low_rank_matrix_operator(
        target,
        protected,
        rank=4,
        beta=0.8,
        steps=200,
        seed=7,
    )
    assert long.final_loss <= short.final_loss + 1e-7
    assert long.final_loss < long.initial_loss
    assert long.steps == 200
    assert 0 < long.best_step < long.steps
    assert long.terminal_loss is not None
    assert long.terminal_loss > 3 * long.final_loss


def test_matrix_optimizer_opt_in_early_stopping_preserves_observed_best() -> None:
    target, protected = _regression_fixture()
    full = fit_low_rank_matrix_operator(
        target,
        protected,
        rank=4,
        beta=0.8,
        steps=200,
        seed=7,
    )
    stopped = fit_low_rank_matrix_operator(
        target,
        protected,
        rank=4,
        beta=0.8,
        steps=200,
        seed=7,
        early_stopping_patience=12,
    )
    assert stopped.steps < 40
    assert stopped.best_step == full.best_step
    torch.testing.assert_close(
        torch.tensor(stopped.final_loss), torch.tensor(full.final_loss), atol=0, rtol=0
    )


def test_matrix_optimizer_validates_early_stopping_controls() -> None:
    target, protected = _regression_fixture()
    for kwargs in (
        {"early_stopping_patience": 0},
        {"early_stopping_patience": True},
        {"minimum_delta": -1e-3},
        {"minimum_delta": float("nan")},
        {"beta": float("nan")},
    ):
        try:
            arguments = {"rank": 2, "beta": 0.5, "steps": 2}
            arguments.update(kwargs)
            fit_low_rank_matrix_operator(
                target,
                protected,
                **arguments,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid early-stopping controls must be rejected")
