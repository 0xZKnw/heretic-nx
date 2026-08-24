from __future__ import annotations

import torch
from torch import nn

from heretic_nx.optimize.attrscan import (
    attribution_reliability,
    gradient_attribution_scores,
    select_top_k,
)
from heretic_nx.optimize.hessian import estimate_reduced_hessian
from heretic_nx.optimize.qcqp import solve_qcqp
from heretic_nx.optimize.robust import cvar, enforce_scenario_constraints, smooth_max


def test_attrscan_matches_central_difference_and_topk() -> None:
    beta_a = torch.tensor(0.0, requires_grad=True)
    beta_b = torch.tensor(0.0, requires_grad=True)
    loss = 0.5 * beta_a**2 + 3 * beta_a + beta_b**2 - beta_b
    scores = gradient_attribution_scores(loss, {"a": beta_a, "b": beta_b})
    assert select_top_k(scores, 1)[0].site_id == "a"
    eps = 0.03
    minus = 0.5 * eps**2 - 3 * eps
    plus = 0.5 * eps**2 + 3 * eps
    diagnostic = attribution_reliability(3.0, minus, plus, eps)
    assert not diagnostic.flagged
    assert abs(diagnostic.central_difference - 3.0) < 1e-6


def test_attrscan_topk_recall_on_mini_transformer() -> None:
    """Gate G0: the cheap gradient scan must recover exact intervention sites."""

    class ResidualSite(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.attention = nn.MultiheadAttention(width, 2, batch_first=True, dropout=0.0)
            self.ffn = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
            self.strength = nn.Parameter(torch.zeros(()))

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            attended, _ = self.attention(hidden, hidden, hidden, need_weights=False)
            update = self.ffn(hidden + attended)
            return hidden + self.strength * update

    class MiniTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sites = nn.ModuleList([ResidualSite(8) for _ in range(5)])
            self.readout = nn.Linear(8, 1, bias=False)

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            for site in self.sites:
                hidden = site(hidden)
            return self.readout(hidden[:, -1]).square().mean()

    torch.manual_seed(139)
    model = MiniTransformer().double()
    hidden = torch.randn(3, 7, 8, dtype=torch.float64)
    parameters = {f"site-{index}": site.strength for index, site in enumerate(model.sites)}
    scores = gradient_attribution_scores(model(hidden), parameters)

    epsilon = 1e-5
    exact: dict[str, float] = {}
    for site_id, parameter in parameters.items():
        with torch.no_grad():
            parameter.fill_(-epsilon)
            minus = float(model(hidden))
            parameter.fill_(epsilon)
            plus = float(model(hidden))
            parameter.zero_()
        exact[site_id] = (plus - minus) / (2.0 * epsilon)

    exact_top2 = {
        site_id
        for site_id, _ in sorted(exact.items(), key=lambda item: -abs(item[1]))[:2]
    }
    scanned_top2 = {score.site_id for score in select_top_k(scores, 2)}
    assert scanned_top2 == exact_top2
    for score in scores:
        diagnostic = attribution_reliability(
            score.raw_gradient,
            -exact[score.site_id] * epsilon,
            exact[score.site_id] * epsilon,
            epsilon,
        )
        assert not diagnostic.flagged


def test_reduced_hessian_recovers_quadratic_interactions() -> None:
    hessian = torch.tensor(
        [[2.0, 0.3, 0.0], [0.3, 1.5, 0.2], [0.0, 0.2, 1.0]]
    )
    gain = torch.tensor([0.4, -0.2, 0.1])

    def gradient(beta: torch.Tensor) -> torch.Tensor:
        return hessian @ beta - gain

    estimate = estimate_reduced_hessian(
        gradient, 3, probes=12, epsilon=0.02, residual_rank=3, seed=139
    )
    torch.testing.assert_close(estimate.dense_estimate, hessian, atol=2e-5, rtol=2e-5)


def test_qcqp_is_feasible_and_matches_unconstrained_solution_when_loose() -> None:
    hessian = torch.diag(torch.tensor([2.0, 4.0]))
    gain = torch.tensor([1.0, 1.0])
    identity = torch.eye(2)
    loose = solve_qcqp(
        gain,
        hessian,
        identity,
        identity,
        capability_budget=10.0,
        risk_budget=10.0,
        beta_max=1.0,
    )
    assert loose.success
    torch.testing.assert_close(loose.beta, torch.tensor([0.5, 0.25]), atol=1e-5, rtol=1e-5)
    tight = solve_qcqp(
        gain,
        hessian,
        identity,
        torch.diag(torch.tensor([1.0, 3.0])),
        capability_budget=0.10,
        risk_budget=0.12,
        beta_max=0.8,
    )
    assert tight.success
    assert tight.capability_cost <= 0.10 + 1e-6
    assert tight.risk_cost <= 0.12 + 1e-6


def test_robust_objectives_and_hard_constraints() -> None:
    losses = torch.tensor([0.1, 0.2, 0.8, 0.4])
    torch.testing.assert_close(cvar(losses, alpha=0.5), torch.tensor(0.6))
    assert float(smooth_max(losses, 0.01)) >= float(losses.max()) - 1e-6
    result = enforce_scenario_constraints(
        torch.tensor([0.01, 0.03]),
        torch.tensor([0.0, 0.02]),
        capability_maximum=0.02,
        risk_maximum=0.0,
    )
    assert not result.feasible
    assert result.violating_scenarios == (1,)
