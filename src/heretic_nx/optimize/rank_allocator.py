"""Exact prefix-constrained rank allocation over a Pareto frontier."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math


def allocate_rank(
    marginal_utilities: Mapping[str, Sequence[float]],
    marginal_costs: Mapping[str, Sequence[float]],
    budget: float,
) -> dict[str, int]:
    if budget < 0:
        raise ValueError("budget must be non-negative")
    sites = tuple(sorted(marginal_utilities))
    choices_by_site: dict[str, list[tuple[float, float, int]]] = {}
    for site in sites:
        utilities = marginal_utilities[site]
        costs = marginal_costs.get(site)
        if costs is None or len(costs) != len(utilities):
            raise ValueError(f"missing or invalid marginal costs for {site}")
        prefix_utility = 0.0
        prefix_cost = 0.0
        choices = [(0.0, 0.0, 0)]
        for rank, (utility_value, cost_value) in enumerate(zip(utilities, costs), start=1):
            utility = float(utility_value)
            cost = float(cost_value)
            if cost <= 0:
                raise ValueError("marginal costs must be positive")
            if not all(map(math.isfinite, (utility, cost))):
                raise ValueError("marginal utilities and costs must be finite")
            prefix_utility += utility
            prefix_cost += cost
            if prefix_cost <= budget:
                choices.append((prefix_cost, prefix_utility, rank))
        choices_by_site[site] = choices

    # State is (cost, utility, rank-prefix tuple). Keeping only nondominated
    # states is exact because future choices add the same options to each state.
    frontier: list[tuple[float, float, tuple[int, ...]]] = [(0.0, 0.0, ())]
    for site in sites:
        expanded = [
            (spent + cost, value + utility, ranks + (rank,))
            for spent, value, ranks in frontier
            for cost, utility, rank in choices_by_site[site]
            if spent + cost <= budget
        ]
        expanded.sort(key=lambda state: (state[0], -state[1], state[2]))
        frontier = []
        best_utility = float("-inf")
        for state in expanded:
            if state[1] > best_utility:
                frontier.append(state)
                best_utility = state[1]

    _cost, _utility, ranks = max(
        frontier,
        key=lambda state: (state[1], -state[0], tuple(-rank for rank in state[2])),
    )
    return dict(zip(sites, ranks))
