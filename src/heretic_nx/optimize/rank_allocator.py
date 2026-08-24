"""Discrete rank allocation by marginal protected utility per unit cost."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def allocate_rank(
    marginal_utilities: Mapping[str, Sequence[float]],
    marginal_costs: Mapping[str, Sequence[float]],
    budget: float,
) -> dict[str, int]:
    if budget < 0:
        raise ValueError("budget must be non-negative")
    allocations = {site: 0 for site in marginal_utilities}
    spent = 0.0

    while True:
        choices: list[tuple[float, str, float]] = []
        for site, utilities in marginal_utilities.items():
            index = allocations[site]
            costs = marginal_costs.get(site)
            if costs is None or len(costs) != len(utilities):
                raise ValueError(f"missing or invalid marginal costs for {site}")
            if index >= len(utilities):
                continue
            cost = float(costs[index])
            utility = float(utilities[index])
            if cost <= 0:
                raise ValueError("marginal costs must be positive")
            if utility > 0 and spent + cost <= budget:
                choices.append((utility / cost, site, cost))
        if not choices:
            break
        _, selected, selected_cost = max(choices, key=lambda item: (item[0], item[1]))
        allocations[selected] += 1
        spent += selected_cost

    return allocations
