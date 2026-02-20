from __future__ import annotations

from typing import Iterable, List

from gabion.common.protocol import RoundResultPayload


def fedavg(results: Iterable[RoundResultPayload], fallback_weights: List[float]) -> List[float]:
    weighted_sum: List[float] = [0.0 for _ in fallback_weights]
    total_samples = 0

    for result in results:
        count = max(0, int(result["sample_count"]))
        weights = result["weights"]
        if len(weights) != len(weighted_sum) or count == 0:
            continue
        total_samples += count
        for i, weight in enumerate(weights):
            weighted_sum[i] += weight * count

    if total_samples == 0:
        return list(fallback_weights)
    return [value / total_samples for value in weighted_sum]
