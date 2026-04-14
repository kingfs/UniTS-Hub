from __future__ import annotations

from typing import Dict, List


def forecast_result(
    mean: List[float],
    quantiles: Dict[str, List[float]] | None = None,
) -> Dict[str, List[float] | Dict[str, List[float]]]:
    return {
        "mean": mean,
        "quantiles": quantiles or {},
    }
