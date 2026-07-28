from __future__ import annotations

import numpy as np

from app.services.hydrology import recommend_drains


def test_single_sloped_basin_recommends_lower_right() -> None:
    height, width = 80, 100
    yy, xx = np.indices((height, width))
    elevation = 1.0 - 0.55 * (xx / (width - 1)) - 0.45 * (yy / (height - 1))

    result = recommend_drains(
        elevation,
        drain_count=1,
        minimum_spacing_ratio=0.1,
    )

    drain = result.drains[0]
    assert drain.x >= int(width * 0.85)
    assert drain.y >= int(height * 0.85)
    # A planar D8 slope forms parallel flow paths instead of one converging basin.
    assert drain.accumulation_cells > 50


def test_two_bowls_produce_spaced_candidates() -> None:
    size = 96
    yy, xx = np.indices((size, size))
    bowl_a = ((xx - 25) ** 2 + (yy - 50) ** 2) / size**2
    bowl_b = ((xx - 72) ** 2 + (yy - 46) ** 2) / size**2
    elevation = np.minimum(bowl_a, bowl_b)
    elevation = (elevation - elevation.min()) / np.ptp(elevation)

    result = recommend_drains(
        elevation,
        drain_count=2,
        minimum_spacing_ratio=0.25,
    )

    first, second = result.drains
    distance = np.hypot(first.x - second.x, first.y - second.y)
    assert distance >= size * 0.25
