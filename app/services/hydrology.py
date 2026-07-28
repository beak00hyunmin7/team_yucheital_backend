from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DrainCandidate:
    x: int
    y: int
    elevation: float
    accumulation_cells: float
    score: float
    local_depression: float


@dataclass(frozen=True)
class HydrologyResult:
    drains: list[DrainCandidate]
    accumulation: np.ndarray
    flow_slope: np.ndarray
    score: np.ndarray


def recommend_drains(
    elevation: np.ndarray,
    *,
    drain_count: int,
    minimum_spacing_ratio: float,
    edge_margin_ratio: float = 0.025,
) -> HydrologyResult:
    """Recommend drain cells using D8 flow routing and terrain-based scoring.

    The DEM is routed to the steepest lower cell among its eight neighbors.
    Strictly descending routing makes the graph acyclic, allowing upstream
    contributing-cell counts to be accumulated in descending elevation order.
    """
    if elevation.ndim != 2:
        raise ValueError("elevation must be a two-dimensional array")
    if min(elevation.shape) < 8:
        raise ValueError("elevation grid is too small")
    if not 1 <= drain_count <= 10:
        raise ValueError("drain_count must be between 1 and 10")

    downstream, flow_slope = _d8_downstream(elevation)
    accumulation = _flow_accumulation(elevation, downstream)
    local_depression = _local_depression(elevation)

    acc_score = np.log1p(accumulation)
    acc_score /= max(float(acc_score.max()), 1e-12)
    low_score = 1.0 - elevation
    depression_score = _normalize(local_depression)

    # Accumulation dominates; low terrain and local depressions break ties and
    # favor sites that naturally retain runoff.
    score = 0.65 * acc_score + 0.25 * low_score + 0.10 * depression_score
    score = np.clip(score, 0.0, 1.0)

    valid = np.ones(elevation.shape, dtype=bool)
    margin = max(1, round(min(elevation.shape) * edge_margin_ratio))
    valid[:margin, :] = False
    valid[-margin:, :] = False
    valid[:, :margin] = False
    valid[:, -margin:] = False

    drains = _select_spaced_candidates(
        score=score,
        valid=valid,
        elevation=elevation,
        accumulation=accumulation,
        local_depression=local_depression,
        count=drain_count,
        spacing=max(1.0, min(elevation.shape) * minimum_spacing_ratio),
    )
    return HydrologyResult(
        drains=drains,
        accumulation=accumulation,
        flow_slope=flow_slope,
        score=score,
    )


def _d8_downstream(elevation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = elevation.shape
    yy, xx = np.indices((height, width))
    downstream = np.full((height, width), -1, dtype=np.int64)
    best_slope = np.zeros((height, width), dtype=np.float64)

    for dy, dx in (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ):
        neighbor = np.roll(elevation, shift=(-dy, -dx), axis=(0, 1))
        valid = np.ones((height, width), dtype=bool)
        if dy < 0:
            valid[: -dy, :] = False
        elif dy > 0:
            valid[height - dy :, :] = False
        if dx < 0:
            valid[:, : -dx] = False
        elif dx > 0:
            valid[:, width - dx :] = False

        distance = np.sqrt(2.0) if dx and dy else 1.0
        slope = (elevation - neighbor) / distance
        better = valid & (slope > best_slope + 1e-12)
        best_slope[better] = slope[better]
        downstream[better] = ((yy + dy) * width + (xx + dx))[better]

    return downstream, best_slope


def _flow_accumulation(elevation: np.ndarray, downstream: np.ndarray) -> np.ndarray:
    flat_elevation = elevation.ravel()
    flat_downstream = downstream.ravel()
    accumulation = np.ones(flat_elevation.size, dtype=np.float64)
    order = np.argsort(flat_elevation, kind="stable")[::-1]

    for source in order:
        target = int(flat_downstream[source])
        if target >= 0:
            accumulation[target] += accumulation[source]
    return accumulation.reshape(elevation.shape)


def _local_depression(elevation: np.ndarray) -> np.ndarray:
    total = np.zeros_like(elevation)
    count = np.zeros_like(elevation)
    height, width = elevation.shape
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        shifted = np.roll(elevation, shift=(-dy, -dx), axis=(0, 1))
        valid = np.ones((height, width), dtype=bool)
        if dy < 0:
            valid[: -dy, :] = False
        elif dy > 0:
            valid[height - dy :, :] = False
        if dx < 0:
            valid[:, : -dx] = False
        elif dx > 0:
            valid[:, width - dx :] = False
        total[valid] += shifted[valid]
        count[valid] += 1.0
    neighbor_mean = total / np.maximum(count, 1.0)
    return np.maximum(neighbor_mean - elevation, 0.0)


def _normalize(values: np.ndarray) -> np.ndarray:
    maximum = float(values.max())
    if maximum <= 1e-12:
        return np.zeros_like(values)
    return values / maximum


def _select_spaced_candidates(
    *,
    score: np.ndarray,
    valid: np.ndarray,
    elevation: np.ndarray,
    accumulation: np.ndarray,
    local_depression: np.ndarray,
    count: int,
    spacing: float,
) -> list[DrainCandidate]:
    ranked = np.argsort(np.where(valid, score, -np.inf).ravel())[::-1]
    height, width = score.shape
    selected: list[DrainCandidate] = []
    spacing_squared = spacing * spacing

    for index in ranked:
        if not np.isfinite(score.ravel()[index]) or not valid.ravel()[index]:
            continue
        y, x = divmod(int(index), width)
        if any((x - item.x) ** 2 + (y - item.y) ** 2 < spacing_squared for item in selected):
            continue
        selected.append(
            DrainCandidate(
                x=x,
                y=y,
                elevation=float(elevation[y, x]),
                accumulation_cells=float(accumulation[y, x]),
                score=float(score[y, x]),
                local_depression=float(local_depression[y, x]),
            )
        )
        if len(selected) == count:
            break

    if len(selected) < count:
        raise ValueError("요청한 개수만큼 간격을 둔 배수구 후보를 찾을 수 없습니다.")
    return selected

