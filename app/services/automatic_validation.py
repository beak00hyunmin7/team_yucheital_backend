from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from app.config import (
    MIN_AUTO_VERIFICATION_CONFIDENCE,
    MIN_CAPTURE_RATIO,
    MIN_MEAN_SUITABILITY,
    MIN_STABILITY_SCORE,
    VALIDATOR_VERSION,
)
from app.services.hydrology import HydrologyResult


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    confidence: float
    count_ok: bool
    spacing_ok: bool
    boundary_ok: bool
    minimum_spacing_ratio_actual: float
    mean_suitability: float
    estimated_capture_ratio: float
    residual_water_ratio: float
    stability_score: float
    failure_reasons: list[str]
    validator_version: str = VALIDATOR_VERSION

    def metrics(self) -> dict[str, float | bool | str | list[str]]:
        return asdict(self)


def smooth_elevation(elevation: np.ndarray) -> np.ndarray:
    """Return a small deterministic perturbation used for consistency checks."""
    padded = np.pad(elevation, 1, mode="edge")
    smoothed = np.zeros_like(elevation, dtype=np.float64)
    for dy in range(3):
        for dx in range(3):
            smoothed += padded[dy : dy + elevation.shape[0], dx : dx + elevation.shape[1]]
    smoothed /= 9.0
    value_range = float(np.ptp(smoothed))
    if value_range <= 1e-12:
        return elevation.copy()
    return (smoothed - float(smoothed.min())) / value_range


def validate_prediction(
    elevation: np.ndarray,
    result: HydrologyResult,
    comparison: HydrologyResult,
    *,
    requested_drain_count: int,
    required_spacing_ratio: float,
    edge_margin_ratio: float = 0.01,
) -> ValidationResult:
    """Validate one prediction without relying on a human reviewer.

    The validator combines hard geometry constraints, a D8 runoff-capture
    simulation, and consistency under a small input perturbation. It is kept
    separate from the prediction adapter so a trained model cannot approve its
    own result using only its confidence score.
    """
    height, width = elevation.shape
    count_ok = len(result.drains) == requested_drain_count
    spacing_actual = _minimum_spacing_ratio(result, height=height, width=width)
    spacing_ok = spacing_actual + 1e-9 >= required_spacing_ratio
    boundary_ok = _boundary_ok(
        result,
        height=height,
        width=width,
        margin_ratio=edge_margin_ratio,
    )
    mean_suitability = (
        float(np.mean([candidate.score for candidate in result.drains]))
        if result.drains
        else 0.0
    )
    capture_ratio = _estimate_capture_ratio(elevation, result)
    stability_score = _stability_score(
        result,
        comparison,
        height=height,
        width=width,
    )

    hard_constraints = count_ok and spacing_ok and boundary_ok
    capture_quality = min(1.0, capture_ratio / max(MIN_CAPTURE_RATIO, 1e-12))
    confidence = (
        0.25 * float(hard_constraints)
        + 0.25 * mean_suitability
        + 0.25 * capture_quality
        + 0.25 * stability_score
    )
    confidence = float(np.clip(confidence, 0.0, 1.0))

    failures: list[str] = []
    if not count_ok:
        failures.append("요청한 배수구 개수와 생성된 개수가 다릅니다.")
    if not spacing_ok:
        failures.append("배수구 간 최소 거리 기준을 만족하지 못했습니다.")
    if not boundary_ok:
        failures.append("배수구가 지형 경계의 안전 여백 안에 있습니다.")
    if mean_suitability < MIN_MEAN_SUITABILITY:
        failures.append("배수구 후보의 평균 적합도 점수가 기준보다 낮습니다.")
    if capture_ratio < MIN_CAPTURE_RATIO:
        failures.append("D8 모의 배수에서 예상 유출수 포착 비율이 기준보다 낮습니다.")
    if stability_score < MIN_STABILITY_SCORE:
        failures.append("작은 입력 변화에 대한 배수구 위치 일관성이 낮습니다.")
    if confidence < MIN_AUTO_VERIFICATION_CONFIDENCE:
        failures.append("종합 자동 검증 신뢰도가 기준보다 낮습니다.")

    return ValidationResult(
        passed=not failures,
        confidence=round(confidence, 6),
        count_ok=count_ok,
        spacing_ok=spacing_ok,
        boundary_ok=boundary_ok,
        minimum_spacing_ratio_actual=round(spacing_actual, 6),
        mean_suitability=round(mean_suitability, 6),
        estimated_capture_ratio=round(capture_ratio, 6),
        residual_water_ratio=round(1.0 - capture_ratio, 6),
        stability_score=round(stability_score, 6),
        failure_reasons=failures,
    )


def _minimum_spacing_ratio(
    result: HydrologyResult, *, height: int, width: int
) -> float:
    if len(result.drains) < 2:
        return 1.0
    distances = []
    for index, first in enumerate(result.drains):
        for second in result.drains[index + 1 :]:
            distances.append(float(np.hypot(first.x - second.x, first.y - second.y)))
    return min(distances) / max(1, min(height, width))


def _boundary_ok(
    result: HydrologyResult, *, height: int, width: int, margin_ratio: float
) -> bool:
    margin = max(1, round(min(height, width) * margin_ratio))
    return all(
        margin <= drain.x < width - margin and margin <= drain.y < height - margin
        for drain in result.drains
    )


def _estimate_capture_ratio(
    elevation: np.ndarray,
    result: HydrologyResult,
) -> float:
    """Estimate uniform-rainfall cells whose D8 path crosses a drain cell."""
    flat_downstream = result.downstream.ravel()
    captured = np.zeros(elevation.size, dtype=bool)
    drain_indices = {
        drain.y * elevation.shape[1] + drain.x for drain in result.drains
    }
    # D8 edges are strictly downhill, so the downstream cell has already been
    # processed when cells are visited from low to high elevation.
    for source in np.argsort(elevation.ravel(), kind="stable"):
        source_index = int(source)
        if source_index in drain_indices:
            captured[source_index] = True
            continue
        target = int(flat_downstream[source_index])
        if target >= 0:
            captured[source_index] = captured[target]
    return float(np.mean(captured))


def _stability_score(
    result: HydrologyResult,
    comparison: HydrologyResult,
    *,
    height: int,
    width: int,
) -> float:
    if not result.drains or len(result.drains) != len(comparison.drains):
        return 0.0
    remaining = list(comparison.drains)
    distances: list[float] = []
    diagonal = max(float(np.hypot(width - 1, height - 1)), 1.0)
    for candidate in result.drains:
        nearest = min(
            remaining,
            key=lambda other: (candidate.x - other.x) ** 2
            + (candidate.y - other.y) ** 2,
        )
        distances.append(float(np.hypot(candidate.x - nearest.x, candidate.y - nearest.y)))
        remaining.remove(nearest)
    mean_normalized_distance = float(np.mean(distances)) / diagonal
    return float(np.exp(-mean_normalized_distance / 0.12))
