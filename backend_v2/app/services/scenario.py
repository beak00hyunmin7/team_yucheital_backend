from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.services.metrics import depth_metrics


class ScenarioEvaluatorUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class UserDrain:
    x: float
    y: float
    capacity_lps: float
    capture_radius_m: float
    drain_type: str


@dataclass(frozen=True)
class ScenarioEvaluation:
    evaluator_version: str
    updated_depth_m: np.ndarray
    baseline_metrics: dict[str, float | int | None]
    scenario_metrics: dict[str, float | int | None]
    drained_volume_m3: float
    warnings: tuple[str, ...]


class ScenarioEvaluator(Protocol):
    version: str

    def evaluate(
        self,
        baseline_depth_m: np.ndarray,
        drains: Sequence[UserDrain],
        *,
        duration_s: float,
        cell_size_m: float | None,
        depth_threshold_m: float,
    ) -> ScenarioEvaluation: ...


class DisabledScenarioEvaluator:
    version = "none"

    def evaluate(
        self,
        baseline_depth_m: np.ndarray,
        drains: Sequence[UserDrain],
        *,
        duration_s: float,
        cell_size_m: float | None,
        depth_threshold_m: float,
    ) -> ScenarioEvaluation:
        raise ScenarioEvaluatorUnavailable(
            "scenario evaluator is disabled; connect a trained surrogate or explicitly "
            "enable capacity_approx"
        )


class CapacityApproxEvaluator:
    """Transparent capacity-limited local removal approximation.

    This evaluator never proposes drain coordinates. It only evaluates the
    user-supplied drains. It is deliberately labeled as an approximation and
    can be replaced through the ``ScenarioEvaluator`` protocol.
    """

    version = "capacity-approx-v1"

    def evaluate(
        self,
        baseline_depth_m: np.ndarray,
        drains: Sequence[UserDrain],
        *,
        duration_s: float,
        cell_size_m: float | None,
        depth_threshold_m: float,
    ) -> ScenarioEvaluation:
        if cell_size_m is None:
            raise ValueError("cell_size_m is required for capacity-based scenario evaluation")
        if cell_size_m <= 0 or duration_s <= 0:
            raise ValueError("cell_size_m and duration_s must be positive")

        updated = np.asarray(baseline_depth_m, dtype=np.float64).copy()
        cell_area = cell_size_m * cell_size_m
        height, width = updated.shape
        total_drained = 0.0
        for drain in drains:
            center_x = drain.x * max(width - 1, 1)
            center_y = drain.y * max(height - 1, 1)
            radius_pixels = max(1.0, drain.capture_radius_m / cell_size_m)
            minimum_x = max(0, int(np.floor(center_x - radius_pixels)))
            maximum_x = min(width - 1, int(np.ceil(center_x + radius_pixels)))
            minimum_y = max(0, int(np.floor(center_y - radius_pixels)))
            maximum_y = min(height - 1, int(np.ceil(center_y + radius_pixels)))
            yy, xx = np.ogrid[minimum_y : maximum_y + 1, minimum_x : maximum_x + 1]
            distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
            mask = distance <= radius_pixels
            region = updated[minimum_y : maximum_y + 1, minimum_x : maximum_x + 1]
            available_volume = np.where(mask, region * cell_area, 0.0)
            distance_weight = np.where(mask, 1.0 - distance / (radius_pixels + 1e-12), 0.0)
            weights = available_volume * (0.25 + 0.75 * distance_weight)
            capacity_volume = drain.capacity_lps / 1000.0 * duration_s
            removed = _remove_weighted_volume(
                available_volume, weights, capacity_volume
            )
            region -= removed / cell_area
            np.maximum(region, 0.0, out=region)
            total_drained += float(removed.sum())

        baseline = depth_metrics(
            baseline_depth_m,
            threshold_m=depth_threshold_m,
            cell_size_m=cell_size_m,
        )
        scenario = depth_metrics(
            updated,
            threshold_m=depth_threshold_m,
            cell_size_m=cell_size_m,
        )
        baseline_area = baseline["flooded_area_m2"]
        scenario_area = scenario["flooded_area_m2"]
        if isinstance(baseline_area, float) and baseline_area > 0 and isinstance(scenario_area, float):
            reduction = (baseline_area - scenario_area) / baseline_area * 100.0
        else:
            reduction = 0.0
        scenario["flooded_area_reduction_percent"] = round(max(0.0, reduction), 6)
        scenario["drained_volume_m3"] = round(total_drained, 6)
        return ScenarioEvaluation(
            evaluator_version=self.version,
            updated_depth_m=updated.astype(np.float32),
            baseline_metrics=baseline,
            scenario_metrics=scenario,
            drained_volume_m3=round(total_drained, 6),
            warnings=(
                "Capacity approximation is not CFD and does not model pipes, infiltration, or obstacles.",
                "Drain coordinates were supplied by the user; this evaluator does not find drains.",
            ),
        )


def build_scenario_evaluator(name: str) -> ScenarioEvaluator:
    normalized = name.strip().lower()
    if normalized == "capacity_approx":
        return CapacityApproxEvaluator()
    if normalized in {"", "none", "disabled"}:
        return DisabledScenarioEvaluator()
    raise ValueError(f"unknown scenario evaluator: {name}")


def _remove_weighted_volume(
    available: np.ndarray, weights: np.ndarray, requested_volume: float
) -> np.ndarray:
    removed = np.zeros_like(available, dtype=np.float64)
    remaining = min(max(float(requested_volume), 0.0), float(available.sum()))
    capacity = available.copy()
    active_weights = weights.copy()
    for _ in range(6):
        if remaining <= 1e-12:
            break
        active = capacity > 1e-12
        if not np.any(active):
            break
        active_weights = np.where(active, active_weights, 0.0)
        weight_sum = float(active_weights.sum())
        if weight_sum <= 1e-12:
            active_weights = active.astype(np.float64)
            weight_sum = float(active_weights.sum())
        allocation = remaining * active_weights / weight_sum
        take = np.minimum(allocation, capacity)
        removed += take
        capacity -= take
        taken = float(take.sum())
        remaining -= taken
        if taken <= 1e-12:
            break
    return removed
