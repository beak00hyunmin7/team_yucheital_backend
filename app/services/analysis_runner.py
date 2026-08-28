from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.ai.interface import DrainageModel
from app.config import MAX_ANALYSIS_ATTEMPTS
from app.services.automatic_validation import (
    ValidationResult,
    smooth_elevation,
    validate_prediction,
)
from app.services.hydrology import HydrologyResult


@dataclass(frozen=True)
class AttemptOutcome:
    attempt_number: int
    result: HydrologyResult
    validation: ValidationResult
    strategy: str


@dataclass(frozen=True)
class AnalysisOutcome:
    attempts: list[AttemptOutcome]
    selected: AttemptOutcome

    @property
    def verified(self) -> bool:
        return self.selected.validation.passed


_STRATEGIES = (
    "balanced_flow_lowland",
    "depression_emphasis",
    "flow_accumulation_emphasis",
)


def run_auto_validated_analysis(
    model: DrainageModel,
    elevation: np.ndarray,
    *,
    drain_count: int,
    minimum_spacing_ratio: float,
) -> AnalysisOutcome:
    comparison_elevation = smooth_elevation(elevation)
    attempts: list[AttemptOutcome] = []

    for attempt_number in range(1, MAX_ANALYSIS_ATTEMPTS + 1):
        result = model.predict(
            elevation,
            drain_count=drain_count,
            minimum_spacing_ratio=minimum_spacing_ratio,
            attempt_number=attempt_number,
        )
        comparison = model.predict(
            comparison_elevation,
            drain_count=drain_count,
            minimum_spacing_ratio=minimum_spacing_ratio,
            attempt_number=attempt_number,
        )
        validation = validate_prediction(
            elevation,
            result,
            comparison,
            requested_drain_count=drain_count,
            required_spacing_ratio=minimum_spacing_ratio,
        )
        outcome = AttemptOutcome(
            attempt_number=attempt_number,
            result=result,
            validation=validation,
            strategy=_STRATEGIES[min(attempt_number - 1, len(_STRATEGIES) - 1)],
        )
        attempts.append(outcome)
        if validation.passed:
            return AnalysisOutcome(attempts=attempts, selected=outcome)

    selected = max(attempts, key=lambda item: item.validation.confidence)
    return AnalysisOutcome(attempts=attempts, selected=selected)
