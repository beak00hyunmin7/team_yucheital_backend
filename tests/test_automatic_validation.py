from __future__ import annotations

import numpy as np

from app.ai.hydrology_model import model
from app.services.analysis_runner import run_auto_validated_analysis
from app.services.automatic_validation import smooth_elevation


def test_smoothing_keeps_shape_and_normalized_range() -> None:
    elevation = np.arange(100, dtype=np.float64).reshape(10, 10)
    smoothed = smooth_elevation(elevation)
    assert smoothed.shape == elevation.shape
    assert float(smoothed.min()) == 0.0
    assert float(smoothed.max()) == 1.0


def test_runner_returns_selected_attempt_with_validation() -> None:
    size = 64
    yy, xx = np.indices((size, size))
    elevation = 1.0 - 0.6 * (xx / (size - 1)) - 0.4 * (yy / (size - 1))
    outcome = run_auto_validated_analysis(
        model,
        elevation,
        drain_count=2,
        minimum_spacing_ratio=0.1,
    )
    assert len(outcome.attempts) >= 1
    assert outcome.selected.validation.confidence >= 0.0
    assert len(outcome.selected.result.drains) == 2
