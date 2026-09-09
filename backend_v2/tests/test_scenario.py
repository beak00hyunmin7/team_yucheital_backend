from __future__ import annotations

import numpy as np
import pytest

from app.services.scenario import (
    CapacityApproxEvaluator,
    DisabledScenarioEvaluator,
    ScenarioEvaluatorUnavailable,
    UserDrain,
)


def test_capacity_evaluator_only_uses_user_supplied_drains() -> None:
    baseline = np.full((64, 64), 0.10, dtype=np.float32)
    drains = [
        UserDrain(
            x=0.5,
            y=0.5,
            capacity_lps=20.0,
            capture_radius_m=12.0,
            drain_type="standard",
        )
    ]
    result = CapacityApproxEvaluator().evaluate(
        baseline,
        drains,
        duration_s=120.0,
        cell_size_m=1.0,
        depth_threshold_m=0.05,
    )
    assert result.evaluator_version == "capacity-approx-v1"
    assert result.updated_depth_m.shape == baseline.shape
    assert float(result.updated_depth_m.sum()) < float(baseline.sum())
    assert result.drained_volume_m3 > 0
    assert "Drain coordinates were supplied by the user" in result.warnings[1]


def test_disabled_evaluator_fails_instead_of_faking_a_result() -> None:
    with pytest.raises(ScenarioEvaluatorUnavailable):
        DisabledScenarioEvaluator().evaluate(
            np.zeros((16, 16), dtype=np.float32),
            [],
            duration_s=120.0,
            cell_size_m=1.0,
            depth_threshold_m=0.05,
        )
