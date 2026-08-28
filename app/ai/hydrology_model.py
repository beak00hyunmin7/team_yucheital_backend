from __future__ import annotations

import numpy as np

from app.config import MODEL_NAME, MODEL_VERSION
from app.services.hydrology import HydrologyResult, recommend_drains


class HydrologySurrogateModel:
    """Executable MVP adapter used until a trained model is supplied.

    Retry attempts use different physically interpretable score mixtures. This
    changes candidate ranking without weakening the user's spacing constraint.
    """

    version = MODEL_VERSION
    name = MODEL_NAME

    _WEIGHTS = (
        (0.65, 0.25, 0.10),
        (0.55, 0.30, 0.15),
        (0.72, 0.18, 0.10),
    )

    def predict(
        self,
        elevation: np.ndarray,
        *,
        drain_count: int,
        minimum_spacing_ratio: float,
        attempt_number: int,
    ) -> HydrologyResult:
        weights = self._WEIGHTS[min(attempt_number - 1, len(self._WEIGHTS) - 1)]
        return recommend_drains(
            elevation,
            drain_count=drain_count,
            minimum_spacing_ratio=minimum_spacing_ratio,
            score_weights=weights,
        )


model = HydrologySurrogateModel()
