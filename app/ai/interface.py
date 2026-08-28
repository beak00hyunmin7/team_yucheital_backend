from __future__ import annotations

from typing import Protocol

import numpy as np

from app.services.hydrology import HydrologyResult


class DrainageModel(Protocol):
    """Minimal contract that a trained PyTorch/ONNX model must implement."""

    version: str
    name: str

    def predict(
        self,
        elevation: np.ndarray,
        *,
        drain_count: int,
        minimum_spacing_ratio: float,
        attempt_number: int,
    ) -> HydrologyResult: ...
