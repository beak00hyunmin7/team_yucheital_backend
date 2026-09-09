from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AssetResponse(BaseModel):
    kind: str
    url: str
    sha256: str


class UploadResponse(BaseModel):
    request_id: str
    upload_id: str
    status: str
    cache_hit: bool
    input_type: str
    original_width: int
    original_height: int
    model_width: int
    model_height: int
    quality_score: float
    quality_report: dict[str, Any]
    preprocessing_version: str
    warnings: list[str]
    assets: list[AssetResponse]
    processing_time_ms: int


class PredictionRequest(BaseModel):
    upload_id: str = Field(min_length=36, max_length=36)
    rain_mm: float = Field(ge=20.0, le=80.0)
    time_s: float = Field(ge=0.0, le=120.0)


class PredictionResponse(BaseModel):
    request_id: str
    result_id: str
    upload_id: str
    status: str
    cache_hit: bool
    model_version: str
    checkpoint_hash: str
    rain_mm: float
    time_s: float
    water_map_url: str
    depth_array_url: str
    metrics: dict[str, Any]
    inference_time_ms: int
    processing_time_ms: int
    drains: list = Field(
        default_factory=list,
        description="Always empty: v2 does not automatically find drain positions.",
    )


class DrainInput(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    capacity_lps: float = Field(gt=0.0, le=1_000_000.0)
    capture_radius_m: float = Field(gt=0.0, le=100_000.0)
    drain_type: str = Field(default="standard", min_length=1, max_length=50)


class ScenarioRequest(BaseModel):
    result_id: str = Field(min_length=36, max_length=36)
    drains: list[DrainInput] = Field(min_length=1, max_length=100)
    duration_s: float | None = Field(default=None, gt=0.0, le=86400.0)


class ScenarioResponse(BaseModel):
    request_id: str
    scenario_id: str
    result_id: str
    status: str
    cache_hit: bool
    evaluator_version: str
    drains: list[DrainInput]
    baseline_metrics: dict[str, Any]
    scenario_metrics: dict[str, Any]
    scenario_map_url: str
    depth_array_url: str
    warnings: list[str]
    processing_time_ms: int


class LiveHealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ReadyHealthResponse(BaseModel):
    status: str
    database: str
    storage: str
    ai_model: str
    model_version: str
    checkpoint_hash: str | None
    checkpoint_name: str
    device: str | None
    condition_dimension: int | None
    fallback: bool = False
    scenario_evaluator: str
    detail: str | None = None
