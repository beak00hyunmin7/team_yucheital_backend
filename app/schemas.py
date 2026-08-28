from __future__ import annotations

from pydantic import BaseModel, Field


class PixelCoordinate(BaseModel):
    x: int
    y: int


class NormalizedCoordinate(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class DrainRecommendation(BaseModel):
    rank: int
    pixel: PixelCoordinate
    normalized: NormalizedCoordinate
    elevation_normalized: float = Field(ge=0.0, le=1.0)
    flow_accumulation_cells: float = Field(ge=1.0)
    catchment_area_m2: float = Field(ge=0.0)
    estimated_peak_flow_lps: float = Field(ge=0.0)
    suitability_score: float = Field(ge=0.0, le=1.0)
    rationale: str


class ImageMetadata(BaseModel):
    filename: str
    original_width: int
    original_height: int
    analysis_width: int
    analysis_height: int
    input_mode: str
    elevation_convention: str


class AnalysisParameters(BaseModel):
    drain_count: int
    rainfall_mm_per_hour: float
    runoff_coefficient: float
    original_cell_size_m: float
    minimum_spacing_ratio: float
    blur_radius: float


class AnalysisResponse(BaseModel):
    request_id: str
    analysis_case_id: str
    status: str
    verification_status: str
    validation_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    retry_count: int = Field(default=0, ge=0)
    selected_attempt_number: int | None = Field(default=None, ge=1)
    training_status: str
    automatic_validation: "AutoValidationResponse"
    model_version: str
    cache_hit: bool = False
    algorithm: str
    image: ImageMetadata
    parameters: AnalysisParameters
    drains: list[DrainRecommendation]
    input_image_url: str
    water_map_url: str
    overlay_image_url: str
    processing_time_ms: int
    assumptions: list[str]
    warnings: list[str]
    overlay_png_base64: str | None = Field(
        default=None,
        description="Base64-encoded PNG with flow heatmap and drain markers.",
    )


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class TrainingDrainPosition(BaseModel):
    order: int = Field(ge=1, le=10)
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class AutoValidationResponse(BaseModel):
    attempt_number: int
    status: str
    passed: bool
    confidence: float = Field(ge=0.0, le=1.0)
    count_ok: bool
    spacing_ok: bool
    boundary_ok: bool
    minimum_spacing_ratio_actual: float = Field(ge=0.0)
    mean_suitability: float = Field(ge=0.0, le=1.0)
    estimated_capture_ratio: float = Field(ge=0.0, le=1.0)
    residual_water_ratio: float = Field(ge=0.0, le=1.0)
    stability_score: float = Field(ge=0.0, le=1.0)
    failure_reasons: list[str]
    validator_version: str


class TrainingDatasetItemResponse(BaseModel):
    item_id: int
    analysis_case_id: str
    status: str
    input_image_url: str
    target_watermap_url: str
    target_positions: list[TrainingDrainPosition]
    source_model_version: str
    validator_version: str
    quality_score: float
    verified_at: str


AnalysisResponse.model_rebuild()
