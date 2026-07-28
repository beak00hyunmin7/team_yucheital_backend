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
    algorithm: str
    image: ImageMetadata
    parameters: AnalysisParameters
    drains: list[DrainRecommendation]
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

