from __future__ import annotations

import base64
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    ALLOWED_CONTENT_TYPES,
    ANALYSIS_MAX_DIMENSION,
    API_PREFIX,
    APP_NAME,
    APP_VERSION,
    MAX_IMAGE_PIXELS,
    MAX_UPLOAD_BYTES,
)
from app.schemas import (
    AnalysisParameters,
    AnalysisResponse,
    DrainRecommendation,
    HealthResponse,
    ImageMetadata,
    NormalizedCoordinate,
    PixelCoordinate,
)
from app.services.hydrology import recommend_drains
from app.services.image_io import InvalidTerrainImage, prepare_dem
from app.services.visualization import render_overlay


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "A baseline backend that interprets grayscale image brightness as terrain elevation, "
        "routes runoff with D8 flow direction, and recommends drain locations."
    ),
)

# Development-friendly default. Replace with the deployed frontend origin in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get(f"{API_PREFIX}/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service=APP_NAME, version=APP_VERSION)


@app.post(
    f"{API_PREFIX}/drainage/analyze",
    response_model=AnalysisResponse,
    tags=["drainage"],
    summary="Analyze a grayscale DEM image and recommend drain locations",
)
async def analyze_drainage(
    image: Annotated[UploadFile, File(description="Grayscale DEM image")],
    drain_count: Annotated[int, Form(ge=1, le=10)] = 3,
    rainfall_mm_per_hour: Annotated[float, Form(gt=0, le=1000)] = 50.0,
    runoff_coefficient: Annotated[float, Form(gt=0, le=1)] = 0.8,
    cell_size_m: Annotated[float, Form(gt=0, le=10000)] = 1.0,
    minimum_spacing_ratio: Annotated[float, Form(ge=0.01, le=0.5)] = 0.10,
    blur_radius: Annotated[float, Form(ge=0, le=10)] = 1.2,
    high_is_bright: Annotated[bool, Form()] = True,
    include_overlay: Annotated[bool, Form()] = True,
) -> AnalysisResponse:
    if image.content_type and image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="PNG, JPEG, WEBP 또는 TIFF 이미지만 업로드할 수 있습니다.",
        )

    data = await image.read(MAX_UPLOAD_BYTES + 1)
    await image.close()
    if not data:
        raise HTTPException(status_code=400, detail="업로드한 이미지가 비어 있습니다.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"파일 크기는 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB 이하여야 합니다.",
        )

    try:
        dem = await run_in_threadpool(
            prepare_dem,
            data,
            max_pixels=MAX_IMAGE_PIXELS,
            analysis_max_dimension=ANALYSIS_MAX_DIMENSION,
            high_is_bright=high_is_bright,
            blur_radius=blur_radius,
        )
        hydrology = await run_in_threadpool(
            recommend_drains,
            dem.elevation,
            drain_count=drain_count,
            minimum_spacing_ratio=minimum_spacing_ratio,
        )
    except InvalidTerrainImage as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    original_cell_area = cell_size_m**2
    analysis_cell_area = original_cell_area * dem.scale_x * dem.scale_y
    recommendations: list[DrainRecommendation] = []
    for rank, candidate in enumerate(hydrology.drains, start=1):
        original_x = min(
            dem.original_width - 1,
            max(0, round((candidate.x + 0.5) * dem.scale_x - 0.5)),
        )
        original_y = min(
            dem.original_height - 1,
            max(0, round((candidate.y + 0.5) * dem.scale_y - 0.5)),
        )
        area_m2 = candidate.accumulation_cells * analysis_cell_area
        peak_flow_lps = runoff_coefficient * rainfall_mm_per_hour * area_m2 / 3600.0
        recommendations.append(
            DrainRecommendation(
                rank=rank,
                pixel=PixelCoordinate(x=original_x, y=original_y),
                normalized=NormalizedCoordinate(
                    x=round(original_x / max(dem.original_width - 1, 1), 6),
                    y=round(original_y / max(dem.original_height - 1, 1), 6),
                ),
                elevation_normalized=round(candidate.elevation, 6),
                flow_accumulation_cells=round(candidate.accumulation_cells, 2),
                catchment_area_m2=round(area_m2, 3),
                estimated_peak_flow_lps=round(peak_flow_lps, 3),
                suitability_score=round(candidate.score, 6),
                rationale=(
                    "상류 기여면적이 크고 상대적으로 낮은 지형으로, "
                    "D8 유향 분석에서 유출수가 집중되는 후보입니다."
                ),
            )
        )

    overlay = None
    if include_overlay:
        png = await run_in_threadpool(
            render_overlay,
            dem.original_rgb,
            hydrology,
            scale_x=dem.scale_x,
            scale_y=dem.scale_y,
        )
        overlay = base64.b64encode(png).decode("ascii")

    warnings = [
        "이 결과는 개념 검증용 지형수문학 기반 추천이며 CFD 해석 결과가 아닙니다.",
        "토질, 침투율, 기존 관로, 배수구 용량, 건축물·도로 장애물은 아직 반영하지 않습니다.",
    ]
    if not high_is_bright:
        warnings.append("이번 요청에서는 어두울수록 고도가 높은 것으로 해석했습니다.")

    return AnalysisResponse(
        request_id=str(uuid4()),
        algorithm="D8 flow direction + flow accumulation + low/depression terrain score (MVP v0.1)",
        image=ImageMetadata(
            filename=image.filename or "upload",
            original_width=dem.original_width,
            original_height=dem.original_height,
            analysis_width=dem.analysis_width,
            analysis_height=dem.analysis_height,
            input_mode="grayscale_dem",
            elevation_convention="bright_is_high" if high_is_bright else "dark_is_high",
        ),
        parameters=AnalysisParameters(
            drain_count=drain_count,
            rainfall_mm_per_hour=rainfall_mm_per_hour,
            runoff_coefficient=runoff_coefficient,
            original_cell_size_m=cell_size_m,
            minimum_spacing_ratio=minimum_spacing_ratio,
            blur_radius=blur_radius,
        ),
        drains=recommendations,
        assumptions=[
            "각 픽셀의 밝기값이 상대 고도를 나타냅니다.",
            "각 셀의 유출수는 인접 8방향 중 가장 가파른 하강 방향으로 이동합니다.",
            "강우가 전체 영역에 균일하게 내리고 유출계수가 일정하다고 가정합니다.",
            "좌표 원점 (0, 0)은 이미지 왼쪽 위입니다.",
        ],
        warnings=warnings,
        overlay_png_base64=overlay,
    )


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {
        "message": APP_NAME,
        "swagger_ui": "/docs",
        "health": f"{API_PREFIX}/health",
    }

