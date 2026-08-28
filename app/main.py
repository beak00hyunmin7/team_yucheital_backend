from __future__ import annotations

import base64
import hashlib
import json
import time
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.ai.active_model import IS_D8_FALLBACK, model
from app.config import (
    ALLOWED_CONTENT_TYPES,
    ANALYSIS_MAX_DIMENSION,
    API_PREFIX,
    APP_NAME,
    APP_VERSION,
    MAX_IMAGE_PIXELS,
    MAX_UPLOAD_BYTES,
    STORAGE_ROOT,
    VALIDATOR_VERSION,
)
from app.database import SessionLocal, get_db, init_db
from app.models import (
    AnalysisAttempt,
    AnalysisCase,
    AnalysisFile,
    AutomaticValidation,
    DrainPosition,
    ModelVersion,
    TrainingDatasetItem,
)
from app.schemas import (
    AnalysisParameters,
    AnalysisResponse,
    AutoValidationResponse,
    DrainRecommendation,
    HealthResponse,
    ImageMetadata,
    NormalizedCoordinate,
    PixelCoordinate,
    TrainingDatasetItemResponse,
    TrainingDrainPosition,
)
from app.services.analysis_runner import AttemptOutcome, run_auto_validated_analysis
from app.services.image_io import InvalidTerrainImage, prepare_dem
from app.services.storage import (
    StoredAsset,
    asset_url,
    ensure_storage,
    read_asset,
    save_original,
    save_png,
)
from app.services.visualization import render_overlay, render_watermap


ALGORITHM = f"{model.name} + D8 자동 검증 (MVP v0.3)"


def _initialize_runtime() -> None:
    ensure_storage()
    init_db()
    with SessionLocal() as session:
        if session.get(ModelVersion, model.version) is None:
            session.add(
                ModelVersion(
                    version=model.version,
                    name=model.name,
                    is_active=True,
                )
            )
            session.commit()


_initialize_runtime()

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "등고선/DEM 입력으로 워터맵과 배수구 위치를 생성하고, 물리 규칙·D8 모의 배수·"
        "입력 변형 일관성으로 자동 검증한 뒤 입력과 결과를 MySQL에 연결해 저장합니다."
    ),
)
app.mount("/files", StaticFiles(directory=STORAGE_ROOT), name="analysis-files")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get(f"{API_PREFIX}/health", response_model=HealthResponse, tags=["system"])
async def health(db: Session = Depends(get_db)) -> HealthResponse:
    db.execute(text("SELECT 1"))
    return HealthResponse(status="ok", service=APP_NAME, version=APP_VERSION)


@app.post(
    f"{API_PREFIX}/drainage/analyze",
    response_model=AnalysisResponse,
    tags=["drainage"],
    summary="워터맵·배수구 위치를 생성하고 자동 검증·저장합니다",
)
async def analyze_drainage(
    image: Annotated[UploadFile, File(description="등고선 또는 grayscale DEM 이미지")],
    drain_count: Annotated[int, Form(ge=1, le=10)] = 3,
    rainfall_mm_per_hour: Annotated[float, Form(gt=0, le=1000)] = 50.0,
    runoff_coefficient: Annotated[float, Form(gt=0, le=1)] = 0.8,
    cell_size_m: Annotated[float, Form(gt=0, le=10000)] = 1.0,
    minimum_spacing_ratio: Annotated[float, Form(ge=0.01, le=0.5)] = 0.10,
    blur_radius: Annotated[float, Form(ge=0, le=10)] = 1.2,
    high_is_bright: Annotated[bool, Form()] = True,
    include_overlay: Annotated[bool, Form()] = False,
    db: Session = Depends(get_db),
) -> AnalysisResponse:
    started_at = time.perf_counter()
    data = await _read_upload(image)

    parameters = AnalysisParameters(
        drain_count=drain_count,
        rainfall_mm_per_hour=rainfall_mm_per_hour,
        runoff_coefficient=runoff_coefficient,
        original_cell_size_m=cell_size_m,
        minimum_spacing_ratio=minimum_spacing_ratio,
        blur_radius=blur_radius,
    )
    parameters_payload = {
        **parameters.model_dump(),
        "high_is_bright": high_is_bright,
    }
    parameters_json = json.dumps(
        parameters_payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    image_hash = hashlib.sha256(data).hexdigest()
    cache_key = hashlib.sha256(
        f"{image_hash}:{model.version}:{VALIDATOR_VERSION}:{parameters_json}".encode()
    ).hexdigest()

    cached = db.scalar(
        select(AnalysisCase)
        .where(
            AnalysisCase.cache_key == cache_key,
            AnalysisCase.status == "COMPLETED",
        )
        .order_by(AnalysisCase.created_at.desc())
    )
    if cached is not None:
        return _response_from_case(cached, include_overlay=include_overlay, cache_hit=True)

    try:
        dem = await run_in_threadpool(
            prepare_dem,
            data,
            max_pixels=MAX_IMAGE_PIXELS,
            analysis_max_dimension=ANALYSIS_MAX_DIMENSION,
            high_is_bright=high_is_bright,
            blur_radius=blur_radius,
        )
    except InvalidTerrainImage as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    case_id = str(uuid4())
    analysis_case = AnalysisCase(
        id=case_id,
        cache_key=cache_key,
        image_hash=image_hash,
        original_filename=image.filename or "upload",
        requested_drain_count=drain_count,
        status="PROCESSING",
        verification_status="AI_GENERATED",
        model_version=model.version,
        algorithm=ALGORITHM,
        parameters_json=parameters_json,
    )
    db.add(analysis_case)
    db.commit()

    try:
        analysis_case.verification_status = "AUTO_VALIDATING"
        db.commit()
        outcome = await run_in_threadpool(
            run_auto_validated_analysis,
            model,
            dem.elevation,
            drain_count=drain_count,
            minimum_spacing_ratio=minimum_spacing_ratio,
        )
        selected = outcome.selected
        recommendations = _build_recommendations(
            selected.result.drains,
            original_width=dem.original_width,
            original_height=dem.original_height,
            scale_x=dem.scale_x,
            scale_y=dem.scale_y,
            analysis_cell_area=(cell_size_m**2) * dem.scale_x * dem.scale_y,
            rainfall_mm_per_hour=rainfall_mm_per_hour,
            runoff_coefficient=runoff_coefficient,
        )

        watermap_png = await run_in_threadpool(
            render_watermap,
            selected.result,
            output_size=(dem.original_width, dem.original_height),
        )
        overlay_png = await run_in_threadpool(
            render_overlay,
            selected.result,
            output_size=(dem.original_width, dem.original_height),
            scale_x=dem.scale_x,
            scale_y=dem.scale_y,
        )
        input_asset, watermap_asset, overlay_asset = _save_case_assets(
            case_id=case_id,
            data=data,
            content_type=image.content_type,
            width=dem.original_width,
            height=dem.original_height,
            watermap_png=watermap_png,
            overlay_png=overlay_png,
        )
        for asset in (input_asset, watermap_asset, overlay_asset):
            db.add(_asset_record(case_id, asset))

        for attempt in outcome.attempts:
            _persist_attempt(
                db,
                analysis_case=analysis_case,
                attempt=attempt,
                original_width=dem.original_width,
                original_height=dem.original_height,
                scale_x=dem.scale_x,
                scale_y=dem.scale_y,
                analysis_cell_area=(cell_size_m**2) * dem.scale_x * dem.scale_y,
                rainfall_mm_per_hour=rainfall_mm_per_hour,
                runoff_coefficient=runoff_coefficient,
                selected_attempt_number=selected.attempt_number,
            )

        verification_status = "AUTO_VERIFIED" if outcome.verified else "AUTO_REJECTED"
        training_status = "READY" if outcome.verified else "EXCLUDED"
        validation_response = _validation_response(selected)
        warnings = [
            "토질, 침투율, 기존 관로, 배수구 용량, 건축물·도로 장애물은 아직 반영하지 않습니다.",
        ]
        if IS_D8_FALLBACK:
            warnings.insert(
                0,
                "현재 분석 모델은 학습 AI를 연결하기 전 사용하는 D8 기반 대리 모델입니다.",
            )
        if not outcome.verified:
            warnings.append(
                "자동 검증을 통과하지 못해 재학습 데이터에서는 제외했습니다: "
                + " ".join(selected.validation.failure_reasons)
            )
        if not high_is_bright:
            warnings.append("이번 요청에서는 어두울수록 고도가 높은 것으로 해석했습니다.")

        processing_time_ms = max(1, round((time.perf_counter() - started_at) * 1000))
        response = AnalysisResponse(
            request_id=case_id,
            analysis_case_id=case_id,
            status="COMPLETED",
            verification_status=verification_status,
            validation_confidence=selected.validation.confidence,
            retry_count=len(outcome.attempts) - 1,
            selected_attempt_number=selected.attempt_number,
            training_status=training_status,
            automatic_validation=validation_response,
            model_version=model.version,
            cache_hit=False,
            algorithm=ALGORITHM,
            image=ImageMetadata(
                filename=image.filename or "upload",
                original_width=dem.original_width,
                original_height=dem.original_height,
                analysis_width=dem.analysis_width,
                analysis_height=dem.analysis_height,
                input_mode="grayscale_dem",
                elevation_convention="bright_is_high" if high_is_bright else "dark_is_high",
            ),
            parameters=parameters,
            drains=recommendations,
            input_image_url=input_asset.url,
            water_map_url=watermap_asset.url,
            overlay_image_url=overlay_asset.url,
            processing_time_ms=processing_time_ms,
            assumptions=[
                "각 픽셀의 밝기값이 상대 고도를 나타냅니다.",
                "각 셀의 유출수는 인접 8방향 중 가장 가파른 하강 방향으로 이동합니다.",
                "강우가 전체 영역에 균일하게 내리고 유출계수가 일정하다고 가정합니다.",
                "좌표 원점 (0, 0)은 이미지 왼쪽 위입니다.",
            ],
            warnings=warnings,
            overlay_png_base64=(
                base64.b64encode(overlay_png).decode("ascii") if include_overlay else None
            ),
        )

        analysis_case.status = "COMPLETED"
        analysis_case.verification_status = verification_status
        analysis_case.validation_confidence = selected.validation.confidence
        analysis_case.retry_count = len(outcome.attempts) - 1
        analysis_case.selected_attempt_number = selected.attempt_number
        analysis_case.processing_time_ms = processing_time_ms
        analysis_case.response_json = response.model_dump_json(
            exclude={"overlay_png_base64"}
        )
        if outcome.verified:
            db.add(
                TrainingDatasetItem(
                    analysis_case_id=case_id,
                    status="READY",
                    input_file_path=input_asset.relative_path,
                    target_watermap_path=watermap_asset.relative_path,
                    target_positions_json=json.dumps(
                        [
                            {
                                "order": item.rank,
                                "x": item.normalized.x,
                                "y": item.normalized.y,
                                "confidence": item.suitability_score,
                            }
                            for item in recommendations
                        ]
                    ),
                    quality_score=selected.validation.confidence,
                    source="AUTO_VERIFIED",
                    validator_version=VALIDATOR_VERSION,
                )
            )
        db.commit()
        return response
    except ValueError as exc:
        analysis_case.status = "FAILED"
        analysis_case.verification_status = "AUTO_REJECTED"
        analysis_case.error_message = str(exc)
        db.commit()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        analysis_case.status = "FAILED"
        analysis_case.error_message = str(exc)
        db.commit()
        raise


@app.get(
    f"{API_PREFIX}/drainage/analyses",
    response_model=list[AnalysisResponse],
    tags=["drainage"],
    summary="최근 완료된 분석 사례를 조회합니다",
)
async def list_analyses(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    db: Session = Depends(get_db),
) -> list[AnalysisResponse]:
    cases = db.scalars(
        select(AnalysisCase)
        .where(AnalysisCase.status == "COMPLETED")
        .order_by(AnalysisCase.created_at.desc())
        .limit(limit)
    ).all()
    return [
        _response_from_case(case, include_overlay=False, cache_hit=False)
        for case in cases
    ]


@app.get(
    f"{API_PREFIX}/drainage/analyses/{{analysis_case_id}}",
    response_model=AnalysisResponse,
    tags=["drainage"],
    summary="저장된 입력·워터맵·배수구 결과를 조회합니다",
)
async def get_analysis(
    analysis_case_id: str,
    include_overlay: bool = False,
    db: Session = Depends(get_db),
) -> AnalysisResponse:
    case = db.get(AnalysisCase, analysis_case_id)
    if case is None or case.status != "COMPLETED" or not case.response_json:
        raise HTTPException(status_code=404, detail="완료된 분석 사례를 찾을 수 없습니다.")
    return _response_from_case(case, include_overlay=include_overlay, cache_hit=False)


@app.get(
    f"{API_PREFIX}/drainage/analyses/{{analysis_case_id}}/validations",
    response_model=list[AutoValidationResponse],
    tags=["validation"],
    summary="자동 검증과 재분석 이력을 조회합니다",
)
async def list_validations(
    analysis_case_id: str,
    db: Session = Depends(get_db),
) -> list[AutoValidationResponse]:
    case = db.get(AnalysisCase, analysis_case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="분석 사례를 찾을 수 없습니다.")
    attempts = sorted(case.attempts, key=lambda item: item.attempt_number)
    return [_validation_from_record(item) for item in attempts]


@app.get(
    f"{API_PREFIX}/training-dataset/items",
    response_model=list[TrainingDatasetItemResponse],
    tags=["training"],
    summary="자동 검증을 통과한 재학습용 데이터 묶음을 조회합니다",
)
async def list_training_items(
    status: str = "READY",
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    db: Session = Depends(get_db),
) -> list[TrainingDatasetItemResponse]:
    items = db.scalars(
        select(TrainingDatasetItem)
        .where(TrainingDatasetItem.status == status.upper())
        .order_by(TrainingDatasetItem.created_at.asc())
        .limit(limit)
    ).all()
    return [
        TrainingDatasetItemResponse(
            item_id=item.id,
            analysis_case_id=item.analysis_case_id,
            status=item.status,
            input_image_url=asset_url(item.input_file_path),
            target_watermap_url=asset_url(item.target_watermap_path),
            target_positions=[
                TrainingDrainPosition.model_validate(position)
                for position in json.loads(item.target_positions_json)
            ],
            source_model_version=item.analysis.model_version,
            validator_version=item.validator_version,
            quality_score=item.quality_score,
            verified_at=item.verified_at.isoformat(),
        )
        for item in items
    ]


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {
        "message": APP_NAME,
        "swagger_ui": "/docs",
        "health": f"{API_PREFIX}/health",
        "analyze": f"{API_PREFIX}/drainage/analyze",
    }


async def _read_upload(image: UploadFile) -> bytes:
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
    return data


def _save_case_assets(
    *,
    case_id: str,
    data: bytes,
    content_type: str | None,
    width: int,
    height: int,
    watermap_png: bytes,
    overlay_png: bytes,
) -> tuple[StoredAsset, StoredAsset, StoredAsset]:
    return (
        save_original(
            case_id=case_id,
            data=data,
            content_type=content_type,
            width=width,
            height=height,
        ),
        save_png(
            case_id=case_id,
            filename="water_map.png",
            kind="WATER_MAP",
            data=watermap_png,
        ),
        save_png(
            case_id=case_id,
            filename="drain_overlay.png",
            kind="DRAIN_OVERLAY",
            data=overlay_png,
        ),
    )


def _persist_attempt(
    db: Session,
    *,
    analysis_case: AnalysisCase,
    attempt: AttemptOutcome,
    original_width: int,
    original_height: int,
    scale_x: float,
    scale_y: float,
    analysis_cell_area: float,
    rainfall_mm_per_hour: float,
    runoff_coefficient: float,
    selected_attempt_number: int,
) -> None:
    recommendations = _build_recommendations(
        attempt.result.drains,
        original_width=original_width,
        original_height=original_height,
        scale_x=scale_x,
        scale_y=scale_y,
        analysis_cell_area=analysis_cell_area,
        rainfall_mm_per_hour=rainfall_mm_per_hour,
        runoff_coefficient=runoff_coefficient,
    )
    attempt_record = AnalysisAttempt(
        analysis_case_id=analysis_case.id,
        attempt_number=attempt.attempt_number,
        status="PASSED" if attempt.validation.passed else "FAILED",
        parameters_json=json.dumps({"strategy": attempt.strategy}),
        positions_json=json.dumps(
            [item.model_dump(mode="json") for item in recommendations]
        ),
    )
    db.add(attempt_record)
    db.flush()
    validation = attempt.validation
    db.add(
        AutomaticValidation(
            analysis_attempt_id=attempt_record.id,
            validator_version=validation.validator_version,
            passed=validation.passed,
            confidence=validation.confidence,
            count_ok=validation.count_ok,
            spacing_ok=validation.spacing_ok,
            boundary_ok=validation.boundary_ok,
            minimum_spacing_ratio_actual=validation.minimum_spacing_ratio_actual,
            mean_suitability=validation.mean_suitability,
            estimated_capture_ratio=validation.estimated_capture_ratio,
            residual_water_ratio=validation.residual_water_ratio,
            stability_score=validation.stability_score,
            failure_reasons_json=json.dumps(validation.failure_reasons),
            metrics_json=json.dumps(validation.metrics()),
        )
    )
    for item in recommendations:
        db.add(
            DrainPosition(
                analysis_case_id=analysis_case.id,
                position_order=item.rank,
                attempt_number=attempt.attempt_number,
                x_normalized=item.normalized.x,
                y_normalized=item.normalized.y,
                x_pixel=item.pixel.x,
                y_pixel=item.pixel.y,
                confidence=item.suitability_score,
                source="AI_PREDICTED",
                is_selected=attempt.attempt_number == selected_attempt_number,
                payload_json=item.model_dump_json(),
            )
        )


def _build_recommendations(
    candidates,
    *,
    original_width: int,
    original_height: int,
    scale_x: float,
    scale_y: float,
    analysis_cell_area: float,
    rainfall_mm_per_hour: float,
    runoff_coefficient: float,
) -> list[DrainRecommendation]:
    recommendations: list[DrainRecommendation] = []
    for rank, candidate in enumerate(candidates, start=1):
        original_x = min(
            original_width - 1,
            max(0, round((candidate.x + 0.5) * scale_x - 0.5)),
        )
        original_y = min(
            original_height - 1,
            max(0, round((candidate.y + 0.5) * scale_y - 0.5)),
        )
        area_m2 = candidate.accumulation_cells * analysis_cell_area
        peak_flow_lps = runoff_coefficient * rainfall_mm_per_hour * area_m2 / 3600.0
        recommendations.append(
            DrainRecommendation(
                rank=rank,
                pixel=PixelCoordinate(x=original_x, y=original_y),
                normalized=NormalizedCoordinate(
                    x=round(original_x / max(original_width - 1, 1), 6),
                    y=round(original_y / max(original_height - 1, 1), 6),
                ),
                elevation_normalized=round(candidate.elevation, 6),
                flow_accumulation_cells=round(candidate.accumulation_cells, 2),
                catchment_area_m2=round(area_m2, 3),
                estimated_peak_flow_lps=round(peak_flow_lps, 3),
                suitability_score=round(candidate.score, 6),
                rationale=(
                    "상류 기여면적이 크고 상대적으로 낮은 지형으로, "
                    "워터맵에서 유출수가 집중되는 후보입니다."
                ),
            )
        )
    return recommendations


def _validation_response(attempt: AttemptOutcome) -> AutoValidationResponse:
    validation = attempt.validation
    return AutoValidationResponse(
        attempt_number=attempt.attempt_number,
        status="PASSED" if validation.passed else "FAILED",
        passed=validation.passed,
        confidence=validation.confidence,
        count_ok=validation.count_ok,
        spacing_ok=validation.spacing_ok,
        boundary_ok=validation.boundary_ok,
        minimum_spacing_ratio_actual=validation.minimum_spacing_ratio_actual,
        mean_suitability=validation.mean_suitability,
        estimated_capture_ratio=validation.estimated_capture_ratio,
        residual_water_ratio=validation.residual_water_ratio,
        stability_score=validation.stability_score,
        failure_reasons=validation.failure_reasons,
        validator_version=validation.validator_version,
    )


def _validation_from_record(attempt: AnalysisAttempt) -> AutoValidationResponse:
    validation = attempt.validation
    return AutoValidationResponse(
        attempt_number=attempt.attempt_number,
        status=attempt.status,
        passed=validation.passed,
        confidence=validation.confidence,
        count_ok=validation.count_ok,
        spacing_ok=validation.spacing_ok,
        boundary_ok=validation.boundary_ok,
        minimum_spacing_ratio_actual=validation.minimum_spacing_ratio_actual,
        mean_suitability=validation.mean_suitability,
        estimated_capture_ratio=validation.estimated_capture_ratio,
        residual_water_ratio=validation.residual_water_ratio,
        stability_score=validation.stability_score,
        failure_reasons=json.loads(validation.failure_reasons_json),
        validator_version=validation.validator_version,
    )


def _asset_record(case_id: str, asset: StoredAsset) -> AnalysisFile:
    return AnalysisFile(
        analysis_case_id=case_id,
        kind=asset.kind,
        storage_path=asset.relative_path,
        mime_type=asset.mime_type,
        file_hash=asset.file_hash,
        width=asset.width,
        height=asset.height,
    )


def _case_asset(case: AnalysisCase, kind: str) -> AnalysisFile:
    for asset in case.files:
        if asset.kind == kind:
            return asset
    raise HTTPException(status_code=500, detail=f"저장된 {kind} 파일이 없습니다.")


def _response_from_case(
    case: AnalysisCase, *, include_overlay: bool, cache_hit: bool
) -> AnalysisResponse:
    if not case.response_json:
        raise HTTPException(status_code=404, detail="저장된 분석 결과가 없습니다.")
    payload = json.loads(case.response_json)
    payload["status"] = case.status
    payload["verification_status"] = case.verification_status
    payload["validation_confidence"] = case.validation_confidence
    payload["retry_count"] = case.retry_count
    payload["selected_attempt_number"] = case.selected_attempt_number
    payload["cache_hit"] = cache_hit
    payload["processing_time_ms"] = case.processing_time_ms or 0
    if include_overlay:
        overlay = _case_asset(case, "DRAIN_OVERLAY")
        payload["overlay_png_base64"] = base64.b64encode(
            read_asset(overlay.storage_path)
        ).decode("ascii")
    else:
        payload["overlay_png_base64"] = None
    return AnalysisResponse.model_validate(payload)
