from __future__ import annotations

import csv
import hashlib
import io
import json
import time
from contextlib import asynccontextmanager
from pathlib import PurePosixPath
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.ai.runtime import DrainageModelRuntime, ModelContractError, ModelUnavailable
from app.config import settings
from app.database import database_is_ready, get_db, initialize_database
from app.models import (
    DrainScenario,
    ModelVersion,
    PredictionResult,
    ProcessingRun,
    ScenarioDrain,
    TerrainUpload,
)
from app.schemas import (
    AssetResponse,
    DrainInput,
    LiveHealthResponse,
    PredictionRequest,
    PredictionResponse,
    ReadyHealthResponse,
    ScenarioRequest,
    ScenarioResponse,
    UploadResponse,
)
from app.services.metrics import (
    array_from_npz,
    array_to_npz,
    depth_metrics,
    render_depth_png,
)
from app.services.preprocessing import (
    InvalidTerrainInput,
    preprocess_terrain,
    preprocessing_cache_key,
)
from app.services.scenario import (
    ScenarioEvaluatorUnavailable,
    UserDrain,
    build_scenario_evaluator,
)
from app.services.singleflight import KeyedLocks
from app.services.storage import FileStorage

ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/tiff",
}
SUFFIX_BY_CONTENT_TYPE = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/tiff": ".tiff",
}

storage = FileStorage(settings.storage_root)
storage.ensure()
runtime = DrainageModelRuntime(
    checkpoint_path=settings.checkpoint_path,
    model_version=settings.model_version,
    image_size=settings.model_image_size,
    concurrency=settings.inference_concurrency,
    wait_timeout_s=settings.inference_wait_timeout_s,
    use_fp16=settings.use_fp16,
    warmup=settings.warmup_model,
)
scenario_evaluator = build_scenario_evaluator(settings.scenario_evaluator)
preprocess_flights = KeyedLocks()
prediction_flights = KeyedLocks()
scenario_flights = KeyedLocks()
DatabaseSession = Annotated[Session, Depends(get_db)]


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    await run_in_threadpool(runtime.try_load)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Upload/preprocess terrain, predict one AI drainage map, and evaluate only "
        "user-supplied drain scenarios. This API never auto-detects drain positions."
    ),
    lifespan=lifespan,
)
app.mount("/files", StaticFiles(directory=settings.storage_root), name="v2-files")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
)


@app.get(
    f"{settings.api_prefix}/health/live",
    response_model=LiveHealthResponse,
    tags=["system"],
)
def live() -> LiveHealthResponse:
    return LiveHealthResponse(
        status="ok", service=settings.app_name, version=settings.app_version
    )


@app.get(
    f"{settings.api_prefix}/health/ready",
    response_model=ReadyHealthResponse,
    tags=["system"],
)
def ready(db: DatabaseSession):
    database_status = "ready"
    storage_status = "ready" if storage.is_writable() else "unavailable"
    detail = None
    try:
        database_is_ready(db)
    except SQLAlchemyError as exc:
        database_status = "unavailable"
        detail = f"database: {type(exc).__name__}"
    model_status = runtime.try_load()
    ready_now = (
        database_status == "ready"
        and storage_status == "ready"
        and model_status.available
    )
    body = ReadyHealthResponse(
        status="ready" if ready_now else "not_ready",
        database=database_status,
        storage=storage_status,
        ai_model="ready" if model_status.available else "unavailable",
        model_version=model_status.model_version,
        checkpoint_hash=model_status.checkpoint_hash,
        checkpoint_name=model_status.checkpoint_name,
        device=model_status.device,
        condition_dimension=model_status.condition_dimension,
        fallback=False,
        scenario_evaluator=scenario_evaluator.version,
        detail=detail or model_status.error,
    )
    if ready_now:
        return body
    return JSONResponse(status_code=503, content=body.model_dump(mode="json"))


@app.post(
    f"{settings.api_prefix}/upload-terrain",
    response_model=UploadResponse,
    tags=["terrain"],
)
def upload_terrain(
    image: Annotated[UploadFile, File(description="PNG/JPEG/WEBP/TIFF terrain image")],
    db: DatabaseSession,
    input_type: Annotated[str, Form()] = "auto",
    high_is_bright: Annotated[bool, Form()] = True,
    blur_radius: Annotated[float, Form(ge=0.0, le=10.0)] = 1.2,
    cell_size_m: Annotated[float | None, Form(gt=0.0, le=10000.0)] = None,
    crs: Annotated[str | None, Form(max_length=100)] = None,
) -> UploadResponse:
    request_id = str(uuid4())
    started_at = time.perf_counter()
    content_type = image.content_type or "application/octet-stream"
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="supported types: PNG, JPEG, WEBP, TIFF")
    data = image.file.read(settings.max_upload_bytes + 1)
    if not data:
        raise HTTPException(status_code=400, detail="uploaded image is empty")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="uploaded image is too large")

    image_hash = hashlib.sha256(data).hexdigest()
    cache_key = preprocessing_cache_key(
        image_hash=image_hash,
        preprocessing_version=settings.preprocessing_version,
        input_type=input_type,
        high_is_bright=high_is_bright,
        blur_radius=blur_radius,
        model_image_size=settings.model_image_size,
        cell_size_m=cell_size_m,
        crs=crs,
    )
    with preprocess_flights.acquire(cache_key):
        cached = db.scalar(
            select(TerrainUpload).where(
                TerrainUpload.preprocess_key == cache_key,
                TerrainUpload.status == "COMPLETED",
            )
        )
        if cached is not None and storage.exists(cached.model_input_path):
            elapsed = _elapsed_ms(started_at)
            _record_run(
                db,
                request_id=request_id,
                resource_type="UPLOAD",
                resource_id=cached.id,
                stage="PREPROCESS",
                status="COMPLETED",
                latency_ms=elapsed,
                cache_hit=True,
            )
            return _upload_response(cached, request_id, True, elapsed)

        try:
            prepared = preprocess_terrain(
                data,
                input_type=input_type,
                high_is_bright=high_is_bright,
                blur_radius=blur_radius,
                max_pixels=settings.max_image_pixels,
                analysis_max_dimension=settings.analysis_max_dimension,
                model_image_size=settings.model_image_size,
            )
        except InvalidTerrainInput as exc:
            _record_run(
                db,
                request_id=request_id,
                resource_type="UPLOAD",
                resource_id=None,
                stage="PREPROCESS",
                status="FAILED",
                latency_ms=_elapsed_ms(started_at),
                error_code="INVALID_TERRAIN",
                details={"message": str(exc)},
            )
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        upload_id = str(uuid4())
        base_path = PurePosixPath("terrain", upload_id)
        suffix = SUFFIX_BY_CONTENT_TYPE[content_type]
        saved_paths: list[str] = []
        try:
            original = storage.save(
                (base_path / "original" / f"input{suffix}").as_posix(), data, content_type
            )
            saved_paths.append(original.relative_path)
            model_input = storage.save(
                (
                    base_path
                    / "preprocessing"
                    / settings.preprocessing_version
                    / "model_input.png"
                ).as_posix(),
                prepared.model_input_png,
                "image/png",
            )
            saved_paths.append(model_input.relative_path)
            elevation = None
            if prepared.elevation_npz is not None:
                elevation = storage.save(
                    (
                        base_path
                        / "preprocessing"
                        / settings.preprocessing_version
                        / "elevation.npz"
                    ).as_posix(),
                    prepared.elevation_npz,
                    "application/x-npz",
                )
                saved_paths.append(elevation.relative_path)

            record = TerrainUpload(
                id=upload_id,
                preprocess_key=cache_key,
                image_hash=image_hash,
                original_filename=image.filename or f"input{suffix}",
                content_type=content_type,
                requested_input_type=input_type,
                resolved_input_type=prepared.resolved_input_type,
                high_is_bright=high_is_bright,
                blur_radius=blur_radius,
                cell_size_m=cell_size_m,
                crs=crs,
                original_width=prepared.original_width,
                original_height=prepared.original_height,
                model_width=prepared.model_width,
                model_height=prepared.model_height,
                quality_score=prepared.quality_score,
                quality_report=prepared.quality_report,
                warnings=list(prepared.warnings),
                preprocessing_version=settings.preprocessing_version,
                original_path=original.relative_path,
                original_hash=original.sha256,
                model_input_path=model_input.relative_path,
                model_input_hash=model_input.sha256,
                elevation_path=elevation.relative_path if elevation else None,
                elevation_hash=elevation.sha256 if elevation else None,
                status="COMPLETED",
            )
            db.add(record)
            db.commit()
        except IntegrityError:
            db.rollback()
            for path in saved_paths:
                storage.delete(path)
            existing = db.scalar(
                select(TerrainUpload).where(TerrainUpload.preprocess_key == cache_key)
            )
            if existing is None:
                raise
            elapsed = _elapsed_ms(started_at)
            return _upload_response(existing, request_id, True, elapsed)
        except Exception:
            db.rollback()
            for path in saved_paths:
                storage.delete(path)
            raise

        elapsed = _elapsed_ms(started_at)
        _record_run(
            db,
            request_id=request_id,
            resource_type="UPLOAD",
            resource_id=upload_id,
            stage="PREPROCESS",
            status="COMPLETED",
            latency_ms=elapsed,
            details={"quality_score": prepared.quality_score},
        )
        return _upload_response(record, request_id, False, elapsed)


@app.post(
    f"{settings.api_prefix}/predict-drainage-map",
    response_model=PredictionResponse,
    tags=["prediction"],
)
def predict_drainage_map(
    body: PredictionRequest, db: DatabaseSession
) -> PredictionResponse:
    request_id = str(uuid4())
    started_at = time.perf_counter()
    upload = db.get(TerrainUpload, body.upload_id)
    if upload is None or upload.status != "COMPLETED":
        raise HTTPException(status_code=404, detail="completed terrain upload not found")
    model_status = runtime.try_load()
    if not model_status.available or not model_status.checkpoint_hash:
        _record_run(
            db,
            request_id=request_id,
            resource_type="PREDICTION",
            resource_id=None,
            stage="INFERENCE",
            status="FAILED",
            latency_ms=_elapsed_ms(started_at),
            error_code="AI_MODEL_UNAVAILABLE",
            details={"message": model_status.error},
        )
        raise HTTPException(
            status_code=503,
            detail={"code": "AI_MODEL_UNAVAILABLE", "message": model_status.error},
        )

    cache_key = _hash_payload(
        {
            "upload_id": upload.id,
            "model_input_hash": upload.model_input_hash,
            "model_version": model_status.model_version,
            "checkpoint_hash": model_status.checkpoint_hash,
            "rain_mm": body.rain_mm,
            "time_s": body.time_s,
        }
    )
    with prediction_flights.acquire(cache_key):
        cached = db.scalar(
            select(PredictionResult).where(
                PredictionResult.cache_key == cache_key,
                PredictionResult.status == "COMPLETED",
            )
        )
        if cached is not None and cached.water_map_path and storage.exists(cached.water_map_path):
            elapsed = _elapsed_ms(started_at)
            _record_run(
                db,
                request_id=request_id,
                resource_type="PREDICTION",
                resource_id=cached.id,
                stage="INFERENCE",
                status="COMPLETED",
                latency_ms=elapsed,
                cache_hit=True,
            )
            return _prediction_response(cached, request_id, True, elapsed)

        result_id = str(uuid4())
        saved_paths: list[str] = []
        try:
            prediction = runtime.predict(
                storage.read(upload.model_input_path),
                rain_mm=body.rain_mm,
                time_s=body.time_s,
            )
            metrics = depth_metrics(
                prediction.depth_m,
                threshold_m=settings.depth_threshold_m,
                cell_size_m=upload.cell_size_m,
            )
            water_map = storage.save(
                f"predictions/{result_id}/water_map.png",
                prediction.water_map_png,
                "image/png",
            )
            saved_paths.append(water_map.relative_path)
            depth_array = storage.save(
                f"predictions/{result_id}/depth_m.npz",
                array_to_npz(prediction.depth_m),
                "application/x-npz",
            )
            saved_paths.append(depth_array.relative_path)
            _register_model_version(db, prediction.checkpoint_hash, prediction.device)
            record = PredictionResult(
                id=result_id,
                cache_key=cache_key,
                upload_id=upload.id,
                model_version=settings.model_version,
                checkpoint_hash=prediction.checkpoint_hash,
                rain_mm=body.rain_mm,
                time_s=body.time_s,
                status="COMPLETED",
                water_map_path=water_map.relative_path,
                water_map_hash=water_map.sha256,
                depth_array_path=depth_array.relative_path,
                depth_array_hash=depth_array.sha256,
                metrics=metrics,
                inference_time_ms=prediction.inference_time_ms,
                processing_time_ms=_elapsed_ms(started_at),
            )
            db.add(record)
            db.commit()
        except (ModelUnavailable, ModelContractError) as exc:
            db.rollback()
            for path in saved_paths:
                storage.delete(path)
            _record_run(
                db,
                request_id=request_id,
                resource_type="PREDICTION",
                resource_id=None,
                stage="INFERENCE",
                status="FAILED",
                latency_ms=_elapsed_ms(started_at),
                error_code="AI_MODEL_UNAVAILABLE",
                details={"message": str(exc)},
            )
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except IntegrityError:
            db.rollback()
            for path in saved_paths:
                storage.delete(path)
            existing = db.scalar(
                select(PredictionResult).where(PredictionResult.cache_key == cache_key)
            )
            if existing is None or existing.status != "COMPLETED":
                raise
            elapsed = _elapsed_ms(started_at)
            return _prediction_response(existing, request_id, True, elapsed)
        except Exception:
            db.rollback()
            for path in saved_paths:
                storage.delete(path)
            raise

        elapsed = _elapsed_ms(started_at)
        _record_run(
            db,
            request_id=request_id,
            resource_type="PREDICTION",
            resource_id=result_id,
            stage="INFERENCE",
            status="COMPLETED",
            latency_ms=elapsed,
            details={"inference_time_ms": prediction.inference_time_ms},
        )
        return _prediction_response(record, request_id, False, elapsed)


@app.post(
    f"{settings.api_prefix}/evaluate-drain-settings",
    response_model=ScenarioResponse,
    tags=["scenario"],
)
def evaluate_drain_settings(
    body: ScenarioRequest, db: DatabaseSession
) -> ScenarioResponse:
    request_id = str(uuid4())
    started_at = time.perf_counter()
    prediction = db.get(PredictionResult, body.result_id)
    if prediction is None or prediction.status != "COMPLETED" or not prediction.depth_array_path:
        raise HTTPException(status_code=404, detail="completed prediction result not found")
    upload = db.get(TerrainUpload, prediction.upload_id)
    if upload is None:
        raise HTTPException(status_code=500, detail="prediction upload metadata is missing")
    duration_s = body.duration_s if body.duration_s is not None else prediction.time_s
    if duration_s <= 0:
        raise HTTPException(
            status_code=422,
            detail="duration_s must be supplied when the prediction time_s is zero",
        )
    canonical_drains = sorted(
        body.drains,
        key=lambda item: (
            item.x,
            item.y,
            item.capacity_lps,
            item.capture_radius_m,
            item.drain_type,
        ),
    )
    cache_key = _hash_payload(
        {
            "prediction_id": prediction.id,
            "prediction_depth_hash": prediction.depth_array_hash,
            "evaluator_version": scenario_evaluator.version,
            "duration_s": duration_s,
            "cell_size_m": upload.cell_size_m,
            "drains": [item.model_dump(mode="json") for item in canonical_drains],
        }
    )
    with scenario_flights.acquire(cache_key):
        cached = db.scalar(
            select(DrainScenario).where(
                DrainScenario.cache_key == cache_key,
                DrainScenario.status == "COMPLETED",
            )
        )
        if cached is not None and cached.scenario_map_path and storage.exists(cached.scenario_map_path):
            elapsed = _elapsed_ms(started_at)
            _record_run(
                db,
                request_id=request_id,
                resource_type="SCENARIO",
                resource_id=cached.id,
                stage="SCENARIO_EVALUATION",
                status="COMPLETED",
                latency_ms=elapsed,
                cache_hit=True,
            )
            return _scenario_response(cached, request_id, True, elapsed)

        baseline_depth = array_from_npz(storage.read(prediction.depth_array_path))
        evaluator_drains = [
            UserDrain(
                x=item.x,
                y=item.y,
                capacity_lps=item.capacity_lps,
                capture_radius_m=item.capture_radius_m,
                drain_type=item.drain_type,
            )
            for item in canonical_drains
        ]
        try:
            evaluation = scenario_evaluator.evaluate(
                baseline_depth,
                evaluator_drains,
                duration_s=duration_s,
                cell_size_m=upload.cell_size_m,
                depth_threshold_m=settings.depth_threshold_m,
            )
        except ScenarioEvaluatorUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        scenario_id = str(uuid4())
        saved_paths: list[str] = []
        try:
            scenario_map = storage.save(
                f"scenarios/{scenario_id}/scenario_map.png",
                render_depth_png(evaluation.updated_depth_m),
                "image/png",
            )
            saved_paths.append(scenario_map.relative_path)
            depth_array = storage.save(
                f"scenarios/{scenario_id}/depth_m.npz",
                array_to_npz(evaluation.updated_depth_m),
                "application/x-npz",
            )
            saved_paths.append(depth_array.relative_path)
            record = DrainScenario(
                id=scenario_id,
                cache_key=cache_key,
                prediction_id=prediction.id,
                evaluator_version=evaluation.evaluator_version,
                duration_s=duration_s,
                status="COMPLETED",
                baseline_metrics=evaluation.baseline_metrics,
                scenario_metrics=evaluation.scenario_metrics,
                warnings=list(evaluation.warnings),
                scenario_map_path=scenario_map.relative_path,
                scenario_map_hash=scenario_map.sha256,
                depth_array_path=depth_array.relative_path,
                depth_array_hash=depth_array.sha256,
                processing_time_ms=_elapsed_ms(started_at),
            )
            db.add(record)
            for order, item in enumerate(canonical_drains, start=1):
                db.add(
                    ScenarioDrain(
                        scenario_id=scenario_id,
                        position_order=order,
                        x_normalized=item.x,
                        y_normalized=item.y,
                        capacity_lps=item.capacity_lps,
                        capture_radius_m=item.capture_radius_m,
                        drain_type=item.drain_type,
                        source="USER",
                    )
                )
            db.commit()
        except IntegrityError:
            db.rollback()
            for path in saved_paths:
                storage.delete(path)
            existing = db.scalar(
                select(DrainScenario).where(DrainScenario.cache_key == cache_key)
            )
            if existing is None or existing.status != "COMPLETED":
                raise
            elapsed = _elapsed_ms(started_at)
            return _scenario_response(existing, request_id, True, elapsed)
        except Exception:
            db.rollback()
            for path in saved_paths:
                storage.delete(path)
            raise

        elapsed = _elapsed_ms(started_at)
        _record_run(
            db,
            request_id=request_id,
            resource_type="SCENARIO",
            resource_id=scenario_id,
            stage="SCENARIO_EVALUATION",
            status="COMPLETED",
            latency_ms=elapsed,
            details={"drain_count": len(canonical_drains)},
        )
        return _scenario_response(record, request_id, False, elapsed)


@app.get(
    f"{settings.api_prefix}/result/{{result_id}}",
    response_model=PredictionResponse,
    tags=["prediction"],
)
def get_result(result_id: str, db: DatabaseSession) -> PredictionResponse:
    record = db.get(PredictionResult, result_id)
    if record is None or record.status != "COMPLETED":
        raise HTTPException(status_code=404, detail="completed prediction result not found")
    return _prediction_response(record, str(uuid4()), False, 0)


@app.get(
    f"{settings.api_prefix}/scenario/{{scenario_id}}",
    response_model=ScenarioResponse,
    tags=["scenario"],
)
def get_scenario(scenario_id: str, db: DatabaseSession) -> ScenarioResponse:
    record = db.get(DrainScenario, scenario_id)
    if record is None or record.status != "COMPLETED":
        raise HTTPException(status_code=404, detail="completed scenario not found")
    return _scenario_response(record, str(uuid4()), False, 0)


@app.get(
    f"{settings.api_prefix}/scenario/{{scenario_id}}/export.csv",
    tags=["scenario"],
)
def export_scenario_csv(scenario_id: str, db: DatabaseSession) -> Response:
    record = db.get(DrainScenario, scenario_id)
    if record is None or record.status != "COMPLETED":
        raise HTTPException(status_code=404, detail="completed scenario not found")
    drains = sorted(record.drains, key=lambda item: item.position_order)
    output = io.StringIO()
    fieldnames = [
        "scenario_id",
        "position_order",
        "x_normalized",
        "y_normalized",
        "capacity_lps",
        "capture_radius_m",
        "drain_type",
        "source",
        "flooded_area_reduction_percent",
        "drained_volume_m3",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    metrics = record.scenario_metrics or {}
    for drain in drains:
        writer.writerow(
            {
                "scenario_id": record.id,
                "position_order": drain.position_order,
                "x_normalized": drain.x_normalized,
                "y_normalized": drain.y_normalized,
                "capacity_lps": drain.capacity_lps,
                "capture_radius_m": drain.capture_radius_m,
                "drain_type": drain.drain_type,
                "source": drain.source,
                "flooded_area_reduction_percent": metrics.get(
                    "flooded_area_reduction_percent"
                ),
                "drained_volume_m3": metrics.get("drained_volume_m3"),
            }
        )
    return Response(
        content=output.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="scenario-{record.id}.csv"'},
    )


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "message": settings.app_name,
        "docs": "/docs",
        "live": f"{settings.api_prefix}/health/live",
        "ready": f"{settings.api_prefix}/health/ready",
    }


def _upload_response(
    record: TerrainUpload,
    request_id: str,
    cache_hit: bool,
    processing_time_ms: int,
) -> UploadResponse:
    assets = [
        AssetResponse(
            kind="ORIGINAL",
            url=storage.public_url(record.original_path),
            sha256=record.original_hash,
        ),
        AssetResponse(
            kind="MODEL_INPUT",
            url=storage.public_url(record.model_input_path),
            sha256=record.model_input_hash,
        ),
    ]
    if record.elevation_path and record.elevation_hash:
        assets.append(
            AssetResponse(
                kind="ELEVATION_ARRAY",
                url=storage.public_url(record.elevation_path),
                sha256=record.elevation_hash,
            )
        )
    return UploadResponse(
        request_id=request_id,
        upload_id=record.id,
        status=record.status,
        cache_hit=cache_hit,
        input_type=record.resolved_input_type,
        original_width=record.original_width,
        original_height=record.original_height,
        model_width=record.model_width,
        model_height=record.model_height,
        quality_score=record.quality_score,
        quality_report=record.quality_report,
        preprocessing_version=record.preprocessing_version,
        warnings=list(record.warnings or []),
        assets=assets,
        processing_time_ms=processing_time_ms,
    )


def _prediction_response(
    record: PredictionResult,
    request_id: str,
    cache_hit: bool,
    processing_time_ms: int,
) -> PredictionResponse:
    if not record.water_map_path or not record.depth_array_path or not record.checkpoint_hash:
        raise HTTPException(status_code=500, detail="prediction assets are incomplete")
    return PredictionResponse(
        request_id=request_id,
        result_id=record.id,
        upload_id=record.upload_id,
        status=record.status,
        cache_hit=cache_hit,
        model_version=record.model_version,
        checkpoint_hash=record.checkpoint_hash,
        rain_mm=record.rain_mm,
        time_s=record.time_s,
        water_map_url=storage.public_url(record.water_map_path),
        depth_array_url=storage.public_url(record.depth_array_path),
        metrics=record.metrics or {},
        inference_time_ms=record.inference_time_ms or 0,
        processing_time_ms=processing_time_ms,
        drains=[],
    )


def _scenario_response(
    record: DrainScenario,
    request_id: str,
    cache_hit: bool,
    processing_time_ms: int,
) -> ScenarioResponse:
    if not record.scenario_map_path or not record.depth_array_path:
        raise HTTPException(status_code=500, detail="scenario assets are incomplete")
    drains = [
        DrainInput(
            x=item.x_normalized,
            y=item.y_normalized,
            capacity_lps=item.capacity_lps,
            capture_radius_m=item.capture_radius_m,
            drain_type=item.drain_type,
        )
        for item in sorted(record.drains, key=lambda item: item.position_order)
    ]
    return ScenarioResponse(
        request_id=request_id,
        scenario_id=record.id,
        result_id=record.prediction_id,
        status=record.status,
        cache_hit=cache_hit,
        evaluator_version=record.evaluator_version,
        drains=drains,
        baseline_metrics=record.baseline_metrics or {},
        scenario_metrics=record.scenario_metrics or {},
        scenario_map_url=storage.public_url(record.scenario_map_path),
        depth_array_url=storage.public_url(record.depth_array_path),
        warnings=list(record.warnings or []),
        processing_time_ms=processing_time_ms,
    )


def _register_model_version(db: Session, checkpoint_hash: str, device: str) -> None:
    existing = db.get(ModelVersion, settings.model_version)
    if existing is not None:
        if existing.checkpoint_hash != checkpoint_hash:
            raise ModelContractError(
                "AI_MODEL_VERSION points to a different checkpoint hash; bump the model version"
            )
        return
    candidate = ModelVersion(
        version=settings.model_version,
        checkpoint_hash=checkpoint_hash,
        checkpoint_name=settings.checkpoint_path.name,
        input_contract={
            "image": f"RGB {settings.model_image_size}x{settings.model_image_size}",
            "conditions": ["rain_mm", "time_s"],
            "condition_dimension": 2,
            "output": "RGB water map and depth_m array",
        },
        device=device,
    )
    db.add(candidate)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = db.get(ModelVersion, settings.model_version)
        if existing is None or existing.checkpoint_hash != checkpoint_hash:
            raise ModelContractError(
                "AI_MODEL_VERSION registration conflicts with another checkpoint"
            )


def _record_run(
    db: Session,
    *,
    request_id: str,
    resource_type: str,
    resource_id: str | None,
    stage: str,
    status: str,
    latency_ms: int,
    cache_hit: bool = False,
    error_code: str | None = None,
    details: dict | None = None,
) -> None:
    db.add(
        ProcessingRun(
            request_id=request_id,
            resource_type=resource_type,
            resource_id=resource_id,
            stage=stage,
            status=status,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
            error_code=error_code,
            details=details or {},
        )
    )
    db.commit()


def _hash_payload(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _elapsed_ms(started_at: float) -> int:
    return max(1, round((time.perf_counter() - started_at) * 1000))
