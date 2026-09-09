from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_origins(value: str | None) -> tuple[str, ...]:
    raw = value or "http://localhost:3000,http://localhost:5173"
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_version: str
    api_prefix: str
    data_root: Path
    storage_root: Path
    database_url: str
    checkpoint_path: Path
    model_version: str
    model_image_size: int
    inference_concurrency: int
    inference_wait_timeout_s: float
    use_fp16: bool
    warmup_model: bool
    preprocessing_version: str
    analysis_max_dimension: int
    max_upload_bytes: int
    max_image_pixels: int
    scenario_evaluator: str
    depth_threshold_m: float
    cors_origins: tuple[str, ...]
    auto_create_tables: bool

    @classmethod
    def from_env(cls) -> Settings:
        package_root = Path(__file__).resolve().parent.parent
        data_root = Path(os.getenv("V2_DATA_ROOT", package_root / "data")).resolve()
        storage_root = Path(
            os.getenv("V2_STORAGE_ROOT", data_root / "storage")
        ).resolve()
        default_db = f"sqlite:///{(data_root / 'terrain_drainage_v2.db').as_posix()}"
        return cls(
            app_name="Terrain Drainage API v2",
            app_version="2.0.0",
            api_prefix="/api/v2",
            data_root=data_root,
            storage_root=storage_root,
            database_url=os.getenv("V2_DATABASE_URL", default_db),
            checkpoint_path=Path(
                os.getenv("AI_MODEL_CHECKPOINT", package_root / "checkpoints" / "best.pt")
            ).resolve(),
            model_version=os.getenv("AI_MODEL_VERSION", "drainage-film-v5"),
            model_image_size=int(os.getenv("AI_MODEL_IMAGE_SIZE", "256")),
            inference_concurrency=max(1, int(os.getenv("AI_INFERENCE_CONCURRENCY", "1"))),
            inference_wait_timeout_s=max(
                0.1, float(os.getenv("AI_INFERENCE_WAIT_TIMEOUT_S", "30"))
            ),
            use_fp16=_as_bool(os.getenv("AI_USE_FP16"), False),
            warmup_model=_as_bool(os.getenv("AI_WARMUP_MODEL"), True),
            preprocessing_version=os.getenv("PREPROCESSING_VERSION", "terrain-preprocess-v2"),
            analysis_max_dimension=int(os.getenv("ANALYSIS_MAX_DIMENSION", "512")),
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", "10485760")),
            max_image_pixels=int(os.getenv("MAX_IMAGE_PIXELS", "16000000")),
            scenario_evaluator=os.getenv(
                "SCENARIO_EVALUATOR", "capacity_approx"
            ).strip().lower(),
            depth_threshold_m=float(os.getenv("DEPTH_THRESHOLD_M", "0.05")),
            cors_origins=_as_origins(os.getenv("CORS_ORIGINS")),
            auto_create_tables=_as_bool(os.getenv("V2_AUTO_CREATE_TABLES"), True),
        )


settings = Settings.from_env()
