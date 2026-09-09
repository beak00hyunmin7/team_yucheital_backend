from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utc_now() -> datetime:
    return datetime.now(UTC)


class TerrainUpload(Base):
    __tablename__ = "v2_terrain_uploads"
    __table_args__ = (
        Index("ix_v2_upload_status_created", "status", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    preprocess_key: Mapped[str] = mapped_column(String(64), unique=True)
    image_hash: Mapped[str] = mapped_column(String(64), index=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))
    requested_input_type: Mapped[str] = mapped_column(String(30))
    resolved_input_type: Mapped[str] = mapped_column(String(30))
    high_is_bright: Mapped[bool] = mapped_column(Boolean)
    blur_radius: Mapped[float] = mapped_column(Float)
    cell_size_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    crs: Mapped[str | None] = mapped_column(String(100), nullable=True)
    original_width: Mapped[int] = mapped_column(Integer)
    original_height: Mapped[int] = mapped_column(Integer)
    model_width: Mapped[int] = mapped_column(Integer)
    model_height: Mapped[int] = mapped_column(Integer)
    quality_score: Mapped[float] = mapped_column(Float)
    quality_report: Mapped[dict] = mapped_column(JSON)
    warnings: Mapped[list] = mapped_column(JSON)
    preprocessing_version: Mapped[str] = mapped_column(String(100))
    original_path: Mapped[str] = mapped_column(String(500))
    original_hash: Mapped[str] = mapped_column(String(64))
    model_input_path: Mapped[str] = mapped_column(String(500))
    model_input_hash: Mapped[str] = mapped_column(String(64))
    elevation_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    elevation_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="COMPLETED")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    predictions: Mapped[list[PredictionResult]] = relationship(
        back_populates="upload", cascade="all, delete-orphan"
    )


class ModelVersion(Base):
    __tablename__ = "v2_model_versions"

    version: Mapped[str] = mapped_column(String(100), primary_key=True)
    checkpoint_hash: Mapped[str] = mapped_column(String(64), unique=True)
    checkpoint_name: Mapped[str] = mapped_column(String(255))
    input_contract: Mapped[dict] = mapped_column(JSON)
    device: Mapped[str] = mapped_column(String(30))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class PredictionResult(Base):
    __tablename__ = "v2_prediction_results"
    __table_args__ = (
        Index("ix_v2_prediction_status_created", "status", "created_at"),
        Index("ix_v2_prediction_upload_created", "upload_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    cache_key: Mapped[str] = mapped_column(String(64), unique=True)
    upload_id: Mapped[str] = mapped_column(
        ForeignKey("v2_terrain_uploads.id", ondelete="CASCADE"), index=True
    )
    model_version: Mapped[str] = mapped_column(String(100))
    checkpoint_hash: Mapped[str] = mapped_column(String(64))
    rain_mm: Mapped[float] = mapped_column(Float)
    time_s: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(30), default="PROCESSING")
    water_map_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    water_map_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    depth_array_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    depth_array_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    inference_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    upload: Mapped[TerrainUpload] = relationship(back_populates="predictions")
    scenarios: Mapped[list[DrainScenario]] = relationship(
        back_populates="prediction", cascade="all, delete-orphan"
    )


class DrainScenario(Base):
    __tablename__ = "v2_drain_scenarios"
    __table_args__ = (
        Index("ix_v2_scenario_prediction_created", "prediction_id", "created_at"),
        Index("ix_v2_scenario_status_created", "status", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    cache_key: Mapped[str] = mapped_column(String(64), unique=True)
    prediction_id: Mapped[str] = mapped_column(
        ForeignKey("v2_prediction_results.id", ondelete="CASCADE"), index=True
    )
    evaluator_version: Mapped[str] = mapped_column(String(100))
    duration_s: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(30), default="PROCESSING")
    baseline_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    scenario_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    warnings: Mapped[list] = mapped_column(JSON)
    scenario_map_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    scenario_map_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    depth_array_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    depth_array_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    prediction: Mapped[PredictionResult] = relationship(back_populates="scenarios")
    drains: Mapped[list[ScenarioDrain]] = relationship(
        back_populates="scenario", cascade="all, delete-orphan"
    )


class ScenarioDrain(Base):
    __tablename__ = "v2_scenario_drains"
    __table_args__ = (
        UniqueConstraint("scenario_id", "position_order", name="uq_v2_scenario_drain_order"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scenario_id: Mapped[str] = mapped_column(
        ForeignKey("v2_drain_scenarios.id", ondelete="CASCADE"), index=True
    )
    position_order: Mapped[int] = mapped_column(Integer)
    x_normalized: Mapped[float] = mapped_column(Float)
    y_normalized: Mapped[float] = mapped_column(Float)
    capacity_lps: Mapped[float] = mapped_column(Float)
    capture_radius_m: Mapped[float] = mapped_column(Float)
    drain_type: Mapped[str] = mapped_column(String(50))
    source: Mapped[str] = mapped_column(String(20), default="USER")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    scenario: Mapped[DrainScenario] = relationship(back_populates="drains")


class ProcessingRun(Base):
    __tablename__ = "v2_processing_runs"
    __table_args__ = (
        Index("ix_v2_run_resource", "resource_type", "resource_id", "created_at"),
        Index("ix_v2_run_stage_status", "stage", "status", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(36), index=True)
    resource_type: Mapped[str] = mapped_column(String(30))
    resource_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    stage: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(30))
    latency_ms: Mapped[int] = mapped_column(Integer)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    error_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    details: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
