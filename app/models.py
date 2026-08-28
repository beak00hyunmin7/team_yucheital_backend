from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    version: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    model_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    analyses: Mapped[list["AnalysisCase"]] = relationship(back_populates="model")


class AnalysisCase(Base):
    __tablename__ = "analysis_cases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    cache_key: Mapped[str] = mapped_column(String(64), index=True)
    image_hash: Mapped[str] = mapped_column(String(64), index=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    requested_drain_count: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(30), default="PROCESSING", index=True)
    verification_status: Mapped[str] = mapped_column(
        String(30), default="AI_GENERATED", index=True
    )
    validation_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    selected_attempt_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_version: Mapped[str] = mapped_column(ForeignKey("model_versions.version"))
    algorithm: Mapped[str] = mapped_column(String(255))
    parameters_json: Mapped[str] = mapped_column(Text)
    response_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    model: Mapped[ModelVersion] = relationship(back_populates="analyses")
    files: Mapped[list["AnalysisFile"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    drain_positions: Mapped[list["DrainPosition"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    attempts: Mapped[list["AnalysisAttempt"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    training_item: Mapped["TrainingDatasetItem | None"] = relationship(
        back_populates="analysis", cascade="all, delete-orphan", uselist=False
    )


class AnalysisFile(Base):
    __tablename__ = "analysis_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_case_id: Mapped[str] = mapped_column(
        ForeignKey("analysis_cases.id"), index=True
    )
    kind: Mapped[str] = mapped_column(String(30))
    storage_path: Mapped[str] = mapped_column(String(500))
    mime_type: Mapped[str] = mapped_column(String(100))
    file_hash: Mapped[str] = mapped_column(String(64))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    analysis: Mapped[AnalysisCase] = relationship(back_populates="files")


class DrainPosition(Base):
    __tablename__ = "drain_positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_case_id: Mapped[str] = mapped_column(
        ForeignKey("analysis_cases.id"), index=True
    )
    position_order: Mapped[int] = mapped_column(Integer)
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)
    x_normalized: Mapped[float] = mapped_column(Float)
    y_normalized: Mapped[float] = mapped_column(Float)
    x_pixel: Mapped[int] = mapped_column(Integer)
    y_pixel: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(30), default="AI_PREDICTED")
    is_selected: Mapped[bool] = mapped_column(Boolean, default=False)
    payload_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    analysis: Mapped[AnalysisCase] = relationship(back_populates="drain_positions")


class AnalysisAttempt(Base):
    __tablename__ = "analysis_attempts"
    __table_args__ = (
        UniqueConstraint(
            "analysis_case_id", "attempt_number", name="uq_case_attempt_number"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_case_id: Mapped[str] = mapped_column(
        ForeignKey("analysis_cases.id"), index=True
    )
    attempt_number: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(30))
    parameters_json: Mapped[str] = mapped_column(Text)
    positions_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    analysis: Mapped[AnalysisCase] = relationship(back_populates="attempts")
    validation: Mapped["AutomaticValidation"] = relationship(
        back_populates="attempt", cascade="all, delete-orphan", uselist=False
    )


class AutomaticValidation(Base):
    __tablename__ = "automatic_validations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_attempt_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_attempts.id"), unique=True, index=True
    )
    validator_version: Mapped[str] = mapped_column(String(80))
    passed: Mapped[bool] = mapped_column(Boolean, index=True)
    confidence: Mapped[float] = mapped_column(Float)
    count_ok: Mapped[bool] = mapped_column(Boolean)
    spacing_ok: Mapped[bool] = mapped_column(Boolean)
    boundary_ok: Mapped[bool] = mapped_column(Boolean)
    minimum_spacing_ratio_actual: Mapped[float] = mapped_column(Float)
    mean_suitability: Mapped[float] = mapped_column(Float)
    estimated_capture_ratio: Mapped[float] = mapped_column(Float)
    residual_water_ratio: Mapped[float] = mapped_column(Float)
    stability_score: Mapped[float] = mapped_column(Float)
    failure_reasons_json: Mapped[str] = mapped_column(Text)
    metrics_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    attempt: Mapped[AnalysisAttempt] = relationship(back_populates="validation")


class TrainingDatasetItem(Base):
    __tablename__ = "training_dataset_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_case_id: Mapped[str] = mapped_column(
        ForeignKey("analysis_cases.id"), unique=True, index=True
    )
    status: Mapped[str] = mapped_column(String(30), default="READY", index=True)
    input_file_path: Mapped[str] = mapped_column(String(500))
    target_watermap_path: Mapped[str] = mapped_column(String(500))
    target_positions_json: Mapped[str] = mapped_column(Text)
    quality_score: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(30), default="AUTO_VERIFIED")
    validator_version: Mapped[str] = mapped_column(String(80))
    verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    analysis: Mapped[AnalysisCase] = relationship(back_populates="training_item")
