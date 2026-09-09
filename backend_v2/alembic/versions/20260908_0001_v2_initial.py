"""Create isolated v2 terrain, prediction, and scenario tables.

Revision ID: 20260908_0001
Revises:
Create Date: 2026-09-08
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260908_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "v2_model_versions",
        sa.Column("version", sa.String(length=100), nullable=False),
        sa.Column("checkpoint_hash", sa.String(length=64), nullable=False),
        sa.Column("checkpoint_name", sa.String(length=255), nullable=False),
        sa.Column("input_contract", sa.JSON(), nullable=False),
        sa.Column("device", sa.String(length=30), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("version"),
        sa.UniqueConstraint("checkpoint_hash"),
    )
    op.create_table(
        "v2_terrain_uploads",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("preprocess_key", sa.String(length=64), nullable=False),
        sa.Column("image_hash", sa.String(length=64), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("content_type", sa.String(length=100), nullable=False),
        sa.Column("requested_input_type", sa.String(length=30), nullable=False),
        sa.Column("resolved_input_type", sa.String(length=30), nullable=False),
        sa.Column("high_is_bright", sa.Boolean(), nullable=False),
        sa.Column("blur_radius", sa.Float(), nullable=False),
        sa.Column("cell_size_m", sa.Float(), nullable=True),
        sa.Column("crs", sa.String(length=100), nullable=True),
        sa.Column("original_width", sa.Integer(), nullable=False),
        sa.Column("original_height", sa.Integer(), nullable=False),
        sa.Column("model_width", sa.Integer(), nullable=False),
        sa.Column("model_height", sa.Integer(), nullable=False),
        sa.Column("quality_score", sa.Float(), nullable=False),
        sa.Column("quality_report", sa.JSON(), nullable=False),
        sa.Column("warnings", sa.JSON(), nullable=False),
        sa.Column("preprocessing_version", sa.String(length=100), nullable=False),
        sa.Column("original_path", sa.String(length=500), nullable=False),
        sa.Column("original_hash", sa.String(length=64), nullable=False),
        sa.Column("model_input_path", sa.String(length=500), nullable=False),
        sa.Column("model_input_hash", sa.String(length=64), nullable=False),
        sa.Column("elevation_path", sa.String(length=500), nullable=True),
        sa.Column("elevation_hash", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("preprocess_key"),
    )
    op.create_index("ix_v2_terrain_uploads_image_hash", "v2_terrain_uploads", ["image_hash"])
    op.create_index(
        "ix_v2_upload_status_created",
        "v2_terrain_uploads",
        ["status", "created_at"],
    )
    op.create_table(
        "v2_prediction_results",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("cache_key", sa.String(length=64), nullable=False),
        sa.Column("upload_id", sa.String(length=36), nullable=False),
        sa.Column("model_version", sa.String(length=100), nullable=False),
        sa.Column("checkpoint_hash", sa.String(length=64), nullable=False),
        sa.Column("rain_mm", sa.Float(), nullable=False),
        sa.Column("time_s", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("water_map_path", sa.String(length=500), nullable=True),
        sa.Column("water_map_hash", sa.String(length=64), nullable=True),
        sa.Column("depth_array_path", sa.String(length=500), nullable=True),
        sa.Column("depth_array_hash", sa.String(length=64), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("inference_time_ms", sa.Integer(), nullable=True),
        sa.Column("processing_time_ms", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["upload_id"], ["v2_terrain_uploads.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("cache_key"),
    )
    op.create_index("ix_v2_prediction_results_upload_id", "v2_prediction_results", ["upload_id"])
    op.create_index(
        "ix_v2_prediction_status_created",
        "v2_prediction_results",
        ["status", "created_at"],
    )
    op.create_index(
        "ix_v2_prediction_upload_created",
        "v2_prediction_results",
        ["upload_id", "created_at"],
    )
    op.create_table(
        "v2_drain_scenarios",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("cache_key", sa.String(length=64), nullable=False),
        sa.Column("prediction_id", sa.String(length=36), nullable=False),
        sa.Column("evaluator_version", sa.String(length=100), nullable=False),
        sa.Column("duration_s", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("baseline_metrics", sa.JSON(), nullable=True),
        sa.Column("scenario_metrics", sa.JSON(), nullable=True),
        sa.Column("warnings", sa.JSON(), nullable=False),
        sa.Column("scenario_map_path", sa.String(length=500), nullable=True),
        sa.Column("scenario_map_hash", sa.String(length=64), nullable=True),
        sa.Column("depth_array_path", sa.String(length=500), nullable=True),
        sa.Column("depth_array_hash", sa.String(length=64), nullable=True),
        sa.Column("processing_time_ms", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["prediction_id"], ["v2_prediction_results.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("cache_key"),
    )
    op.create_index("ix_v2_drain_scenarios_prediction_id", "v2_drain_scenarios", ["prediction_id"])
    op.create_index(
        "ix_v2_scenario_prediction_created",
        "v2_drain_scenarios",
        ["prediction_id", "created_at"],
    )
    op.create_index(
        "ix_v2_scenario_status_created",
        "v2_drain_scenarios",
        ["status", "created_at"],
    )
    op.create_table(
        "v2_processing_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("request_id", sa.String(length=36), nullable=False),
        sa.Column("resource_type", sa.String(length=30), nullable=False),
        sa.Column("resource_id", sa.String(length=36), nullable=True),
        sa.Column("stage", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("cache_hit", sa.Boolean(), nullable=False),
        sa.Column("error_code", sa.String(length=80), nullable=True),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_v2_processing_runs_request_id", "v2_processing_runs", ["request_id"])
    op.create_index(
        "ix_v2_run_resource",
        "v2_processing_runs",
        ["resource_type", "resource_id", "created_at"],
    )
    op.create_index(
        "ix_v2_run_stage_status",
        "v2_processing_runs",
        ["stage", "status", "created_at"],
    )
    op.create_table(
        "v2_scenario_drains",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("scenario_id", sa.String(length=36), nullable=False),
        sa.Column("position_order", sa.Integer(), nullable=False),
        sa.Column("x_normalized", sa.Float(), nullable=False),
        sa.Column("y_normalized", sa.Float(), nullable=False),
        sa.Column("capacity_lps", sa.Float(), nullable=False),
        sa.Column("capture_radius_m", sa.Float(), nullable=False),
        sa.Column("drain_type", sa.String(length=50), nullable=False),
        sa.Column("source", sa.String(length=20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["scenario_id"], ["v2_drain_scenarios.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "scenario_id", "position_order", name="uq_v2_scenario_drain_order"
        ),
    )
    op.create_index("ix_v2_scenario_drains_scenario_id", "v2_scenario_drains", ["scenario_id"])


def downgrade() -> None:
    op.drop_index("ix_v2_scenario_drains_scenario_id", table_name="v2_scenario_drains")
    op.drop_table("v2_scenario_drains")
    op.drop_index("ix_v2_run_stage_status", table_name="v2_processing_runs")
    op.drop_index("ix_v2_run_resource", table_name="v2_processing_runs")
    op.drop_index("ix_v2_processing_runs_request_id", table_name="v2_processing_runs")
    op.drop_table("v2_processing_runs")
    op.drop_index("ix_v2_scenario_status_created", table_name="v2_drain_scenarios")
    op.drop_index("ix_v2_scenario_prediction_created", table_name="v2_drain_scenarios")
    op.drop_index("ix_v2_drain_scenarios_prediction_id", table_name="v2_drain_scenarios")
    op.drop_table("v2_drain_scenarios")
    op.drop_index("ix_v2_prediction_upload_created", table_name="v2_prediction_results")
    op.drop_index("ix_v2_prediction_status_created", table_name="v2_prediction_results")
    op.drop_index("ix_v2_prediction_results_upload_id", table_name="v2_prediction_results")
    op.drop_table("v2_prediction_results")
    op.drop_index("ix_v2_upload_status_created", table_name="v2_terrain_uploads")
    op.drop_index("ix_v2_terrain_uploads_image_hash", table_name="v2_terrain_uploads")
    op.drop_table("v2_terrain_uploads")
    op.drop_table("v2_model_versions")
