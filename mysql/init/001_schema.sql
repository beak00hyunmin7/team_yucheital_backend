CREATE DATABASE IF NOT EXISTS terrain_drainage
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE terrain_drainage;

CREATE TABLE IF NOT EXISTS model_versions (
  version VARCHAR(80) PRIMARY KEY,
  name VARCHAR(200) NOT NULL,
  model_path VARCHAR(500) NULL,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS analysis_cases (
  id CHAR(36) PRIMARY KEY,
  cache_key CHAR(64) NOT NULL,
  image_hash CHAR(64) NOT NULL,
  original_filename VARCHAR(255) NOT NULL,
  requested_drain_count INT NOT NULL,
  status VARCHAR(30) NOT NULL DEFAULT 'PROCESSING',
  verification_status VARCHAR(30) NOT NULL DEFAULT 'AI_GENERATED',
  validation_confidence DOUBLE NULL,
  retry_count INT NOT NULL DEFAULT 0,
  selected_attempt_number INT NULL,
  model_version VARCHAR(80) NOT NULL,
  algorithm VARCHAR(255) NOT NULL,
  parameters_json JSON NOT NULL,
  response_json JSON NULL,
  processing_time_ms INT NULL,
  error_message TEXT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
    ON UPDATE CURRENT_TIMESTAMP(6),
  CONSTRAINT fk_analysis_model_version
    FOREIGN KEY (model_version) REFERENCES model_versions(version),
  INDEX ix_analysis_cache_key (cache_key),
  INDEX ix_analysis_image_hash (image_hash),
  INDEX ix_analysis_status (status),
  INDEX ix_analysis_verification_status (verification_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS analysis_files (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  analysis_case_id CHAR(36) NOT NULL,
  kind VARCHAR(30) NOT NULL,
  storage_path VARCHAR(500) NOT NULL,
  mime_type VARCHAR(100) NOT NULL,
  file_hash CHAR(64) NOT NULL,
  width INT NOT NULL,
  height INT NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  CONSTRAINT fk_analysis_files_case
    FOREIGN KEY (analysis_case_id) REFERENCES analysis_cases(id)
    ON DELETE CASCADE,
  INDEX ix_analysis_files_case (analysis_case_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS analysis_attempts (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  analysis_case_id CHAR(36) NOT NULL,
  attempt_number INT NOT NULL,
  status VARCHAR(30) NOT NULL,
  parameters_json JSON NOT NULL,
  positions_json JSON NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  CONSTRAINT fk_analysis_attempts_case
    FOREIGN KEY (analysis_case_id) REFERENCES analysis_cases(id)
    ON DELETE CASCADE,
  CONSTRAINT uq_case_attempt_number UNIQUE (analysis_case_id, attempt_number),
  INDEX ix_analysis_attempts_case (analysis_case_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS automatic_validations (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  analysis_attempt_id BIGINT NOT NULL,
  validator_version VARCHAR(80) NOT NULL,
  passed BOOLEAN NOT NULL,
  confidence DOUBLE NOT NULL,
  count_ok BOOLEAN NOT NULL,
  spacing_ok BOOLEAN NOT NULL,
  boundary_ok BOOLEAN NOT NULL,
  minimum_spacing_ratio_actual DOUBLE NOT NULL,
  mean_suitability DOUBLE NOT NULL,
  estimated_capture_ratio DOUBLE NOT NULL,
  residual_water_ratio DOUBLE NOT NULL,
  stability_score DOUBLE NOT NULL,
  failure_reasons_json JSON NOT NULL,
  metrics_json JSON NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  CONSTRAINT fk_automatic_validation_attempt
    FOREIGN KEY (analysis_attempt_id) REFERENCES analysis_attempts(id)
    ON DELETE CASCADE,
  CONSTRAINT uq_automatic_validation_attempt UNIQUE (analysis_attempt_id),
  INDEX ix_automatic_validations_passed (passed)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS drain_positions (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  analysis_case_id CHAR(36) NOT NULL,
  position_order INT NOT NULL,
  attempt_number INT NOT NULL DEFAULT 1,
  x_normalized DOUBLE NOT NULL,
  y_normalized DOUBLE NOT NULL,
  x_pixel INT NOT NULL,
  y_pixel INT NOT NULL,
  confidence DOUBLE NOT NULL,
  source VARCHAR(30) NOT NULL DEFAULT 'AI_PREDICTED',
  is_selected BOOLEAN NOT NULL DEFAULT FALSE,
  payload_json JSON NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  CONSTRAINT fk_drain_positions_case
    FOREIGN KEY (analysis_case_id) REFERENCES analysis_cases(id)
    ON DELETE CASCADE,
  INDEX ix_drain_positions_case (analysis_case_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS training_dataset_items (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  analysis_case_id CHAR(36) NOT NULL,
  status VARCHAR(30) NOT NULL DEFAULT 'READY',
  input_file_path VARCHAR(500) NOT NULL,
  target_watermap_path VARCHAR(500) NOT NULL,
  target_positions_json JSON NOT NULL,
  quality_score DOUBLE NOT NULL,
  source VARCHAR(30) NOT NULL DEFAULT 'AUTO_VERIFIED',
  validator_version VARCHAR(80) NOT NULL,
  verified_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  CONSTRAINT fk_training_dataset_case
    FOREIGN KEY (analysis_case_id) REFERENCES analysis_cases(id)
    ON DELETE CASCADE,
  CONSTRAINT uq_training_dataset_case UNIQUE (analysis_case_id),
  INDEX ix_training_dataset_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
