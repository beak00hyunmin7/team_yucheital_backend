from __future__ import annotations

import os
from pathlib import Path


APP_NAME = "Terrain Drainage Recommendation API"
APP_VERSION = "0.3.0"
API_PREFIX = "/api/v1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.getenv("DATA_ROOT", PROJECT_ROOT / "data")).resolve()
STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", DATA_ROOT / "storage")).resolve()
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{(DATA_ROOT / 'terrain_drainage.db').as_posix()}",
)
MODEL_VERSION = os.getenv("MODEL_VERSION", "hydrology-mvp-0.3.0")
MODEL_NAME = os.getenv("MODEL_NAME", "D8 water-map surrogate")
VALIDATOR_VERSION = os.getenv("VALIDATOR_VERSION", "auto-validator-0.3.0")

# "unet"이면 app/ai/unet_model.py의 학습된 U-Net(AI 모델링 팀 제공)을 사용하고,
# 체크포인트/torch를 못 찾으면 "d8" 대리 모델로 자동 폴백한다.
# "d8"로 지정하면 처음부터 대리 모델만 사용한다.
AI_BACKEND = os.getenv("AI_BACKEND", "unet")

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", 16_000_000))
ANALYSIS_MAX_DIMENSION = int(os.getenv("ANALYSIS_MAX_DIMENSION", 512))
MAX_ANALYSIS_ATTEMPTS = int(os.getenv("MAX_ANALYSIS_ATTEMPTS", 3))

# Conservative MVP defaults. Tune these values against a held-out, simulated
# benchmark before using the service for real drainage design.
MIN_MEAN_SUITABILITY = float(os.getenv("MIN_MEAN_SUITABILITY", 0.40))
MIN_CAPTURE_RATIO = float(os.getenv("MIN_CAPTURE_RATIO", 0.005))
MIN_STABILITY_SCORE = float(os.getenv("MIN_STABILITY_SCORE", 0.25))
MIN_AUTO_VERIFICATION_CONFIDENCE = float(
    os.getenv("MIN_AUTO_VERIFICATION_CONFIDENCE", 0.45)
)

ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/tiff",
}
