from __future__ import annotations

import os


APP_NAME = "Terrain Drainage Recommendation API"
APP_VERSION = "0.1.0"
API_PREFIX = "/api/v1"

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", 16_000_000))
ANALYSIS_MAX_DIMENSION = int(os.getenv("ANALYSIS_MAX_DIMENSION", 512))

ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/tiff",
}

