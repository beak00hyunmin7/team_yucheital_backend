from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


def _sample_dem_bytes() -> bytes:
    yy, xx = np.indices((64, 64))
    elevation = 255 - (0.55 * xx + 0.45 * yy) * (255 / 63)
    image = Image.fromarray(np.clip(elevation, 0, 255).astype(np.uint8), mode="L")
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_health() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_dem() -> None:
    response = client.post(
        "/api/v1/drainage/analyze",
        files={"image": ("sample.png", _sample_dem_bytes(), "image/png")},
        data={"drain_count": "2", "include_overlay": "false"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["drains"]) == 2
    assert body["overlay_png_base64"] is None
    assert body["drains"][0]["catchment_area_m2"] > 0


def test_rejects_flat_image() -> None:
    image = Image.new("L", (64, 64), color=127)
    output = BytesIO()
    image.save(output, format="PNG")
    response = client.post(
        "/api/v1/drainage/analyze",
        files={"image": ("flat.png", output.getvalue(), "image/png")},
    )
    assert response.status_code == 422


def test_overlay_is_png() -> None:
    response = client.post(
        "/api/v1/drainage/analyze",
        files={"image": ("sample.png", _sample_dem_bytes(), "image/png")},
        data={"drain_count": "1", "include_overlay": "true"},
    )
    assert response.status_code == 200, response.text
    encoded = response.json()["overlay_png_base64"]
    assert base64.b64decode(encoded).startswith(b"\x89PNG\r\n\x1a\n")
