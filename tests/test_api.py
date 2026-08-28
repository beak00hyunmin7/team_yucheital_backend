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
    assert body["water_map_url"].endswith("/water_map.png")
    assert body["overlay_image_url"].endswith("/drain_overlay.png")
    assert body["verification_status"] == "AUTO_VERIFIED"
    assert body["training_status"] == "READY"
    assert body["automatic_validation"]["passed"] is True
    assert body["automatic_validation"]["failure_reasons"] == []
    assert body["drains"][0]["catchment_area_m2"] > 0

    watermap = client.get(body["water_map_url"])
    assert watermap.status_code == 200
    assert watermap.content.startswith(b"\x89PNG\r\n\x1a\n")

    stored = client.get(f"/api/v1/drainage/analyses/{body['analysis_case_id']}")
    assert stored.status_code == 200
    assert stored.json()["analysis_case_id"] == body["analysis_case_id"]


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


def test_identical_request_uses_cached_analysis() -> None:
    request = {
        "files": {"image": ("cache.png", _sample_dem_bytes(), "image/png")},
        "data": {"drain_count": "3", "rainfall_mm_per_hour": "123"},
    }
    first = client.post("/api/v1/drainage/analyze", **request)
    second = client.post("/api/v1/drainage/analyze", **request)
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["cache_hit"] is True
    assert second.json()["analysis_case_id"] == first.json()["analysis_case_id"]


def test_auto_verified_analysis_registers_training_bundle() -> None:
    analysis = client.post(
        "/api/v1/drainage/analyze",
        files={"image": ("training.png", _sample_dem_bytes(), "image/png")},
        data={"drain_count": "2", "rainfall_mm_per_hour": "222"},
    )
    assert analysis.status_code == 200
    body = analysis.json()

    items = client.get("/api/v1/training-dataset/items?status=READY")
    assert items.status_code == 200
    matching = [
        item
        for item in items.json()
        if item["analysis_case_id"] == body["analysis_case_id"]
    ]
    assert len(matching) == 1
    assert len(matching[0]["target_positions"]) == 2
    assert matching[0]["validator_version"].startswith("auto-validator-")
    assert matching[0]["quality_score"] > 0


def test_validation_history_is_stored() -> None:
    analysis = client.post(
        "/api/v1/drainage/analyze",
        files={"image": ("history.png", _sample_dem_bytes(), "image/png")},
        data={"drain_count": "1", "rainfall_mm_per_hour": "333"},
    )
    case_id = analysis.json()["analysis_case_id"]
    history = client.get(f"/api/v1/drainage/analyses/{case_id}/validations")
    assert history.status_code == 200
    assert len(history.json()) >= 1
    assert history.json()[0]["validator_version"].startswith("auto-validator-")


def test_auto_rejected_analysis_is_excluded_from_training(monkeypatch) -> None:
    import app.services.automatic_validation as validator

    monkeypatch.setattr(validator, "MIN_CAPTURE_RATIO", 1.0)
    analysis = client.post(
        "/api/v1/drainage/analyze",
        files={"image": ("rejected.png", _sample_dem_bytes(), "image/png")},
        data={"drain_count": "2", "rainfall_mm_per_hour": "444"},
    )
    assert analysis.status_code == 200, analysis.text
    body = analysis.json()
    assert body["verification_status"] == "AUTO_REJECTED"
    assert body["training_status"] == "EXCLUDED"
    assert body["retry_count"] == 2
    assert body["automatic_validation"]["passed"] is False

    items = client.get("/api/v1/training-dataset/items?status=READY")
    matching = [
        item for item in items.json() if item["analysis_case_id"] == body["analysis_case_id"]
    ]
    assert matching == []
