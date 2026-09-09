from __future__ import annotations

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.ai.runtime import ModelPrediction, RuntimeStatus
from app.services.metrics import render_depth_png


class FakeRuntime:
    checkpoint_hash = "b" * 64

    def try_load(self) -> RuntimeStatus:
        return self.status()

    def status(self) -> RuntimeStatus:
        return RuntimeStatus(
            available=True,
            model_version=main.settings.model_version,
            checkpoint_hash=self.checkpoint_hash,
            checkpoint_name="test-best.pt",
            device="cpu",
            condition_dimension=2,
            error=None,
        )

    def predict(self, model_input_png: bytes, *, rain_mm: float, time_s: float) -> ModelPrediction:
        assert model_input_png.startswith(b"\x89PNG")
        assert 20.0 <= rain_mm <= 80.0
        assert 0.0 <= time_s <= 120.0
        depth = np.full((256, 256), 0.10, dtype=np.float32)
        return ModelPrediction(
            water_map_png=render_depth_png(depth),
            depth_m=depth,
            inference_time_ms=7,
            checkpoint_hash=self.checkpoint_hash,
            device="cpu",
        )


def _terrain_png() -> bytes:
    yy, xx = np.indices((96, 128))
    values = ((xx + yy) / (128 + 96 - 2) * 255).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(values, mode="L").save(output, format="PNG")
    return output.getvalue()


def test_end_to_end_has_no_automatic_drain_positions(monkeypatch) -> None:
    monkeypatch.setattr(main, "runtime", FakeRuntime())
    with TestClient(main.app) as client:
        upload_response = client.post(
            "/api/v2/upload-terrain",
            files={"image": ("terrain.png", _terrain_png(), "image/png")},
            data={"input_type": "grayscale_dem", "cell_size_m": "1.0"},
        )
        assert upload_response.status_code == 200, upload_response.text
        upload = upload_response.json()
        assert {asset["kind"] for asset in upload["assets"]} == {
            "ORIGINAL",
            "MODEL_INPUT",
            "ELEVATION_ARRAY",
        }

        prediction_response = client.post(
            "/api/v2/predict-drainage-map",
            json={"upload_id": upload["upload_id"], "rain_mm": 45.0, "time_s": 120.0},
        )
        assert prediction_response.status_code == 200, prediction_response.text
        prediction = prediction_response.json()
        assert prediction["drains"] == []
        assert prediction["checkpoint_hash"] == "b" * 64

        scenario_response = client.post(
            "/api/v2/evaluate-drain-settings",
            json={
                "result_id": prediction["result_id"],
                "duration_s": 120.0,
                "drains": [
                    {
                        "x": 0.5,
                        "y": 0.5,
                        "capacity_lps": 15.0,
                        "capture_radius_m": 10.0,
                        "drain_type": "standard",
                    }
                ],
            },
        )
        assert scenario_response.status_code == 200, scenario_response.text
        scenario = scenario_response.json()
        assert len(scenario["drains"]) == 1
        assert scenario["drains"][0]["x"] == 0.5

        csv_response = client.get(
            f"/api/v2/scenario/{scenario['scenario_id']}/export.csv"
        )
        assert csv_response.status_code == 200
        assert "USER" in csv_response.text


def test_prediction_cache_skips_duplicate_inference(monkeypatch) -> None:
    fake = FakeRuntime()
    calls = 0
    original_predict = fake.predict

    def counted_predict(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_predict(*args, **kwargs)

    fake.predict = counted_predict
    monkeypatch.setattr(main, "runtime", fake)
    with TestClient(main.app) as client:
        upload = client.post(
            "/api/v2/upload-terrain",
            files={"image": ("cache.png", _terrain_png(), "image/png")},
            data={"input_type": "grayscale_dem", "cell_size_m": "1.0"},
        ).json()
        payload = {"upload_id": upload["upload_id"], "rain_mm": 55.0, "time_s": 100.0}
        first = client.post("/api/v2/predict-drainage-map", json=payload)
        second = client.post("/api/v2/predict-drainage-map", json=payload)
        assert first.status_code == 200
        assert second.status_code == 200
        assert second.json()["cache_hit"] is True
        assert calls == 1
