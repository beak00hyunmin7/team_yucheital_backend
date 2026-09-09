from __future__ import annotations

import io

import numpy as np
from PIL import Image

from app.services.preprocessing import preprocess_terrain, preprocessing_cache_key


def _gradient_png(width: int = 160, height: int = 96) -> bytes:
    yy, xx = np.indices((height, width))
    values = ((0.7 * xx / max(width - 1, 1) + 0.3 * yy / max(height - 1, 1)) * 255)
    output = io.BytesIO()
    Image.fromarray(values.astype(np.uint8), mode="L").save(output, format="PNG")
    return output.getvalue()


def test_grayscale_dem_produces_separate_model_input_and_numeric_array() -> None:
    prepared = preprocess_terrain(
        _gradient_png(),
        input_type="grayscale_dem",
        high_is_bright=True,
        blur_radius=1.0,
        max_pixels=2_000_000,
        analysis_max_dimension=512,
        model_image_size=256,
    )
    assert prepared.resolved_input_type == "grayscale_dem"
    assert prepared.model_width == 256
    assert prepared.model_height == 256
    assert prepared.model_input_png.startswith(b"\x89PNG\r\n\x1a\n")
    assert prepared.elevation_npz is not None
    with np.load(io.BytesIO(prepared.elevation_npz), allow_pickle=False) as archive:
        elevation = archive["elevation"]
    assert elevation.dtype == np.float32
    assert elevation.shape == (96, 160)
    assert float(elevation.min()) == 0.0
    assert float(elevation.max()) == 1.0


def test_preprocessing_key_changes_with_version_or_options() -> None:
    common = {
        "image_hash": "a" * 64,
        "input_type": "grayscale_dem",
        "high_is_bright": True,
        "blur_radius": 1.2,
        "model_image_size": 256,
    }
    first = preprocessing_cache_key(preprocessing_version="v1", **common)
    second = preprocessing_cache_key(preprocessing_version="v2", **common)
    third = preprocessing_cache_key(
        preprocessing_version="v1", **{**common, "blur_radius": 2.0}
    )
    assert first != second
    assert first != third
