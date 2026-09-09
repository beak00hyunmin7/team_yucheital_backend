from __future__ import annotations

import io

import matplotlib
import numpy as np
from PIL import Image

DEPTH_VMAX_M = 0.6


def depth_metrics(
    depth_m: np.ndarray,
    *,
    threshold_m: float,
    cell_size_m: float | None,
) -> dict[str, float | int | None]:
    values = np.asarray(depth_m, dtype=np.float64)
    flooded = values >= threshold_m
    flooded_pixels = int(np.count_nonzero(flooded))
    flooded_area = (
        round(flooded_pixels * cell_size_m * cell_size_m, 6)
        if cell_size_m is not None
        else None
    )
    water_volume = (
        round(float(values.sum()) * cell_size_m * cell_size_m, 6)
        if cell_size_m is not None
        else None
    )
    return {
        "max_depth_m": round(float(values.max(initial=0.0)), 6),
        "mean_depth_m": round(float(values.mean()) if values.size else 0.0, 6),
        "flooded_pixel_count": flooded_pixels,
        "flooded_area_m2": flooded_area,
        "estimated_water_volume_m3": water_volume,
        "threshold_m": threshold_m,
    }


def render_depth_png(depth_m: np.ndarray) -> bytes:
    values = np.asarray(depth_m, dtype=np.float32)
    normalized = np.sqrt(np.clip(values, 0.0, DEPTH_VMAX_M) / DEPTH_VMAX_M)
    rgb = (
        matplotlib.colormaps["Blues"](normalized)[..., :3] * 255.0
    ).round().astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(output, format="PNG", compress_level=3)
    return output.getvalue()


def array_to_npz(depth_m: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(output, depth_m=np.asarray(depth_m, dtype=np.float32))
    return output.getvalue()


def array_from_npz(data: bytes) -> np.ndarray:
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return np.asarray(archive["depth_m"], dtype=np.float32)
