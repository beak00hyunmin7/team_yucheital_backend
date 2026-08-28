from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from app.services.hydrology import HydrologyResult


def render_watermap(
    result: HydrologyResult,
    *,
    output_size: tuple[int, int],
) -> bytes:
    output = BytesIO()
    _base_canvas(result, output_size).save(output, format="PNG", optimize=True)
    return output.getvalue()


def render_overlay(
    result: HydrologyResult,
    *,
    output_size: tuple[int, int],
    scale_x: float,
    scale_y: float,
) -> bytes:
    # The result view is intentionally based on the water map, not the input
    # contour/DEM image. The original remains available as a separate asset.
    canvas = _base_canvas(result, output_size)
    draw = ImageDraw.Draw(canvas)

    radius = max(8, round(min(canvas.size) * 0.018))
    line_width = max(2, radius // 5)
    for rank, candidate in enumerate(result.drains, start=1):
        x = round((candidate.x + 0.5) * scale_x - 0.5)
        y = round((candidate.y + 0.5) * scale_y - 0.5)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(255, 232, 35),
            outline=(190, 20, 20),
            width=line_width,
        )
        label = f"D{rank}"
        box = draw.textbbox((0, 0), label)
        text_width = box[2] - box[0]
        text_height = box[3] - box[1]
        draw.text(
            (x - text_width / 2, y - text_height / 2 - 1),
            label,
            fill=(20, 20, 20),
        )

    output = BytesIO()
    canvas.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _base_canvas(result: HydrologyResult, output_size: tuple[int, int]) -> Image.Image:
    if result.visualization_image is not None:
        # A trained model (e.g. the U-Net) produced this result: show its
        # actual predicted image instead of a D8-derived heatmap.
        return Image.fromarray(result.visualization_image, mode="RGB").resize(
            output_size, Image.Resampling.BILINEAR
        )
    return _flow_heatmap(result.accumulation, output_size)


def _flow_heatmap(accumulation: np.ndarray, output_size: tuple[int, int]) -> Image.Image:
    normalized = np.log1p(accumulation)
    normalized /= max(float(normalized.max()), 1e-12)
    red = np.clip(2.2 * normalized - 0.25, 0.0, 1.0)
    green = np.clip(1.4 - np.abs(2.2 * normalized - 1.0), 0.0, 1.0)
    blue = np.clip(1.2 - 2.0 * normalized, 0.0, 1.0)
    rgb = (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB").resize(output_size, Image.Resampling.BILINEAR)
