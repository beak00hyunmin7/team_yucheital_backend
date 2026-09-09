from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

SUPPORTED_INPUT_TYPES = {"auto", "grayscale_dem", "contour_image"}


class InvalidTerrainInput(ValueError):
    pass


@dataclass(frozen=True)
class PreprocessedTerrain:
    resolved_input_type: str
    original_width: int
    original_height: int
    model_width: int
    model_height: int
    model_input_png: bytes
    elevation_npz: bytes | None
    quality_score: float
    quality_report: dict[str, object]
    warnings: tuple[str, ...]


def preprocessing_cache_key(
    *,
    image_hash: str,
    preprocessing_version: str,
    input_type: str,
    high_is_bright: bool,
    blur_radius: float,
    model_image_size: int,
    cell_size_m: float | None = None,
    crs: str | None = None,
) -> str:
    payload = {
        "image_hash": image_hash,
        "preprocessing_version": preprocessing_version,
        "input_type": input_type,
        "high_is_bright": high_is_bright,
        "blur_radius": round(blur_radius, 4),
        "model_image_size": model_image_size,
        "cell_size_m": cell_size_m,
        "crs": crs,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def preprocess_terrain(
    data: bytes,
    *,
    input_type: str,
    high_is_bright: bool,
    blur_radius: float,
    max_pixels: int,
    analysis_max_dimension: int,
    model_image_size: int,
) -> PreprocessedTerrain:
    if input_type not in SUPPORTED_INPUT_TYPES:
        raise InvalidTerrainInput(
            f"input_type must be one of {sorted(SUPPORTED_INPUT_TYPES)}"
        )
    if model_image_size < 64 or model_image_size & (model_image_size - 1):
        raise InvalidTerrainInput("model_image_size must be a power of two and at least 64")

    try:
        with Image.open(io.BytesIO(data)) as probe:
            width, height = probe.size
            if width < 16 or height < 16:
                raise InvalidTerrainInput("image width and height must both be at least 16")
            if width * height > max_pixels:
                raise InvalidTerrainInput(
                    f"image has too many pixels; maximum is {max_pixels:,}"
                )
            probe.verify()
        with Image.open(io.BytesIO(data)) as source:
            source = ImageOps.exif_transpose(source)
            rgb = _to_rgb_on_white(source)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidTerrainInput("file is not a supported terrain image") from exc

    resolved_type = _resolve_input_type(rgb, input_type)
    warnings: list[str] = []
    if input_type == "auto":
        warnings.append(
            f"input type was detected as {resolved_type}; specify input_type explicitly if incorrect"
        )

    if resolved_type == "grayscale_dem":
        model_input, elevation, report = _preprocess_grayscale_dem(
            rgb,
            high_is_bright=high_is_bright,
            blur_radius=blur_radius,
            analysis_max_dimension=analysis_max_dimension,
            model_image_size=model_image_size,
        )
        elevation_npz = _array_to_npz(elevation)
    else:
        model_input, report = _preprocess_contour_image(rgb, model_image_size)
        elevation_npz = None
        warnings.append(
            "contour-image mode preserves visible lines but cannot recover absolute elevation values"
        )

    quality_score = float(report["quality_score"])
    if quality_score < 0.20:
        raise InvalidTerrainInput(
            "input quality is too low for inference; improve contrast or upload a DEM/GeoTIFF"
        )

    return PreprocessedTerrain(
        resolved_input_type=resolved_type,
        original_width=width,
        original_height=height,
        model_width=model_image_size,
        model_height=model_image_size,
        model_input_png=_image_to_png(model_input),
        elevation_npz=elevation_npz,
        quality_score=round(quality_score, 6),
        quality_report=report,
        warnings=tuple(warnings),
    )


def _resolve_input_type(image: Image.Image, requested: str) -> str:
    if requested != "auto":
        return requested
    sample = np.asarray(image.resize((64, 64)), dtype=np.int16)
    channel_spread = np.max(sample, axis=2) - np.min(sample, axis=2)
    return "grayscale_dem" if float(channel_spread.mean()) < 3.0 else "contour_image"


def _preprocess_grayscale_dem(
    image: Image.Image,
    *,
    high_is_bright: bool,
    blur_radius: float,
    analysis_max_dimension: int,
    model_image_size: int,
) -> tuple[Image.Image, np.ndarray, dict[str, object]]:
    width, height = image.size
    analysis_size = _fit_size(width, height, analysis_max_dimension)
    grayscale = image.convert("L").resize(analysis_size, Image.Resampling.BILINEAR)
    if blur_radius > 0:
        grayscale = grayscale.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    elevation = np.asarray(grayscale, dtype=np.float32) / 255.0
    if not high_is_bright:
        elevation = 1.0 - elevation
    value_range = float(np.ptp(elevation))
    standard_deviation = float(np.std(elevation))
    if value_range < 0.03:
        raise InvalidTerrainInput(
            "elevation contrast is too low; pixel brightness must represent relative elevation"
        )
    elevation = (elevation - float(elevation.min())) / value_range
    quality_score = min(1.0, 0.65 * (value_range / 0.30) + 0.35 * (standard_deviation / 0.18))
    contour_rgb = _render_contour_rgb(elevation, model_image_size)
    report: dict[str, object] = {
        "quality_score": round(max(0.0, quality_score), 6),
        "value_range": round(value_range, 6),
        "standard_deviation": round(standard_deviation, 6),
        "analysis_width": analysis_size[0],
        "analysis_height": analysis_size[1],
        "assumption": "pixel_brightness_represents_relative_elevation",
    }
    return Image.fromarray(contour_rgb, mode="RGB"), elevation.astype(np.float32), report


def _preprocess_contour_image(
    image: Image.Image, model_image_size: int
) -> tuple[Image.Image, dict[str, object]]:
    model_input = ImageOps.fit(
        image.convert("RGB"),
        (model_image_size, model_image_size),
        method=Image.Resampling.LANCZOS,
    )
    grayscale = np.asarray(model_input.convert("L"), dtype=np.float32) / 255.0
    edge_image = model_input.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_values = np.asarray(edge_image, dtype=np.float32) / 255.0
    contrast = float(np.std(grayscale))
    edge_density = float(np.mean(edge_values > 0.12))
    quality_score = min(1.0, 0.55 * (contrast / 0.20) + 0.45 * (edge_density / 0.18))
    report: dict[str, object] = {
        "quality_score": round(max(0.0, quality_score), 6),
        "contrast_standard_deviation": round(contrast, 6),
        "edge_density": round(edge_density, 6),
        "assumption": "uploaded_image_already_represents_contours",
    }
    return model_input, report


def _render_contour_rgb(elevation: np.ndarray, output_size: int) -> np.ndarray:
    height, width = elevation.shape
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    dpi = 64
    figure = plt.figure(figsize=(output_size / dpi, output_size / dpi), dpi=dpi)
    axes = figure.add_axes([0, 0, 1, 1])
    axes.contourf(grid_x, grid_y, elevation, levels=20, cmap="terrain")
    axes.contour(
        grid_x, grid_y, elevation, levels=20, colors="black", linewidths=0.3
    )
    axes.set_axis_off()
    axes.set_aspect("equal")
    figure.canvas.draw()
    canvas_width, canvas_height = figure.canvas.get_width_height()
    rgba = np.frombuffer(figure.canvas.buffer_rgba(), dtype=np.uint8).reshape(
        canvas_height, canvas_width, 4
    )
    rgb = rgba[:, :, :3].copy()
    plt.close(figure)
    if rgb.shape[:2] != (output_size, output_size):
        return np.asarray(
            Image.fromarray(rgb).resize(
                (output_size, output_size), Image.Resampling.BILINEAR
            ),
            dtype=np.uint8,
        )
    return rgb


def _fit_size(width: int, height: int, maximum: int) -> tuple[int, int]:
    if max(width, height) <= maximum:
        return width, height
    ratio = maximum / max(width, height)
    return max(16, round(width * ratio)), max(16, round(height * ratio))


def _to_rgb_on_white(image: Image.Image) -> Image.Image:
    if "A" not in image.getbands():
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    return Image.alpha_composite(background, rgba).convert("RGB")


def _array_to_npz(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(output, elevation=array.astype(np.float32))
    return output.getvalue()


def _image_to_png(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG", compress_level=3)
    return output.getvalue()
