from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError


@dataclass(frozen=True)
class PreparedDem:
    elevation: np.ndarray
    original_rgb: Image.Image
    original_width: int
    original_height: int
    analysis_width: int
    analysis_height: int
    scale_x: float
    scale_y: float


class InvalidTerrainImage(ValueError):
    """Raised when an upload cannot be interpreted as a grayscale DEM."""


def prepare_dem(
    data: bytes,
    *,
    max_pixels: int,
    analysis_max_dimension: int,
    high_is_bright: bool,
    blur_radius: float,
) -> PreparedDem:
    try:
        with Image.open(BytesIO(data)) as probe:
            width, height = probe.size
            if width < 16 or height < 16:
                raise InvalidTerrainImage("이미지의 가로와 세로는 각각 16픽셀 이상이어야 합니다.")
            if width * height > max_pixels:
                raise InvalidTerrainImage(
                    f"이미지가 너무 큽니다. 최대 허용 픽셀 수는 {max_pixels:,}입니다."
                )
            probe.verify()
        with Image.open(BytesIO(data)) as source:
            source = ImageOps.exif_transpose(source)
            width, height = source.size
            original_rgb = _to_rgb_on_white(source)
            grayscale = _to_grayscale_on_white(source)
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidTerrainImage("PNG, JPEG, WEBP 또는 TIFF 이미지가 아닙니다.") from exc

    analysis_size = _fit_size(width, height, analysis_max_dimension)
    grayscale = grayscale.resize(analysis_size, Image.Resampling.BILINEAR)
    if blur_radius > 0:
        grayscale = grayscale.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    elevation = np.asarray(grayscale, dtype=np.float64) / 255.0
    if not high_is_bright:
        elevation = 1.0 - elevation

    elevation_range = float(np.ptp(elevation))
    if elevation_range < 0.03:
        raise InvalidTerrainImage(
            "고도 대비가 너무 작습니다. 픽셀 밝기가 고도를 나타내는 DEM 이미지를 사용해 주세요."
        )

    # Normalize so elevation-related scores remain comparable across images.
    elevation = (elevation - float(elevation.min())) / elevation_range
    analysis_width, analysis_height = analysis_size
    return PreparedDem(
        elevation=elevation,
        original_rgb=original_rgb,
        original_width=width,
        original_height=height,
        analysis_width=analysis_width,
        analysis_height=analysis_height,
        scale_x=width / analysis_width,
        scale_y=height / analysis_height,
    )


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


def _to_grayscale_on_white(image: Image.Image) -> Image.Image:
    return _to_rgb_on_white(image).convert("L")
