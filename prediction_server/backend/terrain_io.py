"""
사용자가 올린 '등고선 데이터'(고도 격자)를 numpy (H, W) float 배열로 파싱한다.

실무에서 지형/DEM 데이터는 포맷이 제각각이라, 확장자별로 대표적인 것들을 처리한다.

  .npy            numpy 배열 그대로
  .csv .txt .dat  숫자 격자 (콤마 또는 공백 구분). 헤더 줄이 있어도 자동 무시
  .asc .grd       ESRI ASCII Grid (ncols/nrows/cellsize/NODATA_value 헤더 6줄)
  .tif .tiff      단일 밴드 GeoTIFF/일반 TIFF (Pillow로 읽히는 경우)
  .png .jpg ...    회색조 높이맵(밝을수록 높음)으로 해석

반환: (terrain(np.float32 2D), meta(dict))
  meta = {"source_format", "grid_shape", "cellsize_m"(있으면), "nodata_filled"(개수),
          "warnings"(사용자에게 보여줄 주의 문구 목록)}

바꿔야 할 때: 팀에서 쓰는 실제 포맷이 정해지면 아래 _PARSERS에 함수 하나만 추가하면 된다.
GeoTIFF 좌표계/투영까지 제대로 다루려면 rasterio(gdal)를 넣고 _parse_tiff를 교체.
"""
from __future__ import annotations

import io
import re

import numpy as np
from PIL import Image

MAX_GRID_DIM = 512          # 이보다 큰 격자는 등간격 샘플링으로 축소 (등고선 렌더 속도)
MIN_GRID_DIM = 8
# DEM 은 빈칸(바다·측정 안 된 곳)을 -9999, -32768, -3.4e38 같은 값으로 채우는 경우가 많다.
# 지표 고도가 -1000 m 아래인 곳은 없으므로(사해 -430 m) 그보다 낮거나 비정상적으로 큰 값은 빈칸으로 본다.
NODATA_BELOW = -1000.0
NODATA_ABOVE = 1e20


class InvalidTerrainData(ValueError):
    """등고선 데이터로 해석할 수 없는 업로드."""


def load_terrain(raw: bytes, filename: str) -> tuple[np.ndarray, dict]:
    ext = _ext(filename)
    parser = _PARSERS.get(ext)
    if parser is None:
        raise InvalidTerrainData(
            f"지원하지 않는 확장자입니다: '{ext or '없음'}'. "
            "지원: .npy .csv .txt .dat .asc .grd .tif .tiff .png .jpg .jpeg .webp"
        )

    try:
        terrain, meta = parser(raw)
    except InvalidTerrainData:
        raise
    except Exception as exc:  # noqa: BLE001  (파서가 예상 못 한 형식 -> 500 대신 안내)
        raise InvalidTerrainData(f"파일을 '.{ext}' 형식으로 읽지 못했습니다. 파일 내용과 확장자가 맞는지 확인해 주세요.") from exc
    meta.setdefault("warnings", [])
    terrain = np.asarray(terrain, dtype=np.float64)

    if terrain.ndim == 3:
        # (H, W, C) 로 들어온 경우 첫 밴드만 사용
        terrain = terrain[..., 0]
    if terrain.ndim != 2:
        raise InvalidTerrainData(f"2차원 격자가 아닙니다 (shape={terrain.shape}).")
    if min(terrain.shape) < MIN_GRID_DIM:
        raise InvalidTerrainData(f"격자가 너무 작습니다 (shape={terrain.shape}, 최소 {MIN_GRID_DIM}x{MIN_GRID_DIM}).")

    # NODATA / NaN / inf 정리
    # 예전엔 빈칸을 '가장 낮은 고도'로 채워서, 그 자리가 인공 구덩이가 되어 물이 모였다(예측 왜곡).
    # 이제 가장 가까운 정상 칸의 값으로 채운다. -9999 같은 빈칸 표시값도 빈칸으로 본다.
    bad = ~np.isfinite(terrain) | (terrain <= NODATA_BELOW) | (terrain >= NODATA_ABOVE)
    nodata_filled = int(bad.sum())
    if nodata_filled:
        if nodata_filled == terrain.size:
            raise InvalidTerrainData("모든 칸이 빈칸(NaN 또는 -9999 같은 빈칸 값)입니다. 실제 고도 격자를 넣어주세요.")
        from scipy.ndimage import distance_transform_edt
        _, (iy, ix) = distance_transform_edt(bad, return_indices=True)
        terrain = terrain[iy, ix]
        pct = nodata_filled / bad.size * 100
        meta["warnings"].append(
            f"빈칸(NaN 또는 -9999 같은 빈칸 값) {nodata_filled}칸({pct:.1f}%)을 가까운 칸의 고도로 채웠습니다."
            + (" 빈칸이 많아 결과를 믿기 어렵습니다." if pct > 20 else ""))
    meta["nodata_filled"] = meta.get("nodata_filled", 0) + nodata_filled

    if float(np.ptp(terrain)) < 1e-9:
        raise InvalidTerrainData("고도 값이 전부 동일합니다. 실제 지형 격자를 넣어주세요.")

    # 너무 큰 격자 축소
    meta["original_shape"] = list(terrain.shape)
    terrain, cellsize = _downsample(terrain, meta.get("cellsize_m"))
    if cellsize is not None:
        meta["cellsize_m"] = cellsize

    meta["grid_shape"] = list(terrain.shape)
    return terrain.astype(np.float32), meta


# --------------------------------------------------------------------------- #
# 확장자별 파서
# --------------------------------------------------------------------------- #
def _parse_npy(raw: bytes) -> tuple[np.ndarray, dict]:
    try:
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        raise InvalidTerrainData(".npy 를 읽지 못했습니다. 파일이 손상되었거나 숫자 배열(numpy.save 로 저장)이 아닙니다.") from exc
    return arr, {"source_format": "npy"}


def _parse_text_grid(raw: bytes) -> tuple[np.ndarray, dict]:
    text = raw.decode("utf-8-sig", errors="replace")
    delimiter = "," if text.count(",") > text.count("\t") and "," in text else None
    rows = []
    non_numeric = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"[,\s]+", line) if delimiter is None else line.split(",")
        try:
            rows.append([float(p) for p in parts if p != ""])
        except ValueError:
            # 헤더/주석 줄은 건너뜀
            non_numeric += 1
            continue
    if not rows:
        raise InvalidTerrainData("숫자 격자를 찾지 못했습니다.")
    width = max(len(r) for r in rows)
    ragged = sum(1 for r in rows if len(r) != width)
    rows = [r for r in rows if len(r) == width]      # 폭이 안 맞는 줄 제거
    if len(rows) < MIN_GRID_DIM:
        raise InvalidTerrainData("일관된 숫자 행이 너무 적습니다.")
    warnings = []
    # 첫 줄 헤더 1줄은 흔하므로 조용히 넘긴다. 그 외에 버린 줄이 있으면 알린다 (지도가 그만큼 줄어든다)
    if non_numeric > 1:
        warnings.append(f"숫자가 아닌 값이 섞인 줄 {non_numeric}개를 건너뛰었습니다. 지도가 그만큼 줄어들었을 수 있습니다.")
    if ragged:
        warnings.append(f"칸 수가 다른 줄 {ragged}개를 건너뛰었습니다(가장 긴 줄 {width}칸 기준). 지도가 그만큼 줄어들었을 수 있습니다.")
    return np.array(rows, dtype=np.float64), {"source_format": "text", "warnings": warnings}


def _parse_esri_ascii(raw: bytes) -> tuple[np.ndarray, dict]:
    text = raw.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    header = {}
    body_start = 0
    for i, line in enumerate(lines[:8]):
        m = re.match(r"^\s*([A-Za-z_]+)\s+(-?[\d.eE+]+)\s*$", line)
        if not m:
            body_start = i
            break
        header[m.group(1).lower()] = float(m.group(2))
        body_start = i + 1

    values: list[float] = []
    for line in lines[body_start:]:
        values.extend(float(v) for v in line.replace(",", " ").split())

    ncols = int(header.get("ncols", 0))
    nrows = int(header.get("nrows", 0))
    if ncols and nrows and len(values) >= ncols * nrows:
        grid = np.array(values[: ncols * nrows], dtype=np.float64).reshape(nrows, ncols)
    else:
        raise InvalidTerrainData("ESRI ASCII Grid 헤더(ncols/nrows)와 본문 크기가 맞지 않습니다.")

    meta = {"source_format": "esri_ascii"}
    nodata = header.get("nodata_value")
    if nodata is not None:
        grid[grid == nodata] = np.nan
    cellsize = header.get("cellsize")
    if cellsize:
        meta["cellsize_m"] = float(cellsize)
    return grid, meta


def _is_colorful(arr: np.ndarray) -> bool:
    """(H, W, 3+) 이미지가 회색조가 아니라 실제 컬러 그림인지 (채널끼리 눈에 띄게 다른 픽셀이 5% 넘음)."""
    if arr.ndim != 3 or arr.shape[2] < 3:
        return False
    rgb = arr[..., :3].astype(np.float64)
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    scale = max(float(rgb.max()), 1e-9)
    return float(np.mean(spread / scale > 0.08)) > 0.05


_COLOR_IMAGE_MSG = ("컬러 그림이라 고도 데이터로 읽을 수 없습니다(색은 고도가 아닙니다). "
                    "등고선 그림이면 '등고선 이미지' 모드로 올리고, 고도 데이터는 DEM(.tif/.asc/.npy/.csv)이나 "
                    "회색조 높이맵을 쓰세요.")


def _parse_tiff(raw: bytes) -> tuple[np.ndarray, dict]:
    try:
        with Image.open(io.BytesIO(raw)) as im:
            arr = np.array(im)
            tags = dict(getattr(im, "tag_v2", {}) or {})
    except Exception as exc:  # noqa: BLE001
        raise InvalidTerrainData(
            "TIFF 를 읽지 못했습니다. 파일이 손상되었거나, 압축 방식이 특이한 GeoTIFF 일 수 있습니다. "
            "GIS 프로그램에서 .asc(ESRI ASCII Grid)로 저장해서 올려주세요."
        ) from exc
    meta: dict = {"source_format": "tiff", "warnings": []}
    if _is_colorful(arr):
        raise InvalidTerrainData(_COLOR_IMAGE_MSG)
    if arr.dtype == np.uint8:
        meta["warnings"].append("8비트 TIFF 라 고도가 256단계(0~255)로만 저장되어 있습니다. 실제 고도(m)가 아니면 "
                                "분석 지도의 고도 숫자는 의미가 없고, 지형이 계단처럼 거칠 수 있습니다.")
    arr = arr.astype(np.float64)

    # GDAL 이 쓰는 빈칸 값 태그(42113). 있으면 그 값을 빈칸으로 본다
    nod = tags.get(42113)
    if nod is not None:
        try:
            arr[arr == float(str(nod).strip().strip("\x00"))] = np.nan
        except ValueError:
            pass
    # GeoTIFF 셀 크기(ModelPixelScale 33550). 좌표계가 위경도(GeoKey 1024 = 2)면 '도' 단위라 미터로 근사 환산
    scale = tags.get(33550)
    if scale:
        try:
            sx = abs(float(scale[0]))
            geokeys = tags.get(34735) or ()
            geographic = any(geokeys[i] == 1024 and geokeys[i + 3] == 2 for i in range(4, len(geokeys) - 3, 4))
            if geographic or sx < 0.01:
                lat = 0.0
                tie = tags.get(33922)
                if tie and len(tie) >= 5:
                    lat = float(tie[4])
                meters = sx * 111_320.0 * max(np.cos(np.radians(lat)), 0.1)
                meta["cellsize_m"] = meters
                meta["warnings"].append(f"위경도 좌표(셀 {sx:.6g}°)라 셀 크기를 약 {meters:.1f} m 로 환산했습니다.")
            elif sx > 0:
                meta["cellsize_m"] = sx
        except (TypeError, ValueError, IndexError):
            pass
    return arr, meta


def _parse_heightmap_image(raw: bytes) -> tuple[np.ndarray, dict]:
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            mode = im.mode
            if _is_colorful(np.asarray(im.convert("RGB"))):
                raise InvalidTerrainData(_COLOR_IMAGE_MSG)
            arr = np.asarray(im.convert("F"))
    except InvalidTerrainData:
        raise
    except Exception as exc:  # noqa: BLE001  (망가진 그림 파일: 예전엔 서버 에러 500)
        raise InvalidTerrainData("그림 파일을 열 수 없습니다. 파일이 손상되었거나 그림 파일이 아닙니다.") from exc
    warnings = ["높이맵 그림은 밝기를 고도로 읽습니다. 실제 고도(m)가 아니라서 분석 지도의 고도 숫자는 의미가 없습니다."]
    if mode not in ("I;16", "I;16B", "I;16L", "I", "F"):
        warnings.append("8비트 그림이라 고도가 256단계로만 표현되어 지형이 계단처럼 거칠 수 있습니다. 16비트 PNG 를 권장합니다.")
    return arr, {"source_format": "heightmap_image", "warnings": warnings}


_PARSERS = {
    "npy": _parse_npy,
    "csv": _parse_text_grid,
    "txt": _parse_text_grid,
    "dat": _parse_text_grid,
    "asc": _parse_esri_ascii,
    "grd": _parse_esri_ascii,
    "tif": _parse_tiff,
    "tiff": _parse_tiff,
    "png": _parse_heightmap_image,
    "jpg": _parse_heightmap_image,
    "jpeg": _parse_heightmap_image,
    "webp": _parse_heightmap_image,
}


# --------------------------------------------------------------------------- #
def _ext(filename: str) -> str:
    return filename.lower().rsplit(".", 1)[-1] if "." in (filename or "") else ""


def _downsample(terrain: np.ndarray, cellsize: float | None) -> tuple[np.ndarray, float | None]:
    step = max(1, int(np.ceil(max(terrain.shape) / MAX_GRID_DIM)))
    if step == 1:
        return terrain, cellsize
    return terrain[::step, ::step], (cellsize * step if cellsize else None)
