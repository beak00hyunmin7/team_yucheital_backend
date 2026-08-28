from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from app.config import STORAGE_ROOT


@dataclass(frozen=True)
class StoredAsset:
    kind: str
    relative_path: str
    mime_type: str
    file_hash: str
    width: int
    height: int

    @property
    def url(self) -> str:
        return f"/files/{self.relative_path}"


_CONTENT_TYPE_SUFFIX = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/tiff": ".tiff",
}


def ensure_storage() -> None:
    (STORAGE_ROOT / "analysis_cases").mkdir(parents=True, exist_ok=True)


def save_original(
    *, case_id: str, data: bytes, content_type: str | None, width: int, height: int
) -> StoredAsset:
    suffix = _CONTENT_TYPE_SUFFIX.get(content_type or "", ".png")
    return _save(
        case_id=case_id,
        filename=f"input{suffix}",
        kind="INPUT_IMAGE",
        data=data,
        mime_type=content_type or "image/png",
        width=width,
        height=height,
    )


def save_png(*, case_id: str, filename: str, kind: str, data: bytes) -> StoredAsset:
    case_dir = STORAGE_ROOT / "analysis_cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / filename
    path.write_bytes(data)
    with Image.open(path) as image:
        width, height = image.size
    return _asset_from_path(
        path=path,
        kind=kind,
        mime_type="image/png",
        data=data,
        width=width,
        height=height,
    )


def read_asset(relative_path: str) -> bytes:
    path = (STORAGE_ROOT / relative_path).resolve()
    if STORAGE_ROOT not in path.parents:
        raise ValueError("잘못된 저장 경로입니다.")
    return path.read_bytes()


def asset_url(relative_path: str) -> str:
    return f"/files/{relative_path}"


def _save(
    *,
    case_id: str,
    filename: str,
    kind: str,
    data: bytes,
    mime_type: str,
    width: int,
    height: int,
) -> StoredAsset:
    case_dir = STORAGE_ROOT / "analysis_cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / filename
    path.write_bytes(data)
    return _asset_from_path(
        path=path,
        kind=kind,
        mime_type=mime_type,
        data=data,
        width=width,
        height=height,
    )


def _asset_from_path(
    *, path: Path, kind: str, mime_type: str, data: bytes, width: int, height: int
) -> StoredAsset:
    return StoredAsset(
        kind=kind,
        relative_path=path.relative_to(STORAGE_ROOT).as_posix(),
        mime_type=mime_type,
        file_hash=hashlib.sha256(data).hexdigest(),
        width=width,
        height=height,
    )
