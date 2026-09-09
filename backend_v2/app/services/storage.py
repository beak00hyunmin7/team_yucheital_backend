from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class StoredFile:
    relative_path: str
    sha256: str
    size_bytes: int
    mime_type: str


class FileStorage:
    """Atomic local storage with paths constrained below one storage root.

    The interface intentionally mirrors object storage semantics. Replacing the
    implementation with S3/MinIO therefore does not change the API or DB model.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def is_writable(self) -> bool:
        try:
            self.ensure()
            with tempfile.NamedTemporaryFile(dir=self.root, prefix=".ready-", delete=True):
                pass
            return True
        except OSError:
            return False

    def save(self, relative_path: str, data: bytes, mime_type: str) -> StoredFile:
        target = self._resolve(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, target)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return StoredFile(
            relative_path=PurePosixPath(relative_path).as_posix(),
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            mime_type=mime_type,
        )

    def read(self, relative_path: str) -> bytes:
        return self._resolve(relative_path).read_bytes()

    def exists(self, relative_path: str) -> bool:
        return self._resolve(relative_path).is_file()

    def delete(self, relative_path: str) -> None:
        try:
            self._resolve(relative_path).unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def public_url(relative_path: str) -> str:
        return f"/files/{PurePosixPath(relative_path).as_posix()}"

    def _resolve(self, relative_path: str) -> Path:
        relative = PurePosixPath(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("storage path must be relative and may not contain '..'")
        resolved = (self.root / Path(*relative.parts)).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError("storage path escapes the configured root")
        return resolved
