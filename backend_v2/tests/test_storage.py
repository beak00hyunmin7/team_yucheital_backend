from __future__ import annotations

from pathlib import Path

import pytest

from app.services.storage import FileStorage


def test_storage_writes_atomically_and_rejects_path_escape(tmp_path: Path) -> None:
    storage = FileStorage(tmp_path / "assets")
    storage.ensure()
    saved = storage.save("terrain/id/original.png", b"png-data", "image/png")
    assert storage.read(saved.relative_path) == b"png-data"
    assert len(saved.sha256) == 64
    with pytest.raises(ValueError):
        storage.save("../outside.txt", b"bad", "text/plain")
