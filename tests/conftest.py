from __future__ import annotations

import os
import tempfile


_TEST_ROOT = tempfile.TemporaryDirectory(prefix="terrain-drainage-tests-")
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_ROOT.name}/test.db"
os.environ["DATA_ROOT"] = f"{_TEST_ROOT.name}/data"
os.environ["STORAGE_ROOT"] = f"{_TEST_ROOT.name}/storage"
