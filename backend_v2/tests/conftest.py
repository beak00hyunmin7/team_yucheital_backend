from __future__ import annotations

import os
import tempfile

_test_root = tempfile.mkdtemp(prefix="terrain-backend-v2-tests-")
os.environ["V2_DATA_ROOT"] = _test_root
os.environ["V2_STORAGE_ROOT"] = os.path.join(_test_root, "storage")
os.environ["V2_DATABASE_URL"] = f"sqlite:///{os.path.join(_test_root, 'test.db')}"
os.environ["V2_AUTO_CREATE_TABLES"] = "true"
os.environ["AI_WARMUP_MODEL"] = "false"
os.environ["SCENARIO_EVALUATOR"] = "capacity_approx"
