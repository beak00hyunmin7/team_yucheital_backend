from __future__ import annotations

import json

from app.ai.runtime import DrainageModelRuntime
from app.config import settings


def main() -> int:
    runtime = DrainageModelRuntime(
        checkpoint_path=settings.checkpoint_path,
        model_version=settings.model_version,
        image_size=settings.model_image_size,
        concurrency=1,
        use_fp16=False,
        warmup=True,
    )
    status = runtime.try_load()
    print(json.dumps(status.__dict__, ensure_ascii=False, indent=2))
    return 0 if status.available else 1


if __name__ == "__main__":
    raise SystemExit(main())
