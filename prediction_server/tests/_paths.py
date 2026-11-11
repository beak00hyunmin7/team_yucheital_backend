"""테스트가 서버 모듈(backend/)과 AI 모델 코드(서브모듈)를 import 할 수 있게 경로를 잡는다.

torch·DB 없이 도는 테스트만 둔다 (CI 나 AI 체크포인트가 없는 PC 에서도 돌 수 있게).
"""
import os
import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = SERVER_DIR / "backend"
AI_MODEL_DIR = Path(os.environ.get("AI_MODEL_DIR", SERVER_DIR / "ai" / "terrainFlowSim" / "ai_model"))

for p in (BACKEND_DIR, AI_MODEL_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
