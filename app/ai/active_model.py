from __future__ import annotations

import logging

from app.ai.hydrology_model import model as d8_model
from app.config import AI_BACKEND

logger = logging.getLogger(__name__)


def _select_model():
    if AI_BACKEND == "d8":
        logger.info("AI_BACKEND=d8 -> D8 대리 모델을 사용합니다.")
        return d8_model

    from app.ai.unet_model import build_model

    unet_model = build_model()
    if unet_model is not None:
        logger.info("U-Net 모델(%s)을 사용합니다.", unet_model.version)
        return unet_model

    logger.warning(
        "AI_BACKEND=%s 이지만 U-Net을 준비하지 못해 D8 대리 모델로 대체합니다.",
        AI_BACKEND,
    )
    return d8_model


# main.py가 import하는 단일 진입점. 서버 기동 시 한 번만 모델을 고른다
# (app/ai/hydrology_model.py의 기존 `model` 싱글턴 패턴과 동일).
model = _select_model()

# main.py가 "현재 D8 대리 모델을 쓰고 있다"는 경고를 실제로 D8일 때만 보여주기
# 위한 플래그. U-Net이 정상 로드된 경우에도 이 경고가 나가면 사용자가 "AI 모델이
# 안 붙었다"고 오해하게 된다.
IS_D8_FALLBACK = model is d8_model
