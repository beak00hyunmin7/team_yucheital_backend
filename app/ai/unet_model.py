from __future__ import annotations

import logging
import os

import numpy as np
from PIL import Image

from app.services.hydrology import HydrologyResult, recommend_drains_from_score

logger = logging.getLogger(__name__)

MODEL_VERSION = os.environ.get("UNET_MODEL_VERSION", "contour2flow-unet-v1")
MODEL_NAME = "contour2flow U-Net (AI 모델링 팀 제공)"


class UnetDrainageModel:
    """deploy_package의 학습된 U-Net(등고선 -> 예측 배수맵)을 배수구 추천에 연결하는 어댑터.

    DrainageModel 프로토콜(app/ai/interface.py)을 만족시켜 D8 대리 모델과
    동일한 방식으로 app/services/analysis_runner.py에서 호출된다.

    U-Net은 좌표 리스트가 아니라 "예측 배수맵 이미지"를 출력하므로, 이 이미지를
    배수구 위치를 고르는 적합도(score)로 변환한 뒤
    hydrology.recommend_drains_from_score()에 넘겨 실제 배수구 좌표/카탈로그
    (HydrologyResult)를 만든다. D8 흐름누적·다운스트림 그래프는 여전히 원본
    고도에서 계산되므로, 자동 검증(automatic_validation.py)과 워터맵 렌더링은
    그대로 유효하다 — 이 모델은 오직 "어디에 배수구를 둘지"만 AI로 대체한다.
    """

    version = MODEL_VERSION
    name = MODEL_NAME

    def predict(
        self,
        elevation: np.ndarray,
        *,
        drain_count: int,
        minimum_spacing_ratio: float,
        attempt_number: int,
    ) -> HydrologyResult:
        predicted_suitability, predicted_rgb = self._predict_suitability_map(elevation)
        return recommend_drains_from_score(
            elevation,
            predicted_suitability,
            drain_count=drain_count,
            minimum_spacing_ratio=minimum_spacing_ratio,
            visualization_image=predicted_rgb,
        )

    def _predict_suitability_map(
        self, elevation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # 지연 import: torch/torchvision이 없거나 체크포인트가 없어도
        # (AI_BACKEND=d8인) 다른 경로는 이 모듈을 아예 건드리지 않도록.
        from app.ai.unet_infer.inference_api import predict_from_image
        from app.ai.unet_infer.utils.render_fields import render_contour_rgb

        height, width = elevation.shape
        x = np.arange(width, dtype=np.float64)
        y = np.arange(height, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(x, y)

        # 학습 데이터와 같은 방식(matplotlib "terrain" 컬러맵 등고선 렌더)으로
        # 정규화된 고도를 등고선 이미지로 바꿔서 모델에 넣는다. 사용자가 원래
        # 어떤 이미지(등고선/색칠된 지형도/grayscale DEM)를 올렸든, prepare_dem이
        # 만들어낸 0~1 고도 격자를 기준으로 매번 같은 형식의 입력을 재구성하므로
        # 업로드 형식에 관계없이 U-Net이 학습 때 본 것과 같은 스타일의 입력을 받는다.
        contour_rgb = render_contour_rgb(grid_x, grid_y, elevation)
        predicted = predict_from_image(Image.fromarray(contour_rgb))

        # 학습 타깃(data/flow/*.png)은 matplotlib "Blues" 컬러맵으로 그린 수심맵이라
        # 값이 클수록(수심이 깊을수록) 더 진한 파란색 = 밝기(luminance)가 낮다.
        # 배수구 적합도는 "값이 클수록 좋음"이어야 하므로 밝기를 반전한다.
        predicted_gray = np.asarray(predicted.convert("L"), dtype=np.float64)
        suitability = 255.0 - predicted_gray

        if predicted.size != (width, height):
            suitability_img = Image.fromarray(suitability.astype(np.uint8))
            suitability_img = suitability_img.resize((width, height), Image.Resampling.BILINEAR)
            suitability = np.asarray(suitability_img, dtype=np.float64)
            predicted = predicted.resize((width, height), Image.Resampling.BILINEAR)

        # Keep the model's own predicted image (not just the score derived
        # from it) so the water-map/overlay endpoints can show exactly what
        # the U-Net produced, instead of a re-rendered D8 flow-accumulation
        # heatmap that has nothing to do with the model's prediction.
        predicted_rgb = np.asarray(predicted.convert("RGB"), dtype=np.uint8)

        return suitability, predicted_rgb


def build_model() -> UnetDrainageModel | None:
    """체크포인트/torch가 준비돼 있으면 모델을 만들고, 아니면 None + 경고 로그.

    main.py에서 AI_BACKEND=unet일 때 이 함수로 모델을 준비하고, 실패하면
    D8 대리 모델로 자동 폴백한다 (서버가 아예 안 뜨는 것보다는 낫다).
    """
    try:
        from app.ai.unet_infer.inference_api import _load_model
    except ImportError as exc:
        # inference_api.py가 모듈 최상단에서 torch를 import하므로, torch/torchvision이
        # 없으면 CheckpointNotFoundError 같은 이 모듈 안의 이름을 아직 참조할 수 없는
        # 상태로 여기서 바로 실패한다. 그래서 이 import 자체를 별도 try로 분리했다.
        logger.warning(
            "torch/torchvision이 설치되어 있지 않아 D8 대리 모델로 대체합니다: %s", exc
        )
        return None

    try:
        _load_model()
        return UnetDrainageModel()
    except FileNotFoundError as exc:
        # CheckpointNotFoundError(FileNotFoundError의 하위클래스) 포함.
        logger.warning("U-Net 체크포인트를 찾지 못해 D8 대리 모델로 대체합니다: %s", exc)
    except Exception:  # noqa: BLE001 - 모델 로딩 실패는 서버 기동을 막지 않아야 함
        logger.exception("U-Net 모델 로딩 실패, D8 대리 모델로 대체합니다.")
    return None
