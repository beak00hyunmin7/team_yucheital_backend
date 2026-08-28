"""
백엔드 연동용 경량 추론 API.

deploy_package/src(AI 모델링 팀 제공)를 이 백엔드 안으로 그대로 옮겨온 버전입니다.
로직은 원본과 동일하고, import 경로만 이 프로젝트의 app.ai.unet_infer 패키지 구조에
맞게 상대 import로 바꿨습니다.

학습에 쓰인 TranslationModel(옵티마이저 + 판별자까지 포함하는 래퍼) 대신
UnetGenerator만 직접 불러와서 추론만 수행한다. mode: "gan"으로 학습한
체크포인트도 생성자 구조(UnetGenerator)는 mode: "unet"과 동일하므로
코드 변경 없이 그대로 로드된다 (판별자는 추론에 쓰이지 않음).

체크포인트 위치:
    기본값은 이 백엔드와 같은 상위 폴더에 나란히 있는 deploy_package/checkpoints/best.pt
    입니다 (686MB 파일이라 백엔드 저장소 안으로 복사하지 않고 그대로 참조합니다).
    다른 경로를 쓰려면 환경변수 AI_MODEL_CHECKPOINT를 지정하세요.
"""

import io
import os
import threading

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .models.unet_generator import UnetGenerator
from .utils.image_utils import tensor_to_uint8
from .utils.render_fields import render_contour_rgb

# terrain-drainage-fastapi/app/ai/unet_infer/inference_api.py 기준으로
# ../../../../deploy_package/checkpoints/best.pt (Desktop에 두 프로젝트가 나란히
# 있는 구조를 가정). 다르게 배치했다면 AI_MODEL_CHECKPOINT 환경변수로 덮어쓰세요.
_DEFAULT_SIBLING_CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "deploy_package", "checkpoints", "best.pt"
)
CHECKPOINT_PATH = os.environ.get("AI_MODEL_CHECKPOINT", _DEFAULT_SIBLING_CHECKPOINT)
IMAGE_SIZE = int(os.environ.get("AI_MODEL_IMAGE_SIZE", "256"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_loaded_checkpoint_path = None
_model_lock = threading.Lock()

_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


class CheckpointNotFoundError(FileNotFoundError):
    """AI_MODEL_CHECKPOINT 경로에서 체크포인트(.pt)를 찾지 못했을 때."""


def _load_model(checkpoint_path=None):
    global _model, _loaded_checkpoint_path
    path = checkpoint_path or CHECKPOINT_PATH
    with _model_lock:
        if _model is not None and _loaded_checkpoint_path == path:
            return _model
        resolved = os.path.abspath(path)
        if not os.path.isfile(resolved):
            raise CheckpointNotFoundError(
                f"AI 모델 체크포인트를 찾을 수 없습니다: {resolved}\n"
                "deploy_package 폴더 위치가 다르면 AI_MODEL_CHECKPOINT 환경변수로 "
                "checkpoints/best.pt 경로를 지정하세요."
            )
        net = UnetGenerator(in_channels=3, out_channels=3, image_size=IMAGE_SIZE).to(DEVICE)
        state = torch.load(resolved, map_location=DEVICE)
        net.load_state_dict(state["netG"])
        net.eval()
        _model = net
        _loaded_checkpoint_path = path
        return _model


def reload_model(checkpoint_path):
    """서빙 도중 다른 체크포인트로 즉시 교체한다 (프로세스 재시작 불필요)."""
    global _model
    with _model_lock:
        _model = None
    return _load_model(checkpoint_path)


def current_checkpoint():
    """지금 서빙 중인 체크포인트 경로 (헬스체크/버전 응답용)."""
    return _loaded_checkpoint_path or CHECKPOINT_PATH


@torch.no_grad()
def _run(contour_img: Image.Image) -> Image.Image:
    model = _load_model()
    input_t = _transform(contour_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    pred_t = model(input_t)
    return Image.fromarray(tensor_to_uint8(pred_t[0]))


def predict_from_image(image) -> Image.Image:
    """등고선 이미지(경로 1)로 배수 예측.

    image: PIL.Image, 파일 경로(str), 또는 raw bytes(업로드 파일 등) 모두 허용.
    """
    if isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image))
    elif isinstance(image, str):
        image = Image.open(image)
    return _run(image)


def predict_from_terrain(terrain: np.ndarray, dx: float = 10.416666666666666) -> Image.Image:
    """지형 원시 데이터(경로 2)로 배수 예측.

    terrain: (H, W) 고도 배열 [m]
    dx: 셀 크기 [m] (학습 데이터는 1000m 도메인 / 96 격자 = 약 10.4167m 사용)

    학습 때와 동일한 렌더링(render_contour_rgb)으로 먼저 등고선 이미지를 만든 뒤 추론한다.
    """
    ny, nx = terrain.shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dx
    X, Y = np.meshgrid(x, y)
    contour_rgb = render_contour_rgb(X, Y, terrain)
    return _run(Image.fromarray(contour_rgb))
