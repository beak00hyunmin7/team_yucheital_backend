"""
예측 배수맵 웹 백엔드 (FastAPI).

프론트엔드에서 다음을 받는다:
  - mode: "image"  -> 등고선 이미지를 그대로 모델에 입력
          "terrain" -> 등고선 데이터(고도 격자)를 학습 때와 동일한 방식으로
                       등고선 이미지로 렌더링한 뒤 모델에 입력
          "hand"    -> 손그림 사진(파랑 겹동그라미 + 빨강 꼭대기, 모서리 네모 4개)을
                       고도 격자로 읽은 뒤 "terrain" 과 똑같이 처리 (전시용, handdrawn.py)
  - rain_mm: 강우량(mm). FiLM 조건 체크포인트에서 사용
  - time_s : 강우 시작 후 경과 시간(초). cond_dim=2 체크포인트에서 사용
  - dx: (terrain 모드) 격자 셀 크기[m]. 등고선 이미지 모양에는 거의 영향 없음

응답(JSON):
  contour_png_base64     실제로 모델에 들어간 등고선 이미지
  prediction_png_base64  예측 배수맵
  overlay_png_base64     등고선 위에 예측을 겹친 이미지
  + 메타데이터

모델 연결:
  AI 저장소(서브모듈 prediction_server/ai)의 ai_model/src/inference_api.py 의 predict_from_image 를
  그대로 재사용한다. 체크포인트(best.pt)는 저장소에 없고 AI 담당자에게 받아
  prediction_server/checkpoints/best.pt 에 두거나 AI_MODEL_CHECKPOINT 로 지정한다.
  권장 체크포인트는 checkpoints_p2_wet8/best.pt (젖은 픽셀 가중 L1, 128px/10.5M).
  구버전 checkpoints_film 대비: 강우+시점 조건(cond_dim=2), 최대수심 비율 0.49->0.62,
  IoU@5cm 0.378->0.426 (ai_model/IMPROVEMENT_GUIDE.md, eval_p2_wet8/summary.md 참조).
  학습 해상도는 체크포인트에서 자동 판별하므로 AI_MODEL_IMAGE_SIZE 를 줄 필요 없다.
  다른 체크포인트로 바꾸려면 환경변수 AI_MODEL_CHECKPOINT 설정 후 재시작.
"""
from __future__ import annotations

import base64
import hashlib
import io
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from PIL import Image
from sqlalchemy.exc import OperationalError

# --- AI 모델 코드(서브모듈)를 import 경로에 추가하고 기본 체크포인트를 정한다 ----------
# 이 서버는 AI 저장소(team_yucheital_ai_model)의 terrainFlowSim/ai_model 을 가져다 쓴다.
# 기본 위치는 서브모듈 prediction_server/ai (커밋으로 AI 버전이 고정된다).
# 다른 위치를 쓰려면 환경변수 AI_MODEL_DIR (= .../terrainFlowSim/ai_model).
BACKEND_DIR = Path(__file__).resolve().parent
WEBAPP_DIR = BACKEND_DIR.parent
FRONTEND_DIR = WEBAPP_DIR / "frontend"   # 개발·검증용 화면(옛 index.html, /test). 정식 화면은 frontend 저장소
# 팀 프론트엔드(vtkjs.html + style.css) 폴더. 지정하면 이 서버가 화면까지 같이 내준다
# -> 같은 공유기의 다른 PC 는 http://<이 PC IP>:8000 하나만 열면 된다 (Live Server 불필요).
_site_env = os.environ.get("YUCHE_SITE_DIR", "").strip()
SITE_DIR = Path(_site_env).resolve() if _site_env else None
if SITE_DIR is not None and not (SITE_DIR / "vtkjs.html").is_file():
    SITE_DIR = None


def _find_ai_model_dir() -> Path:
    candidates = []
    if os.environ.get("AI_MODEL_DIR"):
        candidates.append(Path(os.environ["AI_MODEL_DIR"]))
    candidates.append(WEBAPP_DIR / "ai" / "terrainFlowSim" / "ai_model")   # 서브모듈 (기본)
    candidates.append(WEBAPP_DIR.parent / "ai_model")                      # 예전 배치(AI 저장소 안의 webapp 옆)
    for c in candidates:
        if (c / "src" / "inference_api.py").is_file():
            return c.resolve()
    raise RuntimeError(
        "AI 모델 코드를 찾을 수 없습니다. 서브모듈을 받아 주세요:\n"
        "  git submodule update --init prediction_server/ai\n"
        "다른 위치를 쓰려면 환경변수 AI_MODEL_DIR 에 .../terrainFlowSim/ai_model 경로를 지정하세요.")


def _default_checkpoint() -> Path:
    # 체크포인트(best.pt)는 저장소에 없다. AI 담당자에게 받아 아래 중 한 곳에 두거나 AI_MODEL_CHECKPOINT 로 지정.
    for c in (WEBAPP_DIR / "checkpoints" / "best.pt",
              AI_MODEL_DIR / "checkpoints_p2_wet8" / "best.pt"):
        if c.is_file():
            return c
    return WEBAPP_DIR / "checkpoints" / "best.pt"


AI_MODEL_DIR = _find_ai_model_dir()
sys.path.insert(0, str(AI_MODEL_DIR))
os.environ.setdefault("AI_MODEL_CHECKPOINT", str(_default_checkpoint()))

from src.inference_api import (  # noqa: E402  (경로 설정 후 import)
    current_checkpoint,
    current_cond_dim,
    current_image_size,
    current_uses_film,
    predict_from_image,
)
from src.utils.render_fields import render_contour_rgb  # noqa: E402
from src.inference_api import _LUT, _LUT_N  # noqa: E402  (수심 역변환 색표)
from src.utils.render_fields import depth_to_norm, norm_to_depth  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

# 검증 탭 전용(ground_truth -> scripts/prepare_openfoam_dataset)은 처음 쓸 때 불러온다.
# 학습 스크립트가 바뀌거나 없어도 예측 서버 기동은 깨지지 않게.
_GT_MOD = None
_GT_STORE = None


def _ground_truth():
    global _GT_MOD, _GT_STORE
    if _GT_MOD is None:
        import src.ground_truth as m
        _GT_MOD, _GT_STORE = m, m.GroundTruth()
    return _GT_MOD


def _gt():
    _ground_truth()
    return _GT_STORE

import contour_color as cc  # noqa: E402  (이미지 색 체계 -> 학습 형식 변환)
from terrain_io import InvalidTerrainData, load_terrain  # noqa: E402
import handdrawn  # noqa: E402  (손그림 사진 -> 고도 격자)
import expo  # noqa: E402  (전시 모드 YUCHE_EXPO=1: 로그인 없이 체험 + 손그림·결과 짝 모으기)
import pond_hold  # noqa: E402  (막힌 웅덩이 물 보정)
import guest_samples  # noqa: E402  (로그인 없이 해 보는 샘플)
from analysis_figure import (  # noqa: E402
    intensity_from_prediction,
    render_analysis_figure,
    render_map_only,
)
import auth  # noqa: E402  (회원가입/로그인 + MySQL)
import history  # noqa: E402  (분석 기록 저장: MySQL + 썸네일 파일). auth.init_db() 전에 import 해야 테이블이 같이 생성됨

RAIN_MIN_MM, RAIN_MAX_MM = 20.0, 80.0          # 학습 데이터가 다룬 강우 범위
TIME_MIN_S, TIME_MAX_S = 0.0, 120.0            # 학습 데이터가 다룬 경과 시간 범위

# 기본 조회 시점.
#
# 주의: 수심은 시간이 갈수록 "줄어든다". 학습 데이터(OpenFOAM)에서 비는 초반에 내리고
# 이후 물이 빠지기 때문에, val 정답의 평균 수심은 t0(18s) 0.039m -> t5(120s) 0.019m 로
# 단조 감소한다. 따라서 "이 지형 침수 예상"을 묻는 일반 조회에 time_s=120 을 쓰면
# 가장 덜 잠긴 상태를 보여주게 된다 (deploy_package/README.md 의 권장값은 이 점에서 틀렸다).
# 기본값은 가장 심한 시점 쪽인 18s 로 둔다.
DEFAULT_TIME_S = 18.0
TIMELAPSE_TIMES_S = [18.0, 38.0, 59.0, 79.0, 100.0, 120.0]   # 학습 데이터의 시점(시간 변화 재생용)
DEFAULT_DX_M = 10.4166667                       # 1000m / 96격자
WET_THRESHOLD_M = 0.05                          # '침수'로 세는 최소 수심 (5cm). 평가 지표 IoU@5cm 와 같은 기준
# 샘플 지형 그림(검증·데모용). 학습 데이터라 저장소에는 없고, 있을 때만 /api/samples 가 동작한다
SAMPLES_DIR = Path(os.environ.get("YUCHE_SAMPLES_DIR", AI_MODEL_DIR / "data" / "contours"))
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

_HEAVY_LOCK = threading.Lock()                  # matplotlib/torch 직렬화 (데모 규모)

app = FastAPI(title="예측 배수맵 API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PATCH", "DELETE"],   # 기록 이름변경/삭제(PATCH/DELETE) 허용
    allow_headers=["*"],
)
# --- [timing] 속도 점검용 시간 로그 (2026-11 마무리 점검) -------------------------------
# 끄려면 환경변수 YUCHE_TIMING=0. 동작은 바꾸지 않고 터미널에 시간만 찍는다.
TIMING_ON = os.environ.get("YUCHE_TIMING", "1") != "0"


class _StepTimer:
    """predict 안에서 단계별 걸린 시간을 모았다가 한 줄로 찍는다."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.t0 = self.last = time.perf_counter()
        self.steps: list[tuple[str, float]] = []

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.steps.append((name, now - self.last))
        self.last = now

    def report(self) -> None:
        if not TIMING_ON:
            return
        total = time.perf_counter() - self.t0
        parts = " | ".join(f"{n} {d:.2f}s" for n, d in self.steps)
        print(f"[timing] predict {self.label} | {parts} | 합계 {total:.2f}s", flush=True)


@app.middleware("http")
async def _timing_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    if TIMING_ON and request.url.path.startswith("/api/"):
        print(f"[timing] {request.method} {request.url.path} -> {response.status_code} "
              f"서버 전체 {time.perf_counter() - t0:.2f}s", flush=True)
    return response


@app.exception_handler(cc.ColorSchemeError)
async def _unreadable_image(_request: Request, exc: cc.ColorSchemeError) -> JSONResponse:
    """색에서 높이를 읽을 수 없는 이미지(지형도·스캔 도면 등).

    detail 은 기존 오류들과 같은 문자열이고, code 를 덧붙여 프런트가 일반 422(파일 형식 오류 등)와
    구분해 안내 화면(다음 행동 3가지)을 띄우게 한다.
    """
    return JSONResponse(status_code=422, content={"detail": str(exc), "code": "unreadable_image"})


@app.exception_handler(OperationalError)
async def _db_unavailable(_request: Request, _exc: OperationalError) -> JSONResponse:
    """요청 도중 DB 연결이 끊겼을 때. 처리 안 된 예외(500)는 CORS 헤더 없이 나가서 브라우저가
    '서버 오류'로만 보이므로, 이유를 알 수 있는 503 JSON 으로 바꿔 보낸다."""
    return JSONResponse(status_code=503, content={
        "detail": "DB 연결이 끊겼습니다. 잠시 후 다시 시도하세요. (WSL/MariaDB 확인)"})


app.include_router(auth.router)
app.include_router(history.router)
auth.init_db()   # 기동 시 테이블 생성 시도. 실패해도 latch 하지 않는다 (auth.db_ready 주석 참조)


# --------------------------------------------------------------------------- #
# 라우트
# --------------------------------------------------------------------------- #
AUTH_REQUIRED = os.environ.get("YUCHE_AUTH_REQUIRED", "0") == "1"
_NO_CACHE = {"Cache-Control": "no-cache"}   # 화면 파일을 고치면 새로고침만으로 반영되게


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    if SITE_DIR is not None:
        return FileResponse(SITE_DIR / "vtkjs.html", headers=_NO_CACHE)
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/test", include_in_schema=False)
def legacy_index() -> FileResponse:
    """AI 저장소에 들어 있는 간단한 테스트 화면."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/vtkjs.html", include_in_schema=False)
def site_page() -> FileResponse:
    if SITE_DIR is None:
        raise HTTPException(404, "YUCHE_SITE_DIR 이 지정되지 않았습니다.")
    return FileResponse(SITE_DIR / "vtkjs.html", headers=_NO_CACHE)


@app.get("/style.css", include_in_schema=False)
def site_css() -> FileResponse:
    if SITE_DIR is None or not (SITE_DIR / "style.css").is_file():
        raise HTTPException(404, "style.css 를 찾을 수 없습니다.")
    return FileResponse(SITE_DIR / "style.css", media_type="text/css", headers=_NO_CACHE)


@app.get("/api/health")
def health() -> dict:
    cond_dim = _safe_cond_dim()
    return {
        "status": "ok",
        "checkpoint": os.path.basename(str(current_checkpoint())),
        "rain_conditioned": cond_dim >= 1,
        "rain_range_mm": [RAIN_MIN_MM, RAIN_MAX_MM],
        "time_conditioned": cond_dim == 2,
        "time_range_s": [TIME_MIN_S, TIME_MAX_S],
        "default_time_s": DEFAULT_TIME_S,
        "timelapse_times": TIMELAPSE_TIMES_S,      # 화면 vtkjs.html TIMELAPSE_TIMES 와 같은 값의 원본
        "drain_ref_time_s": DEFAULT_TIME_S,
        "queue": _PREDICT_GATE.waiting,            # 지금 예측 순서를 기다리는 요청 수
        "image_size": _safe_image_size(),
        "device": _device(),
        "auth_enabled": auth.db_ready(),
        "db": auth.db_status(),
        "db_error": auth.db_error() or None,
        "expo": expo.ENABLED,                     # 전시 모드: 화면이 로그인·기록을 건너뛰고 손그림 결과를 보낸다
    }


@app.get("/api/color-schemes")
def color_schemes() -> dict:
    """등고선 이미지 색 체계 선택지 (낮음->높음 범례색 포함)."""
    return {
        "default": cc.AUTO,
        "schemes": [{"key": k, "label": v[0], "legend": cc._legend(k, False)}
                    for k, v in cc.SCHEMES.items()],
    }


_RV_ANY = re.compile(r"_rv\d+")


def _base_terrain_id(stem: str) -> str:
    """'seoul_sample_0003_rv1_f2_t3' -> 'seoul_sample_0003'.

    학습 때 train/val 을 가르는 것과 같은 함수를 쓴다. 예전의 자체 정규식은 파일명에
    `_t{idx}` 가 붙은 뒤로 증강 접미사를 못 떼서, 같은 지형의 회전본이 서로 다른
    지형으로 집계됐다(샘플 갤러리에 같은 지형이 여러 번 뜨는 원인).
    """
    return _ground_truth().group_key(stem)


@app.get("/api/samples")
def samples(limit: int = 30) -> dict:
    """서로 다른 지형을 지역별로 고르게 섞어서 반환 (같은 지형의 회전본 나열 X)."""
    if not SAMPLES_DIR.is_dir():
        return {"samples": []}

    # 지형별 대표 파일 1개 (가능하면 회전 안 된 _r0)
    canonical: dict[str, str] = {}
    for p in SAMPLES_DIR.glob("*.png"):
        if _RV_ANY.search(p.stem):
            continue  # 강우 변형본은 제외
        base = _base_terrain_id(p.stem)
        # 대표본은 회전 안 된 원본(_r0_t0) 우선. 파일명이 _t{idx} 로 끝나므로
        # 예전처럼 endswith("_r0") 로 보면 절대 안 걸린다.
        if base not in canonical or "_r0_t" in p.stem:
            canonical[base] = p.name

    # 지역별로 모아 균등 간격 샘플링 후 라운드로빈
    by_region: dict[str, list[str]] = {}
    for base, fname in sorted(canonical.items()):
        by_region.setdefault(base.split("_", 1)[0], []).append(fname)

    picked: list[str] = []
    per_region = max(1, limit // max(len(by_region), 1))
    buckets = []
    for files in by_region.values():
        step = max(1, len(files) // per_region)
        buckets.append(files[::step][:per_region])
    for i in range(max((len(b) for b in buckets), default=0)):
        for b in buckets:
            if i < len(b):
                picked.append(b[i])

    return {"samples": picked[:limit]}


@app.get("/api/sample/{name}", include_in_schema=False)
def sample_image(name: str) -> FileResponse:
    path = (SAMPLES_DIR / name).resolve()
    if not str(path).startswith(str(SAMPLES_DIR.resolve())) or not path.is_file():
        raise HTTPException(404, "샘플을 찾을 수 없습니다.")
    guest_samples.remember_file(path)   # 화면이 이 그림을 그대로 다시 올리면 로그인 없이 예측 (전시 관람객)
    return FileResponse(path, media_type="image/png")


_GT_MAX_CACHE: dict[str, float] = {}


def _gt_max_depth(name: str) -> float | None:
    """그 케이스의 정답 최대 수심[m]. 없으면 None."""
    if name in _GT_MAX_CACHE:
        return _GT_MAX_CACHE[name]
    g = _gt().get(name)
    v = float(g["depth_m"].max()) if g is not None else None
    if v is not None:
        _GT_MAX_CACHE[name] = v
    return v


@app.get("/api/testset")
def testset(split: str = "val", limit: int = 24, min_depth_m: float = 0.05) -> dict:
    """검증 탭용 테스트 케이스 목록.

    split="val" 이면 **학습에 쓰이지 않은 지형만** 고른다. 기본 샘플 갤러리(/api/samples)는
    이 구분을 하지 않아서 30개 중 25개가 학습 지형이었고, 그 위에서 잰 성능은 크게
    부풀려진다(MAE 0.0069 vs 0.0115, IoU@5cm 0.58 vs 0.43). 데모와 검증은 분리해야 한다.
    """
    if split not in {"val", "train", "any"}:
        raise HTTPException(400, "split 은 'val' | 'train' | 'any' 여야 합니다.")
    if not SAMPLES_DIR.is_dir():
        return {"cases": [], "split": split}

    # 지형당 1개만, 지역별로 고르게
    by_region: dict[str, dict[str, str]] = {}
    for p in sorted(SAMPLES_DIR.glob("*.png")):
        sp = _ground_truth().split_of(p.name)
        if split != "any" and sp != split:
            continue
        terrain = _base_terrain_id(p.stem)
        region = terrain.split("_", 1)[0]
        slot = by_region.setdefault(region, {})
        if terrain not in slot:
            slot[terrain] = p.name

    # 지역 라운드로빈으로 후보를 먼저 늘어놓는다
    buckets = [list(v.values()) for v in by_region.values()]
    ordered: list[str] = []
    for i in range(max((len(b) for b in buckets), default=0)):
        for b in buckets:
            if i < len(b):
                ordered.append(b[i])

    # 정답에 물이 거의 없는 케이스는 배수 예측 검증에 쓸모가 없다(강원 val 은 대부분
    # 최대수심 1cm 미만). min_depth_m 이상인 케이스를 우선 채우고, 모자라면 나머지로
    # 채운 뒤 몇 개가 걸러졌는지 응답에 밝힌다 - 조용히 고르면 체리피킹이 된다.
    picked, spare, scanned = [], [], 0
    for n in ordered:
        if len(picked) >= limit or scanned >= 400:
            break
        scanned += 1
        d = _gt_max_depth(n)
        if d is None:
            continue
        (picked if d >= min_depth_m else spare).append(n)
    cases = picked[:limit]
    filtered_out = len(spare)
    if len(cases) < limit:
        cases += spare[:limit - len(cases)]

    return {"split": split, "count": len(cases), "min_depth_m": min_depth_m,
            "filtered_out": filtered_out, "scanned": scanned,
            "cases": [{"name": n, "split": _ground_truth().split_of(n),
                       "gt_max_m": _gt_max_depth(n)} for n in cases]}


@app.get("/api/testcase")
def testcase(name: str) -> dict:
    """테스트 케이스 1건: 등고선 · OpenFOAM 정답 · 예측 + 정량 비교.

    조건(rain_mm/time_s)은 사용자가 고른 값이 아니라 **그 샘플의 실제 시뮬레이션 조건**을
    쓴다. 정답과 같은 조건이어야 비교가 성립하기 때문이다.
    """
    path = (SAMPLES_DIR / name).resolve()
    if not str(path).startswith(str(SAMPLES_DIR.resolve())) or not path.is_file():
        raise HTTPException(404, "샘플을 찾을 수 없습니다.")

    gt = _gt().get(name)
    if gt is None:
        raise HTTPException(
            422, f"'{name}' 의 OpenFOAM 원본 정답을 찾지 못했습니다 "
                 f"(파일명 규칙이 다르거나 원본 데이터셋이 없습니다).")

    contour_img = Image.open(path).convert("RGB")
    cond_dim = _safe_cond_dim()
    cond_kwargs: dict = {}
    if cond_dim >= 1:
        cond_kwargs["rain_mm"] = gt["rain_mm"]
    if cond_dim == 2:
        cond_kwargs["time_s"] = gt["time_s"]

    with _HEAVY_LOCK:
        pred_img = predict_from_image(contour_img, **cond_kwargs)
        pred_depth = _ground_truth().depth_from_prediction_image(pred_img, gt["depth_m"].shape)
        truth_png = _ground_truth().render_gt_png_bytes(gt["depth_m"])

    metrics = _ground_truth().compare(pred_depth, gt["depth_m"])
    return {
        "name": name,
        "split": _ground_truth().split_of(name),
        "region": gt["region"],
        "sample": gt["sample"],
        "rain_mm": gt["rain_mm"],
        "time_s": gt["time_s"],
        "grid": list(gt["depth_m"].shape),
        "contour_png_base64": _b64(contour_img),
        "truth_png_base64": base64.b64encode(truth_png).decode("ascii"),
        "prediction_png_base64": _b64(pred_img),
        "metrics": metrics,
        "checkpoint": os.path.basename(str(current_checkpoint())),
    }


def _auth_gate(authorization: str = Header(default="")) -> str | None:
    """인증이 켜져 있으면(=MySQL 연결됨) 토큰 필수, 아니면 통과.

    이 환경의 MariaDB 는 WSL2 안에 있고
    Windows -> 127.0.0.1:3306 포워딩이 수시로 끊긴다(같은 분 안에 20/20 성공 -> 20/20 실패를
    관측). 기동 때 살아 있었으면 인증이 켜진 채 고정되는데, 그 뒤 DB 가 끊기면 **로그인
    자체가 불가능**해져서 토큰을 받을 방법이 없고 예측이 전부 401 로 떨어진다
    ("예측 실패: 세션이 만료되었습니다").

    설계 의도는 "DB 가 없으면 인증을 생략한다" 이므로, 토큰 검증에 실패했을 때 DB 가
    실제로 끊긴 상태인지 확인해서 그렇다면 통과시킨다. DB 조회는 실패 경로에서만 하므로
    정상 상황의 예측 성능에는 영향이 없다.

    반대 방향(기동 때 끊겨 있다가 뒤에 살아남)은 auth.db_ready() 가 맡는다 - 기동 시점의
    판정을 상수로 들고 있으면 그 뒤로 영원히 인증이 꺼진 채 고정되기 때문이다.
    """
    # 토큰 검사는 HMAC 계산뿐이라 DB 가 필요 없다. 유효하면 DB 를 보지 않고 통과.
    token = authorization[7:] if authorization.lower().startswith("bearer ") else authorization
    email = auth.verify_token(token) if token else None
    if email:
        return email
    if expo.ENABLED:    # 전시 모드: 관람객은 가입·로그인 없이 예측 (/api/predict, /api/predict/series)
        return None
    if auth.db_ready():
        raise HTTPException(401, "로그인이 필요합니다.")
    if AUTH_REQUIRED:   # 배포 환경: DB 가 안 되면 비로그인 예측도 막는다
        raise HTTPException(503, "로그인 DB 에 연결할 수 없습니다. 잠시 후 다시 시도하세요.")
    return None         # 개발 환경(기본): 지금처럼 DB 가 없으면 로그인 없이 통과


NEED_LOGIN = "__need_login__"

def _auth_gate_soft(authorization: str = Header(default="")) -> str | None:
    """[11/28] _auth_gate 와 같되, 로그인이 필요한 상황이면 401 대신 NEED_LOGIN 을 돌려준다.
    예측 엔드포인트가 올라온 파일을 보고 샘플이면(guest_samples) 로그인 없이 통과시키려고 쓴다.
    전시 모드(expo)면 _auth_gate 가 토큰 없이 통과시킨다."""
    try:
        return _auth_gate(authorization)
    except HTTPException as exc:
        if exc.status_code == 401:
            return NEED_LOGIN
        raise


def _require_login_unless_sample(user: str | None, raw: bytes) -> None:
    if user == NEED_LOGIN and not guest_samples.is_guest_sample(raw):
        raise HTTPException(401, "로그인이 필요합니다.")


# [속도 개선 2026-11] async def -> def + 예측은 한 번에 하나씩.
# - 안에서 하는 일(색 변환·AI 추론·그림 그리기)이 전부 동기 계산이라, async 로 두면 예측 하나를
#   계산하는 동안 서버 전체(로그인·기록 저장 등)가 멈췄다. def 로 두면 FastAPI 가 스레드에서
#   돌려서 가벼운 요청은 기다리지 않는다.
# - 대신 예측끼리는 _PREDICT_GATE 로 한 번에 하나씩 처리한다 (첫 화면 요청 우선). 둘을 동시에 돌리면 파이썬(GIL)에서
#   계산 스레드끼리 CPU 를 뺏고 뺏기느라 오히려 느려졌다(데이터 모드 첫 화면 3.4초 -> 6~7초 실측).
#   순서대로 하면 뒤 요청이 앞 요청의 색 변환 캐시도 그대로 쓴다.
from predict_gate import PriorityGate  # noqa: E402  (첫 화면 요청을 시간 변화 요청보다 먼저)

_PREDICT_GATE = PriorityGate()
PREDICT_WAIT_S = float(os.environ.get("YUCHE_PREDICT_WAIT_S", "60"))   # 앞 예측을 기다리는 최대 시간


@app.post("/api/predict")
def predict(
    mode: str = Form(...),
    rain_mm: float = Form(50.0),
    time_s: float = Form(DEFAULT_TIME_S),
    dx: float = Form(DEFAULT_DX_M),
    color_scheme: str = Form(cc.AUTO),
    reverse: bool = Form(False),
    light: bool = Form(False),
    no_analysis: bool = Form(False),
    no_map: bool = Form(False),
    mark_x: float | None = Form(None),
    mark_y: float | None = Form(None),
    file: UploadFile = File(...),
    _user: str | None = Depends(_auth_gate_soft),
) -> dict:
    # 샘플(전시 관람객용)은 로그인 없이. 그 밖의 파일은 로그인 필요 (guest_samples 설명)
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    file.file.close()
    _require_login_unless_sample(_user, raw)
    file = _reupload(raw, file.filename)
    # mark_x / mark_y (선택, 그림 좌표 0~1 · 왼쪽 위 기준): 분석 지도의 ★를 이 자리에 그린다 (고도격자·손그림 모드).
    # 웹 화면은 120초까지 물이 남는 곳(침수 위험지역)으로 배수구를 다시 고르므로, 그 자리를 넘겨서 ★를 맞춘다.
    # 보내지 않으면 지금처럼 예측 유출이 가장 큰 칸에 그린다. drain_candidate 응답은 어느 쪽이든 그대로다.
    # no_analysis / no_map (데이터 모드만 의미 있음): 첫 화면은 '사용자 시점' 요청과 '배수구 기준 시점(18초)'
    # 요청을 같이 보내는데, 화면은 앞 요청의 지도(map)와 뒤 요청의 분석 지도(analysis)만 쓴다.
    # 쓰지 않는 그림을 빼달라는 표시. 기본값 False 라 보내지 않으면 지금처럼 둘 다 그린다.
    if not _PREDICT_GATE.acquire(high=not light, timeout=PREDICT_WAIT_S):
        raise HTTPException(503, "다른 예측이 진행 중입니다. 잠시 후 다시 시도하세요.",
                            headers={"Retry-After": "10"})
    try:
        mark = None
        if mark_x is not None and mark_y is not None:
            mark = (float(np.clip(mark_x, 0.0, 1.0)), float(np.clip(mark_y, 0.0, 1.0)))
        return _predict(mode, rain_mm, time_s, dx, color_scheme, reverse,
                        light or no_analysis, light or no_map, light, file, mark)
    finally:
        _PREDICT_GATE.release()


@app.post("/api/predict/series")
def predict_series(
    mode: str = Form(...),
    rain_mm: float = Form(50.0),
    times: str = Form(...),
    dx: float = Form(DEFAULT_DX_M),
    color_scheme: str = Form(cc.AUTO),
    reverse: bool = Form(False),
    file: UploadFile = File(...),
    _user: str | None = Depends(_auth_gate_soft),
) -> dict:
    """시간 변화 장면 여러 개를 한 번에 (light 와 같은 내용). times = "38,59,79,100,120".

    화면은 첫 화면 뒤에 같은 파일을 시점마다 한 번씩(5번) 올렸다. 폰은 업로드와 ngrok 왕복이 5번 쌓여
    노트북보다 몇 초씩 늦었다 -> 파일은 한 번 올리고 장면은 한 번에 받는다. 서버 계산량은 같다
    (같은 파일이라 읽기·등고선 그리기는 캐시, 장면당 약 0.1초). 실패한 시점은 error 로 알려 주고 나머지는 계속.
    """
    try:
        ts = [float(t) for t in times.split(",") if t.strip()][:12]
    except ValueError as exc:
        raise HTTPException(400, "times 는 '38,59,79' 처럼 쉼표로 구분한 숫자여야 합니다.") from exc
    if not ts:
        raise HTTPException(400, "times 가 비었습니다.")
    # [11/28 수정] 업로드는 여기서 한 번만 읽는다. _predict 는 파일을 읽고 닫기 때문에, 예전처럼 같은
    # UploadFile 을 seek(0) 해서 다시 넘기면 두 번째 시점부터 "closed file" 오류(500)가 나서 화면이
    # 시점별 요청으로 되돌아갔다. 시점마다 메모리 사본(_reupload)을 새로 만들어 넘긴다.
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    file.file.close()
    _require_login_unless_sample(_user, raw)   # 샘플은 로그인 없이
    frames: dict[str, dict] = {}
    for t in ts:
        # 문은 장면마다 잡고 놓는다: 그 사이 다른 사람의 첫 화면 요청이 끼어들 수 있게 (첫 화면 우선)
        if not _PREDICT_GATE.acquire(high=False, timeout=PREDICT_WAIT_S):
            frames[f"{t:g}"] = {"error": "다른 예측이 진행 중입니다. 잠시 후 다시 시도하세요.", "status": 503}
            continue
        try:
            frames[f"{t:g}"] = _predict(mode, rain_mm, t, dx, color_scheme, reverse, True, True, True,
                                        _reupload(raw, file.filename), None)
        except HTTPException as exc:
            frames[f"{t:g}"] = {"error": exc.detail, "status": exc.status_code}
        finally:
            _PREDICT_GATE.release()
    return {"frames": frames}


def _reupload(raw: bytes, filename: str | None) -> UploadFile:
    """같은 업로드 내용을 _predict 에 여러 번 넘기기 위한 메모리 사본 (_predict 가 읽고 닫아도 다음 사본은 새것)."""
    return UploadFile(io.BytesIO(raw), filename=filename)


def _predict(mode: str, rain_mm: float, time_s: float, dx: float, color_scheme: str,
             reverse: bool, skip_analysis: bool, skip_map: bool, light: bool,
             file: UploadFile, mark: tuple[float, float] | None = None) -> dict:
    # light=True: 시간 변화(타임랩스)용 요청. 화면이 오버레이·3D 격자만 쓰므로
    # 데이터 모드의 분석 지도·지도 그림(요청당 약 1.4초)을 건너뛴다. 기본값 False 라 기존 화면은 그대로.
    if mode not in {"image", "terrain", "hand"}:
        raise HTTPException(400, "mode 는 'image', 'terrain', 'hand' 중 하나여야 합니다.")
    # 손그림은 사진을 고도 격자로 읽은 뒤부터 고도격자 모드와 같은 길을 간다
    grid_mode = mode in {"terrain", "hand"}

    tag = " light" if light else (" -분석지도" if skip_analysis else "") + (" -지도" if skip_map else "")
    tm = _StepTimer(f"{mode} t={time_s:g}s{tag}")
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    file.file.close()
    if not raw:
        raise HTTPException(400, "빈 파일입니다.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"파일이 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB 를 초과합니다.")

    # 사용자에게 보여줄 주의 문구 (응답의 "warnings"). 화면은 결과 안내와 알림으로 띄운다.
    warns: list[str] = []
    # 예전엔 0~500 mm 를 그대로 받고 모델 안에서 조용히 20~80 으로 잘려서, 응답에는 원래 값이 찍혔다
    rain_in = float(rain_mm)
    rain_mm = float(np.clip(rain_in, RAIN_MIN_MM, RAIN_MAX_MM))
    if rain_mm != rain_in:
        warns.append(f"강우량 {rain_in:g} mm 는 학습 범위({RAIN_MIN_MM:g}~{RAIN_MAX_MM:g} mm) 밖이라 "
                     f"{rain_mm:g} mm 로 계산했습니다.")
    time_s = float(np.clip(time_s, TIME_MIN_S, TIME_MAX_S))
    meta: dict = {"mode": mode, "filename": file.filename}
    tm.mark("업로드읽기")
    color_info: dict | None = None
    original_img: Image.Image | None = None
    hand_preview_b64: str | None = None

    try:
        if mode == "image":
            try:
                contour_img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(422, f"이미지를 열 수 없습니다: {exc}") from exc
            meta["source_format"] = "contour_image"
            iw, ih = contour_img.size
            if max(iw, ih) / max(min(iw, ih), 1) > 1.2:
                warns.append(f"그림이 정사각형이 아닙니다({iw}x{ih}). 모델은 정사각형 지형으로 학습되어 "
                             "정사각형으로 늘려서 예측하므로 모양이 찌그러질 수 있습니다. 정사각형으로 잘라서 넣는 것을 권장합니다.")

            # 모델은 색을 곧 높이로 읽는다. 학습 때 쓴 terrain 색 체계가 아니면 높낮이가
            # 뒤집혀 나오므로, 색을 높이로 복원해 학습 형식으로 다시 그려서 넣는다.
            # [속도 개선 2026-11] 같은 그림이 시간 변화용으로 6~7번 연달아 오는데, 색 체계 판별·변환이
            # 요청마다 약 2초씩 걸렸다. 결과는 그림·색 체계·반전 여부로만 정해지므로 기억해 두고 재사용한다.
            # 캐시 확인도 잠금 안에서 한다: 화면이 첫 예측과 기준 시점(18초) 예측을 동시에 보내므로,
            # 밖에서 확인하면 두 요청이 같은 변환을 둘 다 계산한다. 안에서 확인하면 뒤 요청은 앞 요청이
            # 끝나길 기다렸다가 결과를 그대로 쓴다.
            conv_key = (hashlib.md5(raw).hexdigest(), color_scheme, bool(reverse))
            with _HEAVY_LOCK:   # ColorSchemeError 는 위 핸들러가 code=unreadable_image 로 422 응답
                conv = _CONV_CACHE.get(conv_key)
                if conv is None:
                    conv = cc.convert_to_training_style(contour_img, color_scheme, reverse)
                    if len(_CONV_CACHE) >= _CONV_CACHE_MAX:
                        _CONV_CACHE.pop(next(iter(_CONV_CACHE)))
                    _CONV_CACHE[conv_key] = conv
            if conv.converted:
                original_img = contour_img
                contour_img = conv.image
            color_info = {
                "scheme": conv.scheme,
                "label": cc.SCHEMES[conv.scheme][0],
                "auto": conv.auto,
                "reversed": conv.reversed,
                "converted": conv.converted,
                "match_frac": None if conv.match_frac is None else round(conv.match_frac, 3),
                "legend": conv.legend,
                "warnings": conv.warnings,
            }
            meta["color_scheme"] = conv.scheme
            meta["color_reversed"] = conv.reversed
            tm.mark("이미지열기+색변환")
        else:
            # 같은 파일이 시점마다 7번 오므로 읽기·등고선 그리기 결과를 기억한다. dx 가 다르면 다른 그림이다
            suffix = "#hand" if mode == "hand" else Path(file.filename or "").suffix.lower()
            t_key = (hashlib.md5(raw).hexdigest(), suffix, float(dx))
            cached = _TERRAIN_CACHE.get(t_key)
            if cached is None:
                if mode == "hand":
                    terrain, tmeta = _read_hand(raw)
                else:
                    terrain, tmeta = load_terrain(raw, file.filename or "upload")
            else:
                terrain, tmeta, contour_img = cached
            hand_preview_b64 = tmeta.get("hand_preview_png_base64")
            meta.update({k: v for k, v in tmeta.items() if k != "hand_preview_png_base64" and not k.startswith("_")})
            effective_dx = float(tmeta.get("cellsize_m") or dx)
            meta["dx_m"] = effective_dx
            warns.extend(tmeta.get("warnings") or [])
            if mode == "terrain":
                # 셀 크기를 파일에서 읽었으면 축소 후 격자·셀 크기가 서로 맞고, 아니면 입력한 셀 크기는 원본 격자 기준이다
                dom_shape = terrain.shape if tmeta.get("cellsize_m") else tuple(tmeta.get("original_shape") or terrain.shape)
                warns.extend(_domain_warnings(dom_shape, effective_dx, bool(tmeta.get("cellsize_m"))))
            tm.mark("파일읽기" if mode == "terrain" else "손그림읽기")
            if cached is None:
                contour_img = _render_contour(terrain, effective_dx)
                if len(_TERRAIN_CACHE) >= _TERRAIN_CACHE_MAX:
                    _TERRAIN_CACHE.pop(next(iter(_TERRAIN_CACHE)))
                _TERRAIN_CACHE[t_key] = (terrain, tmeta, contour_img)
            tm.mark("등고선그리기")

        # 체크포인트가 요구하는 조건만 넘긴다 (0=없음, 1=rain, 2=rain+time).
        cond_dim = _safe_cond_dim()
        cond_kwargs: dict = {}
        if cond_dim >= 1:
            cond_kwargs["rain_mm"] = rain_mm
        if cond_dim == 2:
            cond_kwargs["time_s"] = time_s
        meta["rain_conditioned"] = cond_dim >= 1
        meta["rain_mm"] = rain_mm if cond_dim >= 1 else None
        meta["time_conditioned"] = cond_dim == 2
        meta["time_s"] = time_s if cond_dim == 2 else None

        with _HEAVY_LOCK:
            pred_img = predict_from_image(contour_img, **cond_kwargs)
        tm.mark("AI추론")
        if cond_dim == 2 and pond_hold.ENABLED and time_s > POND_REF_TIME_S:
            try:
                pred_img, meta["pond_hold"] = _apply_pond_hold(
                    pred_img, contour_img, terrain if grid_mode else None, rain_mm, time_s)
            except Exception as exc:  # noqa: BLE001  (보정은 부가 기능: 실패해도 원래 예측을 준다)
                print(f"[pond_hold] 웅덩이 보정을 건너뜁니다: {exc!r}")
            tm.mark("웅덩이보정")
        overlay_img = _make_overlay(contour_img, pred_img)
        tm.mark("오버레이")

        # 등고선 데이터 모드에서만: 실제 등고선(m) + 음영기복 + 축척 + 배수구 후보 지도
        analysis_b64 = None
        map_b64 = None
        drain = None
        if grid_mode:
            with _HEAVY_LOCK:
                # render_contour_rgb 가 등고선 이미지를 상하반전(matplotlib Y축 위쪽)해서
                # 만들기 때문에, 그 이미지를 통과한 예측도 terrain 배열 기준으로는 상하반전
                # 상태다. terrain(등고선 선)과 겹쳐 그리려면 예측을 다시 뒤집어 맞춘다.
                inten = np.flipud(intensity_from_prediction(pred_img, terrain.shape))
                r, c = np.unravel_index(int(np.argmax(inten)), inten.shape)
                mr, mc = int(r), int(c)
                if mark is not None:
                    # 그림 좌표(위 = 0) -> terrain 배열 칸(0행 = 아래쪽). drain_candidate 의 image 와 반대 계산
                    th, tw = terrain.shape
                    mc = int(round(mark[0] * (tw - 1)))
                    mr = int(round((1.0 - mark[1]) * (th - 1)))
                    meta["analysis_mark"] = "client"
                # 제목은 "AI 예측 배수맵" 한 줄만 둔다 (조건 부제목은 길어서 그림 밖으로 잘렸다.
                # 강우·시간 조건은 웹 화면 결과 설명에 이미 나온다)
                png = None if skip_analysis else render_analysis_figure(
                    terrain, inten, effective_dx,
                    mark_idx=(mr, mc),
                )
            if not skip_analysis:
                analysis_b64 = base64.b64encode(png).decode("ascii")
                tm.mark("분석지도")
            if not skip_map:
                with _HEAVY_LOCK:
                    map_b64 = base64.b64encode(render_map_only(terrain, inten)).decode("ascii")
                tm.mark("지도")
            h, w = terrain.shape
            drain = {
                "pixel": {"x": int(c), "y": int(r)},
                "normalized": {"x": round(c / max(w - 1, 1), 4), "y": round(r / max(h - 1, 1), 4)},
                # 화면 그림(입력 등고선·예측·오버레이) 위 위치. 왼쪽 위 = (0, 0).
                # 그림은 terrain 배열과 상하가 반대라서 y 를 뒤집는다.
                "image": {"x": round(c / max(w - 1, 1), 4), "y": round(1 - r / max(h - 1, 1), 4)},
            }
        else:
            # 이미지 모드: 고도값이 없어 분석 지도는 못 그리지만, 예측 그림에서 유출이 가장
            # 강한 점은 찾을 수 있다. 예측 그림은 입력 이미지와 같은 방향이라 뒤집기 보정이 필요 없다.
            ph, pw = pred_img.height, pred_img.width
            inten = intensity_from_prediction(pred_img, (ph, pw))
            r, c = np.unravel_index(int(np.argmax(inten)), inten.shape)
            pos = {"x": round(c / max(pw - 1, 1), 4), "y": round(r / max(ph - 1, 1), 4)}
            drain = {"pixel": {"x": int(c), "y": int(r)}, "normalized": pos, "image": pos}

        # 지도 위 요약 숫자. 예측 그림을 수심[m]으로 되돌려(학습 때 쓴 Blues 역변환) 잰다.
        # 이미지 모드는 격자 정보가 없어 학습 격자(96x96)로 환산하므로 면적 비율만 의미가 있다.
        gh, gw = terrain.shape if grid_mode else (96, 96)
        depth_grid = np.ascontiguousarray(np.flipud(
            _depth_from_image(pred_img.resize((gw, gh), Image.BICUBIC))))
        stats = {
            "max_depth_cm": round(float(depth_grid.max()) * 100.0, 1),
            "wet_area_pct": round(float((depth_grid > WET_THRESHOLD_M).mean()) * 100.0, 1),
            "wet_threshold_cm": WET_THRESHOLD_M * 100.0,
        }
        tm.mark("배수구+통계")

    except InvalidTerrainData as exc:
        raise HTTPException(422, str(exc)) from exc
    except (HTTPException, cc.ColorSchemeError):
        raise
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    result = {
        # light(시간 변화용)는 화면이 overlay 와 grid3d.depth_u16 만 쓴다 -> 나머지 그림은 만들지 않는다
        "contour_png_base64": None if light else _b64(contour_img),
        "original_png_base64": _b64(original_img) if (original_img is not None and not light) else None,
        "color_info": color_info,
        # 손그림 모드: '이렇게 읽었어요' 그림 (왼쪽 = 반듯하게 편 사진, 오른쪽 = 읽은 높이 층)
        "hand_preview_png_base64": None if light else hand_preview_b64,
        "prediction_png_base64": None if light else _b64(pred_img),
        "overlay_png_base64": _b64(overlay_img),
        "analysis_png_base64": analysis_b64,
        "map_jpg_base64": map_b64,
        "stats": stats,
        "grid3d": _grid3d(contour_img, pred_img, terrain if grid_mode else None,
                          effective_dx if grid_mode else None, skip_height=light),
        "drain_candidate": drain,
        "checkpoint": os.path.basename(str(current_checkpoint())),
        "meta": {**meta, "warnings": warns},
        "warnings": warns,
        "notes": [
            "예측 배수맵은 색이 진한(파란) 곳일수록 지표 유출수가 모이는 침수·배수 취약 지점입니다.",
            "모델은 학습 데이터와 같은 스타일의 등고선(채색 20단계 + 검은 등고선)에서 가장 정확합니다.",
            "경과 시간이 짧을수록 물이 많습니다 - 비가 초반에 내리고 이후 빠지는 시뮬레이션이라, 가장 심한 상태는 20초 부근입니다.",
        ],
    }
    # 전시 모드: 손그림의 첫 화면 요청(시간 변화·기준 시점·분석 지도 요청 제외)마다 편 손그림 + '읽었어요' 그림을 저장하고
    # 번호(expo_id)를 준다. 화면은 이 번호로 결과 그림을 보낸다 (/api/expo/result). 원본 사진·파일 이름은 저장하지 않는다
    if expo.ENABLED and mode == "hand" and not skip_map:
        quad = tmeta.get("_hand_quad")
        if quad:
            try:
                result["expo_id"] = expo.save_drawing(handdrawn.flatten_original(raw, quad), hand_preview_b64)
            except Exception as exc:  # noqa: BLE001  (저장 실패가 체험을 막으면 안 된다)
                print(f"[expo] 손그림 저장 실패: {exc!r}")
    tm.mark("응답만들기(그림base64+3D)")
    tm.report()
    return result


# --------------------------------------------------------------------------- #
# 헬퍼
# --------------------------------------------------------------------------- #
GRID3D_N = 96                       # 3D 보기용 격자 한 변 (학습 격자와 같은 96)
_HEIGHT_CACHE: dict[str, np.ndarray] = {}   # 시점별 예측(같은 지형 6~7번 요청)에서 높이 복원을 한 번만 하도록
_CONV_CACHE: dict = {}              # (그림 md5, 색 체계, 반전) -> cc.Conversion. 시간 변화 요청마다 색 변환을 다시 하지 않도록
_CONV_CACHE_MAX = 16
_TERRAIN_CACHE: dict = {}           # (파일 md5, 확장자, dx) -> (terrain, tmeta, 등고선 그림)
_TERRAIN_CACHE_MAX = 16


_LUT_TREE = cKDTree(_LUT.astype(np.float64))

# --- 웅덩이 보정 (backend/pond_hold.py 설명 참고) ---------------------------------------
POND_REF_TIME_S = DEFAULT_TIME_S    # 이 시점부터 웅덩이 물을 붙잡는다 (그 전엔 아직 모이는 중)
POND_MASK_MIN_M = 0.01              # 웅덩이 칸 중 이만큼 이상 물이 남은 칸을 '물이 갇힌 곳'으로 화면에 알려 준다
_POND_REF_CACHE: dict = {}          # (등고선 그림 md5, 강우, 시점) -> 그 시점 예측 수심. 시간 변화 6장이 서로의 기준이 된다
_POND_PIT_CACHE: dict = {}          # (등고선 그림 md5, 크기, 데이터 모드) -> 웅덩이 깊이
_POND_CACHE_MAX = 48


def _cache_put(cache: dict, key, value) -> None:
    if len(cache) >= _POND_CACHE_MAX:
        cache.pop(next(iter(cache)))
    cache[key] = value


def _apply_pond_hold(pred_img: Image.Image, contour_img: Image.Image,
                     terrain: np.ndarray | None, rain_mm: float, time_s: float) -> tuple[Image.Image, dict]:
    """18초 뒤 시점 예측에서 막힌 웅덩이의 물이 앞 시점들의 최대보다 줄지 않게 한다. 예측 그림 방향 그대로 다룬다."""
    w, h = pred_img.size
    ckey = hashlib.md5(contour_img.tobytes()).hexdigest()
    metric = terrain is not None

    pkey = (ckey, (w, h), metric)
    pits = _POND_PIT_CACHE.get(pkey)
    if pits is None:
        if metric:   # 예측 그림은 terrain 배열과 상하가 반대다
            z = np.nan_to_num(np.ascontiguousarray(np.flipud(np.asarray(terrain, dtype=np.float32))))
        else:
            with _HEAVY_LOCK:
                z, _frac = cc._to_height(contour_img, "terrain")
        z = np.asarray(Image.fromarray(np.asarray(z, dtype=np.float32)).resize((w, h), Image.BILINEAR),
                       dtype=np.float64)
        pits = pond_hold.pit_depth(z)
        _cache_put(_POND_PIT_CACHE, pkey, pits)

    # 기준 = 지금 시점보다 앞선 재생 시점들(18초부터)의 예측. 웅덩이마다 그중 가장 많았던 물 부피를 붙잡는다.
    # 웅덩이 물은 절반쯤이 18초 뒤에 가장 많아지므로(경사면 물이 늦게 모임) 18초 하나만 보면
    # 38초에 더 고인 물이 59초에 다시 줄어든다. (계산 방식은 pond_hold.py 설명)
    ref_times = [t for t in TIMELAPSE_TIMES_S if POND_REF_TIME_S <= t < time_s] or [POND_REF_TIME_S]
    refs = []
    for t in ref_times:
        rkey = (ckey, round(float(rain_mm), 3), float(t))
        d = _POND_REF_CACHE.get(rkey)
        if d is None:
            with _HEAVY_LOCK:
                d = _depth_from_image(predict_from_image(contour_img, rain_mm=rain_mm, time_s=t))
            _cache_put(_POND_REF_CACHE, rkey, d)
        refs.append(d)

    cur = _depth_from_image(pred_img)
    new, info, keep = pond_hold.hold(cur, refs, pits, metric, times=[*ref_times, time_s])
    info["ref_times_s"] = ref_times
    # 물이 갇힌 웅덩이 칸 (보정 뒤 1cm 이상 남은 웅덩이 칸) -> 화면이 이 칸으로 침수 위험지역을 정한다.
    # 3D 격자와 같은 96x96 · 그림 방향, 칸 i 는 바이트 i>>3 의 (i&7) 번째 비트 (화면 unpackMask 와 같은 순서)
    # 작아서 건너뛰었거나 다 빠진 웅덩이는 빼고, 지금 물을 붙잡고 있는 웅덩이 칸만 (keep)
    held = Image.fromarray((keep & (new >= POND_MASK_MIN_M)).astype(np.uint8) * 255)
    held = np.asarray(held.resize((GRID3D_N, GRID3D_N), Image.NEAREST)) > 127
    info["mask_n"] = GRID3D_N
    info["mask_b64"] = base64.b64encode(np.packbits(held.reshape(-1), bitorder="little").tobytes()).decode("ascii")
    if info["applied"]:
        # 올린 칸만 학습 색표(Blues)로 다시 칠한다. 나머지 픽셀은 모델 출력 그대로 둔다
        rgb = np.asarray(pred_img.convert("RGB")).copy()
        up = new > cur + 1e-4
        idx = np.clip(np.rint(depth_to_norm(new[up]) * (_LUT_N - 1)), 0, _LUT_N - 1).astype(int)
        rgb[up] = np.rint(_LUT[idx]).astype(np.uint8)
        pred_img = Image.fromarray(rgb)
    return pred_img, info


def _depth_from_image(img: Image.Image) -> np.ndarray:
    """예측 그림 -> 수심[m]. ai_model 의 _image_to_depth 와 같은 결과.

    원본은 (픽셀 수 x 256색 x 3) 배열을 한 번에 만들어 512x512 에서 1.5GB 를 쓴다.
    KD-tree 로 가장 가까운 색을 찾고, 1위·2위 거리 차가 0.001 미만인 픽셀만 원래 식(float32 argmin)으로
    다시 골라 동점 처리까지 같게 맞춘다.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    flat = rgb.reshape(-1, 3)
    d, idx = _LUT_TREE.query(flat.astype(np.float64), k=2)
    best = idx[:, 0].copy()
    amb = np.nonzero(d[:, 1] - d[:, 0] < 1e-3)[0]
    if len(amb):
        sub_ = flat[amb]
        best[amb] = ((sub_[:, None, :] - _LUT[None, :, :]) ** 2).sum(-1).argmin(1)
    return norm_to_depth((best / (_LUT_N - 1)).reshape(rgb.shape[:2]))


def _u16_b64(arr01: np.ndarray, scale: float) -> str:
    """0 이상 실수 배열 -> uint16(리틀엔디언) base64. 화면이 Uint16Array 로 바로 읽는다."""
    q = np.clip(np.rint(np.asarray(arr01, dtype=np.float64) * scale), 0, 65535).astype("<u2")
    return base64.b64encode(q.tobytes()).decode("ascii")


def _grid3d(contour_img: Image.Image, pred_img: Image.Image,
            terrain: np.ndarray | None, dx: float | None, skip_height: bool = False) -> dict | None:
    """웹 화면의 3D 지형 보기용 격자. 실패해도 예측 자체는 돌려줘야 하므로 None 으로 넘긴다.

    둘 다 '그림 방향'(0행 = 화면 위쪽, 입력 등고선·예측 그림과 같은 방향)이다.
      height_u16 : 높이 0..1 정규화 x 65535      (GRID3D_N x GRID3D_N, 행 우선)
      depth_u16  : 수심[m] / depth_unit_m          (같은 격자)
    데이터 모드는 실제 고도 격자를 쓰고, 이미지 모드는 모델에 넣은 등고선 그림의 색을 높이로 되읽는다.
    """
    try:
        import hashlib

        from scipy.ndimage import gaussian_filter
        n = GRID3D_N
        relief_m = extent_m = None
        if terrain is not None:
            t = np.nan_to_num(np.ascontiguousarray(np.flipud(np.asarray(terrain, dtype=np.float32))))
            h = np.asarray(Image.fromarray(t).resize((n, n), Image.BILINEAR), dtype=np.float64)
            relief_m = round(float(h.max() - h.min()), 2)
            if dx:
                extent_m = round(float(max(terrain.shape) * dx), 1)
        else:
            small = contour_img.convert("RGB").resize((cc.WORK_SIZE, cc.WORK_SIZE), Image.BILINEAR)
            key = hashlib.md5(small.tobytes()).hexdigest()
            with _HEAVY_LOCK:   # 동시에 온 같은 그림 요청이 높이 복원을 두 번 하지 않도록 (위 색 변환 캐시와 같은 이유)
                h = _HEIGHT_CACHE.get(key)
                if h is None:
                    hh, _frac = cc._to_height(contour_img, "terrain")
                    # 채색이 20단계라 계단처럼 읽힌다 -> 살짝 뭉개서 완만한 면으로 만든다
                    hh = gaussian_filter(hh.astype(np.float32), sigma=cc.WORK_SIZE / n * 1.6)
                    h = np.asarray(Image.fromarray(hh).resize((n, n), Image.BILINEAR), dtype=np.float64)
                    if len(_HEIGHT_CACHE) >= 8:
                        _HEIGHT_CACHE.pop(next(iter(_HEIGHT_CACHE)))
                    _HEIGHT_CACHE[key] = h
        lo, hi = float(h.min()), float(h.max())
        hn = (h - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(h)

        depth = _depth_from_image(pred_img.resize((n, n), Image.BICUBIC))   # 그림 방향 수심[m]
        unit = 1e-4                                                       # 0.1mm 단위 (최대 6.5m)
        return {
            "n": n,
            "height_u16": None if skip_height else _u16_b64(hn, 65535.0),   # 지형은 첫 응답에서 한 번만
            "depth_u16": _u16_b64(np.clip(depth, 0.0, None), 1.0 / unit),
            "depth_unit_m": unit,
            "max_depth_m": round(float(np.max(depth)), 4),
            "relief_m": relief_m,
            "extent_m": extent_m,
        }
    except Exception as exc:  # noqa: BLE001  (3D 는 부가 기능: 실패해도 예측 결과는 그대로 준다)
        print(f"[grid3d] 3D 격자를 만들지 못했습니다: {exc!r}")
        return None


TRAIN_EXTENT_M = 1000.0            # 학습 지형 한 변 (96칸 x 10.42 m)


def _domain_warnings(shape: tuple[int, int], dx: float, dx_from_file: bool) -> list[str]:
    """학습 데이터 범위(1 km 정사각 지형)에서 벗어난 입력이면 알린다.

    모델은 축척을 모른다: 서버가 고도를 0~1 로 맞춰 등고선 그림을 그리므로, 100 m 지도든 10 km 지도든
    1 km 지형으로 보고 예측한다. 그래서 크기가 많이 다르면 결과를 그대로 믿으면 안 된다.
    """
    out: list[str] = []
    h, w = shape
    extent = max(h, w) * dx
    basis = "파일에 적힌 셀 크기" if dx_from_file else "입력한 셀 크기"
    if not (TRAIN_EXTENT_M * 0.5 <= extent <= TRAIN_EXTENT_M * 2.0):
        size = f"{extent / 1000:.1f} km" if extent >= 1000 else f"{extent:.0f} m"
        out.append(f"지도 한 변이 약 {size}입니다({basis} {dx:.3g} m 기준). 모델은 1 km 지형으로 학습되어 "
                   "축척을 구분하지 못하므로 결과가 실제와 다를 수 있습니다. 1 km 안팎으로 잘라서 넣는 것을 권장합니다.")
    if min(h, w) < 32:
        out.append(f"격자가 {h}x{w} 칸으로 작아서 결과가 뭉개질 수 있습니다(학습 격자 96x96).")
    if max(h, w) / max(min(h, w), 1) > 1.2:
        out.append(f"지도가 정사각형이 아닙니다({h}x{w}). 모델은 정사각형 지형으로 학습되어 가장자리 결과가 부정확할 수 있습니다.")
    return out


HAND_GRID_N = 96                    # 손그림을 읽는 격자 = 학습 격자 (96칸 x 10.42 m = 1 km)
# 손그림 땅 기울기[m] (위쪽 높고 아래쪽 낮게). 0 이면 평평 -> 산 밖 평지 전체가 물로 나온다 (handdrawn.TILT_M 참고)
HAND_TILT_M = float(os.environ.get("YUCHE_HAND_TILT_M", str(handdrawn.TILT_M)))


def _read_hand(raw: bytes) -> tuple[np.ndarray, dict]:
    """손그림 사진 -> (고도 격자, 메타). 학습 지형과 같은 1 km / 96칸으로 만든다."""
    terrain, preview, info, hwarns = handdrawn.read_drawing(raw, n=HAND_GRID_N, tilt_m=HAND_TILT_M)
    meta = {
        "source_format": "hand_drawing",
        "cellsize_m": TRAIN_EXTENT_M / HAND_GRID_N,
        "hand_levels": info["levels"],
        "hand_markers_found": info["markers_found"],
        "hand_tilt_m": info["tilt_m"],
        "warnings": hwarns,
        "hand_preview_png_base64": _b64(preview),
        "_hand_quad": info.get("quad"),   # 전시 모드가 원본 사진에서 다시 펼 자리 (응답에는 넣지 않음)
    }
    return terrain, meta


def _render_contour(terrain: np.ndarray, dx: float) -> Image.Image:
    ny, nx = terrain.shape
    xs = np.arange(nx) * dx
    ys = np.arange(ny) * dx
    X, Y = np.meshgrid(xs, ys)
    with _HEAVY_LOCK:
        rgb = render_contour_rgb(X, Y, terrain)
    return Image.fromarray(rgb)


def _make_overlay(contour_img: Image.Image, pred_img: Image.Image) -> Image.Image:
    size = (256, 256)
    # 배경은 등고선을 회색조로 (terrain 컬러맵의 파란 저지대가 '물'처럼 보여 혼동되므로).
    gray = np.asarray(contour_img.convert("L").resize(size)).astype(np.float64)
    base = np.stack([gray] * 3, axis=-1) * 0.35 + 255.0 * 0.65   # 옅게 깔기
    pred = np.asarray(pred_img.convert("RGB").resize(size)).astype(np.float64)

    # 예측 배수맵은 흰 배경 + 파란 유출. (255 - 밝기평균) 이 곧 유출 강도.
    # 절대값 기준 고정 게인 -> 유출 거의 없는 예측은 옅게, 강한 곳만 진하게.
    intensity = np.clip(255.0 - pred.mean(axis=2), 0.0, 255.0) / 255.0
    alpha = np.clip(intensity * 1.8, 0.0, 0.9)[..., None]
    highlight = np.array([12.0, 90.0, 235.0])
    blended = base * (1.0 - alpha) + highlight * alpha
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def _b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _safe_uses_film() -> bool:
    try:
        return bool(current_uses_film())
    except Exception:  # noqa: BLE001  (체크포인트 로드 전/문제 시)
        return False


def _safe_cond_dim() -> int:
    """체크포인트가 요구하는 조건 개수. 0=없음, 1=rain_mm, 2=rain_mm+time_s."""
    try:
        return int(current_cond_dim())
    except Exception:  # noqa: BLE001
        return 0


def _safe_image_size() -> int | None:
    try:
        return int(current_image_size())
    except Exception:  # noqa: BLE001
        return None


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------- #
# 화면 폴더(YUCHE_SITE_DIR)의 나머지 파일 (logo.png 등). [11/27]
# ngrok 터널 하나(8000)로 화면·API·카메라를 한 주소에서 쓰려면 화면이 쓰는 파일을 이 서버가 다 내줘야 한다.
# 맨 끝에 둬서 위의 API 경로가 먼저 잡히게 하고, 폴더 밖·정해진 확장자 밖은 내주지 않는다.
# --------------------------------------------------------------------------- #
_SITE_ASSET_EXT = {".png", ".jpg", ".jpeg", ".webp", ".svg", ".ico", ".js", ".css", ".json"}


# --------------------------------------------------------------------------- #
# 전시 모드 (expo.py 설명)
# --------------------------------------------------------------------------- #
@app.post("/api/expo/result")
def expo_result(
    expo_id: str = Form(...),
    meta: str = Form("{}"),
    result_image: UploadFile = File(...),
) -> dict:
    """화면이 만든 결과 그림(18초 지도 + 빨간 빗금 + 배수구 동그라미)을 손그림과 짝으로 저장. 전시 모드에서만, 로그인 불필요."""
    if not expo.ENABLED:
        raise HTTPException(404, "전시 모드가 아닙니다.")
    raw = result_image.file.read(expo.RESULT_MAX_BYTES + 1)
    result_image.file.close()
    try:
        expo.save_result(expo_id, raw, meta)
    except expo.ExpoError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True}


@app.get("/expo/gallery", include_in_schema=False)
def expo_gallery(key: str = "") -> HTMLResponse:
    """팀 전용 모아보기: 날짜별 [손그림 | 읽은 그림 | 결과]. key = YUCHE_EXPO_KEY."""
    if not expo.key_ok(key):
        raise HTTPException(403, "키가 맞지 않습니다.")
    return HTMLResponse(expo.gallery_html(key), headers=_NO_CACHE)


@app.get("/expo/gallery.zip", include_in_schema=False)
def expo_gallery_zip(key: str = "") -> Response:
    if not expo.key_ok(key):
        raise HTTPException(403, "키가 맞지 않습니다.")
    name = time.strftime("expo_%Y%m%d_%H%M.zip")
    return Response(expo.zip_bytes(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{name}"', **_NO_CACHE})


@app.get("/expo/file/{day}/{name}", include_in_schema=False)
def expo_file(day: str, name: str, key: str = "") -> FileResponse:
    if not expo.key_ok(key):
        raise HTTPException(403, "키가 맞지 않습니다.")
    path = expo.file_path(day, name)
    if path is None:
        raise HTTPException(404, "없는 파일입니다.")
    return FileResponse(path, headers=_NO_CACHE)


@app.get("/{asset}", include_in_schema=False)
def site_asset(asset: str) -> FileResponse:
    if SITE_DIR is None or Path(asset).suffix.lower() not in _SITE_ASSET_EXT:
        raise HTTPException(404, "없는 파일입니다.")
    target = (SITE_DIR / asset).resolve()
    if target.parent != SITE_DIR or not target.is_file():
        raise HTTPException(404, "없는 파일입니다.")
    return FileResponse(target, headers=_NO_CACHE)
