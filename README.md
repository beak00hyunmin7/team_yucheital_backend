# 지형 워터맵·배수구 추천 백엔드

등고선 또는 grayscale DEM 이미지와 배수구 개수를 받아 다음 작업을 수행하는.
`FastAPI + MySQL` 백엔드입니다.

1. 워터맵과 배수구 후보 생성
2. 물리·지형 규칙, D8 모의 배수, 입력 변형 일관성으로 자동 검증
3. 실패 시 최대 2회 다른 분석 전략으로 자동 재분석
4. 워터맵 위에 배수구 위치 표시
5. 입력 이미지·워터맵·표시 결과·좌표·검증 근거를 하나의 사례로 저장
6. 자동 검증을 통과한 사례만 재학습 대기 데이터로 등록

별도의 사용자·전문가 승인 단계는 없습니다.

> 현재 포함된 분석기는 학습된 AI 모델이 아니라 실행 가능한 D8 수문 분석 대리
> 모델입니다. 실제 PyTorch/ONNX 모델은 `app/ai/interface.py` 규격에 맞춰 교체합니다.

## 바로 실행하기: FastAPI + MySQL

필요 프로그램은 Docker Desktop입니다.

```bash
cp .env.example .env
docker compose up --build
```

Windows 명령 프롬프트에서는 다음을 사용할 수 있습니다.

```bat
copy .env.example .env
run_windows.bat
```

실행 후 접속 주소:

- Swagger API 테스트: <http://127.0.0.1:8000/docs>
- 서버·DB 상태 확인: <http://127.0.0.1:8000/api/v1/health>

최초 실행 시 MySQL의 `terrain_drainage` 데이터베이스와 7개 테이블이 자동으로
생성됩니다. 이미지 파일과 MySQL 데이터는 Docker volume에 유지됩니다.

공유 서버나 외부에서 접속 가능한 환경에 배포하기 전에는 `.env`의 두 비밀번호를
반드시 변경해야 합니다.

## 분석 API 사용

`POST /api/v1/drainage/analyze`에 `multipart/form-data`로 요청합니다.

| 필드 | 기본값 | 설명 |
|---|---:|---|
| `image` | 필수 | PNG/JPEG/WEBP/TIFF 등고선 또는 grayscale DEM |
| `drain_count` | 3 | 배수구 개수, 1~10 |
| `rainfall_mm_per_hour` | 50 | 설계 강우강도(mm/h) |
| `runoff_coefficient` | 0.8 | 유출계수 |
| `cell_size_m` | 1.0 | 원본 픽셀 1개의 실제 길이(m) |
| `minimum_spacing_ratio` | 0.10 | 짧은 이미지 변 대비 최소 간격 비율 |
| `blur_radius` | 1.2 | DEM 노이즈 완화값 |
| `high_is_bright` | true | 밝은 픽셀을 높은 고도로 해석 |
| `include_overlay` | false | 결과 PNG를 Base64로도 반환 |

응답의 핵심 부분은 다음과 같습니다.

```json
{
  "analysis_case_id": "c81555be-0f55-4ecb-a765-f02d1befa9dd",
  "status": "COMPLETED",
  "verification_status": "AUTO_VERIFIED",
  "validation_confidence": 0.88,
  "retry_count": 0,
  "training_status": "READY",
  "water_map_url": "/files/analysis_cases/.../water_map.png",
  "overlay_image_url": "/files/analysis_cases/.../drain_overlay.png",
  "automatic_validation": {
    "passed": true,
    "estimated_capture_ratio": 0.035,
    "residual_water_ratio": 0.965,
    "stability_score": 0.95,
    "failure_reasons": []
  },
  "drains": [
    {
      "rank": 1,
      "normalized": {"x": 0.72, "y": 0.81},
      "suitability_score": 0.91
    }
  ]
}
```

프론트엔드는 `overlay_image_url`을 그대로 표시하거나, `water_map_url` 위에
`drains[].normalized` 좌표를 직접 그리면 됩니다.

## 자동 검증과 재분석

검증기는 예측 AI의 점수만 신뢰하지 않고 아래 기준을 따로 계산합니다.

- 요청 개수 일치
- 배수구 간 최소 거리 충족
- 지형 경계 안전 여백 충족
- 후보의 평균 지형 적합도
- D8 모의 배수에서 배수구를 통과하는 예상 유출수 비율
- 입력을 조금 평활화해도 비슷한 위치가 나오는지에 대한 일관성
- 위 결과를 합친 종합 신뢰도

기준 미달이면 최대 3회까지 분석 전략을 바꾸어 다시 계산합니다. 모든 시도가
실패하면 결과는 `AUTO_REJECTED`로 저장하고 사용자에게 반환하지만, 재학습 데이터에는
넣지 않습니다.

```text
AI_GENERATED
→ AUTO_VALIDATING
→ AUTO_VERIFIED → training_dataset_items: READY
                 또는
→ 재분석 → AUTO_REJECTED → 학습 제외
```

검증 임계값은 `compose.yaml`의 환경변수로 조정할 수 있습니다. 실제 운영값은 별도
시뮬레이션 정답 데이터로 보정해야 합니다.

## 저장 구조

분석 이미지 파일:

```text
data/storage/analysis_cases/{analysis_case_id}/
├── input.png
├── water_map.png
└── drain_overlay.png
```

MySQL 테이블:

| 테이블 | 저장 내용 |
|---|---|
| `model_versions` | 사용 모델과 활성 버전 |
| `analysis_cases` | 요청, 상태, 캐시 키, 선택 시도, 최종 검증 결과 |
| `analysis_files` | 입력·워터맵·결과 이미지 경로와 해시 |
| `analysis_attempts` | 최초 분석과 자동 재분석 시도별 좌표 |
| `automatic_validations` | 시도별 규칙·시뮬레이션·일관성 지표 |
| `drain_positions` | 모든 시도의 배수구 좌표와 최종 선택 여부 |
| `training_dataset_items` | 자동 검증을 통과한 재학습용 묶음 |

모든 자료는 `analysis_case_id`로 연결됩니다. 이미지 자체는 MySQL에 넣지 않고 파일
저장소에 두며, MySQL에는 경로·해시·좌표·검증 정보를 저장합니다.

## 주요 API

| 기능 | API |
|---|---|
| 분석·자동 검증·저장 | `POST /api/v1/drainage/analyze` |
| 최근 분석 목록 | `GET /api/v1/drainage/analyses` |
| 분석 사례 상세 | `GET /api/v1/drainage/analyses/{id}` |
| 자동 검증·재분석 이력 | `GET /api/v1/drainage/analyses/{id}/validations` |
| 재학습 대기 데이터 | `GET /api/v1/training-dataset/items?status=READY` |
| 저장 이미지 | `GET /files/analysis_cases/{id}/{filename}` |

## 실제 AI 연결

`app/ai/interface.py`의 `DrainageModel` 규격을 구현한 뒤
`app/ai/hydrology_model.py`에서 내보내는 `model` 객체를 교체합니다.

필수 출력:

- 워터맵 생성에 쓰는 2차원 flow/score 배열
- 배수구 후보의 `x`, `y`, 적합도, 고도, 누적 유량
- 각 셀의 하류 연결 정보: 자동 모의 배수 검증에 사용
- 모델 버전

AI 추가 학습은 API 요청 중 실행하지 않습니다. `training_dataset_items`의 `READY`
자료를 별도 오프라인 학습 작업이 읽고, 고정 평가셋에서 기존 모델보다 좋아진 경우에만
새 모델을 배포해야 합니다.

## 개발 테스트

테스트는 MySQL과 동일한 SQLAlchemy 모델을 임시 SQLite DB에 만들어 빠르게 실행합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest
```

MySQL 원본 스키마는 `mysql/init/001_schema.sql`에 있습니다.

## 현재 한계

1. 일반 등고선 선화는 고도값을 직접 포함하지 않으므로 OCR·등고선 topology·고도 보간
   또는 DEM 변환 과정이 별도로 필요합니다.
2. 현재 모델과 검증 시뮬레이션은 CFD가 아니라 D8 기반 MVP입니다.
3. 자동 검증 결과는 실제 정답이 아니라 고신뢰도 의사 라벨입니다. 실측 또는 고품질
   시뮬레이션 정답과 함께 학습해야 오류가 강화되는 것을 줄일 수 있습니다.
4. 운영 배포 전 인증, 접근 권한, 악성 파일 검사, rate limit, 백업, 마이그레이션 도구를
   추가해야 합니다.
