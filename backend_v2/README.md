# 지형 배수맵 백엔드 v2

기존 루트의 `app/`를 변경하지 않고 새로 만든 독립 실행형 백엔드입니다. 기존 계획인
`업로드 → 전처리 → AI 배수맵 → 사용자 배수구 설정 → 시나리오 비교`를 따릅니다.

## v1과 달라진 핵심

- AI는 배수맵과 수심 배열만 생성합니다.
- 배수구 위치를 D8이나 규칙으로 자동 선정하지 않습니다.
- AI 응답의 `drains`는 항상 빈 배열입니다.
- 시나리오 API는 사용자가 전송한 배수구 좌표만 평가합니다.
- 첨부된 v5 체크포인트의 RGB 256×256, `rain_mm`, `time_s`, FiLM `cond_dim=2`
  계약을 엄격하게 검사합니다.
- AI 로딩에 실패하면 D8로 전환하지 않고 readiness와 예측 API가 503을 반환합니다.
- 원본, 모델 입력 PNG, 고도 배열, 배수맵, 수심 배열을 별도 파일로 저장합니다.
- 같은 업로드·조건·시나리오는 해시 캐시와 single-flight로 중복 계산을 줄입니다.
- 요청당 AI 추론은 한 번만 실행합니다.

## 저장 구조

```text
data/storage/
├── terrain/{upload_id}/
│   ├── original/input.{ext}
│   └── preprocessing/{version}/
│       ├── model_input.png
│       └── elevation.npz
├── predictions/{result_id}/
│   ├── water_map.png
│   └── depth_m.npz
└── scenarios/{scenario_id}/
    ├── scenario_map.png
    └── depth_m.npz
```

이미지와 배열은 파일 저장소에 두고 DB에는 경로, 해시, 버전, 지표와 관계만
저장합니다. CSV는 원본 저장 형식으로 사용하지 않고 요청 시 DB에서 생성합니다.

## 실행

### Docker Compose

```bash
cd backend_v2
cp .env.example .env
```

`.env`에서 다음 값을 실제 환경에 맞게 변경합니다.

```dotenv
MYSQL_PASSWORD=충분히_긴_비밀번호
MYSQL_ROOT_PASSWORD=충분히_긴_루트_비밀번호
AI_CHECKPOINT_HOST_DIR=/absolute/path/to/1_ai_model_deploy_package/checkpoints
```

CPU 실행:

```bash
docker compose up --build
```

NVIDIA Container Toolkit이 설치된 GPU 서버:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

접속 주소:

- Swagger: <http://127.0.0.1:8001/docs>
- Liveness: <http://127.0.0.1:8001/api/v2/health/live>
- Readiness: <http://127.0.0.1:8001/api/v2/health/ready>

체크포인트를 찾지 못하거나 구조가 다르면 liveness는 200, readiness는 503입니다.
API는 실행 중이지만 AI 요청을 받을 준비가 되지 않았다는 뜻입니다.

### 로컬 실행

```bash
cd backend_v2
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export AI_MODEL_CHECKPOINT=/absolute/path/to/best.pt
alembic upgrade head
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

Windows PowerShell에서는 환경변수를 다음처럼 지정합니다.

```powershell
$env:AI_MODEL_CHECKPOINT="C:\path\to\best.pt"
```

## API 순서

1. `POST /api/v2/upload-terrain`
2. `POST /api/v2/predict-drainage-map`
3. 프론트에서 배수구를 클릭·드래그
4. `POST /api/v2/evaluate-drain-settings`
5. `GET /api/v2/result/{result_id}` 또는 `GET /api/v2/scenario/{scenario_id}`

요청 예시는 [docs/api.md](docs/api.md)에 있습니다.

## 시나리오 평가 주의사항

기본 `capacity_approx` 평가기는 사용자가 지정한 위치 주변에서 배수구 용량과 지속시간만큼
물을 제거하는 명시적 근사 계산입니다. 배수구를 찾지 않으며 CFD, 관로, 장애물, 침투를
모델링하지 않습니다. 응답에 `capacity-approx-v1`과 경고가 포함됩니다.

학습된 시나리오 surrogate가 준비되면 `ScenarioEvaluator` 인터페이스를 구현하여
교체해야 합니다. 근사 결과를 제공하지 않으려면 다음처럼 비활성화할 수 있습니다.

```dotenv
SCENARIO_EVALUATOR=none
```

이때 시나리오 평가 요청은 임의 결과 대신 503을 반환합니다.

## 테스트

```bash
python -m pip install -r requirements-test.txt
python -m ruff check app tests
python -m pytest -q
```

실제 체크포인트 계약과 워밍업 확인:

```bash
AI_MODEL_CHECKPOINT=/absolute/path/to/best.pt python -m scripts.verify_checkpoint
```

체크포인트 파일은 저장소에 커밋하지 않습니다.

## 문서

- [API 사용법](docs/api.md)
- [점검 목록](docs/audit_checklist.md)
- [개선 내역](docs/improvement_report.md)
- [검증 결과](docs/verification_results.md)
