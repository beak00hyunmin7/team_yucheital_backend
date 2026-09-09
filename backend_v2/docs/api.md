# API 사용 순서

## 1. 업로드와 전처리

```bash
curl -X POST http://127.0.0.1:8001/api/v2/upload-terrain \
  -F "image=@terrain.png" \
  -F "input_type=grayscale_dem" \
  -F "high_is_bright=true" \
  -F "blur_radius=1.2" \
  -F "cell_size_m=1.0"
```

`input_type`은 `grayscale_dem`, `contour_image`, `auto` 중 하나입니다. 자동 판별이
틀릴 가능성이 있으므로 운영 UI에서는 사용자가 입력 종류를 확인하도록 권장합니다.

응답에는 원본과 전처리 모델 입력이 별도 자산으로 포함됩니다.

```json
{
  "upload_id": "...",
  "input_type": "grayscale_dem",
  "quality_score": 0.91,
  "assets": [
    {"kind": "ORIGINAL", "url": "/files/terrain/.../original/input.png"},
    {"kind": "MODEL_INPUT", "url": "/files/terrain/.../model_input.png"},
    {"kind": "ELEVATION_ARRAY", "url": "/files/terrain/.../elevation.npz"}
  ]
}
```

## 2. AI 기본 배수맵

```bash
curl -X POST http://127.0.0.1:8001/api/v2/predict-drainage-map \
  -H "Content-Type: application/json" \
  -d '{"upload_id":"UPLOAD_ID","rain_mm":45.0,"time_s":120.0}'
```

`rain_mm`은 20~80mm, `time_s`는 0~120초 범위입니다. 이는 제공된 v5 모델의 학습
입력 범위입니다. 응답의 `drains`는 자동 탐색이 없음을 명시하기 위해 항상 `[]`입니다.

## 3. 사용자 배수구 시나리오

```bash
curl -X POST http://127.0.0.1:8001/api/v2/evaluate-drain-settings \
  -H "Content-Type: application/json" \
  -d '{
    "result_id":"RESULT_ID",
    "duration_s":120,
    "drains":[
      {
        "x":0.35,
        "y":0.62,
        "capacity_lps":15.0,
        "capture_radius_m":10.0,
        "drain_type":"standard"
      }
    ]
  }'
```

`x`, `y`는 왼쪽 위가 `(0, 0)`, 오른쪽 아래가 `(1, 1)`인 정규화 좌표입니다.
`capacity_approx` 평가를 사용하려면 업로드 때 `cell_size_m`을 제공해야 합니다.

## 4. 조회와 CSV 내보내기

```text
GET /api/v2/result/{result_id}
GET /api/v2/scenario/{scenario_id}
GET /api/v2/scenario/{scenario_id}/export.csv
```

CSV는 중복 원본으로 보관하지 않고 DB의 사용자 배수구와 시나리오 지표를 요청 시
생성합니다.
