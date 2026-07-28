# 지형 배수구 추천 FastAPI

밝기값이 상대 고도를 나타내는 2D DEM 이미지를 받아 지표 유출 방향과 누적 유량을 계산하고, 배수구 후보 위치를 JSON과 오버레이 이미지로 반환하는 백엔드 MVP입니다.

## 어떤 환경으로 실행하나요?

- **권장 개발 환경:** VS Code 또는 PyCharm + Python 3.11/3.12
- **Jupyter Notebook:** DEM 전처리와 알고리즘 실험에는 유용하지만 API 서버의 기본 실행 환경으로는 비권장
- **OBS Studio:** 화면 녹화·송출 도구이므로 백엔드 개발 및 실행과 무관
- **Docker:** 팀원별 환경 차이 없이 실행하거나 서버에 배포할 때 권장

현재 설치된 것이 Jupyter뿐이어도 Jupyter의 Terminal을 열어 아래 명령으로 실행할 수 있습니다. Windows에서는 `run_windows.bat`를 더블 클릭해도 됩니다.

## 1. 실행

### Windows PowerShell

```powershell
cd terrain-drainage-fastapi
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Python 3.12가 없다면 `py -3.11`을 사용해도 됩니다.

### macOS/Linux

```bash
cd terrain-drainage-fastapi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

실행 후 다음 주소를 엽니다.

- Swagger 테스트 화면: <http://127.0.0.1:8000/docs>
- 상태 확인: <http://127.0.0.1:8000/api/v1/health>

## 2. 샘플 DEM 만들기

```bash
python scripts/generate_sample_dem.py
```

생성 파일은 `samples/sample_dem.png`입니다. 흰색에 가까울수록 높은 지형이고 검은색에 가까울수록 낮은 지형입니다.

## 3. Swagger에서 테스트

1. `/docs` 접속
2. `POST /api/v1/drainage/analyze` 선택
3. `Try it out` 클릭
4. `image`에 `samples/sample_dem.png` 업로드
5. 기본값을 유지하고 `Execute` 클릭
6. 응답의 `drains[].pixel`에서 추천 좌표 확인
7. `overlay_png_base64`는 프런트엔드에서 `data:image/png;base64,` 뒤에 붙여 표시

프런트엔드 예시:

```javascript
const form = new FormData();
form.append("image", fileInput.files[0]);
form.append("drain_count", "3");
form.append("rainfall_mm_per_hour", "50");

const response = await fetch("http://127.0.0.1:8000/api/v1/drainage/analyze", {
  method: "POST",
  body: form,
});
const result = await response.json();
overlayImage.src = `data:image/png;base64,${result.overlay_png_base64}`;
```

## 4. 요청 파라미터

| 필드 | 기본값 | 의미 |
|---|---:|---|
| `image` | 필수 | PNG/JPEG/WEBP/TIFF 형식의 DEM 이미지 |
| `drain_count` | 3 | 추천할 배수구 개수(1~10) |
| `rainfall_mm_per_hour` | 50 | 설계 강우강도(mm/h) |
| `runoff_coefficient` | 0.8 | 강우 중 지표 유출로 전환되는 비율(0~1) |
| `cell_size_m` | 1.0 | 원본 이미지 한 픽셀의 실제 한 변 길이(m) |
| `minimum_spacing_ratio` | 0.10 | 배수구 간 최소 간격/이미지 짧은 변 |
| `blur_radius` | 1.2 | 영상 노이즈를 줄이는 Gaussian blur 반경 |
| `high_is_bright` | true | 밝을수록 높은 지형인지 여부 |
| `include_overlay` | true | 결과 오버레이 이미지를 응답에 포함할지 여부 |

## 5. 계산 방식

1. 이미지를 grayscale DEM으로 변환하고 높이를 0~1로 정규화합니다.
2. 각 셀에서 인접 8방향 중 가장 가파르게 낮아지는 방향을 선택하는 D8 방식으로 유향을 계산합니다.
3. 상류에서 해당 셀로 들어오는 기여 셀 수를 누적합니다.
4. 누적 유량 65%, 낮은 고도 25%, 국부 함몰도 10%를 결합해 적합도 점수를 만듭니다.
5. 지정한 최소 간격을 만족하는 상위 후보를 배수구 위치로 반환합니다.
6. 첨두 유량은 단순화한 Rational Method로 추정합니다.

`Q(L/s) = 유출계수 × 강우강도(mm/h) × 집수면적(m²) / 3600`

## 6. 중요한 입력 조건

이 MVP는 **등고선만 그려진 일반 지도 이미지**를 바로 해석하지 않습니다. 등고선의 숫자 고도, 등고 간격, 축척이 없는 선 그림만으로는 실제 고도장을 유일하게 복원할 수 없기 때문입니다. 먼저 아래 중 하나가 필요합니다.

- GIS에서 추출한 DEM/DTM raster
- 각 픽셀의 밝기가 상대 고도를 뜻하도록 만든 height map
- 추후 추가할 등고선 OCR + 고도 보간 전처리 모듈의 출력

## 7. 현재 한계와 다음 개발 순서

현재 결과는 CFD가 아니라 지형수문학 기반 초기 후보 추천입니다. 실제 설계용으로 발전시키려면 다음 순서가 적합합니다.

1. GeoTIFF/실수형 DEM 입력 및 실제 좌표계 지원
2. 함몰부 보정(Priority-Flood), D-infinity 또는 MFD 유향 적용
3. 토지피복별 침투율·조도·유출계수 적용
4. 건물/도로/배수관로/설치 금지영역 마스크 반영
5. 배수구 용량과 비용 제약을 포함한 최적화
6. 검증용 SWMM 또는 CFD 결과와 비교하고, 필요하면 대리모델 학습

## 8. 테스트

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

## 9. Docker 실행

```bash
docker compose up --build
```

