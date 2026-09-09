# 검증 결과

검증일: 2026-09-08

## 자동 검사

| 검사 | 결과 |
|---|---|
| Python compileall | 통과 |
| Ruff 정적 검사 | 통과 |
| Pytest | 7개 통과 |
| Alembic upgrade | 통과 |
| Alembic downgrade | 통과 |
| Alembic 모델·마이그레이션 차이 검사 | 차이 없음 |
| Git diff whitespace 검사 | 통과 |

Pytest 범위:

- grayscale DEM 전처리와 256×256 모델 입력 생성
- 원본과 전처리 수치 배열의 분리 가능성
- 전처리 버전·옵션별 캐시 키
- 사용자 배수구만 사용하는 용량 근사 평가
- 평가기 비활성화 시 임의 결과 대신 오류 반환
- 원자적 파일 저장과 상위 경로 이탈 차단
- 업로드 → 예측 → 사용자 시나리오 → CSV API 흐름
- 동일 AI 요청의 캐시 적중과 모델 1회 호출

## 모델 런타임 검사

제공 모델과 같은 RGB 입력, FiLM `cond_dim=2`, `ngf=64`, 256×256 구조로 합성한
체크포인트를 사용하여 다음 항목을 확인했습니다.

- 엄격한 `state_dict` 로딩 통과
- 체크포인트 SHA-256 생성 통과
- CPU 워밍업 통과
- `rain_mm`, `time_s` 조건 추론 통과
- RGB 배수맵 생성 통과
- 256×256 수심 배열 역변환 통과
- 합성 체크포인트 크기: 217,866,157 bytes
- 검사 환경 CPU 단일 추론: 169ms

위 시간은 랜덤 합성 가중치와 현재 검사 CPU에서 측정한 런타임 확인값이며 실제 모델의
정확도 또는 운영 성능 목표값이 아닙니다.

## 첨부 체크포인트 확인 상태

작업 공간에 복사된 첨부 ZIP은 중앙 디렉터리가 없는 불완전한 ZIP으로 확인됐고,
추출된 `best.pt`도 162,922,496 bytes 지점에서 끝나 PyTorch가 다음 오류를 반환했습니다.

```text
PytorchStreamReader failed reading zip archive: failed finding central directory
```

따라서 현재 작업 공간의 첨부 복사본으로는 실제 가중치 워밍업까지 완료할 수 없었습니다.
코드의 v5 구조와 추론 경로는 동일 계약의 합성 체크포인트로 검증했으며, 배포 전 완전한
`best.pt`를 다시 받아 다음 명령을 반드시 통과해야 합니다.

```bash
AI_MODEL_CHECKPOINT=/absolute/path/to/best.pt python -m scripts.verify_checkpoint
```
