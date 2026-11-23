# team_yucheital_backend

레인스코프(유체이탈) 침수 예측 웹서비스의 백엔드 저장소입니다.

**서버 코드는 전부 [`prediction_server/`](prediction_server/README.md) 에 있습니다.**
설치·실행 방법도 그 폴더의 README 를 보세요.

| 구성 | 위치 |
|---|---|
| 예측 서버 (FastAPI: 예측 · 회원가입/로그인 · 분석 기록) | `prediction_server/backend/` |
| AI 모델 코드 (서브모듈) | `prediction_server/ai` → [team_yucheital_ai_model](https://github.com/beak00hyunmin7/team_yucheital_ai_model) |
| 화면 | [team_yucheital_frontend](https://github.com/beak00hyunmin7/team_yucheital_frontend) |

```powershell
git clone --recurse-submodules https://github.com/beak00hyunmin7/team_yucheital_backend.git
```

> 예전 서버(`app/` 배수구 위치 추천, `backend_v2/` 시나리오 비교)와 Docker 설정은 2026-11-23 정리하면서 삭제했습니다.
> 필요하면 git 기록에서 되살릴 수 있습니다 (정리 직전 커밋: `c885e91`).
