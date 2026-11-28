"""전시 모드 (YUCHE_EXPO=1): 관람객이 가입·로그인 없이 체험하고, 손그림 예측 1번마다
'손그림 + 결과 이미지' 짝을 서버에 모은다. 꺼져 있으면(기본) 이 모듈은 아무것도 하지 않는다.

저장 위치: prediction_server/data/expo/YYYYMMDD/   (data/ 는 git 에 올라가지 않는다)
  {expo_id}_drawing.jpg : 네모 4개 안쪽을 반듯하게 편 손그림 (1024x1024, 원본 사진에서 다시 폄)
  {expo_id}_read.png    : 서버가 만든 '이렇게 읽었어요' 그림 (hand_preview)
  {expo_id}_result.jpg  : 화면이 보낸 결과 그림 (18초 지도 + 빨간 빗금 + 배수구 동그라미)
  records.jsonl         : 결과가 들어온 짝마다 한 줄 (시각, 강우량, 배수구, 빗금 면적, 파일 이름)

지키는 것
  - 원본 사진은 저장하지 않는다 (노트북 카메라에 얼굴·손이 찍힐 수 있음). 네모 4개를 못 찾으면
    펴서 잘라낼 수 없으므로 아무것도 저장하지 않는다 (사진 전체를 저장하게 되므로)
  - 업로드 파일 이름은 저장하지 않는다
  - 결과는 발급 후 10분 안에, 한 번만, 3MB 이하만 받는다

모아보기(팀 전용): GET /expo/gallery?key=... (YUCHE_EXPO_KEY). 전시 후 삭제: python clear_expo_data.py
"""
from __future__ import annotations

import base64
import html
import io
import json
import os
import re
import secrets
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path

from PIL import Image

ENABLED = os.environ.get("YUCHE_EXPO", "0") == "1"
KEY = os.environ.get("YUCHE_EXPO_KEY", "")
_BASE = Path(__file__).resolve().parent.parent
DIR = Path(os.environ.get("YUCHE_EXPO_DIR", str(_BASE / "data" / "expo"))).resolve()   # 시험용으로만 바꾼다

RESULT_TTL_S = 600            # 결과는 손그림 저장 후 10분 안에만
RESULT_MAX_BYTES = 3 * 1024 * 1024
DRAWING_SIZE = 1024
_ID_RE = re.compile(r"^[0-9a-f]{20}$")
_DAY_RE = re.compile(r"^\d{8}$")
_NAME_RE = re.compile(r"^[0-9a-f]{20}_(drawing\.jpg|read\.png|result\.jpg)$")
_LOCK = threading.Lock()


class ExpoError(ValueError):
    """400 으로 돌려줄 사유."""


def _day_dir(when: float | None = None) -> Path:
    return DIR / datetime.fromtimestamp(when or time.time()).strftime("%Y%m%d")


def save_drawing(flat: Image.Image, preview_png_b64: str | None) -> str:
    """편 손그림(1024)과 '읽었어요' 그림을 저장하고 expo_id 를 돌려준다."""
    expo_id = secrets.token_hex(10)
    d = _day_dir()
    d.mkdir(parents=True, exist_ok=True)
    flat.convert("RGB").resize((DRAWING_SIZE, DRAWING_SIZE), Image.LANCZOS).save(d / f"{expo_id}_drawing.jpg", "JPEG", quality=90)
    if preview_png_b64:
        (d / f"{expo_id}_read.png").write_bytes(base64.b64decode(preview_png_b64))
    return expo_id


def _find_drawing(expo_id: str) -> Path | None:
    if not _ID_RE.match(expo_id or ""):
        return None
    now = time.time()
    for when in (now, now - 86400):   # 자정을 넘긴 경우 어제 폴더도
        p = _day_dir(when) / f"{expo_id}_drawing.jpg"
        if p.is_file():
            return p
    return None


def _num(v, lo=-1e9, hi=1e9):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if lo <= x <= hi else None


def save_result(expo_id: str, image_bytes: bytes, meta_json: str) -> dict:
    """결과 그림을 짝으로 저장하고 records.jsonl 에 한 줄. 조건이 안 맞으면 ExpoError."""
    drawing = _find_drawing(expo_id)
    if drawing is None:
        raise ExpoError("없는 체험 번호입니다.")
    if time.time() - drawing.stat().st_mtime > RESULT_TTL_S:
        raise ExpoError("체험 번호가 만료되었습니다 (10분).")
    if not image_bytes:
        raise ExpoError("결과 그림이 비었습니다.")
    if len(image_bytes) > RESULT_MAX_BYTES:
        raise ExpoError("결과 그림이 3MB 를 넘습니다.")
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception as exc:  # noqa: BLE001
        raise ExpoError("결과 그림을 열 수 없습니다.") from exc
    try:
        meta = json.loads(meta_json or "{}")
        if not isinstance(meta, dict):
            raise ValueError
    except ValueError as exc:
        raise ExpoError("meta 는 JSON 객체여야 합니다.") from exc
    drain = meta.get("drain") if isinstance(meta.get("drain"), dict) else {}
    rec = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "expo_id": expo_id,
        "rain_mm": _num(meta.get("rain_mm"), 0, 1000),
        "drain": {"x": _num(drain.get("x"), 0, 1), "y": _num(drain.get("y"), 0, 1)} if drain else None,
        "risk_area_pct": _num(meta.get("risk_area_pct"), 0, 100),
        "risk_threshold": meta.get("risk_threshold") if isinstance(meta.get("risk_threshold"), (dict, int, float)) else None,
        "files": {"drawing": drawing.name, "read": f"{expo_id}_read.png", "result": f"{expo_id}_result.jpg"},
    }
    d = drawing.parent
    out = d / f"{expo_id}_result.jpg"
    with _LOCK:   # 같은 번호가 동시에 두 번 와도 한 번만
        if out.exists():
            raise ExpoError("이미 결과가 저장된 체험 번호입니다.")
        img.convert("RGB").save(out, "JPEG", quality=88)
        with open(d / "records.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def key_ok(key: str | None) -> bool:
    return bool(KEY) and bool(key) and secrets.compare_digest(str(key), KEY)


def file_path(day: str, name: str) -> Path | None:
    if not (_DAY_RE.match(day or "") and _NAME_RE.match(name or "")):
        return None
    p = DIR / day / name
    return p if p.is_file() else None


def _pairs():
    """날짜(최근 먼저) -> [(expo_id, 손그림 시각, 결과 있음)]."""
    out = []
    if not DIR.is_dir():
        return out
    for d in sorted((p for p in DIR.iterdir() if p.is_dir() and _DAY_RE.match(p.name)), reverse=True):
        rows = []
        for p in sorted(d.glob("*_drawing.jpg"), key=lambda q: q.stat().st_mtime, reverse=True):
            eid = p.name.split("_")[0]
            rows.append((eid, datetime.fromtimestamp(p.stat().st_mtime).strftime("%H:%M:%S"),
                         (d / f"{eid}_read.png").is_file(), (d / f"{eid}_result.jpg").is_file()))
        out.append((d.name, rows))
    return out


def gallery_html(key: str) -> str:
    q = "key=" + html.escape(key, quote=True)
    days = _pairs()
    total = sum(len(r) for _, r in days)
    parts = [f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>전시 체험 모아보기</title><style>
body{{margin:0;font:15px/1.5 "Malgun Gothic","Apple SD Gothic Neo",sans-serif;background:#f3f5f8;color:#1c2430}}
.wrap{{max-width:1100px;margin:0 auto;padding:24px 16px 48px}} h1{{margin:0 0 4px;font-size:24px}} .sub{{color:#5f6b7a;margin:0 0 18px}}
a.dl{{display:inline-block;padding:8px 14px;border-radius:8px;background:#1f6fd1;color:#fff;text-decoration:none;font-weight:700}}
h2{{margin:28px 0 10px;font-size:18px}} .row{{display:grid;grid-template-columns:90px repeat(3,minmax(0,1fr));gap:10px;align-items:center;background:#fff;border:1px solid #d5dbe3;border-radius:10px;padding:10px;margin-bottom:10px}}
.row img{{width:100%;height:auto;border-radius:6px;background:#eef1f5}} .t{{font-size:13px;color:#5f6b7a}} .none{{display:flex;align-items:center;justify-content:center;aspect-ratio:1;border:1px dashed #c4ccd6;border-radius:6px;color:#8a95a3;font-size:13px}}
.head{{display:grid;grid-template-columns:90px repeat(3,minmax(0,1fr));gap:10px;font-size:12px;color:#5f6b7a;padding:0 10px}}
@media (max-width:640px){{.row,.head{{grid-template-columns:repeat(3,minmax(0,1fr))}} .t{{grid-column:1/-1}} .head span:first-child{{display:none}}}}
</style></head><body><div class="wrap"><h1>전시 체험 모아보기</h1>
<p class="sub">손그림 {total}건 · 결과가 없는 짝은 '결과 없음'으로 보여요.</p><a class="dl" href="/expo/gallery.zip?{q}">전체 zip 내려받기</a>"""]
    for day, rows in days:
        parts.append(f"<h2>{day[:4]}-{day[4:6]}-{day[6:]} · {len(rows)}건</h2>"
                     "<div class='head'><span>시각</span><span>손그림</span><span>읽은 그림</span><span>결과</span></div>")
        for eid, hhmm, has_read, has_result in rows:
            img = lambda n: f"<img loading='lazy' src='/expo/file/{day}/{eid}_{n}?{q}' alt=''>"
            parts.append(f"<div class='row'><span class='t'>{hhmm}</span>{img('drawing.jpg')}"
                         f"{img('read.png') if has_read else '<span class=none>없음</span>'}"
                         f"{img('result.jpg') if has_result else '<span class=none>결과 없음</span>'}</div>")
    if not days:
        parts.append("<p class='sub' style='margin-top:20px'>아직 저장된 체험이 없어요.</p>")
    parts.append("</div></body></html>")
    return "".join(parts)


def zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        if DIR.is_dir():
            for p in sorted(DIR.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(DIR).as_posix())
    return buf.getvalue()


def counts() -> tuple[int, int]:
    """(손그림 짝 수, 결과 수)."""
    if not DIR.is_dir():
        return 0, 0
    return len(list(DIR.rglob("*_drawing.jpg"))), len(list(DIR.rglob("*_result.jpg")))
