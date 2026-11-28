"""전시 모드(YUCHE_EXPO) 켬/끔 시험.

  (1) 토큰 없이 예측     (2) 손그림만 expo_id     (3) 결과 저장 후 짝 생성
  (4) 키 없이 갤러리 403  (5) 원본 사진을 저장하지 않음  (6) 10분 지난 번호 거부

실제 예측(AI 모델)을 돌리므로 체크포인트(prediction_server/checkpoints/best.pt 등)가 있어야 한다. 없으면 건너뛴다.
저장 폴더는 임시 폴더로 바꿔서 실제 data/expo 를 건드리지 않는다.
"""
import hashlib
import io
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

import _paths  # noqa: F401
import expo

HERE = Path(__file__).resolve().parent
HAND = HERE / "handdrawn_samples" / "demo1.jpg"

try:
    import app  # noqa: E402  (모델 코드·체크포인트를 찾는다)
    from fastapi import HTTPException, UploadFile
    APP_OK = HAND.is_file()
except Exception as exc:  # noqa: BLE001
    APP_OK = False
    APP_ERR = exc


def _upload(raw: bytes, name: str) -> "UploadFile":
    return UploadFile(io.BytesIO(raw), filename=name)


def _predict(mode, raw, name, light=False, no_map=False):
    return app.predict(mode=mode, rain_mm=50.0, time_s=18.0, dx=10.4166667, color_scheme="auto", reverse=False,
                       light=light, no_analysis=True, no_map=no_map, mark_x=None, mark_y=None,
                       file=_upload(raw, name), _user=None)


def _jpeg(size=(320, 320)) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, (40, 90, 200)).save(buf, "JPEG")
    return buf.getvalue()


@unittest.skipUnless(APP_OK, "예측 서버(app)를 불러올 수 없거나 손그림 시험 사진이 없음")
class ExpoTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._saved = (expo.ENABLED, expo.KEY, expo.DIR, app.auth.db_ready)
        expo.DIR = self.tmp / "expo"
        expo.KEY = "test-key"
        app.auth.db_ready = lambda: True          # DB 가 붙어 있는 상황 (꺼짐이면 로그인 필요)
        self.hand = HAND.read_bytes()
        buf = io.BytesIO()
        np.save(buf, (np.random.RandomState(0).rand(96, 96) * 20).astype(np.float32))
        self.npy = buf.getvalue()

    def tearDown(self):
        expo.ENABLED, expo.KEY, expo.DIR, app.auth.db_ready = self._saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    # (1) 토큰 없이 예측
    def test_1_no_token(self):
        expo.ENABLED = False
        with self.assertRaises(HTTPException) as cm:
            app._auth_gate("")
        self.assertEqual(cm.exception.status_code, 401)
        expo.ENABLED = True
        self.assertIsNone(app._auth_gate(""))

    # (2) 손그림 첫 화면 요청만 expo_id
    def test_2_expo_id_only_for_hand_first_screen(self):
        expo.ENABLED = True
        self.assertIn("expo_id", _predict("hand", self.hand, "photo.jpg"))
        self.assertNotIn("expo_id", _predict("terrain", self.npy, "t.npy"))
        self.assertNotIn("expo_id", _predict("hand", self.hand, "photo.jpg", light=True))     # 시간 변화
        self.assertNotIn("expo_id", _predict("hand", self.hand, "photo.jpg", no_map=True))    # 기준 시점·분석 지도
        series = app.predict_series(mode="hand", rain_mm=50.0, times="38", dx=10.4166667, color_scheme="auto",
                                    reverse=False, file=_upload(self.hand, "photo.jpg"), _user=None)
        self.assertNotIn("expo_id", series["frames"]["38"])
        self.assertEqual(len(list(expo.DIR.rglob("*_drawing.jpg"))), 1)   # 위 다섯 요청 중 손그림 첫 화면 하나만 저장
        expo.ENABLED = False
        self.assertNotIn("expo_id", _predict("hand", self.hand, "photo.jpg"))
        self.assertEqual(len(list(expo.DIR.rglob("*_drawing.jpg"))), 1)   # 꺼짐이면 저장 안 함

    # (3) 결과 저장 -> 짝 + records.jsonl, 한 번만, 3MB 넘으면 거부, 전시 모드 꺼짐이면 404
    def test_3_result_pair(self):
        expo.ENABLED = True
        eid = _predict("hand", self.hand, "photo.jpg")["expo_id"]
        meta = json.dumps({"rain_mm": 50, "drain": {"x": 0.4, "y": 0.6}, "risk_area_pct": 12.5, "risk_threshold": {"abs_cm": 3}})
        self.assertEqual(app.expo_result(expo_id=eid, meta=meta, result_image=_upload(_jpeg(), "r.jpg")), {"ok": True})
        day = next(expo.DIR.iterdir())
        names = sorted(p.name for p in day.iterdir())
        self.assertEqual(names, sorted([f"{eid}_drawing.jpg", f"{eid}_read.png", f"{eid}_result.jpg", "records.jsonl"]))
        rec = json.loads((day / "records.jsonl").read_text(encoding="utf-8").strip())
        self.assertEqual((rec["expo_id"], rec["rain_mm"], rec["risk_area_pct"]), (eid, 50.0, 12.5))
        self.assertEqual(rec["files"]["result"], f"{eid}_result.jpg")
        with self.assertRaises(HTTPException) as cm:   # 같은 번호 두 번째
            app.expo_result(expo_id=eid, meta="{}", result_image=_upload(_jpeg(), "r.jpg"))
        self.assertEqual(cm.exception.status_code, 400)
        eid2 = _predict("hand", self.hand, "photo.jpg")["expo_id"]
        with self.assertRaises(HTTPException) as cm:   # 3MB 초과
            app.expo_result(expo_id=eid2, meta="{}", result_image=_upload(b"\xff" * (3 * 1024 * 1024 + 10), "r.jpg"))
        self.assertEqual(cm.exception.status_code, 400)
        with self.assertRaises(HTTPException) as cm:   # 없는 번호
            app.expo_result(expo_id="0" * 20, meta="{}", result_image=_upload(_jpeg(), "r.jpg"))
        self.assertEqual(cm.exception.status_code, 400)
        expo.ENABLED = False
        with self.assertRaises(HTTPException) as cm:
            app.expo_result(expo_id=eid2, meta="{}", result_image=_upload(_jpeg(), "r.jpg"))
        self.assertEqual(cm.exception.status_code, 404)

    # (4) 키 없이 갤러리 403 (전시 모드 켬/끔 둘 다), 맞는 키면 결과 없는 짝은 '결과 없음'
    def test_4_gallery_key(self):
        for on in (True, False):
            expo.ENABLED = on
            for k in ("", "wrong"):
                with self.assertRaises(HTTPException) as cm:
                    app.expo_gallery(key=k)
                self.assertEqual(cm.exception.status_code, 403)
                with self.assertRaises(HTTPException):
                    app.expo_gallery_zip(key=k)
        expo.ENABLED = True
        _predict("hand", self.hand, "photo.jpg")
        page = app.expo_gallery(key="test-key").body.decode("utf-8")
        self.assertIn("결과 없음", page)
        expo.KEY = ""   # 키를 설정하지 않았으면 누구도 못 본다
        with self.assertRaises(HTTPException):
            app.expo_gallery(key="")

    # (5) 원본 사진·파일 이름을 저장하지 않음 (편 손그림 1024 와 읽은 그림만)
    def test_5_no_original_photo(self):
        from PIL import Image
        expo.ENABLED = True
        eid = _predict("hand", self.hand, "내얼굴_사진.jpg")["expo_id"]
        files = list(expo.DIR.rglob("*"))
        up = hashlib.sha256(self.hand).hexdigest()
        for p in files:
            if p.is_file():
                self.assertNotEqual(hashlib.sha256(p.read_bytes()).hexdigest(), up)
                self.assertNotIn("내얼굴", p.name)
                self.assertNotIn(b"\xeb\x82\xb4\xec\x96\xbc\xea\xb5\xb4", p.read_bytes())   # '내얼굴' UTF-8
        drawing = next(expo.DIR.rglob(f"{eid}_drawing.jpg"))
        with Image.open(drawing) as im:
            self.assertEqual(im.size, (1024, 1024))
        self.assertEqual(sorted(p.name for p in files if p.is_file()), sorted([f"{eid}_drawing.jpg", f"{eid}_read.png"]))

    # (6) 발급 후 10분 지난 번호는 거부
    def test_6_expired(self):
        expo.ENABLED = True
        eid = _predict("hand", self.hand, "photo.jpg")["expo_id"]
        drawing = next(expo.DIR.rglob(f"{eid}_drawing.jpg"))
        old = time.time() - expo.RESULT_TTL_S - 5
        os.utime(drawing, (old, old))
        with self.assertRaises(HTTPException) as cm:
            app.expo_result(expo_id=eid, meta="{}", result_image=_upload(_jpeg(), "r.jpg"))
        self.assertEqual(cm.exception.status_code, 400)
        self.assertFalse(any(expo.DIR.rglob("*_result.jpg")))


if __name__ == "__main__":
    unittest.main()
