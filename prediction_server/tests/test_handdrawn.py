"""손그림 모드: 실제 학습지 사진(마커로 그리고 폰으로 찍은 것)을 층수대로 읽는지.

photo1 산 셋(2층·2층·3층) / photo2 봉우리 둘인 큰 산 하나, 비스듬히 찍음(4층)
photo3 산 둘 + 튄 잉크 점, 비스듬히 찍음(3층) / photo4 산 하나(4층, 왼쪽 위 빈 네모 구멍이 작음)
demo1~6 시연용 학습지 (11/25). 종이를 90도·180도 돌려 들고 찍었다 (빈 네모가 사진의 오른쪽 위·왼쪽 아래).
  예전 코드로는 2장은 네모 4개를 못 찾고(오른쪽 아래 네모가 회색으로 옅음) 2장은 빈 네모를 못 알아봐서
  같은 종이도 찍을 때마다 결과 방향이 달라졌다.
"""
import io
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

import _paths  # noqa: F401
import handdrawn as hd

SAMPLES = Path(__file__).resolve().parent / "handdrawn_samples"


def _read(name, tilt_m=0.0):
    return hd.read_drawing((SAMPLES / name).read_bytes(), tilt_m=tilt_m)


def _hills(terrain):
    """따로 떨어진 산 개수 (첫 동그라미보다 높은 덩어리 수)."""
    return ndi.label(terrain > hd.LEVEL_M * 1.1)[1]


class HandDrawnPhotoTest(unittest.TestCase):
    def test_levels_and_hills(self):
        cases = {"photo1.jpg": (3, 3), "photo2.jpg": (4, 1), "photo3.jpg": (3, 2), "photo4.jpg": (4, 1)}
        for name, (levels, hills) in cases.items():
            with self.subTest(name):
                terrain, preview, info, warns = _read(name)
                self.assertEqual(terrain.shape, (96, 96))
                self.assertTrue(info["markers_found"])
                self.assertEqual(info["levels"], levels)
                self.assertEqual(_hills(terrain), hills)

    def test_rotated_photo_gives_same_terrain(self):
        raw = (SAMPLES / "photo1.jpg").read_bytes()
        base = hd.read_drawing(raw, tilt_m=0)[0]
        for ang in (90, 180, 270):
            buf = io.BytesIO()
            Image.open(io.BytesIO(raw)).rotate(ang, expand=True).save(buf, "PNG")
            t = hd.read_drawing(buf.getvalue(), tilt_m=0)[0]
            self.assertGreater(np.corrcoef(base.ravel(), t.ravel())[0, 1], 0.98, ang)

    def test_top_of_photo_is_north(self):
        # photo1: 아래쪽 산이 가장 높다(3층). 서버 격자는 0행이 남쪽(그림 아래)이므로 앞쪽 행에 최고점
        terrain = _read("photo1.jpg")[0]
        r, _ = np.unravel_index(np.argmax(terrain), terrain.shape)
        self.assertLess(r, 48)

    def test_plain_is_tilted_not_flat(self):
        # 그림 밖 땅이 완전히 평평하면 AI 가 평지 전체를 물웅덩이로 본다 -> 위쪽 높고 아래쪽 낮게
        flat = _read("photo3.jpg", tilt_m=0)[0]
        tilted = _read("photo3.jpg", tilt_m=15)[0]
        plain = flat == 0
        self.assertGreater(plain.mean(), 0.3)
        self.assertAlmostEqual(float(tilted[0][plain[0]].min()), 0.0, places=3)     # 아래(남)쪽 끝 = 0 m
        self.assertAlmostEqual(float(tilted[-1][plain[-1]].max()), 15.0, places=3)  # 위(북)쪽 끝 = 15 m
        self.assertEqual(len(np.unique(flat[plain])), 1)                           # 평지에 잡음 없음

    def test_blank_paper_is_rejected(self):
        buf = io.BytesIO()
        Image.new("RGB", (800, 600), "white").save(buf, "PNG")
        with self.assertRaises(hd.HandDrawnError):
            hd.read_drawing(buf.getvalue())

    def test_not_an_image_is_rejected(self):
        with self.assertRaises(hd.HandDrawnError):
            hd.read_drawing(b"not an image")


class HandDrawnDemoSheetTest(unittest.TestCase):
    """시연용 학습지 사진: 네모 4개와 빈 네모(방향)를 늘 찾고, 층수가 맞는지."""
    LEVELS = {"demo1.jpg": 4, "demo2.jpg": 4, "demo3.jpg": 4, "demo4.jpg": 3, "demo5.jpg": 4, "demo6.jpg": 3}

    def test_markers_hollow_and_levels(self):
        for name, levels in self.LEVELS.items():
            with self.subTest(name):
                terrain, preview, info, warns = _read(name)
                self.assertTrue(info["markers_found"], warns)
                self.assertTrue(info["hollow_found"], warns)
                self.assertEqual(info["levels"], levels)

    def test_markers_found_at_camera_sizes(self):
        # 웹 화면 카메라 판별은 640~960px 로 줄여서 같은 규칙을 쓴다 -> 줄여도 같은 모서리를 찾아야 한다
        for name in list(self.LEVELS) + ["photo1.jpg", "photo2.jpg", "photo3.jpg"]:
            img = hd._open((SAMPLES / name).read_bytes())
            q0, h0 = hd.find_markers(np.asarray(img).astype(np.int16))
            for W in (960, 640):
                s = W / max(img.size)
                small = img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.BILINEAR)
                with self.subTest(name=name, W=W):
                    found = hd.find_markers(np.asarray(small).astype(np.int16))
                    self.assertIsNotNone(found)
                    q, hollow = found
                    self.assertTrue(hollow)
                    self.assertLess(np.abs(q / np.array(small.size) - q0 / np.array(img.size)).max(), 0.03)

    def test_turned_sheet_gives_same_terrain(self):
        raw = (SAMPLES / "demo4.jpg").read_bytes()
        base = hd.read_drawing(raw, tilt_m=0)[0]
        for ang in (90, 180, 270):
            buf = io.BytesIO()
            Image.open(io.BytesIO(raw)).rotate(ang, expand=True).save(buf, "JPEG", quality=92)
            t = hd.read_drawing(buf.getvalue(), tilt_m=0)[0]
            self.assertGreater(np.corrcoef(base.ravel(), t.ravel())[0, 1], 0.98, ang)

    def test_warm_light_keeps_levels(self):
        # 노란 조명(파랑이 약해짐)을 흉내: 종이 흰색에 맞춘 색 보정으로 층수가 그대로여야 한다
        for name, levels in (("photo1.jpg", 3), ("demo2.jpg", 4)):
            img = Image.open(SAMPLES / name).convert("RGB")
            r, g, b = img.split()
            warm = Image.merge("RGB", (r.point(lambda v: min(255, v * 1.08)), g, b.point(lambda v: v * 0.8)))
            buf = io.BytesIO()
            warm.save(buf, "JPEG", quality=90)
            with self.subTest(name):
                self.assertEqual(hd.read_drawing(buf.getvalue())[2]["levels"], levels)

    def test_small_sheet_in_landscape_frame(self):
        # 노트북 카메라처럼 가로 1280x720 화면 가운데에 종이가 작게(높이 70%) 잡혀도 네모와 방향을 찾는다
        img = hd._open((SAMPLES / "demo1.jpg").read_bytes())
        s = 720 * 0.7 / img.size[1]
        im = img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.BILINEAR)
        frame = Image.new("RGB", (1280, 720), (40, 38, 36))
        frame.paste(im, ((1280 - im.size[0]) // 2, (720 - im.size[1]) // 2))
        found = hd.find_markers(np.asarray(frame).astype(np.int16))
        self.assertIsNotNone(found)
        self.assertTrue(found[1])


if __name__ == "__main__":
    unittest.main()
