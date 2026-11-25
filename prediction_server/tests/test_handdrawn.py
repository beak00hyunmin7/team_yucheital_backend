"""손그림 모드: 실제 학습지 사진(마커로 그리고 폰으로 찍은 것)을 층수대로 읽는지.

photo1 산 셋(2층·2층·3층) / photo2 봉우리 둘인 큰 산 하나, 비스듬히 찍음(4층)
photo3 산 둘 + 튄 잉크 점, 비스듬히 찍음(3층) / photo4 산 하나(4층, 왼쪽 위 빈 네모 구멍이 작음)
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


def _read(name):
    return hd.read_drawing((SAMPLES / name).read_bytes())


def _hills(terrain):
    """따로 떨어진 산 개수 (첫 동그라미 안쪽 덩어리 수)."""
    return ndi.label(terrain > hd.LEVEL_M * 0.5)[1]


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
        base = hd.read_drawing(raw)[0]
        for ang in (90, 180, 270):
            buf = io.BytesIO()
            Image.open(io.BytesIO(raw)).rotate(ang, expand=True).save(buf, "PNG")
            t = hd.read_drawing(buf.getvalue())[0]
            self.assertGreater(np.corrcoef(base.ravel(), t.ravel())[0, 1], 0.98, ang)

    def test_top_of_photo_is_north(self):
        # photo1: 아래쪽 산이 가장 높다(3층). 서버 격자는 0행이 남쪽(그림 아래)이므로 앞쪽 행에 최고점
        terrain = _read("photo1.jpg")[0]
        r, _ = np.unravel_index(np.argmax(terrain), terrain.shape)
        self.assertLess(r, 48)

    def test_blank_paper_is_rejected(self):
        buf = io.BytesIO()
        Image.new("RGB", (800, 600), "white").save(buf, "PNG")
        with self.assertRaises(hd.HandDrawnError):
            hd.read_drawing(buf.getvalue())

    def test_not_an_image_is_rejected(self):
        with self.assertRaises(hd.HandDrawnError):
            hd.read_drawing(b"not an image")


if __name__ == "__main__":
    unittest.main()
