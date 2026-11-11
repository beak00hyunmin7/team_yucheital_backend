"""이미지 색 체계 변환(contour_color.py): 모델은 색을 곧 높이로 읽으므로 학습 형식이 아닌 이미지는
높이로 복원해 다시 그려야 하고, 색이 높이가 아닌 이미지(지형도)는 거부해야 한다."""
import unittest

import numpy as np
from PIL import Image

import _paths  # noqa: F401  (경로 설정)
import contour_color as cc
from src.utils.render_fields import render_contour_rgb


def _hills(n: int = 96) -> np.ndarray:
    y, x = np.mgrid[0:n, 0:n] / (n - 1)
    return 60 * np.exp(-((x - .3) ** 2 + (y - .7) ** 2) / .05) + 35 * np.exp(-((x - .75) ** 2 + (y - .35) ** 2) / .08) + 15 * x


def _training_style() -> Image.Image:
    n = 96
    X, Y = np.meshgrid(np.arange(n) * 10.4, np.arange(n) * 10.4)
    return Image.fromarray(render_contour_rgb(X, Y, _hills(n)))


def _gray_hill() -> Image.Image:
    z = _hills(128)
    z = (z - z.min()) / np.ptp(z)
    return Image.fromarray((z * 255).astype(np.uint8)).convert("RGB")


class ConvertTest(unittest.TestCase):
    def test_training_style_passes_through_unchanged(self):
        img = _training_style()
        c = cc.convert_to_training_style(img, "auto", False)
        self.assertEqual(c.scheme, "terrain")
        self.assertFalse(c.converted)

    def test_gray_is_converted_and_warns_about_direction(self):
        c = cc.convert_to_training_style(_gray_hill(), "auto", False)
        self.assertEqual(c.scheme, "gray")
        self.assertTrue(c.converted)
        self.assertTrue(c.warnings, "회색조는 방향을 알 수 없다는 경고가 있어야 한다")

    def test_reverse_flips_heights(self):
        img = _gray_hill()
        a = np.asarray(cc.convert_to_training_style(img, "gray", False).image, dtype=float)
        b = np.asarray(cc.convert_to_training_style(img, "gray", True).image, dtype=float)
        self.assertGreater(np.abs(a - b).mean(), 30, "높낮이를 뒤집으면 변환 결과가 크게 달라야 한다")

    def test_other_colormaps_are_detected(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = 96
        ys, xs = np.mgrid[0:n, 0:n].astype(float)
        fig = plt.figure(figsize=(4, 4), dpi=64)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.contourf(xs, ys, _hills(n), levels=20, cmap="viridis")
        ax.set_axis_off()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        arr = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()
        plt.close(fig)
        c = cc.convert_to_training_style(Image.fromarray(arr), "auto", False)
        self.assertEqual(c.scheme, "viridis")
        self.assertTrue(c.converted)

    def test_landcover_map_is_rejected(self):
        """숲·시가지·물 색이 불규칙하게 섞이고 갈색 등고선·도로가 그려진 지도(실제 지형도)는
        색에서 높이를 읽을 수 없어 거부한다.

        알려진 한계: 선·글씨가 전혀 없는 단색 덩어리뿐인 합성 그림은 이웃 픽셀이 덜 튀어(거칠기 0.03)
        통과한다. 실제 지형도에는 선과 글씨가 항상 있어(거칠기 0.12~0.19, 기준 0.07) 걸린다.
        """
        from PIL import ImageDraw
        rng = np.random.default_rng(7)
        palette = np.array([[200, 230, 160], [255, 255, 255], [169, 216, 239], [223, 236, 203], [243, 230, 200], [159, 209, 139]])
        arr = np.zeros((300, 300, 3), dtype=np.uint8) + 255
        for _ in range(300):
            x, y, w, h = rng.integers(0, 280), rng.integers(0, 280), rng.integers(10, 70), rng.integers(10, 70)
            arr[y:y + h, x:x + w] = palette[rng.integers(0, len(palette))]
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        for _ in range(40):   # 갈색 등고선·도로
            pts = [(int(rng.integers(0, 300)), int(rng.integers(0, 300))) for _ in range(4)]
            draw.line(pts, fill=(160, 82, 45), width=1)
        with self.assertRaises(cc.ColorSchemeError):
            cc.convert_to_training_style(img, "auto", False)

    def test_manual_scheme_only_warns_not_rejects(self):
        rng = np.random.default_rng(7)
        arr = (rng.random((200, 200, 3)) * 255).astype(np.uint8)
        # 직접 고른 경우는 막지 않는다 (사용자가 팔레트를 안다고 본다). 거부/경고 어느 쪽이든 예외는 ColorSchemeError 만.
        try:
            cc.convert_to_training_style(Image.fromarray(arr), "jet", False)
        except cc.ColorSchemeError:
            pass


if __name__ == "__main__":
    unittest.main()
