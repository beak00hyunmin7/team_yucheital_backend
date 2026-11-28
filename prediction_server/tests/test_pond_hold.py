"""웅덩이 보정: 막힌 웅덩이 물만 부피로 붙잡고, 빠질 수 있는 물은 그대로 둔다."""
import unittest

import numpy as np
from scipy import ndimage

import _paths  # noqa: F401
import pond_hold


def _bowl_on_slope(n=32):
    """완만한 경사면 + 가운데 깊이 0.3 m 웅덩이."""
    y, x = np.mgrid[0:n, 0:n].astype(float)
    r = np.hypot(x - n / 2, y - n / 2)
    return (x + y) * 0.01 - np.clip(0.3 * (1 - r / 5), 0, None)


class PitDepthTest(unittest.TestCase):
    def test_slope_has_no_pit(self):
        y, x = np.mgrid[0:20, 0:20].astype(float)
        self.assertEqual(pond_hold.pit_depth(x + y).max(), 0.0)

    def test_closed_bowl_depth(self):
        z = np.ones((9, 9))
        z[4, 4] = 0.2          # 가장자리보다 0.8 m 낮은 구멍
        pd = pond_hold.pit_depth(z)
        self.assertAlmostEqual(pd[4, 4], 0.8)
        self.assertEqual(pd[0, 0], 0.0)

    def test_edge_is_outlet(self):
        z = np.ones((9, 9))
        z[4, :5] = 0.2          # 가장자리까지 이어진 낮은 골 -> 물이 빠진다
        self.assertEqual(pond_hold.pit_depth(z)[4, 4], 0.0)


class HoldTest(unittest.TestCase):
    def setUp(self):
        self.z = _bowl_on_slope()
        self.pits = pond_hold.pit_depth(self.z)
        self.pit = self.pits > pond_hold.MIN_PIT_M
        self.assertGreater(self.pits.max(), 0.1)

    def _pit_volume(self, d):
        return float(np.minimum(d, self.pits)[self.pit].sum())

    def test_keeps_peak_volume_without_inflating(self):
        # 시점마다 젖은 칸이 다르다: 칸별 최댓값을 모으면 부피가 늘어나는 상황
        a = np.zeros(self.z.shape); b = np.zeros(self.z.shape)
        idx = np.argwhere(self.pit)
        a[tuple(idx[::2].T)] = 0.05
        b[tuple(idx[1::2].T)] = 0.05
        cur = np.zeros(self.z.shape)
        out, info, _ = pond_hold.hold(cur, [a, b], self.pits, metric=True)
        peak = max(self._pit_volume(a), self._pit_volume(b))
        self.assertAlmostEqual(self._pit_volume(out), peak, delta=peak * 0.02)
        self.assertTrue(info["applied"])

    def test_capped_at_pit_capacity_and_slope_untouched(self):
        ref = np.full(self.z.shape, 1.0)   # 웅덩이 용량보다 훨씬 많이 고였던 물 -> 넘치는 몫은 빠진다
        cur = np.full(self.z.shape, 0.01)
        out, _, _ = pond_hold.hold(cur, [ref], self.pits, metric=True)
        cap = np.maximum(0.01, self.pits)
        self.assertTrue(np.all(out[self.pit] <= cap[self.pit] + 1e-9))            # 용량은 넘지 않는다
        inner = pond_hold.edge_weight(self.pit) >= 1.0
        np.testing.assert_allclose(out[inner], cap[inner], atol=1e-6)             # 안쪽은 가득
        np.testing.assert_array_equal(out[~self.pit], cur[~self.pit])

    def test_never_lowers(self):
        cur = np.full(self.z.shape, 0.4)
        out, info, _ = pond_hold.hold(cur, [np.zeros(self.z.shape)], self.pits, metric=True)
        np.testing.assert_array_equal(out, cur)
        self.assertFalse(info["applied"])

    def test_relative_height_mode_keeps_volume(self):
        zr = (self.z - self.z.min()) / (self.z.max() - self.z.min())
        pr = pond_hold.pit_depth(zr)
        m = pr > pond_hold.MIN_PIT_REL
        ref = np.where(m, 0.1, 0.0)
        out, _, _ = pond_hold.hold(np.zeros(zr.shape), [ref], pr, metric=False)
        self.assertAlmostEqual(out[m].sum(), ref[m].sum(), places=6)

    def test_keeps_prediction_shape(self):
        # 보정한 물은 평평하게 채우지 않고, 물이 가장 많던 시점의 깊이 모양(골이 깊음)을 따른다
        ref = np.where(self.pit, self.pits * 0.5, 0.0)       # 깊은 곳일수록 깊은 예측
        out, _, _ = pond_hold.hold(np.zeros(self.z.shape), [ref], self.pits, metric=True)
        inner = pond_hold.edge_weight(self.pit) >= 1.0
        ratio = out[inner] / ref[inner]
        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-6)               # 안쪽은 예측 모양 그대로(같은 비율)
        self.assertAlmostEqual(out[self.pit].sum(), ref[self.pit].sum(), delta=ref.sum() * 0.01)

    def test_edge_blends_into_outside(self):
        # 가장자리 칸은 바깥(보정 안 한 예측)에 가깝고, 안쪽으로 갈수록 보정이 다 들어간다
        w = pond_hold.edge_weight(self.pit)
        edge = self.pit & (ndimage.binary_erosion(self.pit) == 0)
        self.assertTrue(np.all(w[edge] < 0.2))
        self.assertTrue(np.all(w[~self.pit] == 0))


class SmallPondTest(unittest.TestCase):
    """[11/28] 작은 웅덩이는 보정 안 하거나, 시간이 지나며 빠지다가 보정이 없어진다."""

    def _bowl(self, n, r):
        y, x = np.mgrid[0:n, 0:n].astype(float)
        return (x + y) * 0.001 - np.clip(0.3 * (1 - np.hypot(x - n / 2, y - n / 2) / r), 0, None)

    def test_tiny_pit_ignored(self):
        z = self._bowl(128, 1.2)                     # 지도의 0.2% 보다 작은 웅덩이
        pits = pond_hold.pit_depth(z)
        ref = np.where(pits > 0, 0.2, 0.0)
        out, info, keep = pond_hold.hold(np.zeros(z.shape), [ref], pits, metric=True, times=[18, 120])
        self.assertFalse(info["applied"])
        self.assertFalse(keep.any())

    def test_small_pit_drains_then_correction_gone(self):
        z = self._bowl(128, 5)                       # 약 0.5% 웅덩이 -> tau 약 21초
        pits = pond_hold.pit_depth(z)
        pit = pits > pond_hold.MIN_PIT_M
        ref = np.where(pit, np.minimum(pits, 0.2), 0.0)
        cur = np.zeros(z.shape)
        out38, _, keep38 = pond_hold.hold(cur, [ref], pits, metric=True, times=[18, 38])
        out59, _, _ = pond_hold.hold(cur, [ref], pits, metric=True, times=[18, 59])
        out120, info, keep120 = pond_hold.hold(cur, [ref], pits, metric=True, times=[18, 120])
        v = lambda d: float(d[pit].sum())
        self.assertGreater(v(out38), v(out59))       # 시간이 지나며 조금씩 준다
        self.assertGreater(v(out59), 0)
        self.assertTrue(keep38.any())
        self.assertEqual(v(out120), 0.0)             # 다 빠지면 보정 없음
        self.assertFalse(keep120.any())
        self.assertEqual(info["drained_small"], 1)

    def test_big_pit_does_not_drain(self):
        z = self._bowl(128, 20)                      # 지도의 수 % -> 빠지지 않는다
        pits = pond_hold.pit_depth(z)
        pit = pits > pond_hold.MIN_PIT_M
        ref = np.where(pit, np.minimum(pits, 0.1), 0.0)
        out, _, _ = pond_hold.hold(np.zeros(z.shape), [ref], pits, metric=True, times=[18, 120])
        self.assertAlmostEqual(float(out[pit].sum()), float(np.minimum(ref, pits)[pit].sum()), delta=float(ref.sum()) * 0.02)


if __name__ == "__main__":
    unittest.main()
