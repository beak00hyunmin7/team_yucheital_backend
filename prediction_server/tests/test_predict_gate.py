"""예측 순서 문: 한 번에 하나씩 + 첫 화면 요청이 시간 변화(light) 요청보다 먼저."""
import threading
import time
import unittest

import _paths  # noqa: F401
from predict_gate import PriorityGate


class PriorityGateTest(unittest.TestCase):
    def test_one_at_a_time(self):
        g, active, peak, lock = PriorityGate(), [0], [0], threading.Lock()

        def work(high):
            self.assertTrue(g.acquire(high, 5))
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.02)
            with lock:
                active[0] -= 1
            g.release()
        ts = [threading.Thread(target=work, args=(i % 2 == 0,)) for i in range(12)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(peak[0], 1)

    def test_first_screen_goes_before_waiting_light(self):
        g, order = PriorityGate(), []
        self.assertTrue(g.acquire(True, 1))          # 누가 예측 중
        def run(name, high, delay):
            time.sleep(delay)
            self.assertTrue(g.acquire(high, 5))
            order.append(name)
            time.sleep(0.01)
            g.release()
        ts = [threading.Thread(target=run, args=(f"light{i}", False, 0.0)) for i in range(3)]
        ts.append(threading.Thread(target=run, args=("first", True, 0.05)))   # 늦게 왔지만 첫 화면
        [t.start() for t in ts]
        time.sleep(0.15)
        self.assertEqual(g.waiting, 4)
        g.release()
        [t.join() for t in ts]
        self.assertEqual(order[0], "first")
        self.assertEqual(g.waiting, 0)

    def test_timeout(self):
        g = PriorityGate()
        self.assertTrue(g.acquire(True, 1))
        t0 = time.monotonic()
        self.assertFalse(g.acquire(False, 0.1))
        self.assertGreaterEqual(time.monotonic() - t0, 0.09)
        self.assertEqual(g.waiting, 0)
        g.release()


if __name__ == "__main__":
    unittest.main()
