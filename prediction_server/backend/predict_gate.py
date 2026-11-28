"""예측 순서 문: 한 번에 하나씩, 첫 화면 요청 우선 (app.py 가 쓴다. torch 없이 테스트할 수 있게 따로 둔다)."""
from __future__ import annotations

import threading
import time


class PriorityGate:
    """예측을 한 번에 하나씩 처리하되, 첫 화면 요청을 시간 변화(light) 요청보다 먼저 처리하는 문.

    한 사람의 예측은 '첫 화면 1번 + 시간 변화 5번'이다. 그냥 도착 순서대로 처리하면 앞사람의 시간 변화 5장이
    끝날 때까지 뒷사람은 첫 화면조차 못 본다. 첫 화면(light 가 아닌 요청)이 기다리고 있으면 light 요청은
    양보한다 -> 여러 명이 동시에 눌러도 각자 첫 화면은 빨리 뜨고, 시간 변화는 그 뒤에 채워진다.
    (스레드로 동시에 돌리면 GIL 때문에 오히려 느려져서 여전히 한 번에 하나씩이다)
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._busy = False
        self._high_waiting = 0
        self.waiting = 0            # 지금 줄 서 있는 요청 수 (/api/health 로 보여 준다)

    def acquire(self, high: bool, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._cv:
            self.waiting += 1
            if high:
                self._high_waiting += 1
            try:
                while self._busy or (not high and self._high_waiting > 0):
                    remain = deadline - time.monotonic()
                    if remain <= 0:
                        return False
                    self._cv.wait(remain)
                self._busy = True
                return True
            finally:
                self.waiting -= 1
                if high:
                    self._high_waiting -= 1

    def release(self) -> None:
        with self._cv:
            self._busy = False
            self._cv.notify_all()
