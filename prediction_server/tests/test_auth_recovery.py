"""회원가입 500 수정: DB 연결이 순간 거부돼도 복구하고, 끝내 안 되면 500 이 아니라 503 으로 안내한다."""
import time
import unittest
from unittest import mock

from fastapi import HTTPException
from sqlalchemy.exc import OperationalError

import _paths  # noqa: F401
import auth


class FakeDB:
    """connection() 이 fail_times 번 실패한 뒤 성공하는 세션 흉내."""

    def __init__(self, fail_times: int):
        self.fail_times, self.calls, self.rollbacks = fail_times, 0, 0

    def connection(self):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise OperationalError("SELECT 1", {}, Exception("(2003) WinError 10061"))

    def rollback(self):
        self.rollbacks += 1


class ConnectWithRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.pokes = []
        p1 = mock.patch.object(auth, "_poke_wsl", lambda: self.pokes.append(1))
        p2 = mock.patch.object(time, "sleep", lambda s: None)   # 테스트가 느려지지 않게
        p1.start(); p2.start()
        self.addCleanup(p1.stop); self.addCleanup(p2.stop)

    def test_healthy_connection_does_not_touch_wsl(self):
        db = FakeDB(0)
        auth.connect_with_recovery(db)
        self.assertEqual((db.calls, len(self.pokes)), (1, 0))

    def test_recovers_after_one_refusal_and_wakes_wsl_once(self):
        db = FakeDB(1)
        auth.connect_with_recovery(db)
        self.assertEqual((db.calls, len(self.pokes)), (2, 1))

    def test_persistent_failure_is_503_not_500(self):
        db = FakeDB(99)
        with self.assertRaises(HTTPException) as cm:
            auth.connect_with_recovery(db)
        self.assertEqual(cm.exception.status_code, 503)
        self.assertEqual(db.calls, 3)


class EmailNormalizeTest(unittest.TestCase):
    def test_email_regex_accepts_normal_and_rejects_bad(self):
        self.assertTrue(auth._EMAIL_RE.match("a@b.com"))
        self.assertFalse(auth._EMAIL_RE.match("nope"))


if __name__ == "__main__":
    unittest.main()
