"""로그인 없이 해 보는 샘플: 내용이 샘플과 바이트 단위로 같을 때만 통과."""
import os
import tempfile
import unittest
from pathlib import Path

import _paths  # noqa: F401
import guest_samples as gs

SITE = Path(__file__).resolve().parents[3] / "team_yucheital_frontend" / "project_yucheital" / "yuche_frontend" / "vtkjs.html"


class GuestSampleTest(unittest.TestCase):
    def test_random_file_needs_login(self):
        self.assertFalse(gs.is_guest_sample(b"not a sample"))
        self.assertFalse(gs.is_guest_sample(b""))

    def test_served_sample_image_is_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.png"
            p.write_bytes(os.urandom(256))
            self.assertFalse(gs.is_guest_sample(p.read_bytes()))
            gs.remember_file(p)
            self.assertTrue(gs.is_guest_sample(p.read_bytes()))
            self.assertFalse(gs.is_guest_sample(p.read_bytes() + b"x"))   # 한 바이트만 달라도 안 됨

    @unittest.skipUnless(SITE.is_file(), "화면 저장소(vtkjs.html)가 옆에 없음")
    def test_builtin_hashes_match_site(self):
        # 화면의 기본 지형 4개가 만드는 .npy 지문 = 서버 목록 (화면 샘플이 바뀌면 이 테스트가 알려 준다)
        hashes = gs.builtin_from_site(SITE)
        self.assertEqual(len(hashes), 4)
        self.assertEqual(set(hashes.values()), gs.BUILTIN_SHA256)


if __name__ == "__main__":
    unittest.main()
