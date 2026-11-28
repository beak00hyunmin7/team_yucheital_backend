"""전시가 끝난 뒤 모은 체험 자료(data/expo 전체)를 지운다.

    python clear_expo_data.py

몇 건이 있는지 보여 주고, y 를 입력해야 지운다 (그 밖의 입력은 취소).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))
import expo  # noqa: E402


def main() -> int:
    if not expo.DIR.is_dir():
        print(f"지울 자료가 없습니다: {expo.DIR}")
        return 0
    drawings, results = expo.counts()
    days = sorted(p.name for p in expo.DIR.iterdir() if p.is_dir())
    print(f"폴더: {expo.DIR}")
    print(f"날짜 {len(days)}개 ({', '.join(days) or '-'}) · 손그림 {drawings}건 · 결과 {results}건")
    answer = input("전부 지울까요? 되돌릴 수 없습니다. 지우려면 y 입력: ").strip().lower()
    if answer != "y":
        print("취소했습니다. 아무것도 지우지 않았습니다.")
        return 1
    shutil.rmtree(expo.DIR)
    print("지웠습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
