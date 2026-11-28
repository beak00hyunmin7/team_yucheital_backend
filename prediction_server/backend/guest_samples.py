"""로그인 없이 해 볼 수 있는 샘플 (전시 관람객용).

예측은 로그인해야 하지만, '샘플로 해보기' 는 처음 온 사람도 바로 눌러 볼 수 있어야 한다.
서버는 올라온 파일이 샘플인지 이름으로는 믿을 수 없으므로, 파일 내용(SHA-256)이 샘플과
**바이트 단위로 같을 때만** 로그인 없이 통과시킨다. 그래서 샘플이 아닌 파일은 여전히 로그인이 필요하다.

샘플은 두 종류다.
  1. 화면에 들어 있는 기본 지형 4개 (서울 · 골짜기 · 저지대 · 산지): 화면(vtkjs.html 의 BUILTIN_SAMPLES)이
     높이 값으로 .npy 를 직접 만들어 올린다. 만드는 식이 정해져 있어 결과 바이트도 늘 같다 -> 아래 BUILTIN_SHA256.
     화면의 샘플 값이 바뀌면 `python backend/guest_samples.py <vtkjs.html 경로>` 로 새 값을 뽑아 바꾼다.
  2. 학습 데이터 샘플 그림 (/api/sample/{이름}): 화면이 서버에서 받은 그림을 그대로 다시 올린다
     -> 서버가 그 그림을 내줄 때 지문을 기억해 둔다 (remember_file).

끄기: 환경변수 YUCHE_GUEST_SAMPLES=0 (샘플도 로그인 필요)
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
import struct
import sys
import threading
from pathlib import Path

import numpy as np

ENABLED = os.environ.get("YUCHE_GUEST_SAMPLES", "1") != "0"

# 화면(2026-11-28 판)의 기본 지형 4개가 만드는 .npy 의 SHA-256 (브라우저에서 만든 바이트로 확인)
BUILTIN_SHA256 = {
    "72ec01c7ea6df8e3976b4c9bab3d9bc233afb2a1846994907914a8e754b495da",   # 서울
    "ff494bc9731ad1f933b88d888e4d8a631725c49b65d7aa98bb8daf95c5e6d33d",   # 골짜기
    "340124686b5a7b4a30e3ae0ad62bfb1405587e7620d3179d3291faa6276411da",   # 저지대
    "065ae064117534eb5c018511d2ecf7b4016e521a3f47e2ff65809dc731df80a2",   # 산지
}

_served: set[str] = set()          # 이번 실행에서 /api/sample 로 내준 그림들의 지문
_served_lock = threading.Lock()
_MAX_SERVED = 5000


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def remember_file(path: Path) -> None:
    """/api/sample 로 내주는 그림의 지문을 기억한다 (화면은 받은 그대로 다시 올린다)."""
    try:
        h = sha256(Path(path).read_bytes())
    except OSError:
        return
    with _served_lock:
        if len(_served) < _MAX_SERVED:
            _served.add(h)


def is_guest_sample(raw: bytes) -> bool:
    if not ENABLED or not raw:
        return False
    h = sha256(raw)
    if h in BUILTIN_SHA256:
        return True
    with _served_lock:
        return h in _served


# ---- 화면의 기본 지형 -> .npy 바이트 (화면 sampleNpyFile 과 같은 식). 지문 다시 뽑을 때 ----
SAMPLE_N = 96


def builtin_npy_bytes(lo: float, hi: float, b64: str) -> bytes:
    q = np.frombuffer(base64.b64decode(b64), dtype="<u2").astype(np.float64)
    z = (lo + (q / 65535.0) * (hi - lo)).astype("<f4")            # JS: Float32Array 에 float64 값을 넣음
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }" % (SAMPLE_N, SAMPLE_N)
    pre = 10
    header += " " * ((64 - ((pre + len(header) + 1) % 64)) % 64) + "\n"
    return b"\x93NUMPY" + bytes([1, 0]) + struct.pack("<H", len(header)) + header.encode("latin-1") + z.tobytes()


def builtin_from_site(html_path: str | Path) -> dict[str, str]:
    """vtkjs.html 의 BUILTIN_SAMPLES -> {이름: SHA-256}."""
    text = Path(html_path).read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r'name:\s*"([^"]+)".*?lo:\s*([-\d.]+),\s*hi:\s*([-\d.]+),\s*b64:\s*"([A-Za-z0-9+/=]+)"', text):
        out[m.group(1)] = sha256(builtin_npy_bytes(float(m.group(2)), float(m.group(3)), m.group(4)))
    return out


if __name__ == "__main__":
    for name, h in builtin_from_site(sys.argv[1]).items():
        print(f'    "{h}",   # {name}')
