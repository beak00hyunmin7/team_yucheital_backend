"""
확장자별 검증용 테스트 입력 생성기 (마무리 점검용).

make_test_input.py 와 **같은 지형 하나**를 기준으로, 서버(terrain_io.py)가 받는
모든 확장자 + 축척/비트수/에러 케이스 파일을 한 번에 만든다.
결과 폴더에 목록.csv (파일 / 무엇을 확인하나 / 넣을 모드 / 기대 결과) 도 같이 저장한다.

사용법 (prediction_server 폴더에서):
    .venv\\Scripts\\python.exe make_all_test_inputs.py --out C:\\yuche\\test_input
"""
import argparse
import csv
import os
import struct
import sys

import numpy as np
from PIL import Image
from PIL.TiffImagePlugin import ImageFileDirectory_v2

HERE = os.path.dirname(os.path.abspath(__file__))
AI_MODEL = os.path.join(HERE, "ai", "terrainFlowSim", "ai_model")
sys.path.insert(0, AI_MODEL)
sys.path.insert(0, HERE)

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from make_test_input import DX, N, make_terrain  # noqa: E402  (같은 지형 생성 로직 재사용)
from src.utils.render_fields import render_contour_rgb  # noqa: E402

ROWS = []  # 목록.csv 내용


def add(path, what, mode, expect):
    ROWS.append([os.path.relpath(path, OUT), what, mode, expect])


def to_u8(z):
    return np.round((z - z.min()) / np.ptp(z) * 255).astype(np.uint8)


def to_u16(z):
    return np.round((z - z.min()) / np.ptp(z) * 65535).astype(np.uint16)


def write_asc(path, z, cellsize, nodata=None):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"ncols {z.shape[1]}\nnrows {z.shape[0]}\n")
        f.write("xllcorner 0.0\nyllcorner 0.0\n")
        f.write(f"cellsize {cellsize}\n")
        if nodata is not None:
            f.write(f"NODATA_value {nodata}\n")
        np.savetxt(f, z, fmt="%.3f")


def write_geotiff_latlon(path, z, lon0=127.0, lat0=37.5, deg=1 / 3600):
    """위경도 좌표계(EPSG:4326) GeoTIFF. 셀 크기가 '도' 단위(1/3600° ≈ 30 m)."""
    ifd = ImageFileDirectory_v2()
    ifd[33550] = (deg, deg, 0.0)                       # ModelPixelScale
    ifd.tagtype[33550] = 12                            # DOUBLE
    ifd[33922] = (0.0, 0.0, 0.0, lon0, lat0, 0.0)      # ModelTiepoint
    ifd.tagtype[33922] = 12
    # GeoKeyDirectory: 버전, 키 3개 (모델=지리좌표, 래스터=PixelIsArea, EPSG 4326)
    ifd[34735] = (1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326)
    ifd.tagtype[34735] = 3                             # SHORT
    Image.fromarray(z.astype(np.float32), mode="F").save(path, tiffinfo=ifd)


def main():
    global OUT
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(HERE, "test_input"))
    p.add_argument("--big-mb", type=float, default=26.0, help="용량 초과 테스트 파일 크기(MB)")
    args = p.parse_args()
    OUT = os.path.abspath(args.out)

    d = {k: os.path.join(OUT, k) for k in
         ("01_확장자_같은지형", "02_tif_종류", "03_축척_크기", "04_모양", "05_에러_입력")}
    for v in d.values():
        os.makedirs(v, exist_ok=True)

    z = make_terrain()                     # 96x96, dx 10.4167 m (=1 km), 고도 60~100 m
    X, Y = np.meshgrid(np.arange(N) * DX, np.arange(N) * DX)
    ok = "정상 예측 (01 기준 결과와 비슷)"

    # ---------------- 01 같은 지형, 확장자만 다르게 ----------------
    o = d["01_확장자_같은지형"]
    f = os.path.join(o, "contour.png")
    Image.fromarray(render_contour_rgb(X, Y, z)).save(f)
    add(f, "기준: 학습과 같은 등고선 그림", "등고선 이미지", "정상 예측 (이 결과가 비교 기준)")

    f = os.path.join(o, "terrain.npy"); np.save(f, z)
    add(f, "numpy 배열", "등고선 데이터", ok)
    f = os.path.join(o, "terrain.csv"); np.savetxt(f, z, delimiter=",", fmt="%.3f")
    add(f, "콤마 구분 숫자", "등고선 데이터", ok)
    f = os.path.join(o, "terrain_header.csv")
    with open(f, "w", encoding="utf-8") as fh:
        fh.write(",".join(f"c{i}" for i in range(N)) + "\n")
        np.savetxt(fh, z, delimiter=",", fmt="%.3f")
    add(f, "첫 줄에 헤더(글자)가 있는 csv", "등고선 데이터", ok + " (헤더 자동 무시)")
    f = os.path.join(o, "terrain.txt"); np.savetxt(f, z, delimiter=" ", fmt="%.3f")
    add(f, "공백 구분 숫자", "등고선 데이터", ok)
    f = os.path.join(o, "terrain.dat"); np.savetxt(f, z, delimiter="\t", fmt="%.3f")
    add(f, "탭 구분 숫자", "등고선 데이터", ok)
    f = os.path.join(o, "terrain.asc"); write_asc(f, z, DX)
    add(f, "ESRI ASCII Grid (cellsize 10.4 m)", "등고선 데이터", ok + ", 셀 크기 10.4 m로 읽힘")
    f = os.path.join(o, "terrain.grd"); write_asc(f, z, DX)
    add(f, "ESRI ASCII Grid (.grd 확장자)", "등고선 데이터", ok)
    f = os.path.join(o, "terrain_float32.tif")
    Image.fromarray(z.astype(np.float32), mode="F").save(f)
    add(f, "32비트 실수 TIFF (실제 DEM 표준)", "등고선 데이터", ok)
    f = os.path.join(o, "terrain_float32.tiff")
    Image.fromarray(z.astype(np.float32), mode="F").save(f)
    add(f, "32비트 실수 TIFF (.tiff 확장자)", "등고선 데이터", ok)
    f = os.path.join(o, "heightmap_8bit.png"); Image.fromarray(to_u8(z)).save(f)
    add(f, "회색조 높이맵 8비트 (밝을수록 높음)", "등고선 데이터", ok + " (고도는 0~255로 바뀜)")
    f = os.path.join(o, "heightmap_16bit.png"); Image.fromarray(to_u16(z)).save(f)
    add(f, "회색조 높이맵 16비트", "등고선 데이터", ok)
    f = os.path.join(o, "heightmap.jpg"); Image.fromarray(to_u8(z)).save(f, quality=90)
    add(f, "회색조 높이맵 JPG (압축 손실)", "등고선 데이터", ok + " (약간 거칠 수 있음)")
    f = os.path.join(o, "heightmap.jpeg"); Image.fromarray(to_u8(z)).save(f, quality=90)
    add(f, "회색조 높이맵 JPEG 확장자", "등고선 데이터", ok)
    f = os.path.join(o, "heightmap.webp"); Image.fromarray(to_u8(z)).save(f, lossless=True)
    add(f, "회색조 높이맵 WEBP", "등고선 데이터", ok)

    # ---------------- 02 tif 종류 ----------------
    o = d["02_tif_종류"]
    f = os.path.join(o, "tif_uint8_256단계.tif"); Image.fromarray(to_u8(z)).save(f)
    add(f, "8비트 tif: 고도를 256단계로만 저장", "등고선 데이터", "예측은 되지만 고도 0~255로 바뀜, 경고 없음")
    f = os.path.join(o, "tif_int16_미터.tif")
    Image.fromarray(np.round(z).astype(np.int16)).save(f)
    add(f, "16비트 정수 tif (1 m 단위로 반올림)", "등고선 데이터", ok + " (1 m 계단)")
    f = os.path.join(o, "tif_uint16.tif"); Image.fromarray(to_u16(z)).save(f)
    add(f, "16비트 부호없는 tif", "등고선 데이터", ok)
    zn = z.copy(); zn[:12, :20] = -9999.0
    f = os.path.join(o, "tif_nodata_-9999.tif")
    Image.fromarray(zn.astype(np.float32), mode="F").save(f)
    add(f, "빈칸을 -9999로 채운 tif (DEM에서 흔함)", "등고선 데이터",
        "문제 예상: -9999를 진짜 고도로 읽어 왼쪽 위가 깊은 구덩이가 됨")
    zn = z.copy(); zn[:12, :20] = np.nan
    f = os.path.join(o, "tif_nodata_NaN.tif")
    Image.fromarray(zn.astype(np.float32), mode="F").save(f)
    add(f, "빈칸을 NaN으로 채운 tif", "등고선 데이터", "빈칸을 최저 고도로 채움 → 왼쪽 위에 물이 고일 수 있음")
    f = os.path.join(o, "tif_위경도_EPSG4326.tif"); write_geotiff_latlon(f, z)
    add(f, "위경도 좌표 GeoTIFF (셀 크기가 '도' 단위)", "등고선 데이터",
        "예측은 됨. 서버가 좌표 정보를 아예 안 읽음(셀 크기 모름)")
    f = os.path.join(o, "tif_RGB_컬러.tif")
    Image.fromarray(render_contour_rgb(X, Y, z)).save(f)
    add(f, "등고선 컬러 그림을 tif로 저장", "등고선 데이터",
        "문제 예상: 빨강 채널만 고도로 읽음 → 엉뚱한 지형")

    # ---------------- 03 축척 / 크기 ----------------
    o = d["03_축척_크기"]
    f = os.path.join(o, "asc_셀1m_96m지도.asc"); write_asc(f, z, 1.0)
    add(f, "같은 지형, 셀 1 m (지도 96 m)", "등고선 데이터",
        "축척 확인: 결과가 01과 똑같으면 축척을 무시하는 것 (경고 필요)")
    f = os.path.join(o, "asc_셀30m_2.9km지도.asc"); write_asc(f, z, 30.0)
    add(f, "같은 지형, 셀 30 m (지도 2.9 km)", "등고선 데이터",
        "축척 확인: 결과가 01과 똑같으면 축척을 무시하는 것 (경고 필요)")
    big = np.kron(z, np.ones((21, 21), dtype=np.float32))  # 2016x2016
    f = os.path.join(o, "npy_큰격자_2016x2016.npy"); np.save(f, big)
    add(f, "큰 격자 2016x2016 (512로 자동 축소)", "등고선 데이터", ok + " (축소 후), 시간 기록")
    big_asc = np.kron(z, np.ones((6, 6), dtype=np.float32))  # 576x576, 셀 10 m → 5.8 km
    f = os.path.join(o, "asc_큰지도_5.8km.asc"); write_asc(f, big_asc, 10.0)
    add(f, "셀 10 m, 576x576 → 5.8 km 지도", "등고선 데이터",
        "도메인 밖(1 km 학습): 경고 없이 예측되면 경고 추가 필요")
    small = z[::8, ::8]  # 12x12
    f = os.path.join(o, "npy_작은격자_12x12.npy"); np.save(f, small)
    add(f, "작은 격자 12x12 (최소 8은 통과)", "등고선 데이터", "예측은 되지만 뭉개짐")
    f = os.path.join(o, "npy_고도차_500m.npy"); np.save(f, (z - z.min()) / np.ptp(z) * 500 + 60)
    add(f, "같은 모양, 고저차 500 m (산지)", "등고선 데이터",
        "결과가 01과 똑같으면 고저차를 무시하는 것 (모델은 등고선 모양만 봄)")

    # ---------------- 04 모양 ----------------
    o = d["04_모양"]
    rect = np.concatenate([z, z[:, ::-1]], axis=1)  # 96x192
    f = os.path.join(o, "npy_직사각형_96x192.npy"); np.save(f, rect)
    add(f, "직사각형 격자 (가로 2배)", "등고선 데이터", "흰 여백 생길 수 있음 (학습은 정사각형)")
    f = os.path.join(o, "npy_3밴드_96x96x3.npy"); np.save(f, np.stack([z, z, z], axis=-1))
    add(f, "3차원 배열 (밴드 3개)", "등고선 데이터", ok + " (첫 밴드만 사용)")
    f = os.path.join(o, "contour_직사각형.png")
    Xr, Yr = np.meshgrid(np.arange(rect.shape[1]) * DX, np.arange(rect.shape[0]) * DX)
    Image.fromarray(render_contour_rgb(Xr, Yr, rect, figsize=(8, 4))).save(f)
    add(f, "직사각형 등고선 그림", "등고선 이미지", "늘어나거나 잘리는지 확인")
    f = os.path.join(o, "contour.png_데이터모드로넣기.png")
    Image.fromarray(render_contour_rgb(X, Y, z)).save(f)
    add(f, "등고선 그림을 '데이터' 모드로 잘못 넣기", "등고선 데이터 (일부러 틀리게)",
        "문제 예상: 그림 밝기를 고도로 읽어 엉뚱한 결과, 경고 없음")
    f = os.path.join(o, "heightmap_이미지모드로넣기.png"); Image.fromarray(to_u8(z)).save(f)
    add(f, "높이맵을 '이미지' 모드로 잘못 넣기", "등고선 이미지 (일부러 틀리게)",
        "색 체계 안 맞음 → 거부되거나 엉뚱한 결과")

    # ---------------- 05 에러 입력 ----------------
    o = d["05_에러_입력"]
    f = os.path.join(o, "빈파일.csv"); open(f, "wb").close()
    add(f, "0바이트 파일", "등고선 데이터", "에러 안내: 숫자 격자를 찾지 못했습니다")
    f = os.path.join(o, "빈파일.png"); open(f, "wb").close()
    add(f, "0바이트 이미지", "등고선 이미지", "에러 안내: 빈 파일입니다 (400)")
    f = os.path.join(o, "너무작음_5x5.npy"); np.save(f, z[:5, :5])
    add(f, "5x5 격자", "등고선 데이터", "에러 안내: 격자가 너무 작습니다")
    f = os.path.join(o, "평평함_전부70m.npy"); np.save(f, np.full((96, 96), 70.0, np.float32))
    add(f, "고도 전부 같음", "등고선 데이터", "에러 안내: 고도 값이 전부 동일합니다")
    f = os.path.join(o, "1차원배열.npy"); np.save(f, z.ravel())
    add(f, "1차원 배열", "등고선 데이터", "에러 안내: 2차원 격자가 아닙니다")
    f = os.path.join(o, "깨진파일.npy")
    with open(f, "wb") as fh:
        fh.write(np.random.default_rng(0).bytes(4096))
    add(f, "내용이 망가진 npy", "등고선 데이터", "에러 안내: .npy 를 읽지 못했습니다")
    f = os.path.join(o, "깨진파일.tif")
    with open(f, "wb") as fh:
        fh.write(b"II*\x00" + np.random.default_rng(1).bytes(4096))
    add(f, "내용이 망가진 tif", "등고선 데이터", "에러 안내: TIFF 를 읽지 못했습니다")
    f = os.path.join(o, "깨진파일.png")
    with open(f, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + np.random.default_rng(2).bytes(4096))
    add(f, "내용이 망가진 png", "등고선 데이터 / 이미지 둘 다",
        "이미지 모드: 안내(422). 데이터 모드: 문제 예상 - 처리 안 된 서버 에러(500)")
    f = os.path.join(o, "asc_헤더크기틀림.asc")
    with open(f, "w", encoding="utf-8") as fh:
        fh.write("ncols 200\nnrows 200\nxllcorner 0\nyllcorner 0\ncellsize 10\n")
        np.savetxt(fh, z, fmt="%.3f")
    add(f, "헤더는 200x200인데 본문은 96x96", "등고선 데이터", "에러 안내: 헤더와 본문 크기가 맞지 않습니다")
    f = os.path.join(o, "csv_줄길이제각각.csv")
    with open(f, "w", encoding="utf-8") as fh:
        for i, row in enumerate(z):
            r = row if i % 3 else row[:50]
            fh.write(",".join(f"{v:.3f}" for v in r) + "\n")
    add(f, "줄마다 칸 수가 다름", "등고선 데이터", "짧은 줄을 조용히 버림 → 지도가 세로로 줄어듦 (경고 없음)")
    f = os.path.join(o, "csv_글자섞임.csv")
    with open(f, "w", encoding="utf-8") as fh:
        for i, row in enumerate(z):
            vals = [f"{v:.3f}" for v in row]
            if i % 10 == 0:
                vals[5] = "abc"
            fh.write(",".join(vals) + "\n")
    add(f, "중간중간 숫자 대신 글자", "등고선 데이터", "글자 있는 줄을 조용히 버림 (경고 없음)")
    f = os.path.join(o, "지원안함.bmp"); Image.fromarray(to_u8(z)).save(f)
    add(f, "지원하지 않는 확장자 .bmp", "등고선 데이터", "에러 안내: 지원하지 않는 확장자입니다")
    f = os.path.join(o, "확장자없음"); np.save(f + ".npy", z); os.replace(f + ".npy", f)
    add(f, "확장자가 없는 파일", "등고선 데이터", "에러 안내: 지원하지 않는 확장자 '없음'")
    f = os.path.join(o, f"용량초과_{int(args.big_mb)}MB.csv")
    target = int(args.big_mb * 1024 * 1024)
    line = (",".join(f"{v:.3f}" for v in z[0]) + "\n").encode()
    with open(f, "wb") as fh:
        written = 0
        while written < target:
            fh.write(line); written += len(line)
    add(f, f"{args.big_mb:.0f} MB 파일", "등고선 데이터", "에러 안내: 25MB 를 초과합니다 (업로드 시간도 기록)")

    with open(os.path.join(OUT, "목록.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["파일", "무엇을 확인하나", "넣을 모드", "기대 결과"])
        w.writerows(ROWS)

    print(f"테스트 파일 {len(ROWS)}개 생성 → {OUT}")
    print("목록.csv 에 파일별 '넣을 모드'와 '기대 결과'가 있습니다.")


if __name__ == "__main__":
    main()
